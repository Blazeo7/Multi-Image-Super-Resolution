import torch
from torch import nn


class HighResNetEncoder(nn.Module):
    def __init__(
        self,
        num_res_blocks,
        in_channels,
        out_channels,
    ):
        super().__init__()
        self.conv1 = nn.Conv2d(2 * in_channels, out_channels, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1)
        self.prelu = nn.PReLU()
        self.residual_blocks = nn.ModuleList(
            [
                nn.Sequential(
                    nn.Conv2d(out_channels, out_channels, 3, padding=1),
                    nn.PReLU(),
                    nn.Conv2d(out_channels, out_channels, 3, padding=1),
                    nn.PReLU(),
                )
                for _ in range(num_res_blocks)
            ]
        )

    def forward(self, x):
        """
        Args:
            x: (B, num_lr, C, H, W)

        Returns:
            (B, num_lr, hidden_dim, H, W)
        """
        B, K, C, H, W = x.shape

        lr_ref, _ = torch.median(x, dim=1, keepdim=True)  # (B, 1, C_in, H, W)
        lr_ref = lr_ref.expand(-1, x.shape[1], -1, -1, -1)  # (B, num_lr, C_in, H, W)
        lr = torch.cat([x, lr_ref], dim=2)  # (B, num_lr, 2*C_in, H, W)

        # collapse dimensions for proper convolution
        lr = lr.view(B * K, 2 * C, H, W)

        f = self.prelu(self.conv1(lr))

        for rblock in self.residual_blocks:
            f = f + rblock(f)

        f = self.conv2(f)

        # restore initial shape
        f = f.view(B, K, -1, H, W)
        return f


class TransformerEncoder(nn.Module):
    def __init__(
        self,
        embed_dim,
        num_heads,
        ffn_hidden_dim,
    ):
        super().__init__()

        self.mha = nn.MultiheadAttention(embed_dim=embed_dim, num_heads=num_heads, batch_first=True)
        self.ffn = nn.Sequential(
            nn.Linear(embed_dim, ffn_hidden_dim),
            nn.GELU(),
            nn.Linear(ffn_hidden_dim, embed_dim),
        )
        self.norm1 = nn.LayerNorm(embed_dim)
        self.norm2 = nn.LayerNorm(embed_dim)

    def forward(self, x, padding_mask):
        """
        Args:
            x: (B, S, embed_dim)
            padding_mask: (B, S) where True indicates positions to be masked

        Returns:
            (B, S, embed_dim)
        """

        residual = x

        # self-attention
        x = self.norm1(x)
        attn_out, _ = self.mha(x, x, x, key_padding_mask=padding_mask)
        residual = residual + attn_out

        # ffn
        x = self.norm2(residual)
        residual = residual + self.ffn(x)

        return residual


class TransformerMISR(nn.Module):
    def __init__(
        self,
        upscale_factor,
        hidden_dim,
        num_transformer_blocks,
        mha_heads,
        num_residual_blocks,
        in_channels=3,
    ):
        super().__init__()

        self.encoder = HighResNetEncoder(
            num_res_blocks=num_residual_blocks,
            in_channels=in_channels,
            out_channels=hidden_dim,
        )
        self.transformers = nn.ModuleList(
            [TransformerEncoder(hidden_dim, mha_heads, ffn_hidden_dim=128) for _ in range(num_transformer_blocks)]
        )
        self.decoder = nn.Sequential(
            nn.Conv2d(hidden_dim, in_channels * (upscale_factor**2), kernel_size=1),
            nn.PReLU(),
            nn.PixelShuffle(upscale_factor),
        )

        self.x0 = nn.Parameter(torch.empty(1, 1, hidden_dim), requires_grad=True)
        nn.init.uniform_(self.x0)

        self.scale_factor = upscale_factor
        self.hidden_dim = hidden_dim
        self.num_residual_blocks = num_residual_blocks
        self.num_channels = in_channels

    def forward(self, x, padding_mask):
        """
        Args:
            x: (B, num_lr, C_in, H, W)
            padding_mask: (B, num_lr, H, W)

        Returns:
            (B, C_out, H*scale_factor, W*scale_factor)
        """

        f = self.encoder.forward(x)  # (B, num_lr, hidden_dim, H, W)

        B, num_lr, hidden_dim, H, W = f.shape
        L = B * H * W

        # collapse for transformer input
        f = f.permute(0, 3, 4, 1, 2)  # (B, H, W, num_lr, hidden_dim)
        f = f.reshape(L, num_lr, hidden_dim)  # (L, num_lr, hidden_dim)

        x0_expanded = self.x0.expand(L, -1, -1)  # (L, 1, hidden_dim)
        f = torch.cat([x0_expanded, f], dim=1)  # (L, num_lr + 1, hidden_dim)

        # adjust padding mask to match the added x0 embedding and reshape for transformer
        padding_mask = torch.cat(
            [torch.zeros((B, 1, H, W), dtype=padding_mask.dtype, device=padding_mask.device), padding_mask], dim=1
        )  # (B, num_lr + 1, H, W)
        padding_mask = padding_mask.permute(0, 2, 3, 1)  # (B, H, W, num_lr + 1)
        padding_mask = padding_mask.reshape(B * H * W, num_lr + 1)  # (L, num_lr + 1)

        for t in self.transformers:
            f = t.forward(f, padding_mask=padding_mask)  # (L, num_lr + 1, hidden_dim)

        # reconstruct
        f_spatial = f.reshape(B, H, W, num_lr + 1, hidden_dim)
        z = f_spatial[:, :, :, 0, :]  # (B, H, W, hidden_dim)

        # sub-pixel convolution
        out = self.decoder(z.permute(0, 3, 1, 2))  # (B, C_out, H*scale_factor, W*scale_factor)
        padding_mask = padding_mask.reshape(B, H, W, num_lr + 1).permute(0, 3, 1, 2)  # (B, num_lr, H, W)

        return out, padding_mask

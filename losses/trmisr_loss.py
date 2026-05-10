import torch
import torch.nn as nn


class TRMISRLoss(nn.Module):
    def __init__(self, max_shift: int = 2, loss_type: str = "l2", y_channel_only: bool = False):
        """
        Shift and brightness invariant loss that works for YCbCr by computing brightness
        offsets per-channel.
        """
        super().__init__()
        self.c = max_shift
        self.loss_type = loss_type.lower()
        self.y_channel_only = y_channel_only

    def forward(self, sr: torch.Tensor, hr: torch.Tensor) -> torch.Tensor:
        """
        Args:
            sr: Model output (B, C, H, W) - Expects YCbCr
            hr: Ground Truth (B, C, H, W) - Expects YCbCr
        """
        # Isolate the Y (Luminance) channel.
        if self.y_channel_only:
            sr = sr[:, 0:1, :, :]
            hr = hr[:, 0:1, :, :]

        c = self.c
        sr_cropped = sr[:, :, c:-c, c:-c]
        H_c, W_c = sr_cropped.shape[-2:]

        best_loss = None

        # Search for the best alignment u,v within the HR ground truth
        for u in range(2 * c + 1):
            for v in range(2 * c + 1):
                # Extract a candidate crop from HR
                hr_crop = hr[:, :, u : u + H_c, v : v + W_c]

                # Compute spatial difference
                diff = hr_crop - sr_cropped

                # Compute Brightness Correction (b_uv) PER CHANNEL
                # dim=(2, 3) averages over H and W, but keeps B and C separate
                # Output shape: (B, C, 1, 1)
                b_uv = diff.mean(dim=(2, 3), keepdim=True)

                # Apply Brightness Correction
                # Y channel gets its luminance shift, Cb/Cr get their color shifts
                corrected_diff = diff - b_uv

                # Calculate Pixel Loss
                if self.loss_type == "l1":
                    loss_uv = torch.abs(corrected_diff).mean(dim=(1, 2, 3))
                else:
                    loss_uv = (corrected_diff**2).mean(dim=(1, 2, 3))

                if best_loss is None:
                    best_loss = loss_uv
                else:
                    best_loss = torch.minimum(best_loss, loss_uv)

        return best_loss.mean()

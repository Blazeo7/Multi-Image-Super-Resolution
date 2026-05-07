from torch import nn


class PixelShuffleNet(nn.Module):
    """
    A simple convolutional network with sub-pixel upsampling.
    """

    def __init__(self, out_channels=3, scale_factor=2, num_frames=5):
        super(PixelShuffleNet, self).__init__()
        in_channels = num_frames * 3  # 15

        self.features = nn.Sequential(
            nn.Conv2d(in_channels, 64, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 64, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 64, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 64, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
        )

        self.upsample = nn.Sequential(
            nn.Conv2d(64, out_channels * (scale_factor**2), kernel_size=3, padding=1),
            nn.PixelShuffle(
                scale_factor
            ),  # (B, out_channels*4, H, W) -> (B, out_channels, H*2, W*2) for scale_factor=2
        )

    def forward(self, x):
        x = self.features(x)
        x = self.upsample(x)
        return x

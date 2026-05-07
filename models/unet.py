import torch
import torch.nn as nn


def double_convolution(in_channels, out_channels):
    return nn.Sequential(
        nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
        nn.ReLU(inplace=True),
        nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
        nn.ReLU(inplace=True)
    )


class UNet(nn.Module):
    def __init__(self, out_channels, num_frames=6):
        super(UNet, self).__init__()
        in_channels = num_frames*3
        self.max_pool2d = nn.MaxPool2d(kernel_size=2, stride=2)
        self.down_conv1 = double_convolution(in_channels, 64)
        self.down_conv2 = double_convolution(64, 128)
        self.down_conv3 = double_convolution(128, 256)
        self.down_conv4 = double_convolution(256, 512)
        self.down_conv5 = double_convolution(512, 1024)
        self.up_transpose1 = nn.ConvTranspose2d(1024, 512, kernel_size=2, stride=2)
        self.up_conv1 = double_convolution(1024, 512)
        self.up_transpose2 = nn.ConvTranspose2d(512, 256, kernel_size=2, stride=2)
        self.up_conv2 = double_convolution(512, 256)
        self.up_transpose3 = nn.ConvTranspose2d(256, 128, kernel_size=2, stride=2)
        self.up_conv3 = double_convolution(256, 128)
        self.up_transpose4 = nn.ConvTranspose2d(128, 64, kernel_size=2, stride=2)
        self.up_conv4 = double_convolution(128, 64)

        self.sr_upsample = nn.ConvTranspose2d(64, 64, kernel_size=2, stride=2)
        self.out = nn.Conv2d(64, out_channels, kernel_size=1)

    def forward(self, x):
        down1 = self.down_conv1(x)
        pool1 = self.max_pool2d(down1)
        down2 = self.down_conv2(pool1)
        pool2 = self.max_pool2d(down2)
        down3 = self.down_conv3(pool2)
        pool3 = self.max_pool2d(down3)
        down4 = self.down_conv4(pool3)
        pool4 = self.max_pool2d(down4)
        down5 = self.down_conv5(pool4)
        up1 = self.up_transpose1(down5)
        x = self.up_conv1(torch.cat([down4, up1], 1))
        up2 = self.up_transpose2(x)
        x = self.up_conv2(torch.cat([down3, up2], 1))
        up3 = self.up_transpose3(x)
        x = self.up_conv3(torch.cat([down2, up3], 1))
        up4 = self.up_transpose4(x)
        x = self.up_conv4(torch.cat([down1, up4], 1))
        x = self.sr_upsample(x)
        return self.out(x)

if __name__ == '__main__':
    input_image = torch.rand((1, 15, 512, 512))
    model = UNet(out_channels=3)
    # Total parameters and trainable parameters.
    total_params = sum(p.numel() for p in model.parameters())
    print(f"{total_params:,} total parameters.")
    total_trainable_params = sum(
        p.numel() for p in model.parameters() if p.requires_grad)
    print(f"{total_trainable_params:,} training parameters.")
    outputs = model(input_image)
    print(outputs.shape)


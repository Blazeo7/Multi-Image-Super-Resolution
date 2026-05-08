import kornia.filters as KF
import kornia.losses as KL
import torch
import torch.nn as nn
import torch.nn.functional as F


class HuberSobelSSIMLoss(nn.Module):
    def __init__(
        self,
        w_huber: float = 1.0,
        w_sobel: float = 0.1,
        w_ssim: float = 0.1,
        huber_delta: float = 0.1,
        window_size: int = 11,
        grayscale: bool = False,
    ):
        super().__init__()
        self.w_huber = w_huber
        self.w_sobel = w_sobel
        self.w_ssim = w_ssim
        self.grayscale = grayscale
        self.huber = nn.HuberLoss(delta=huber_delta)
        self.ssim = KL.SSIMLoss(window_size)
        self.spatial_grad = KF.SpatialGradient()

    def forward(self, sr: torch.Tensor, hr: torch.Tensor):
        loss_huber = self.huber(sr, hr)

        sr_y = sr if self.grayscale else sr[:, 0:1]
        hr_y = hr if self.grayscale else hr[:, 0:1]

        loss_sobel = F.l1_loss(self.spatial_grad(sr_y), self.spatial_grad(hr_y))
        loss_ssim = self.ssim(sr_y, hr_y)
        total_loss = self.w_huber * loss_huber + self.w_sobel * loss_sobel + self.w_ssim * loss_ssim

        return total_loss, {
            "loss": total_loss.detach().mean(),
            "huber": loss_huber.detach().mean(),
            "sobel": loss_sobel.detach().mean(),
            "ssim": loss_ssim.detach().mean(),
        }

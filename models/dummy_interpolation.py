import torch
import torch.nn.functional as F
from torch import nn


class DummyInterpolationAverage(nn.Module):
    def __init__(self, upscale_factor, interpolation_type, take_n_input_samples):
        super().__init__()
        self.upscale_factor = upscale_factor
        self.interpolation_type = interpolation_type
        self.take_n_input_samples = take_n_input_samples

    def forward(self, x, padding_mask=None):
        """
        Args:
            x: (B, num_lr, C_in, H, W)
            padding_mask: (B, num_lr, H, W) - True means masked (invalid)
        """
        B, num_lr, C_in, H, W = x.shape
        num_lr = self.take_n_input_samples if num_lr < self.take_n_input_samples else num_lr
        x_flat = x.view(B * num_lr, C_in, H, W)
        if padding_mask is not None:
            # padding_mask is True for invalid pixels. We want 1.0 for valid, 0.0 for invalid.
            valid_mask_flat = (~padding_mask).view(B * num_lr, 1, H, W).float()
        else:
            # Fallback: if a pixel is exactly 0 across all channels, consider it black/invalid
            valid_mask_flat = (x_flat.abs().sum(dim=1, keepdim=True) > 1e-5).float()
        x_up = F.interpolate(
            x_flat,
            scale_factor=self.upscale_factor,
            mode=self.interpolation_type,
            align_corners=False
        )
        mask_up = F.interpolate(
            valid_mask_flat,
            scale_factor=self.upscale_factor,
            mode='nearest'
        )
        _, _, H_up, W_up = x_up.shape
        x_up = x_up.view(B, num_lr, C_in, H_up, W_up)
        mask_up = mask_up.view(B, num_lr, 1, H_up, W_up)
        x_up_masked = x_up * mask_up
        sum_x = x_up_masked.sum(dim=1)  # Shape: (B, C_in, H_up, W_up)
        count_valid = mask_up.sum(dim=1)  # Shape: (B, 1, H_up, W_up)
        out = sum_x / torch.clamp(count_valid, min=1.0)

        return out

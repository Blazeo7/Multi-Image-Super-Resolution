import torch
from torch import nn


def init_icnr(tensor, upscale_factor=2, initializer=nn.init.kaiming_normal_):
    out_channels, in_channels, h, w = tensor.shape
    assert out_channels % (upscale_factor**2) == 0

    # base kernel shape: one filter per output pixel
    base_out = out_channels // (upscale_factor**2)
    sub_kernel = torch.empty(base_out, in_channels, h, w)
    initializer(sub_kernel)

    # repeat each filter upscale_factor^2 times contiguously
    icnr = sub_kernel.repeat_interleave(upscale_factor**2, dim=0)

    with torch.no_grad():
        tensor.copy_(icnr)

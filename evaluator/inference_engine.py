from omegaconf import DictConfig
import torch
from torch import nn


class InferenceEngine:
    def __init__(self, model: nn.Module, cfg: DictConfig):
        self.cfg = cfg
        self.device = self.cfg.hardware.device
        self.model = model.to(self.device)

        self.model.eval()

    @torch.no_grad()
    def forward(self, lr_stack: torch.Tensor) -> torch.Tensor:
        lr_stack = lr_stack.to(self.device)
        sr = self.model(lr_stack)
        return torch.clamp(sr, 0.0, 1.0)

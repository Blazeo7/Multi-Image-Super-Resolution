import torch
from omegaconf import DictConfig
from torch import nn


class InferenceEngine:
    def __init__(self, model: nn.Module, cfg: DictConfig):
        self.cfg = cfg
        self.model = model
        self.model.eval()

    @torch.no_grad()
    def forward(self, lr_stack: torch.Tensor, masks: torch.Tensor) -> torch.Tensor:
        sr = self.model(lr_stack, masks)
        return torch.clamp(sr, 0.0, 1.0)

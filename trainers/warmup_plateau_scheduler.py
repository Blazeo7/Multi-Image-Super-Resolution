import torch
from torch.optim.lr_scheduler import ReduceLROnPlateau


class WarmupPlateauWrapper:
    def __init__(self, optimizer, warmup_steps, base_lrs: dict, plateau_config: dict):
        self.optimizer = optimizer
        self.warmup_steps = warmup_steps
        self.current_step = 0
        self.base_lrs = base_lrs

        self.plateau_scheduler = ReduceLROnPlateau(optimizer, **plateau_config)

    def step(self, metrics=None):
        # warmup phase
        if self.current_step < self.warmup_steps:
            self.current_step += 1

            progress = self.current_step / self.warmup_steps

            for param_group in self.optimizer.param_groups:
                name = param_group["name"]
                param_group["lr"] = progress * self.base_lrs[name]

        # plateau phase
        elif metrics is not None:
            self.plateau_scheduler.step(metrics)

    def state_dict(self):
        return {
            "plateau_state": self.plateau_scheduler.state_dict(),
            "current_step": self.current_step,
            "warmup_steps": self.warmup_steps,
        }

    def load_state_dict(self, state_dict):
        self.plateau_scheduler.load_state_dict(state_dict["plateau_state"])
        self.current_step = state_dict["current_step"]
        self.warmup_steps = state_dict["warmup_steps"]

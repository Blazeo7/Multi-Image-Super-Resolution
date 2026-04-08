from dataclasses import dataclass, field
from typing import Any, Dict, Optional

import numpy as np


@dataclass
class TrainerConfig:
    max_epochs: int = 100
    max_steps: int = -1
    gradient_accumulation_steps: int = 1

    validation_interval: int = 1
    max_patience: int = 5
    save_max_score: bool = False

    chkpt_interval: int = 1
    ckpt_dir: str = "checkpoints"
    resume_from: Optional[str] = None

    # lr scheduler
    scheduler: Dict[str, Any] = field(
        default_factory=lambda: {
            "_target_": "torch.optim.lr_scheduler.StepLR",
            "step_size": 10,
            "gamma": 0.1,
        }
    )
    step_on_batch: bool = True


class TrainerState:
    def __init__(self, maximize_score) -> None:
        self.epochs_trained = 0
        self.steps_trained = 0

        self.patience = 0

        self.best_score = -np.inf if maximize_score else np.inf
        self.best_score_epoch = 0

    def load_state_dict(self, state_dict: dict) -> None:
        self.epochs_trained = state_dict["epochs_trained"]
        self.steps_trained = state_dict["steps_trained"]

        self.best_score = state_dict["best_score"]
        self.best_score_epoch = state_dict["best_score_epoch"]

        self.patience = state_dict["patience"]

    def state_dict(self) -> dict:
        return {
            "epochs_trained": self.epochs_trained,
            "steps_trained": self.steps_trained,
            "patience": self.patience,
            "best_score": self.best_score,
            "best_score_epoch": self.best_score_epoch,
        }

import random
from pathlib import Path

import accelerate
import hydra
import numpy as np
import torch
from omegaconf import DictConfig
from safetensors.torch import load_file

from evaluator import Evaluator

np.random.seed(42)
random.seed(42)


@hydra.main(version_base=None, config_path="configs", config_name="evaluate")
def main(cfg: DictConfig) -> None:
    """
    Main function to evaluate a model on a test dataset.
    """
    accelerator = accelerate.Accelerator()
    model = hydra.utils.instantiate(cfg.model)

    dataset = hydra.utils.instantiate(cfg.dataset)
    dataloader = hydra.utils.instantiate(cfg.dataloader, dataset=dataset)
    # pass accelerator to logger for distributed logging
    ml_logger = hydra.utils.instantiate(cfg.logger, accelerator=accelerator)

    model, dataset, dataloader = accelerator.prepare(model, dataset, dataloader)

    if cfg.load_ckpt:
        weights_path = Path(cfg.checkpoint_path) / "model.safetensors"
        state_dict = load_file(weights_path, device=str(accelerator.device))
        model.load_state_dict(state_dict)

    evaluator = Evaluator(
        model,
        dataset,
        cfg=cfg.evaluator,
        logger=ml_logger,
        loader=dataloader,
        accelerator=accelerator,
        metric_prefix=cfg.eval_metric_prefix,
    )
    evaluator.run()


if __name__ == "__main__":
    main()

import random

import hydra
import numpy as np
import torch
from accelerate import Accelerator
from omegaconf import DictConfig
from torch.utils.data import ConcatDataset, Subset
from torchinfo import summary

from loggers.base_logger import BaseLogger
from trainers import BasicTrainer as Trainer

random.seed(42)
np.random.seed(42)

torch.cuda.empty_cache()


@hydra.main(config_path="configs", config_name="train_dual_ds", version_base=None)
def main(cfg: DictConfig) -> None:
    accelerator = Accelerator(
        gradient_accumulation_steps=cfg.trainer.gradient_accumulation_steps,
    )

    logger: BaseLogger = hydra.utils.instantiate(cfg.logger)
    logger.log_hyperparameters(dict(cfg))

    model = hydra.utils.instantiate(cfg.model)

    val_dataset = hydra.utils.instantiate(cfg.val_dataset)

    train_ds_main = hydra.utils.instantiate(cfg.train_dataset_main)
    train_ds_scnd = hydra.utils.instantiate(cfg.train_dataset_secondary)

    if cfg.debug:
        train_ds_main = Subset(train_ds_main, range(min(len(train_ds_main), cfg.num_debug_samples)))
        train_ds_scnd = Subset(train_ds_scnd, range(min(len(train_ds_scnd), cfg.num_debug_samples)))
        val_dataset = Subset(val_dataset, range(min(len(val_dataset), cfg.num_debug_samples)))

    train_ds = ConcatDataset([train_ds_main, train_ds_scnd])

    main_idx = list(range(len(train_ds_main)))
    scnd_idx = list(range(len(train_ds_main), len(train_ds_main) + len(train_ds_scnd)))

    sampler = hydra.utils.instantiate(cfg.sampler, main_indices=main_idx, secondary_indices=scnd_idx)

    train_loader = hydra.utils.instantiate(cfg.dataloader, dataset=train_ds, batch_sampler=sampler)
    val_loader = hydra.utils.instantiate(cfg.dataloader, dataset=val_dataset)

    optimizer = hydra.utils.instantiate(cfg.optimizer, params=model.parameter_groups(), _convert_="all")

    loss_fn = hydra.utils.instantiate(cfg.loss)

    model, optimizer, train_loader, val_loader = accelerator.prepare(model, optimizer, train_loader, val_loader)

    trainer = Trainer(
        accelerator=accelerator,
        model=model,
        optimizer=optimizer,
        loss_fn=loss_fn,
        cfg=cfg.trainer,
        logger=logger,
    )

    summary(model)
    trainer.train(train_loader, val_loader)


if __name__ == "__main__":
    main()

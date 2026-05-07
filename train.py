import hydra
from accelerate import Accelerator
from omegaconf import DictConfig
from torchinfo import summary

from loggers.base_logger import BaseLogger
from trainers import BasicTrainer as Trainer


@hydra.main(config_path="configs", config_name="train", version_base=None)
def main(cfg: DictConfig) -> None:
    accelerator = Accelerator(
        gradient_accumulation_steps=cfg.trainer.gradient_accumulation_steps,
    )

    logger: BaseLogger = hydra.utils.instantiate(cfg.logger)
    logger.log_hyperparameters(dict(cfg))

    model = hydra.utils.instantiate(cfg.model)

    train_dataset = hydra.utils.instantiate(cfg.train_dataset)
    val_dataset = hydra.utils.instantiate(cfg.val_dataset)

    train_loader = hydra.utils.instantiate(cfg.dataloader, dataset=train_dataset)
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

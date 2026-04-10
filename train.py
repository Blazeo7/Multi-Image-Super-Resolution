import hydra
from accelerate import Accelerator
from omegaconf import DictConfig

# NOTE: if there will be more trainers, consider using hydra instantiation for them as well
from trainers import BasicTrainer as Trainer


@hydra.main(config_path="configs", config_name="config", version_base=None)
def main(cfg: DictConfig) -> None:
    accelerator = Accelerator(
        gradient_accumulation_steps=cfg.trainer.gradient_accumulation_steps,
    )

    logger = hydra.utils.instantiate(cfg.logger)

    model = hydra.utils.instantiate(cfg.model)

    train_dataset = hydra.utils.instantiate(cfg.dataset, split="train")
    val_dataset = hydra.utils.instantiate(cfg.dataset, split="dev")

    train_loader = hydra.utils.instantiate(cfg.dataloader, dataset=train_dataset)
    val_loader = hydra.utils.instantiate(cfg.dataloader, dataset=val_dataset)

    optimizer = hydra.utils.instantiate(cfg.optimizer, params=model.parameters())

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

    trainer.train(train_loader, val_loader)


if __name__ == "__main__":
    main()

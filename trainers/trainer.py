import torch

from .base_trainer import Trainer


class BasicTrainer(Trainer):
    """
    TODO: rename to unet trainer
    """

    def __init__(self, *args, **kwargs):
        super(BasicTrainer, self).__init__(*args, **kwargs)

    def training_step(self, batch, batch_idx):
        inputs, targets = batch

        self.optimizer.zero_grad()
        preds = self.model.forward(inputs)
        loss = self.loss_fn(preds, targets)
        self.accelerator.backward(loss)
        self.optimizer.step()

        return {"loss": loss.detach()}

    def validation_step(self, batch, batch_idx):
        inputs, targets = batch

        preds = self.model.forward(inputs)
        loss = self.loss_fn(preds, targets)

        return {"loss": loss}

    def validation_epoch_end(self, outputs):
        losses = []
        for o in outputs:
            losses.append(o["loss"])

        avg_loss = torch.mean(torch.stack(losses))
        self.logger.info(f"Epoch validation average loss: {avg_loss}")

        self.logger.log_metrics(
            {
                f"Val Epoch/loss": avg_loss,
                f"Train_Epoch/lr": self.optimizer.param_groups[0]["lr"],
            },
            step=self.state.epochs_trained,
        )

        return avg_loss

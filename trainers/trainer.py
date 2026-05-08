import torch

from .base_trainer import Trainer


class BasicTrainer(Trainer):
    def __init__(self, *args, **kwargs):
        super(BasicTrainer, self).__init__(*args, **kwargs)

    def training_step(self, batch, batch_idx):
        inputs, targets, masks = batch

        preds = self.model.forward(inputs, masks)
        loss_res = self.loss_fn(preds, targets)

        if isinstance(loss_res, tuple):
            loss, _ = loss_res
        elif isinstance(loss_res, dict):
            loss = loss_res["loss"]
        else:
            loss = loss_res

        if torch.isnan(loss) or torch.isinf(loss):
            self.logger.warning(f"Bad loss at batch {batch_idx}: {loss.item()}")
            self.optimizer.zero_grad()
            return {"loss": 0.0}

        self.accelerator.backward(loss)

        return {"loss": loss.detach().item()}

    def validation_step(self, batch, batch_idx):
        inputs, targets, masks = batch

        preds = self.model.forward(inputs, masks)
        loss_res = self.loss_fn(preds, targets)

        # check if loss returns (total_loss, loss_dict)
        if isinstance(loss_res, (tuple, list)):
            _, loss_dict = loss_res
        elif isinstance(loss_res, dict):
            loss_dict = loss_res
        else:
            loss_dict = {"loss": loss_res}

        return {k: v.detach() if torch.is_tensor(v) else torch.tensor(v) for k, v in loss_dict.items()}


    def validation_epoch_end(self, outputs):
        if not outputs:
            return 0.0

        keys = outputs[0].keys()
        metrics = {}

        # aggregate the losses
        for key in keys:
            values = [o[key] for o in outputs if key in o]
            if values:
                avg_val = torch.stack(values).float().mean()
                metrics[f"Val_Epoch/{key}"] = avg_val

        # report lr
        for i, pgroup in enumerate(self.optimizer.param_groups):
            name = pgroup.get("name", f"group_{i}")
            metrics[f"Val_Epoch/lr_{name}"] = pgroup["lr"]

        # report losses
        self.logger.info(f"Epoch validation average loss: {metrics.get('Val_Epoch/loss', 'N/A')}")
        self.logger.log_metrics(
            metrics,
            step=self.state.epochs_trained,
        )

        return metrics.get("Val_Epoch/loss", 0.0)

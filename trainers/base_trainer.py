import inspect
import os
from pathlib import Path
from typing import Optional

import hydra
import torch
import tqdm
from accelerate import Accelerator
from torch.nn import Module
from torch.optim import Optimizer

from loggers.base_logger import BaseLogger

from .utils import TrainerConfig, TrainerState


class Trainer:
    def __init__(
        self,
        accelerator: Accelerator,
        model: Module,
        optimizer: Optimizer,
        loss_fn: Module,
        cfg: TrainerConfig,
        logger: BaseLogger,
    ):
        self.accelerator = accelerator
        self.cfg = cfg

        # Component Setup
        self.model = model
        self.optimizer = optimizer
        self.loss_fn = loss_fn
        self.device = accelerator.device

        # Logging Setup
        self.logger = logger

        # State Management
        self.state = TrainerState(maximize_score=self.cfg.save_max_score)
        self.accelerator.register_for_checkpointing(self.state)

        # Training Control Variables
        self.lr_scheduler = None

    def set_models_to_train_mode(self):
        self.model.train()

    def set_models_to_eval_mode(self):
        self.model.eval()

    def _instantiate_lr_scheduler(self, total_steps):
        scheduler_cfg = self.cfg.scheduler

        # Resolve the actual class/target from the config
        target_cls = hydra.utils.get_class(scheduler_cfg._target_)

        # Get the list of parameters the scheduler's __init__ accepts
        signature = inspect.signature(target_cls.__init__)

        kwargs = {"optimizer": self.optimizer}

        # Only add total_steps if the class explicitly asks for it
        if "total_steps" in signature.parameters:
            kwargs["total_steps"] = total_steps

        return hydra.utils.instantiate(scheduler_cfg, **kwargs)

    def _check_improvement(self, score, best_score):
        if self.cfg.save_max_score:
            return score > best_score
        else:
            return score < best_score

    def _early_stop_check(self, score: float) -> bool:
        if self._check_improvement(score, self.state.best_score):
            self.state.best_score = score
            self.state.best_score_epoch = self.state.epochs_trained
            self.state.patience = 0
            self.logger.info(
                f"New best score: {self.state.best_score:.4f} at epoch {self.state.best_score_epoch}. Saving checkpoint..."
            )
            self._save_checkpoint(self.state.epochs_trained, True)
        else:
            self.state.patience += 1
            self.logger.info(
                f"No improvement in score. Patience counter: {self.state.patience}/{self.cfg.max_patience}"
            )

            if self.state.patience >= self.cfg.max_patience:
                self.logger.info(f"Early stopping triggered after {self.state.patience} epochs without improvement.")
                return True

        return False

    def train(self, train_loader, val_loader):
        """
        Train loop entry point

        Args:
            train_loader: DataLoader for training data
            val_loader: DataLoader for validation data

        Notes:
            ``.zero_grad()``, ``.backward()``, and ``.step()`` are expected to be called inthe ``training_step()`` method of the child implementations.

            The step should be implemented as follows:

            .. code-block:: python
                self.optimizer.zero_grad()
                loss = self.loss_fn(predictions, targets)
                self.accelerator.backward(loss)
                self.optimizer.step()

        Todo:
            Validation loop
            LR Scheduler step
            LR Decay step
        """
        early_stop_mark = torch.zeros(1, device=self.device)

        if self.cfg.resume_from:
            self._load_checkpoint(self.cfg.resume_from)

        steps_per_epoch = len(train_loader)
        update_steps_per_epoch = steps_per_epoch // self.cfg.gradient_accumulation_steps
        update_steps_per_epoch = max(update_steps_per_epoch, 1)

        if self.cfg.max_steps > 0:
            max_steps = self.cfg.max_steps
            max_epochs = self.cfg.max_steps // update_steps_per_epoch + int(
                self.cfg.max_steps % update_steps_per_epoch > 0
            )
        else:
            max_steps = self.cfg.max_epochs * update_steps_per_epoch
            max_epochs = self.cfg.max_epochs

        self.lr_scheduler = self._instantiate_lr_scheduler(max_steps)

        self.logger.info("Training control variables:")
        self.logger.info(f"`steps_per_epoch`: {steps_per_epoch}")
        self.logger.info(f"Gradient accumulation steps: {self.cfg.gradient_accumulation_steps}")
        self.logger.info(f"`update_steps_per_epoch`: {update_steps_per_epoch}")
        self.logger.info(f"`max_steps`: {max_steps}")
        self.logger.info(f"`max_epochs`: {max_epochs}")

        for epoch in range(self.state.epochs_trained + 1, max_epochs + 1):
            self.logger.info(f"{'=' * 9} Epoch {epoch}/{max_epochs} {'=' * 9}")
            self.logger.info("Begin training...")

            self.set_models_to_train_mode()

            trainloader_bar = tqdm.tqdm(
                train_loader,
                desc="",
                dynamic_ncols=True,
                bar_format="{l_bar}{r_bar}",
                disable=not self.accelerator.is_main_process,
                position=0,
                leave=True,
            )

            training_epoch_output = []

            for batch_idx, batch in enumerate(trainloader_bar):
                with self.accelerator.accumulate(self.model):
                    loss_dict = self.training_step(batch, batch_idx)
                    training_epoch_output.append(loss_dict)

                    if self.cfg.step_on_batch and self.accelerator.sync_gradients:
                        self.lr_scheduler.step()

                self.state.steps_trained += 1

            self.state.epochs_trained += 1
            self.training_epoch_end(training_epoch_output)

            if epoch % self.cfg.chkpt_interval == 0 and self.accelerator.is_main_process:
                self._save_checkpoint(self.state.epochs_trained)

            if epoch % self.cfg.validation_interval == 0:
                score = self.validate(val_loader)

                if score is not None:
                    if not self.cfg.step_on_batch:
                        self.lr_scheduler.step()

                    should_stop = self._early_stop_check(score)
                    if should_stop:
                        early_stop_mark += 1

            self.accelerator.wait_for_everyone()

            reduced_early_stop_mark = self.accelerator.reduce(early_stop_mark, reduction="sum")
            if reduced_early_stop_mark != 0:
                break

    @torch.no_grad()
    def validate(self, val_loader):
        self.logger.info("Begin validation...")

        self.set_models_to_eval_mode()
        validation_output = []

        devloader_bar = tqdm.tqdm(
            val_loader,
            desc="",
            dynamic_ncols=True,
            bar_format="{l_bar}{r_bar}",
            disable=not self.accelerator.is_main_process,
            position=0,
            leave=True,
        )

        for batch_idx, batch in enumerate(devloader_bar):
            step_output = self.validation_step(batch, batch_idx)

            # concatenates outputs from each gpu
            gathered_step_output = self.accelerator.gather_for_metrics(step_output)
            validation_output.append(gathered_step_output)

        self.logger.info("Validation steps completed, beginning validation epoch end...")

        if self.accelerator.is_local_main_process:
            score = self.validation_epoch_end(validation_output)
            return score
        else:
            return None

    def training_step(self, batch, batch_idx):
        raise NotImplementedError

    def training_epoch_end(self, training_epoch_output):
        """Implement the logic of the end of a training epoch. Please override this function if you want to do something.

        When the training epoch ends, this function will be called. The input is a list of the loss dict of each step
        in a training epoch. You may want to log the epoch-level training loss here.

        .. code-block:: python
            for epoch in range(start_epoch, end_epoch):
                self.model.train()

                training_epoch_output = []
                for batch, batch_index in dataloader:
                    loss = training_step(batch, batch_idx)
                    training_epoch_output.append(loss)

                training_epoch_end(training_epoch_output)

                save_checkpoint()

                if some_condition:
                    score = validate()
                    if score > best_score:
                        save_checkpoint(best=True)

        Args:
            training_epoch_output: the output of the training epoch. It may a list of the output of each batch.
        """
        loss_keys = training_epoch_output[0].keys()

        # Compute mean loss on all loss items on a epoch
        for key in loss_keys:
            loss_items = [step_out[key] for step_out in training_epoch_output]
            loss_mean = torch.mean(torch.tensor(loss_items))

            if self.accelerator.is_local_main_process:
                self.logger.info(f"Training Loss '{key}' on epoch {self.state.epochs_trained}: {loss_mean}")

                # log training loss
                metrics = {f"Train_Epoch/{key}": loss_mean.item()}

                # log learning rates
                for i, pgroup in enumerate(self.optimizer.param_groups):
                    name = pgroup.get("name", f"group_{i}")
                    metrics[f"Train_Epoch/lr_{name}"] = pgroup["lr"]

                self.logger.log_metrics(
                    metrics,
                    step=self.state.epochs_trained,
                )

    @torch.no_grad()
    def validation_step(self, batch, batch_idx):
        raise NotImplementedError

    def validation_epoch_end(self, outputs):
        raise NotImplementedError

    def _save_checkpoint(self, epoch, best=False):
        ckpts_dir = os.path.join(os.getcwd(), "checkpoints")
        os.makedirs(ckpts_dir, exist_ok=True)

        if best:
            ckpt_path = os.path.join(ckpts_dir, "best")

            # also log to logger if available
            unwrapped_model = self.accelerator.unwrap_model(self.model)
            self.logger.log_model(unwrapped_model, "best_model")
        else:
            ckpt_path = os.path.join(ckpts_dir, str(epoch))

        os.makedirs(ckpt_path, exist_ok=True)
        self.accelerator.save_state(ckpt_path)

    def _load_checkpoint(self, ckpt_path):
        self.accelerator.load_state(ckpt_path)

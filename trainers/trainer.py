from .base_trainer import Trainer


class BasicTrainer(Trainer):
    def __init__(self, *args, **kwargs):
        super(BasicTrainer, self).__init__(*args, **kwargs)

    def training_step(self, batch, batch_idx):
        pass

    def validation_step(self, batch, batch_idx):
        pass

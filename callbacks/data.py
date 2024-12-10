from pytorch_lightning.callbacks import Callback


class IncreaseDataEpoch(Callback):
    def __init__(self):
        super().__init__()

    def on_train_epoch_start(self, trainer, pl_module):
        epoch = pl_module.current_epoch
        if hasattr(trainer.datamodule.train_dataset, "shared_epoch"):
            trainer.datamodule.train_dataset.shared_epoch.set_value(epoch)

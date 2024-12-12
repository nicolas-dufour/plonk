import logging
from pytorch_lightning.callbacks import Callback
import torch

log = logging.getLogger(__name__)


class FixNANinGrad(Callback):
    def __init__(self, monitor):
        super().__init__()
        self.monitor = monitor
        self.continuous_nan_batchs = 0

    def on_before_optimizer_step(self, trainer, pl_module, optimizer) -> None:
        has_nan = []
        is_inf = []
        for name, param in pl_module.named_parameters():
            if param.grad is not None:
                if torch.isnan(param.grad).any():
                    has_nan.append(name)
                if torch.isinf(param.grad).any():
                    is_inf.append(name)
                torch.nan_to_num(param.grad, nan=0, posinf=0, neginf=0, out=param.grad)
        if len(has_nan) > 0:
            print(f"Found NaN in {has_nan}")
        if len(is_inf) > 0:
            print(f"Found Inf in {is_inf}")

    def on_train_batch_end(
        self,
        trainer,
        pl_module,
        outputs,
        batch,
        batch_idx,
    ) -> None:
        logs = trainer.callback_metrics
        i = 0
        found_metric = False
        while i < len(self.monitor) and not found_metric:
            if self.monitor[i] in logs.keys():
                current = logs[self.monitor[i]].squeeze()
                found_metric = True
            else:
                i += 1
        if not found_metric:
            raise ValueError("Asked metric not in logs")

        if not torch.isfinite(current):
            self.continuous_nan_batchs += 1
            if self.continuous_nan_batchs >= 5:
                trainer.should_stop = True
                log.info("Training interrupted because of NaN in {self.monitor}")
        else:
            self.continuous_nan_batchs = 0

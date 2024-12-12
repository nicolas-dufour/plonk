from pytorch_lightning import Callback
import copy
import itertools
import torch
import contextlib
from torch.distributed.fsdp import FullyShardedDataParallel


class EMACallback(Callback):
    def __init__(
        self,
        module_attr_name,
        ema_module_attr_name,
        decay=0.999,
        start_ema_step=0,
        init_ema_random=True,
    ):
        super().__init__()
        self.decay = decay
        self.module_attr_name = module_attr_name
        self.ema_module_attr_name = ema_module_attr_name
        self.start_ema_step = start_ema_step
        self.init_ema_random = init_ema_random

    def on_train_start(self, trainer, pl_module):
        if pl_module.global_step == 0:
            if not hasattr(pl_module, self.module_attr_name):
                raise ValueError(
                    f"Module {pl_module} does not have attribute {self.module_attr_name}"
                )
            if not hasattr(pl_module, self.ema_module_attr_name):
                pl_module.add_module(
                    self.ema_module_attr_name,
                    copy.deepcopy(getattr(pl_module, self.module_attr_name))
                    .eval()
                    .requires_grad_(False),
                )
            self.reset_ema(pl_module)

    def on_train_batch_end(self, trainer, pl_module, outputs, batch, batch_idx):
        if pl_module.global_step == self.start_ema_step:
            self.reset_ema(pl_module)
        elif (
            pl_module.global_step < self.start_ema_step
            and pl_module.global_step % 100 == 0
        ):
            ## slow ema updates for visualisation
            self.update_ema(pl_module, decay=0.9)
        elif pl_module.global_step > self.start_ema_step:
            self.update_ema(pl_module, decay=self.decay)

    def update_ema(self, pl_module, decay=0.999):
        ema_module = getattr(pl_module, self.ema_module_attr_name)
        module = getattr(pl_module, self.module_attr_name)
        context_manager = self.get_model_context_manager(module)
        with context_manager:
            with torch.no_grad():
                ema_params = ema_module.state_dict()
                for name, param in itertools.chain(
                    module.named_parameters(), module.named_buffers()
                ):
                    if name in ema_params:
                        if param.requires_grad:
                            ema_params[name].copy_(
                                ema_params[name].detach().lerp(param.detach(), decay)
                            )

    def get_model_context_manager(self, module):
        fsdp_enabled = is_model_fsdp(module)
        model_context_manager = contextlib.nullcontext()
        if fsdp_enabled:
            model_context_manager = module.summon_full_params(module)
        return model_context_manager

    def reset_ema(self, pl_module):
        ema_module = getattr(pl_module, self.ema_module_attr_name)
        if self.init_ema_random:
            ema_module.init_weights()
        else:
            module = getattr(pl_module, self.module_attr_name)
            context_manager = self.get_model_context_manager(module)
            with context_manager:
                ema_params = ema_module.state_dict()
                for name, param in itertools.chain(
                    module.named_parameters(), module.named_buffers()
                ):
                    if name in ema_params:
                        ema_params[name].copy_(param.detach())


def is_model_fsdp(model: torch.nn.Module) -> bool:
    try:
        if isinstance(model, FullyShardedDataParallel):
            return True

        # Check if model is wrapped with FSDP
        for _, obj in model.named_children():
            if isinstance(obj, FullyShardedDataParallel):
                return True
        return False
    except ImportError:
        return False

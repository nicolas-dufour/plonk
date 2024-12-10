import os
from models.module import DiffGeolocalizer
import hydra
import wandb
from os.path import isfile, join
from shutil import copyfile

import torch

from omegaconf import OmegaConf
from omegaconf import open_dict
from hydra.core.hydra_config import HydraConfig
from hydra.utils import instantiate
from pytorch_lightning.callbacks import LearningRateMonitor
from lightning_fabric.utilities.rank_zero import _get_rank

from models.module import DiffGeolocalizer

torch.set_float32_matmul_precision("high")  # TODO do we need that?

# Registering the "eval" resolver allows for advanced config
# interpolation with arithmetic operations in hydra:
# https://omegaconf.readthedocs.io/en/2.3_branch/how_to_guides.html
OmegaConf.register_new_resolver("eval", eval)


def load_model(cfg, dict_config, wandb_id):
    logger = instantiate(cfg.logger, id=open(wandb_id, "r").read(), resume="allow")
    model = DiffGeolocalizer.load_from_checkpoint(cfg.checkpoint, cfg=cfg.model)
    trainer = instantiate(cfg.trainer, strategy=cfg.trainer.strategy, logger=logger)
    return trainer, model


def hydra_boilerplate(cfg):
    dict_config = OmegaConf.to_container(cfg, resolve=True)
    trainer, model = load_model(cfg, dict_config, cfg.wandb_id)
    return trainer, model


import copy


def generate_datamodules(cfg_):
    for f in os.listdir(cfg_.test_dir):
        cfg = copy.deepcopy(cfg_)
        # open join(f, directory) with OmegaConf
        with open_dict(cfg):
            cfg_new = OmegaConf.load(join(cfg.test_dir, f))
            cfg.datamodule = cfg_new.datamodule
            cfg.dataset = cfg_new.dataset
            cfg.dataset.test_transform = cfg_.dataset.test_transform

        datamodule = instantiate(cfg.datamodule)
        yield datamodule


if __name__ == "__main__":
    import sys

    sys.argv = (
        [sys.argv[0]]
        + ["+pt_model_path=${hydra:runtime.config_sources}"]
        + sys.argv[1:]
    )

    @hydra.main(version_base=None)
    def main(cfg):
        # print(hydra.runtime.config_sources)
        with open_dict(cfg):
            path = cfg.pt_model_path[1]["path"]
            cfg.wandb_id = join(path, "wandb_id.txt")
            cfg.checkpoint = join(path, "last.ckpt")
            cfg.computer.devices = 1

        (
            trainer,
            model,
        ) = hydra_boilerplate(cfg)
        for datamodule in generate_datamodules(cfg):
            model.datamodule = datamodule
            model.datamodule.setup()
            print("Testing on", datamodule.test_dataset.class_name)
            trainer.test(model, datamodule=datamodule)

    main()

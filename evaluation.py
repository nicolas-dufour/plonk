import os
from models.module import DiffGeolocalizer
import hydra
from os.path import join

import torch

from omegaconf import OmegaConf
from omegaconf import open_dict
from hydra.utils import instantiate

from models.eval_best_model import EvalModule

torch.set_float32_matmul_precision("high")

# Registering the "eval" resolver allows for advanced config
# interpolation with arithmetic operations in hydra:
# https://omegaconf.readthedocs.io/en/2.3_branch/how_to_guides.html
OmegaConf.register_new_resolver("eval", eval)


def load_model(cfg, dict_config, wandb_id):
    logger = instantiate(cfg.logger, id=wandb_id, resume="allow")
    log_dict = {"model": dict_config["model"], "dataset": dict_config["dataset"]}
    logger._wandb_init.update({"config": log_dict})
    model = EvalModule(cfg.model)
    trainer = instantiate(
        cfg.trainer, strategy=cfg.trainer.strategy
    )  # , logger=logger)
    return trainer, model


def hydra_boilerplate(cfg):
    dict_config = OmegaConf.to_container(cfg, resolve=True)
    trainer, model = load_model(cfg, dict_config, cfg.wandb_id)
    return trainer, model


import copy


def init_datamodule(cfg):
    datamodule = instantiate(cfg.datamodule)
    return datamodule


if __name__ == "__main__":
    import sys

    sys.argv = (
        [sys.argv[0]]
        + ["+pt_model_path=${hydra:runtime.config_sources}"]
        + sys.argv[1:]
    )

    @hydra.main(config_path="configs", config_name="config", version_base=None)
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
        datamodule = init_datamodule(cfg)
        trainer.test(model, datamodule=datamodule)

    main()

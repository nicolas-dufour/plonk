import sys
import os

from plonk.models.networks.mlp import GeoAdaLNMLP
from huggingface_hub import PyTorchModelHubMixin
import torch
import argparse

models_overrides = {
    "YFCC100M_geoadalnmlp_r3_small_sigmoid_flow_riemann_10M_10M": "YFCC100M_geoadalnmlp_r3_small_sigmoid_flow_riemann",
    "iNaturalist_geoadalnmlp_r3_small_sigmoid_flow_riemann_-7_3": "iNaturalist_geoadalnmlp_r3_small_sigmoid_flow_riemann",
    "osv_5m_geoadalnmlp_r3_small_sigmoid_flow_riemann_-7_3": "osv_5m_geoadalnmlp_r3_small_sigmoid_flow_riemann",
}


class Plonk(
    GeoAdaLNMLP,
    PyTorchModelHubMixin,
    repo_url="https://github.com/nicolas-dufour/plonk",
    tags=["plonk", "geolocalization", "diffusion"],
    license="mit",
):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)


def upload_model(checkpoint_dir, repo_name):
    import hydra
    from omegaconf import OmegaConf

    hydra.initialize(version_base=None, config_path=f"../configs")
    cfg = hydra.compose(
        config_name="config",
        overrides=[
            f"exp={models_overrides[checkpoint_dir]}",
        ],
    )
    network_config = cfg.model.network
    serialized_network_config = OmegaConf.to_container(network_config, resolve=True)
    print(serialized_network_config)
    del serialized_network_config["_target_"]
    model = Plonk(**serialized_network_config)
    ckpt = torch.load(f"checkpoints/{checkpoint_dir}/last.ckpt")
    ckpt_state_dict = ckpt["state_dict"]
    ckpt_state_dict = {k: v for k, v in ckpt_state_dict.items() if "ema_network" in k}
    ckpt_state_dict = {
        k.replace("ema_network.", ""): v for k, v in ckpt_state_dict.items()
    }
    model.load_state_dict(ckpt_state_dict)
    model.push_to_hub(repo_name, commit_message="Fixed ckpt keys")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint_dir", type=str, required=True)
    parser.add_argument("--repo_name", type=str, required=True)
    args = parser.parse_args()
    upload_model(args.checkpoint_dir, args.repo_name)

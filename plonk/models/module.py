from typing import Any
import pytorch_lightning as L
import torch
import torch.nn as nn
from hydra.utils import instantiate
import copy
import pandas as pd
import numpy as np
from tqdm import tqdm
from plonk.utils.manifolds import Sphere
from torch.func import jacrev, vjp, vmap
from torchdiffeq import odeint
from geoopt import ProductManifold, Euclidean
from plonk.models.samplers.riemannian_flow_sampler import ode_riemannian_flow_sampler


class DiffGeolocalizer(L.LightningModule):
    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg
        self.network = instantiate(cfg.network)
        # self.network = torch.compile(self.network, fullgraph=True)
        self.input_dim = cfg.network.input_dim
        self.train_noise_scheduler = instantiate(cfg.train_noise_scheduler)
        self.inference_noise_scheduler = instantiate(cfg.inference_noise_scheduler)
        self.data_preprocessing = instantiate(cfg.data_preprocessing)
        self.cond_preprocessing = instantiate(cfg.cond_preprocessing)
        self.preconditioning = instantiate(cfg.preconditioning)

        self.ema_network = copy.deepcopy(self.network).requires_grad_(False)
        self.ema_network.eval()
        self.postprocessing = instantiate(cfg.postprocessing)
        self.val_sampler = instantiate(cfg.val_sampler)
        self.test_sampler = instantiate(cfg.test_sampler)
        self.loss = instantiate(cfg.loss)(
            self.train_noise_scheduler,
        )
        self.val_metrics = instantiate(cfg.val_metrics)
        self.test_metrics = instantiate(cfg.test_metrics)
        self.manifold = instantiate(cfg.manifold) if hasattr(cfg, "manifold") else None

        self.interpolant = cfg.interpolant

    def training_step(self, batch, batch_idx):
        with torch.no_grad():
            batch = self.data_preprocessing(batch)
            batch = self.cond_preprocessing(batch)
        batch_size = batch["x_0"].shape[0]
        loss = self.loss(self.preconditioning, self.network, batch).mean()
        self.log(
            "train/loss",
            loss,
            sync_dist=True,
            on_step=True,
            on_epoch=True,
            batch_size=batch_size,
        )
        return loss

    def on_before_optimizer_step(self, optimizer):
        if self.global_step == 0:
            no_grad = []
            for name, param in self.network.named_parameters():
                if param.grad is None:
                    no_grad.append(name)
            if len(no_grad) > 0:
                print("Parameters without grad:")
                print(no_grad)

    def on_validation_start(self):
        self.validation_generator = torch.Generator(device=self.device).manual_seed(
            3407
        )
        self.validation_generator_ema = torch.Generator(device=self.device).manual_seed(
            3407
        )

    def validation_step(self, batch, batch_idx):
        batch = self.data_preprocessing(batch)
        batch = self.cond_preprocessing(batch)
        batch_size = batch["x_0"].shape[0]
        loss = self.loss(
            self.preconditioning,
            self.network,
            batch,
            generator=self.validation_generator,
        ).mean()
        self.log(
            "val/loss",
            loss,
            sync_dist=True,
            on_step=False,
            on_epoch=True,
            batch_size=batch_size,
        )
        if hasattr(self, "ema_model"):
            loss_ema = self.loss(
                self.preconditioning,
                self.ema_network,
                batch,
                generator=self.validation_generator_ema,
            ).mean()
            self.log(
                "val/loss_ema",
                loss_ema,
                sync_dist=True,
                on_step=False,
                on_epoch=True,
                batch_size=batch_size,
            )
        # nll = -self.compute_exact_loglikelihood(batch).mean()
        # self.log(
        #     "val/nll",
        #     nll,
        #     sync_dist=True,
        #     on_step=False,
        #     on_epoch=True,
        #     batch_size=batch_size,
        # )

    # def on_validation_epoch_end(self):
    #     metrics = self.val_metrics.compute()
    #     for metric_name, metric_value in metrics.items():
    #         self.log(
    #             f"val/{metric_name}",
    #             metric_value,
    #             sync_dist=True,
    #             on_step=False,
    #             on_epoch=True,
    #         )

    def on_test_start(self):
        self.test_generator = torch.Generator(device=self.device).manual_seed(3407)

    def test_step_simple(self, batch, batch_idx):
        batch = self.data_preprocessing(batch)
        batch = self.cond_preprocessing(batch)
        batch_size = batch["x_0"].shape[0]
        if isinstance(self.manifold, Sphere):
            x_N = self.manifold.random_base(
                batch_size,
                self.input_dim,
                device=self.device,
            )
            x_N = x_N.reshape(batch_size, self.input_dim)
        else:
            x_N = torch.randn(
                batch_size,
                self.input_dim,
                device=self.device,
                generator=self.test_generator,
            )
        cond = batch[self.cfg.cond_preprocessing.output_key]

        samples = self.sample(
            x_N=x_N,
            cond=cond,
            stage="val",
            generator=self.test_generator,
            cfg=self.cfg.cfg_rate,
        )
        self.test_metrics.update({"gps": samples}, batch)
        if self.cfg.compute_nll:
            nll = -self.compute_exact_loglikelihood(batch, cfg=0).mean()
            self.log(
                "test/NLL",
                nll,
                sync_dist=True,
                on_step=False,
                on_epoch=True,
                batch_size=batch_size,
            )

    def test_best_nll(self, batch, batch_idx):
        batch = self.data_preprocessing(batch)
        batch = self.cond_preprocessing(batch)
        batch_size = batch["x_0"].shape[0]
        num_sample_per_cond = 32
        if isinstance(self.manifold, Sphere):
            x_N = self.manifold.random_base(
                batch_size * num_sample_per_cond,
                self.input_dim,
                device=self.device,
            )
            x_N = x_N.reshape(batch_size * num_sample_per_cond, self.input_dim)
        else:
            x_N = torch.randn(
                batch_size * num_sample_per_cond,
                self.input_dim,
                device=self.device,
                generator=self.test_generator,
            )
        cond = (
            batch[self.cfg.cond_preprocessing.output_key]
            .unsqueeze(1)
            .repeat(1, num_sample_per_cond, 1)
            .view(-1, batch[self.cfg.cond_preprocessing.output_key].shape[-1])
        )
        samples = self.sample_distribution(
            x_N,
            cond,
            sampling_batch_size=32768,
            stage="val",
            generator=self.test_generator,
            cfg=0,
        )
        samples = samples.view(batch_size * num_sample_per_cond, -1)
        batch_swarm = {"gps": samples, "emb": cond}
        nll_batch = -self.compute_exact_loglikelihood(batch_swarm, cfg=0)
        nll_batch = nll_batch.view(batch_size, num_sample_per_cond, -1)
        nll_best = nll_batch[
            torch.arange(batch_size), nll_batch.argmin(dim=1).squeeze(1)
        ]
        self.log(
            "test/best_nll",
            nll_best.mean(),
            sync_dist=True,
            on_step=False,
            on_epoch=True,
        )
        samples = samples.view(batch_size, num_sample_per_cond, -1)[
            torch.arange(batch_size), nll_batch.argmin(dim=1).squeeze(1)
        ]
        self.test_metrics.update({"gps": samples}, batch)

    def test_step(self, batch, batch_idx):
        if self.cfg.compute_swarms:
            self.test_best_nll(batch, batch_idx)
        else:
            self.test_step_simple(batch, batch_idx)

    def on_test_epoch_end(self):
        metrics = self.test_metrics.compute()
        for metric_name, metric_value in metrics.items():
            self.log(
                f"test/{metric_name}",
                metric_value,
                sync_dist=True,
                on_step=False,
                on_epoch=True,
            )

    def configure_optimizers(self):
        if self.cfg.optimizer.exclude_ln_and_biases_from_weight_decay:
            parameters_names_wd = get_parameter_names(self.network, [nn.LayerNorm])
            parameters_names_wd = [
                name for name in parameters_names_wd if "bias" not in name
            ]
            optimizer_grouped_parameters = [
                {
                    "params": [
                        p
                        for n, p in self.network.named_parameters()
                        if n in parameters_names_wd
                    ],
                    "weight_decay": self.cfg.optimizer.optim.weight_decay,
                    "layer_adaptation": True,
                },
                {
                    "params": [
                        p
                        for n, p in self.network.named_parameters()
                        if n not in parameters_names_wd
                    ],
                    "weight_decay": 0.0,
                    "layer_adaptation": False,
                },
            ]
            optimizer = instantiate(
                self.cfg.optimizer.optim, optimizer_grouped_parameters
            )
        else:
            optimizer = instantiate(self.cfg.optimizer.optim, self.network.parameters())
        if "lr_scheduler" in self.cfg:
            scheduler = instantiate(self.cfg.lr_scheduler)(optimizer)
            return [optimizer], [{"scheduler": scheduler, "interval": "step"}]
        else:
            return optimizer

    def lr_scheduler_step(self, scheduler, metric):
        scheduler.step(self.global_step)

    def sample(
        self,
        batch_size=None,
        cond=None,
        x_N=None,
        num_steps=None,
        stage="test",
        cfg=0,
        generator=None,
        return_trajectories=False,
        postprocessing=True,
    ):
        if x_N is None:
            assert batch_size is not None
            if isinstance(self.manifold, Sphere):
                x_N = self.manifold.random_base(
                    batch_size, self.input_dim, device=self.device
                )
                x_N = x_N.reshape(batch_size, self.input_dim)
            else:
                x_N = torch.randn(batch_size, self.input_dim, device=self.device)
        batch = {"y": x_N}
        if stage == "val":
            sampler = self.val_sampler
        elif stage == "test":
            sampler = self.test_sampler
        else:
            raise ValueError(f"Unknown stage {stage}")
        batch[self.cfg.cond_preprocessing.input_key] = cond
        batch = self.cond_preprocessing(batch, device=self.device)
        if num_steps is None:
            output = sampler(
                self.ema_model,
                batch,
                conditioning_keys=self.cfg.cond_preprocessing.output_key,
                scheduler=self.inference_noise_scheduler,
                cfg_rate=cfg,
                generator=generator,
                return_trajectories=return_trajectories,
            )
        else:
            output = sampler(
                self.ema_model,
                batch,
                conditioning_keys=self.cfg.cond_preprocessing.output_key,
                scheduler=self.inference_noise_scheduler,
                num_steps=num_steps,
                cfg_rate=cfg,
                generator=generator,
                return_trajectories=return_trajectories,
            )
        if return_trajectories:
            return (
                self.postprocessing(output[0]) if postprocessing else output[0],
                [
                    self.postprocessing(frame) if postprocessing else frame
                    for frame in output[1]
                ],
            )
        else:
            return self.postprocessing(output) if postprocessing else output

    def sample_distribution(
        self,
        x_N,
        cond,
        sampling_batch_size=2048,
        num_steps=None,
        stage="test",
        cfg=0,
        generator=None,
        return_trajectories=False,
    ):
        if return_trajectories:
            x_0 = []
            trajectories = []
            i = -1
            for i in range(x_N.shape[0] // sampling_batch_size):
                x_N_batch = x_N[i * sampling_batch_size : (i + 1) * sampling_batch_size]
                cond_batch = cond[
                    i * sampling_batch_size : (i + 1) * sampling_batch_size
                ]
                out, trajectories = self.sample(
                    cond=cond_batch,
                    x_N=x_N_batch,
                    num_steps=num_steps,
                    stage=stage,
                    cfg=cfg,
                    generator=generator,
                    return_trajectories=return_trajectories,
                )
                x_0.append(out)
                trajectories.append(trajectories)
            if x_N.shape[0] % sampling_batch_size != 0:
                x_N_batch = x_N[(i + 1) * sampling_batch_size :]
                cond_batch = cond[(i + 1) * sampling_batch_size :]
                out, trajectories = self.sample(
                    cond=cond_batch,
                    x_N=x_N_batch,
                    num_steps=num_steps,
                    stage=stage,
                    cfg=cfg,
                    generator=generator,
                    return_trajectories=return_trajectories,
                )
                x_0.append(out)
                trajectories.append(trajectories)
            x_0 = torch.cat(x_0, dim=1)
            trajectories = [torch.cat(frame, dim=1) for frame in trajectories]
            return x_0, trajectories
        else:
            x_0 = []
            i = -1
            for i in range(x_N.shape[0] // sampling_batch_size):
                x_N_batch = x_N[i * sampling_batch_size : (i + 1) * sampling_batch_size]
                cond_batch = cond[
                    i * sampling_batch_size : (i + 1) * sampling_batch_size
                ]
                out = self.sample(
                    cond=cond_batch,
                    x_N=x_N_batch,
                    num_steps=num_steps,
                    stage=stage,
                    cfg=cfg,
                    generator=generator,
                    return_trajectories=return_trajectories,
                )
                x_0.append(out)
            if x_N.shape[0] % sampling_batch_size != 0:
                x_N_batch = x_N[(i + 1) * sampling_batch_size :]
                cond_batch = cond[(i + 1) * sampling_batch_size :]
                out = self.sample(
                    cond=cond_batch,
                    x_N=x_N_batch,
                    num_steps=num_steps,
                    stage=stage,
                    cfg=cfg,
                    generator=generator,
                    return_trajectories=return_trajectories,
                )
                x_0.append(out)
            x_0 = torch.cat(x_0, dim=0)
            return x_0

    def model(self, *args, **kwargs):
        return self.preconditioning(self.network, *args, **kwargs)

    def ema_model(self, *args, **kwargs):
        return self.preconditioning(self.ema_network, *args, **kwargs)

    def compute_exact_loglikelihood(
        self,
        batch=None,
        x_1=None,
        cond=None,
        t1=1.0,
        num_steps=1000,
        rademacher=False,
        data_preprocessing=True,
        cfg=0,
    ):
        nfe = [0]
        if batch is None:
            batch = {"x_0": x_1, "emb": cond}
        if data_preprocessing:
            batch = self.data_preprocessing(batch)
        batch = self.cond_preprocessing(batch)
        timesteps = self.inference_noise_scheduler(
            torch.linspace(0, t1, 2).to(batch["x_0"])
        )
        with torch.inference_mode(mode=False):

            def odefunc(t, tensor):
                nfe[0] += 1
                t = t.to(tensor)
                gamma = self.inference_noise_scheduler(t)
                x = tensor[..., : self.input_dim]
                y = batch["emb"]

                def vecfield(x, y):
                    if cfg > 0:
                        batch_vecfield = {
                            "y": x,
                            "emb": y,
                            "gamma": gamma.reshape(-1),
                        }
                        model_output_cond = self.ema_model(batch_vecfield)
                        batch_vecfield_uncond = {
                            "y": x,
                            "emb": torch.zeros_like(y),
                            "gamma": gamma.reshape(-1),
                        }
                        model_output_uncond = self.ema_model(batch_vecfield_uncond)
                        model_output = model_output_cond + cfg * (
                            model_output_cond - model_output_uncond
                        )

                    else:
                        batch_vecfield = {
                            "y": x,
                            "emb": y,
                            "gamma": gamma.reshape(-1),
                        }
                        model_output = self.ema_model(batch_vecfield)

                    if self.interpolant == "flow_matching":
                        d_gamma = self.inference_noise_scheduler.derivative(t).reshape(
                            -1, 1
                        )
                        return d_gamma * model_output
                    elif self.interpolant == "diffusion":
                        alpha_t = self.inference_noise_scheduler.alpha(t).reshape(-1, 1)
                        return (
                            -1 / 2 * (alpha_t * x - torch.abs(alpha_t) * model_output)
                        )
                    else:
                        raise ValueError(f"Unknown interpolant {self.interpolant}")

                if rademacher:
                    v = torch.randint_like(x, 2) * 2 - 1
                else:
                    v = None
                dx, div = output_and_div(vecfield, x, y, v=v)
                div = div.reshape(-1, 1)
                del t, x
                return torch.cat([dx, div], dim=-1)

            x_1 = batch["x_0"]
            state1 = torch.cat([x_1, torch.zeros_like(x_1[..., :1])], dim=-1)
            with torch.no_grad():
                if False and isinstance(self.manifold, Sphere):
                    print("Riemannian flow sampler")
                    product_man = ProductManifold(
                        (self.manifold, self.input_dim), (Euclidean(), 1)
                    )
                    state0 = ode_riemannian_flow_sampler(
                        odefunc,
                        state1,
                        manifold=product_man,
                        scheduler=self.inference_noise_scheduler,
                        num_steps=num_steps,
                    )
                else:
                    print("ODE solver")
                    state0 = odeint(
                        odefunc,
                        state1,
                        t=torch.linspace(0, t1, 2).to(batch["x_0"]),
                        atol=1e-6,
                        rtol=1e-6,
                        method="dopri5",
                        options={"min_step": 1e-5},
                    )[-1]
        x_0, logdetjac = state0[..., : self.input_dim], state0[..., -1]
        if self.manifold is not None:
            x_0 = self.manifold.projx(x_0)
            logp0 = self.manifold.base_logprob(x_0)
        else:
            logp0 = (
                -1 / 2 * (x_0**2).sum(dim=-1)
                - self.input_dim
                * torch.log(torch.tensor(2 * np.pi, device=x_0.device))
                / 2
            )
        print(f"nfe: {nfe[0]}")
        logp1 = logp0 + logdetjac
        logp1 = logp1 / (self.input_dim * np.log(2))
        return logp1


def get_parameter_names(model, forbidden_layer_types):
    """
    Returns the names of the model parameters that are not inside a forbidden layer.
    Taken from HuggingFace transformers.
    """
    result = []
    for name, child in model.named_children():
        result += [
            f"{name}.{n}"
            for n in get_parameter_names(child, forbidden_layer_types)
            if not isinstance(child, tuple(forbidden_layer_types))
        ]
    # Add model specific parameters (defined with nn.Parameter) since they are not in any child.
    result += list(model._parameters.keys())
    return result


# for likelihood computation
def div_fn(u):
    """Accepts a function u:R^D -> R^D."""
    J = jacrev(u, argnums=0)
    return lambda x, y: torch.trace(J(x, y).squeeze(0))


def output_and_div(vecfield, x, y, v=None):
    if v is None:
        dx = vecfield(x, y)
        div = vmap(div_fn(vecfield))(x, y)
    else:
        vecfield_x = lambda x: vecfield(x, y)
        dx, vjpfunc = vjp(vecfield_x, x)
        vJ = vjpfunc(v)[0]
        div = torch.sum(vJ * v, dim=-1)
    return dx, div


class VonFisherGeolocalizer(L.LightningModule):
    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg
        self.network = instantiate(cfg.network)
        # self.network = torch.compile(self.network, fullgraph=True)
        self.input_dim = cfg.network.input_dim
        self.data_preprocessing = instantiate(cfg.data_preprocessing)
        self.cond_preprocessing = instantiate(cfg.cond_preprocessing)
        self.preconditioning = instantiate(cfg.preconditioning)

        self.ema_network = copy.deepcopy(self.network).requires_grad_(False)
        self.ema_network.eval()
        self.postprocessing = instantiate(cfg.postprocessing)
        self.val_sampler = instantiate(cfg.val_sampler)
        self.test_sampler = instantiate(cfg.test_sampler)
        self.loss = instantiate(cfg.loss)()
        self.val_metrics = instantiate(cfg.val_metrics)
        self.test_metrics = instantiate(cfg.test_metrics)

    def training_step(self, batch, batch_idx):
        with torch.no_grad():
            batch = self.data_preprocessing(batch)
            batch = self.cond_preprocessing(batch)
        batch_size = batch["x_0"].shape[0]
        loss = self.loss(self.preconditioning, self.network, batch).mean()
        self.log(
            "train/loss",
            loss,
            sync_dist=True,
            on_step=True,
            on_epoch=True,
            batch_size=batch_size,
        )
        return loss

    def on_before_optimizer_step(self, optimizer):
        if self.global_step == 0:
            no_grad = []
            for name, param in self.network.named_parameters():
                if param.grad is None:
                    no_grad.append(name)
            if len(no_grad) > 0:
                print("Parameters without grad:")
                print(no_grad)

    def on_validation_start(self):
        self.validation_generator = torch.Generator(device=self.device).manual_seed(
            3407
        )
        self.validation_generator_ema = torch.Generator(device=self.device).manual_seed(
            3407
        )

    def validation_step(self, batch, batch_idx):
        batch = self.data_preprocessing(batch)
        batch = self.cond_preprocessing(batch)
        batch_size = batch["x_0"].shape[0]
        loss = self.loss(
            self.preconditioning,
            self.network,
            batch,
            generator=self.validation_generator,
        ).mean()
        self.log(
            "val/loss",
            loss,
            sync_dist=True,
            on_step=False,
            on_epoch=True,
            batch_size=batch_size,
        )
        if hasattr(self, "ema_model"):
            loss_ema = self.loss(
                self.preconditioning,
                self.ema_network,
                batch,
                generator=self.validation_generator_ema,
            ).mean()
            self.log(
                "val/loss_ema",
                loss_ema,
                sync_dist=True,
                on_step=False,
                on_epoch=True,
                batch_size=batch_size,
            )

    def on_test_start(self):
        self.test_generator = torch.Generator(device=self.device).manual_seed(3407)

    def test_step(self, batch, batch_idx):
        batch = self.data_preprocessing(batch)
        batch = self.cond_preprocessing(batch)
        batch_size = batch["x_0"].shape[0]
        cond = batch[self.cfg.cond_preprocessing.output_key]

        samples = self.sample(cond=cond, stage="test")
        self.test_metrics.update({"gps": samples}, batch)
        nll = -self.compute_exact_loglikelihood(batch).mean()
        self.log(
            "test/NLL",
            nll,
            sync_dist=True,
            on_step=False,
            on_epoch=True,
            batch_size=batch_size,
        )

    def on_test_epoch_end(self):
        metrics = self.test_metrics.compute()
        for metric_name, metric_value in metrics.items():
            self.log(
                f"test/{metric_name}",
                metric_value,
                sync_dist=True,
                on_step=False,
                on_epoch=True,
            )

    def configure_optimizers(self):
        if self.cfg.optimizer.exclude_ln_and_biases_from_weight_decay:
            parameters_names_wd = get_parameter_names(self.network, [nn.LayerNorm])
            parameters_names_wd = [
                name for name in parameters_names_wd if "bias" not in name
            ]
            optimizer_grouped_parameters = [
                {
                    "params": [
                        p
                        for n, p in self.network.named_parameters()
                        if n in parameters_names_wd
                    ],
                    "weight_decay": self.cfg.optimizer.optim.weight_decay,
                    "layer_adaptation": True,
                },
                {
                    "params": [
                        p
                        for n, p in self.network.named_parameters()
                        if n not in parameters_names_wd
                    ],
                    "weight_decay": 0.0,
                    "layer_adaptation": False,
                },
            ]
            optimizer = instantiate(
                self.cfg.optimizer.optim, optimizer_grouped_parameters
            )
        else:
            optimizer = instantiate(self.cfg.optimizer.optim, self.network.parameters())
        if "lr_scheduler" in self.cfg:
            scheduler = instantiate(self.cfg.lr_scheduler)(optimizer)
            return [optimizer], [{"scheduler": scheduler, "interval": "step"}]
        else:
            return optimizer

    def lr_scheduler_step(self, scheduler, metric):
        scheduler.step(self.global_step)

    def sample(
        self,
        batch_size=None,
        cond=None,
        postprocessing=True,
        stage="val",
    ):
        batch = {}
        if stage == "val":
            sampler = self.val_sampler
        elif stage == "test":
            sampler = self.test_sampler
        else:
            raise ValueError(f"Unknown stage {stage}")
        batch[self.cfg.cond_preprocessing.input_key] = cond
        batch = self.cond_preprocessing(batch, device=self.device)
        output = sampler(
            self.ema_model,
            batch,
        )
        return self.postprocessing(output) if postprocessing else output

    def model(self, *args, **kwargs):
        return self.preconditioning(self.network, *args, **kwargs)

    def ema_model(self, *args, **kwargs):
        return self.preconditioning(self.ema_network, *args, **kwargs)

    def compute_exact_loglikelihood(
        self,
        batch=None,
    ):
        batch = self.data_preprocessing(batch)
        batch = self.cond_preprocessing(batch)
        return -self.loss(self.preconditioning, self.ema_network, batch)


class RandomGeolocalizer(L.LightningModule):
    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg
        self.test_metrics = instantiate(cfg.test_metrics)
        self.data_preprocessing = instantiate(cfg.data_preprocessing)
        self.cond_preprocessing = instantiate(cfg.cond_preprocessing)
        self.postprocessing = instantiate(cfg.postprocessing)

    def test_step(self, batch, batch_idx):
        batch = self.data_preprocessing(batch)
        batch = self.cond_preprocessing(batch)
        batch_size = batch["x_0"].shape[0]
        samples = torch.randn(batch_size, 3, device=self.device)
        samples = samples / samples.norm(dim=-1, keepdim=True)
        samples = self.postprocessing(samples)
        self.test_metrics.update({"gps": samples}, batch)

    def on_test_epoch_end(self):
        metrics = self.test_metrics.compute()
        for metric_name, metric_value in metrics.items():
            self.log(
                f"test/{metric_name}",
                metric_value,
                sync_dist=True,
                on_step=False,
                on_epoch=True,
            )

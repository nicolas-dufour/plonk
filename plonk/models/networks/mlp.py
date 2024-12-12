import torch.nn as nn
from plonk.models.positional_embeddings import FourierEmbedding, PositionalEmbedding
from plonk.models.networks.transformers import FusedMLP
import torch
import torch.nn.functional as F
import numpy as np
from einops import rearrange


class TimeEmbedder(nn.Module):
    def __init__(
        self,
        noise_embedding_type: str,
        dim: int,
        time_scaling: float,
        expansion: int = 4,
    ):
        super().__init__()
        self.encode_time = (
            PositionalEmbedding(num_channels=dim, endpoint=True)
            if noise_embedding_type == "positional"
            else FourierEmbedding(num_channels=dim)
        )
        self.time_scaling = time_scaling
        self.map_time = nn.Sequential(
            nn.Linear(dim, dim * expansion),
            nn.SiLU(),
            nn.Linear(dim * expansion, dim * expansion),
        )

    def forward(self, t):
        time = self.encode_time(t * self.time_scaling)
        time_mean = time.mean(dim=-1, keepdim=True)
        time_std = time.std(dim=-1, keepdim=True)
        time = (time - time_mean) / time_std
        return self.map_time(time)


def get_timestep_embedding(timesteps, embedding_dim, dtype=torch.float32):
    assert len(timesteps.shape) == 1
    timesteps = timesteps * 1000.0

    half_dim = embedding_dim // 2
    emb = np.log(10000) / (half_dim - 1)
    emb = (torch.arange(half_dim, dtype=dtype, device=timesteps.device) * -emb).exp()
    emb = timesteps.to(dtype)[:, None] * emb[None, :]
    emb = torch.cat([emb.sin(), emb.cos()], dim=-1)
    if embedding_dim % 2 == 1:  # zero pad
        emb = F.pad(emb, (0, 1))
    assert emb.shape == (timesteps.shape[0], embedding_dim)
    return emb


class AdaLNMLPBlock(nn.Module):
    def __init__(self, dim, expansion):
        super().__init__()
        self.mlp = FusedMLP(
            dim, dropout=0.0, hidden_layer_multiplier=expansion, activation=nn.GELU
        )
        self.ada_map = nn.Sequential(nn.SiLU(), nn.Linear(dim, dim * 3))
        self.ln = nn.LayerNorm(dim, elementwise_affine=False)

        nn.init.zeros_(self.mlp[-1].weight)
        nn.init.zeros_(self.mlp[-1].bias)

    def forward(self, x, y):
        gamma, mu, sigma = self.ada_map(y).chunk(3, dim=-1)
        x_res = (1 + gamma) * self.ln(x) + mu
        x = x + self.mlp(x_res) * sigma
        return x


class GeoAdaLNMLP(nn.Module):
    def __init__(self, input_dim, dim, depth, expansion, cond_dim):
        super().__init__()
        self.time_embedder = TimeEmbedder("positional", dim // 4, 1000, expansion=4)
        self.cond_mapper = nn.Linear(cond_dim, dim)
        self.initial_mapper = nn.Linear(input_dim, dim)
        self.blocks = nn.ModuleList(
            [AdaLNMLPBlock(dim, expansion) for _ in range(depth)]
        )
        self.final_adaln = nn.Sequential(
            nn.SiLU(),
            nn.Linear(dim, dim * 2),
        )
        self.final_ln = nn.LayerNorm(dim, elementwise_affine=False)
        self.final_linear = nn.Linear(dim, input_dim)

    def forward(self, batch):
        x = batch["y"]
        x = self.initial_mapper(x)
        gamma = batch["gamma"]
        cond = batch["emb"]
        t = self.time_embedder(gamma)
        cond = self.cond_mapper(cond)
        cond = cond + t
        for block in self.blocks:
            x = block(x, cond)
        gamma_last, mu_last = self.final_adaln(cond).chunk(2, dim=-1)
        x = (1 + gamma_last) * self.final_ln(x) + mu_last
        x = self.final_linear(x)
        return x


class GeoAdaLNMLPVonFisher(nn.Module):
    def __init__(self, input_dim, dim, depth, expansion, cond_dim):
        super().__init__()
        self.cond_mapper = nn.Linear(cond_dim, dim)
        self.blocks = nn.ModuleList(
            [AdaLNMLPBlock(dim, expansion) for _ in range(depth)]
        )
        self.final_adaln = nn.Sequential(
            nn.SiLU(),
            nn.Linear(dim, dim * 2),
        )
        self.final_ln = nn.LayerNorm(dim, elementwise_affine=False)
        self.mu_predictor = nn.Sequential(
            FusedMLP(dim, dropout=0.0, hidden_layer_multiplier=2, activation=nn.GELU),
            nn.Linear(dim, input_dim),
        )
        self.kappa_predictor = nn.Sequential(
            FusedMLP(dim, dropout=0.0, hidden_layer_multiplier=2, activation=nn.GELU),
            nn.Linear(dim, 1),
            torch.nn.Softplus(),
        )
        self.init_registers = torch.nn.Parameter(torch.randn(dim), requires_grad=True)
        torch.nn.init.trunc_normal_(
            self.init_registers, std=0.02, a=-2 * 0.02, b=2 * 0.02
        )

    def forward(self, batch):
        cond = batch["emb"]
        cond = self.cond_mapper(cond)
        x = self.init_registers.unsqueeze(0).repeat(cond.shape[0], 1)
        for block in self.blocks:
            x = block(x, cond)
        gamma_last, mu_last = self.final_adaln(cond).chunk(2, dim=-1)
        x = (1 + gamma_last) * self.final_ln(x) + mu_last
        mu = self.mu_predictor(x)
        mu = mu / mu.norm(dim=-1, keepdim=True)
        kappa = self.kappa_predictor(x)
        return mu, kappa


class GeoAdaLNMLPVonFisherMixture(nn.Module):
    def __init__(self, input_dim, dim, depth, expansion, cond_dim, num_mixtures=3):
        super().__init__()
        self.cond_mapper = nn.Linear(cond_dim, dim)
        self.blocks = nn.ModuleList(
            [AdaLNMLPBlock(dim, expansion) for _ in range(depth)]
        )
        self.final_adaln = nn.Sequential(
            nn.SiLU(),
            nn.Linear(dim, dim * 2),
        )
        self.final_ln = nn.LayerNorm(dim, elementwise_affine=False)
        self.mu_predictor = nn.Sequential(
            FusedMLP(dim, dropout=0.0, hidden_layer_multiplier=2, activation=nn.GELU),
            nn.Linear(dim, input_dim * num_mixtures),
        )
        self.kappa_predictor = nn.Sequential(
            FusedMLP(dim, dropout=0.0, hidden_layer_multiplier=2, activation=nn.GELU),
            nn.Linear(dim, num_mixtures),
            torch.nn.Softplus(),
        )
        self.mixture_weights = nn.Sequential(
            FusedMLP(dim, dropout=0.0, hidden_layer_multiplier=2, activation=nn.GELU),
            nn.Linear(dim, num_mixtures),
            torch.nn.Softmax(dim=-1),
        )
        self.num_mixtures = num_mixtures
        self.init_registers = torch.nn.Parameter(torch.randn(dim), requires_grad=True)
        torch.nn.init.trunc_normal_(
            self.init_registers, std=0.02, a=-2 * 0.02, b=2 * 0.02
        )

    def forward(self, batch):
        cond = batch["emb"]
        cond = self.cond_mapper(cond)
        x = self.init_registers.unsqueeze(0).repeat(cond.shape[0], 1)
        for block in self.blocks:
            x = block(x, cond)
        gamma_last, mu_last = self.final_adaln(cond).chunk(2, dim=-1)
        x = (1 + gamma_last) * self.final_ln(x) + mu_last
        mu = self.mu_predictor(x)
        mu = rearrange(mu, "b (n d) -> b n d", n=self.num_mixtures)
        mu = mu / mu.norm(dim=-1, keepdim=True)
        kappa = self.kappa_predictor(x)
        weights = self.mixture_weights(x)
        return mu, kappa, weights

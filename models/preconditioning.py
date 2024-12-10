import torch
from torch import nn

# ----------------------------------------------------------------------------
# Improved preconditioning proposed in the paper "Elucidating the Design
# Space of Diffusion-Based Generative networks" (EDM).


class EDMPrecond(torch.nn.Module):
    def __init__(
        self,
        network,
        label_dim=0,  # Number of class labels, 0 = unconditional.
        sigma_min=0,  # Minimum supported noise level.
        sigma_max=float("inf"),  # Maximum supported noise level.
        sigma_data=0.5,  # Expected standard deviation of the training data.
    ):
        super().__init__()
        self.label_dim = label_dim
        self.sigma_min = sigma_min
        self.sigma_max = sigma_max
        self.sigma_data = sigma_data
        self.network = network

    def forward(self, x, sigma, conditioning=None, **network_kwargs):
        x = x.to(torch.float32)
        sigma = sigma.to(torch.float32).reshape(-1, 1, 1, 1)
        conditioning = (
            None
            if self.label_dim == 0
            else torch.zeros([1, self.label_dim], device=x.device)
            if conditioning is None
            else conditioning.to(torch.float32)
        )

        c_skip = self.sigma_data**2 / (sigma**2 + self.sigma_data**2)
        c_out = sigma * self.sigma_data / (sigma**2 + self.sigma_data**2).sqrt()
        c_in = 1 / (self.sigma_data**2 + sigma**2).sqrt()
        c_noise = sigma.log() / 4

        F_x = self.network(
            (c_in * x),
            c_noise.flatten(),
            conditioning=conditioning,
            **network_kwargs,
        )
        D_x = c_skip * x + c_out * F_x.to(torch.float32)
        return D_x

    def round_sigma(self, sigma):
        return torch.as_tensor(sigma)


class DDPMPrecond(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, network, batch):
        F_x = network(batch)
        return F_x

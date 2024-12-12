import torch
import torch.nn as nn
import numpy as np


class PositionalEmbedding(nn.Module):
    """
    Taken from https://github.com/NVlabs/edm
    """

    def __init__(self, num_channels, max_positions=10000, endpoint=False):
        super().__init__()
        self.num_channels = num_channels
        self.max_positions = max_positions
        self.endpoint = endpoint
        freqs = torch.arange(start=0, end=self.num_channels // 2, dtype=torch.float32)
        freqs = 2 * freqs / self.num_channels
        freqs = (1 / self.max_positions) ** freqs
        self.register_buffer("freqs", freqs)

    def forward(self, x):
        x = torch.outer(x, self.freqs)
        out = torch.cat([x.cos(), x.sin()], dim=1)
        return out.to(x.dtype)


# ----------------------------------------------------------------------------
# Timestep embedding used in the NCSN++ architecture.
class FourierEmbedding(nn.Module):
    """
    Taken from https://github.com/NVlabs/edm
    """

    def __init__(self, num_channels, scale=16):
        super().__init__()
        self.register_buffer("freqs", torch.randn(num_channels // 2) * scale)

    def forward(self, x):
        x = x.ger((2 * np.pi * self.freqs).to(x.dtype))
        x = torch.cat([x.cos(), x.sin()], dim=1)
        return x

import torch.nn as nn
import torch
import numpy as np

class UnormGPS(nn.Module):
    def __init__(self):
        super().__init__()
        self.register_buffer("gps_normalize", torch.Tensor([np.pi * 0.5, np.pi]).unsqueeze(0))

    def forward(self, x):
        """Unormalize latitude longtitude radians to -1, 1."""
        x = torch.clamp(x, -1, 1)
        return x * self.gps_normalize

class CartesiantoGPS(nn.Module):
    def __init__(self):
        super().__init__()
    def forward(self, cartesian):
        x = cartesian[:, 0]
        y = cartesian[:, 1]
        z = cartesian[:, 2]
        lat = z.arcsin()
        lon = y.atan2(x)
        return torch.stack([lat, lon], dim=-1)
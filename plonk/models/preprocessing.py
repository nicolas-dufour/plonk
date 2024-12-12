import torch
from torch import nn
import numpy as np


class NormGPS(nn.Module):
    def __init__(self, input_key="gps", output_key="x_0", normalize=True):
        super().__init__()
        self.input_key = input_key
        self.output_key = output_key
        self.normalize = normalize
        if self.normalize:
            self.register_buffer(
                "gps_normalize", 1 / torch.Tensor([np.pi * 0.5, np.pi]).unsqueeze(0)
            )

    def forward(self, batch):
        """Normalize latitude longtitude radians to -1, 1."""  # not used currently
        x = batch[self.input_key]
        if self.normalize:
            x = x * self.gps_normalize
        batch[self.output_key] = x
        return batch

class GPStoCartesian(nn.Module):
    def __init__(self, input_key="gps", output_key="x_0"):
        super().__init__()
        self.input_key = input_key
        self.output_key = output_key

    def forward(self, batch):
        """Project latitude longtitude radians to 3D coordinates."""
        x = batch[self.input_key]
        lat, lon = x[:, 0], x[:, 1]
        x = torch.stack([lat.cos() * lon.cos(), lat.cos() * lon.sin(), lat.sin()], dim=-1)
        batch[self.output_key] = x
        return batch

class PrecomputedPreconditioning:
    def __init__(
        self,
        input_key="emb",
        output_key="emb",
    ):
        self.input_key = input_key
        self.output_key = output_key

    def __call__(self, batch, device=None):
        batch[self.output_key] = batch[self.input_key]
        return batch

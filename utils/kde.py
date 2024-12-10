import torch
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt


class BatchedKDE(nn.Module):
    def __init__(self, bandwith=0.0):
        super().__init__()
        self.bandwidth = bandwith
        self.X = None

    def fit(self, X: torch.Tensor):
        self.mu = X
        self.nmu2 = torch.sum(X * X, dim=-1, keepdim=True)
        b, n, d = X.shape
        if self.bandwidth == 0:
            q = torch.quantile(X.view(b, -1), 0.75) - torch.quantile(
                X.view(b, -1), 0.25
            )
            self.bandwidth = (
                0.9 * torch.min(torch.std(X, dim=(1, 2)), q / 1.34) / pow(n, 0.2)
            )

    def score(self, X):
        nx2 = torch.sum(X * X, dim=-1, keepdim=True)
        dot = torch.einsum("bnd, bmd -> bnm", X, self.mu)
        dist = nx2 + self.nmu2.transpose(1, 2) - 2 * dot
        return torch.sum(
            torch.exp(-dist / self.bandwidth.unsqueeze(-1).unsqueeze(-1)), dim=-1
        )

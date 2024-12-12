import torch
from plonk.utils.manifolds import Sphere, geodesic
from torch.func import vjp, jvp, vmap, jacrev


class DDPMLoss:
    def __init__(
        self,
        scheduler,
        cond_drop_rate=0.0,
        conditioning_key="label",
    ):
        self.scheduler = scheduler
        self.cond_drop_rate = cond_drop_rate
        self.conditioning_key = conditioning_key

    def __call__(self, preconditioning, network, batch, generator=None):
        x_0 = batch["x_0"]
        batch_size = x_0.shape[0]
        device = x_0.device
        t = torch.rand(batch_size, device=device, dtype=x_0.dtype, generator=generator)
        gamma = self.scheduler(t).unsqueeze(-1)
        n = torch.randn(x_0.shape, dtype=x_0.dtype, device=device, generator=generator)
        y = torch.sqrt(gamma) * x_0 + torch.sqrt(1 - gamma) * n
        batch["y"] = y
        conditioning = batch[self.conditioning_key]
        if conditioning is not None and self.cond_drop_rate > 0:
            drop_mask = (
                torch.rand(batch_size, device=device, generator=generator)
                < self.cond_drop_rate
            )
            conditioning[drop_mask] = torch.zeros_like(conditioning[drop_mask])
            batch[self.conditioning_key] = conditioning.detach()
        batch["gamma"] = gamma.squeeze(-1).squeeze(-1).squeeze(-1)
        D_n = preconditioning(network, batch)
        loss = (D_n - n) ** 2
        return loss


class FlowMatchingLoss:
    def __init__(
        self,
        scheduler,
        cond_drop_rate=0.0,
        conditioning_key="label",
    ):
        self.scheduler = scheduler
        self.cond_drop_rate = cond_drop_rate
        self.conditioning_key = conditioning_key

    def __call__(self, preconditioning, network, batch, generator=None):
        x_0 = batch["x_0"]
        batch_size = x_0.shape[0]
        device = x_0.device
        t = torch.rand(batch_size, device=device, dtype=x_0.dtype, generator=generator)
        gamma = self.scheduler(t).unsqueeze(-1)
        n = torch.randn(x_0.shape, dtype=x_0.dtype, device=device, generator=generator)
        y = gamma * x_0 + (1 - gamma) * n
        batch["y"] = y
        conditioning = batch[self.conditioning_key]
        if conditioning is not None and self.cond_drop_rate > 0:
            drop_mask = (
                torch.rand(batch_size, device=device, generator=generator)
                < self.cond_drop_rate
            )
            conditioning[drop_mask] = torch.zeros_like(conditioning[drop_mask])
            batch[self.conditioning_key] = conditioning.detach()
        batch["gamma"] = gamma.squeeze(-1).squeeze(-1).squeeze(-1)
        D_n = preconditioning(network, batch)
        loss = (D_n - (x_0 - n)) ** 2
        return loss


class RiemannianFlowMatchingLoss:
    def __init__(
        self,
        scheduler,
        cond_drop_rate=0.0,
        conditioning_key="label",
    ):
        self.scheduler = scheduler
        self.cond_drop_rate = cond_drop_rate
        self.conditioning_key = conditioning_key
        self.manifold = Sphere()
        self.manifold_dim = 3

    def __call__(self, preconditioning, network, batch, generator=None):
        x_1 = batch["x_0"]
        batch_size = x_1.shape[0]
        device = x_1.device
        t = torch.rand(batch_size, device=device, dtype=x_1.dtype, generator=generator)
        gamma = self.scheduler(t).unsqueeze(-1)
        x_0 = self.manifold.random_base(x_1.shape[0], self.manifold_dim).to(x_1)

        def cond_u(x0, x1, t):
            path = geodesic(self.manifold, x0, x1)
            x_t, u_t = jvp(path, (t,), (torch.ones_like(t).to(t),))
            return x_t, u_t

        y, u_t = vmap(cond_u)(x_0, x_1, gamma)
        y = y.reshape(batch_size, self.manifold_dim)
        u_t = u_t.reshape(batch_size, self.manifold_dim)
        batch["y"] = y
        conditioning = batch[self.conditioning_key]
        if conditioning is not None and self.cond_drop_rate > 0:
            drop_mask = (
                torch.rand(batch_size, device=device, generator=generator)
                < self.cond_drop_rate
            )
            conditioning[drop_mask] = torch.zeros_like(conditioning[drop_mask])
            batch[self.conditioning_key] = conditioning.detach()
        batch["gamma"] = gamma.squeeze(-1).squeeze(-1).squeeze(-1)
        D_n = preconditioning(network, batch)
        diff = D_n - u_t
        loss = self.manifold.inner(y, diff, diff).mean() / self.manifold_dim
        return loss


class VonFisherLoss:
    def __init__(self, dim=3):
        self.dim = dim

    def __call__(self, preconditioning, network, batch, generator=None):
        x = batch["x_0"]
        mu, kappa = preconditioning(network, batch)
        loss = (
            torch.log((kappa + 1e-8))
            - torch.log(torch.tensor(4 * torch.pi, dtype=kappa.dtype))
            - log_sinh(kappa)
            + kappa * (mu * x).sum(dim=-1, keepdim=True)
        )
        return -loss


class VonFisherMixtureLoss:
    def __init__(self, dim=3):
        self.dim = dim

    def __call__(self, preconditioning, network, batch, generator=None):
        x = batch["x_0"]
        mu_mixture, kappa_mixture, weights = preconditioning(network, batch)
        loss = 0
        for i in range(mu_mixture.shape[1]):
            mu = mu_mixture[:, i]
            kappa = kappa_mixture[:, i].unsqueeze(1)
            loss += weights[:, i].unsqueeze(1) * (
                kappa
                * torch.exp(kappa * ((mu * x).sum(dim=-1, keepdim=True) - 1))
                / (1e-8 + 2 * torch.pi * (1 - torch.exp(-2 * kappa)))
            )
        return -torch.log(loss)


def log_sinh(x):
    return x + torch.log(1e-8 + (1 - torch.exp(-2 * x)) / 2)

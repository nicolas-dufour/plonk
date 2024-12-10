"""
Generate multivariate von Mises Fisher samples.
PyTorch implementation of the original code from:
https://github.com/clara-labs/spherecluster
"""

import torch

__all__ = ["sample_vMF"]


def vMF_sampler(
    net,
    batch,
):
    mu, kappa = net(batch)
    return sample_vMF(mu.T, kappa.squeeze(1))


def vMF_mixture_sampler(
    net,
    batch,
):
    mu_mixture, kappa_mixture, weights = net(batch)
    # Sample mixture component indices based on weights
    indices = torch.multinomial(weights, num_samples=1).squeeze()
    # Select corresponding mu and kappa
    mu = mu_mixture[torch.arange(mu_mixture.shape[0]), indices]
    kappa = kappa_mixture[torch.arange(kappa_mixture.shape[0]), indices]
    return sample_vMF(mu.T, kappa)


def sample_vMF(mu, kappa, num_samples=1):
    """Generate N-dimensional samples from von Mises Fisher
    distribution around center mu ∈ R^N with concentration kappa.
    mu and kappa may be vectors,
    mu should have shape (N,) or (N, 1), kappa should be scalar or vector of length N.
    """
    if len(mu.shape) == 1:
        mu = mu.unsqueeze(1)

    if isinstance(kappa, torch.Tensor):
        dim = mu.shape[0]
        assert mu.shape[1] == kappa.size(0)
    else:
        dim = mu.shape[0]
        mu = mu.repeat(1, num_samples)
        kappa = torch.full((num_samples,), kappa, device=mu.device, dtype=mu.dtype)

    # sample offset from center (on sphere) with spread kappa
    w = _sample_weight(kappa, dim)

    # sample a point v on the unit sphere that's orthogonal to mu
    v = _sample_orthonormal_to(mu)

    # compute new point
    result = v * torch.sqrt(1.0 - w**2).unsqueeze(0) + w.unsqueeze(0) * mu
    return result.T


def _sample_weight(kappa, dim):
    """Rejection sampling scheme for sampling distance from center on
    surface of the sphere.
    """
    dim = dim - 1  # since S^{n-1}
    try:
        size = kappa.size(0)
    except AttributeError:
        size = 1

    b = dim / (torch.sqrt(4.0 * kappa**2 + dim**2) + 2 * kappa)
    x = (1.0 - b) / (1.0 + b)
    c = kappa * x + dim * torch.log(1 - x**2)

    w = torch.zeros_like(kappa)
    idx = torch.zeros_like(kappa, dtype=torch.bool)

    while True:
        where_zero = ~idx
        if torch.all(idx):
            return w

        z = (
            torch.distributions.Beta(dim / 2.0, dim / 2.0)
            .sample((size,))
            .to(kappa.device)
        )
        _w = (1.0 - (1.0 + b) * z) / (1.0 - (1.0 - b) * z)
        u = torch.rand(size, device=kappa.device)

        _idx = kappa * _w + dim * torch.log(1.0 - x * _w) - c >= torch.log(u)

        if not torch.any(_idx):
            continue

        w[where_zero] = _w[where_zero]
        idx[_idx] = True


def _sample_orthonormal_to(mu):
    """Sample point on sphere orthogonal to mu."""
    v = torch.randn(mu.shape[0], mu.shape[1], device=mu.device)
    proj_mu_v = mu * ((v * mu).sum(dim=0)) / torch.norm(mu, dim=0) ** 2
    orthto = v - proj_mu_v
    return orthto / torch.norm(orthto, dim=0)

import torch
from plonk.utils.manifolds import Sphere
from tqdm.auto import tqdm


def riemannian_flow_sampler(
    net,
    batch,
    manifold=Sphere(),
    conditioning_keys=None,
    scheduler=None,
    num_steps=250,
    cfg_rate=0,
    generator=None,
    return_trajectories=False,
):
    if scheduler is None:
        raise ValueError("Scheduler must be provided")

    x_cur = batch["y"].to(torch.float32)
    if return_trajectories:
        traj = [x_cur.detach()]
    step_indices = torch.arange(num_steps + 1, dtype=torch.float32, device=x_cur.device)
    steps = 1 - step_indices / num_steps
    gammas = scheduler(steps)
    dtype = torch.float32
    if cfg_rate > 0 and conditioning_keys is not None:
        stacked_batch = {}
        stacked_batch[conditioning_keys] = torch.cat(
            [batch[conditioning_keys], torch.zeros_like(batch[conditioning_keys])],
            dim=0,
        )
    for step, (gamma_now, gamma_next) in enumerate(zip(gammas[:-1], gammas[1:])):
        with torch.cuda.amp.autocast(dtype=dtype):
            if cfg_rate > 0 and conditioning_keys is not None:
                stacked_batch["y"] = torch.cat([x_cur, x_cur], dim=0)
                stacked_batch["gamma"] = gamma_now.expand(x_cur.shape[0] * 2)
                denoised_all = net(stacked_batch)
                denoised_cond, denoised_uncond = denoised_all.chunk(2, dim=0)
                denoised = denoised_cond * (1 + cfg_rate) - denoised_uncond * cfg_rate
            else:
                batch["y"] = x_cur
                batch["gamma"] = gamma_now.expand(x_cur.shape[0])
                denoised = net(batch)

        dt = gamma_next - gamma_now
        x_next = x_cur + dt * denoised  # manifold.expmap(x_cur, dt * denoised)
        x_next = manifold.projx(x_next)
        x_cur = x_next
        if return_trajectories:
            traj.append(x_cur.detach().to(torch.float32))

    if return_trajectories:
        return x_cur.to(torch.float32), traj
    else:
        return x_cur.to(torch.float32)


def ode_riemannian_flow_sampler(
    odefunc,
    x_1,
    manifold=Sphere(),
    scheduler=None,
    num_steps=1000,
):
    if scheduler is None:
        raise ValueError("Scheduler must be provided")

    x_cur = x_1.to(torch.float32)
    steps = (
        torch.arange(num_steps + 1, dtype=torch.float32, device=x_cur.device)
        / num_steps
    )
    dtype = torch.float32
    for step, (t_now, t_next) in enumerate(zip(steps[:-1], steps[1:]), total=num_steps):
        with torch.cuda.amp.autocast(dtype=dtype):
            denoised = odefunc(t_now, x_cur)
        gamma_now = scheduler(t_now)
        gamma_next = scheduler(t_next)
        dt = gamma_next - gamma_now
        x_next = x_cur + dt * denoised  # manifold.expmap(x_cur, dt * denoised)
        x_next = manifold.projx(x_next)
        x_cur = x_next
    return x_cur.to(torch.float32)

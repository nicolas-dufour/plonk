import torch


def flow_sampler(
    net,
    batch,
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
    dtype = (
        torch.float32
    )  # torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
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
        x_next = x_cur + dt * denoised
        x_cur = x_next
        if return_trajectories:
            traj.append(x_cur.detach().to(torch.float32))

    if return_trajectories:
        return x_cur.to(torch.float32), traj
    else:
        return x_cur.to(torch.float32)


def circular_transformation(x, min_val=-1, max_val=1):
    return (x - min_val) % (max_val - min_val) + min_val

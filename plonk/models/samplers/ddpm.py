import torch


def ddpm_sampler(
    net,
    batch,
    conditioning_keys=None,
    scheduler=None,
    uncond_tokens=None,
    num_steps=1000,
    cfg_rate=0,
    generator=None,
    use_confidence_sampling=False,
    use_uncond_token=True,
    confidence_value=1.0,
    unconfidence_value=0.0,
):
    if scheduler is None:
        raise ValueError("Scheduler must be provided")

    x_cur = batch["y"].to(torch.float32)
    latents = batch["previous_latents"]
    if use_confidence_sampling:
        batch["confidence"] = (
            torch.ones(x_cur.shape[0], device=x_cur.device) * confidence_value
        )
    step_indices = torch.arange(num_steps + 1, dtype=torch.float32, device=x_cur.device)
    steps = 1 - step_indices / num_steps
    gammas = scheduler(steps)
    latents_cond = latents_uncond = latents
    # dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
    dtype = torch.float32
    if cfg_rate > 0 and conditioning_keys is not None:
        stacked_batch = {}
        for key in conditioning_keys:
            if f"{key}_mask" in batch:
                if use_confidence_sampling and not use_uncond_token:
                    stacked_batch[f"{key}_mask"] = torch.cat(
                        [batch[f"{key}_mask"], batch[f"{key}_mask"]], dim=0
                    )
                else:
                    if (
                        batch[f"{key}_mask"].shape[1]
                        > uncond_tokens[f"{key}_mask"].shape[1]
                    ):
                        uncond_mask = (
                            torch.zeros_like(batch[f"{key}_mask"])
                            if batch[f"{key}_mask"].dtype == torch.bool
                            else torch.ones_like(batch[f"{key}_mask"]) * -torch.inf
                        )
                        uncond_mask[:, : uncond_tokens[f"{key}_mask"].shape[1]] = (
                            uncond_tokens[f"{key}_mask"]
                        )
                    else:
                        uncond_mask = uncond_tokens[f"{key}_mask"]
                        batch[f"{key}_mask"] = torch.cat(
                            [
                                batch[f"{key}_mask"],
                                torch.zeros(
                                    batch[f"{key}_mask"].shape[0],
                                    uncond_tokens[f"{key}_embeddings"].shape[1]
                                    - batch[f"{key}_mask"].shape[1],
                                    device=batch[f"{key}_mask"].device,
                                    dtype=batch[f"{key}_mask"].dtype,
                                ),
                            ],
                            dim=1,
                        )
                    stacked_batch[f"{key}_mask"] = torch.cat(
                        [batch[f"{key}_mask"], uncond_mask], dim=0
                    )
            if f"{key}_embeddings" in batch:
                if use_confidence_sampling and not use_uncond_token:
                    stacked_batch[f"{key}_embeddings"] = torch.cat(
                        [
                            batch[f"{key}_embeddings"],
                            batch[f"{key}_embeddings"],
                        ],
                        dim=0,
                    )
                else:
                    if (
                        batch[f"{key}_embeddings"].shape[1]
                        > uncond_tokens[f"{key}_embeddings"].shape[1]
                    ):
                        uncond_tokens[f"{key}_embeddings"] = torch.cat(
                            [
                                uncond_tokens[f"{key}_embeddings"],
                                torch.zeros(
                                    uncond_tokens[f"{key}_embeddings"].shape[0],
                                    batch[f"{key}_embeddings"].shape[1]
                                    - uncond_tokens[f"{key}_embeddings"].shape[1],
                                    uncond_tokens[f"{key}_embeddings"].shape[2],
                                    device=uncond_tokens[f"{key}_embeddings"].device,
                                ),
                            ],
                            dim=1,
                        )
                    elif (
                        batch[f"{key}_embeddings"].shape[1]
                        < uncond_tokens[f"{key}_embeddings"].shape[1]
                    ):
                        batch[f"{key}_embeddings"] = torch.cat(
                            [
                                batch[f"{key}_embeddings"],
                                torch.zeros(
                                    batch[f"{key}_embeddings"].shape[0],
                                    uncond_tokens[f"{key}_embeddings"].shape[1]
                                    - batch[f"{key}_embeddings"].shape[1],
                                    batch[f"{key}_embeddings"].shape[2],
                                    device=batch[f"{key}_embeddings"].device,
                                ),
                            ],
                            dim=1,
                        )
                    stacked_batch[f"{key}_embeddings"] = torch.cat(
                        [
                            batch[f"{key}_embeddings"],
                            uncond_tokens[f"{key}_embeddings"],
                        ],
                        dim=0,
                    )
            elif key not in batch:
                raise ValueError(f"Key {key} not in batch")
            else:
                if isinstance(batch[key], torch.Tensor):
                    if use_confidence_sampling and not use_uncond_token:
                        stacked_batch[key] = torch.cat([batch[key], batch[key]], dim=0)
                    else:
                        stacked_batch[key] = torch.cat(
                            [batch[key], uncond_tokens], dim=0
                        )
                elif isinstance(batch[key], list):
                    if use_confidence_sampling and not use_uncond_token:
                        stacked_batch[key] = [*batch[key], *batch[key]]
                    else:
                        stacked_batch[key] = [*batch[key], *uncond_tokens]
                else:
                    raise ValueError(
                        "Conditioning must be a tensor or a list of tensors"
                    )
        if use_confidence_sampling:
            stacked_batch["confidence"] = torch.cat(
                [
                    torch.ones(x_cur.shape[0], device=x_cur.device) * confidence_value,
                    torch.ones(x_cur.shape[0], device=x_cur.device)
                    * unconfidence_value,
                ],
                dim=0,
            )
    for step, (gamma_now, gamma_next) in enumerate(zip(gammas[:-1], gammas[1:])):
        with torch.cuda.amp.autocast(dtype=dtype):
            if cfg_rate > 0 and conditioning_keys is not None:
                stacked_batch["y"] = torch.cat([x_cur, x_cur], dim=0)
                stacked_batch["gamma"] = gamma_now.expand(x_cur.shape[0] * 2)
                stacked_batch["previous_latents"] = (
                    torch.cat([latents_cond, latents_uncond], dim=0)
                    if latents is not None
                    else None
                )
                denoised_all, latents_all = net(stacked_batch)
                denoised_cond, denoised_uncond = denoised_all.chunk(2, dim=0)
                latents_cond, latents_uncond = latents_all.chunk(2, dim=0)
                denoised = denoised_cond * (1 + cfg_rate) - denoised_uncond * cfg_rate
            else:
                batch["y"] = x_cur
                batch["gamma"] = gamma_now.expand(x_cur.shape[0])
                batch["previous_latents"] = latents
                denoised, latents = net(
                    batch,
                )
        x_pred = (x_cur - torch.sqrt(1 - gamma_now) * denoised) / torch.sqrt(gamma_now)
        x_pred = torch.clamp(x_pred, -1, 1)
        noise_pred = (x_cur - torch.sqrt(gamma_now) * x_pred) / torch.sqrt(
            1 - gamma_now
        )

        log_alpha_t = torch.log(gamma_now) - torch.log(gamma_next)
        alpha_t = torch.clip(torch.exp(log_alpha_t), 0, 1)
        x_mean = torch.rsqrt(alpha_t) * (
            x_cur - torch.rsqrt(1 - gamma_now) * (1 - alpha_t) * noise_pred
        )
        var_t = 1 - alpha_t
        eps = torch.randn(x_cur.shape, device=x_cur.device, generator=generator)
        x_next = x_mean + torch.sqrt(var_t) * eps
        x_cur = x_next
    return x_cur.to(torch.float32)

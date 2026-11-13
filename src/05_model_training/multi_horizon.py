"""Shared forecast head conditioned on patch lead, without future encoder inputs."""

import torch


def weighted_label_mass(labels, weights):
    """Effective supervised point count; safe to compute on CPU before transfer."""
    return sum(
        weight * torch.isfinite(labels[:, :, lead:]).sum()
        for lead, weight in enumerate(weights)
        if lead < labels.shape[2]
    )


def weighted_training_loss(model, hidden, labels, loc, scale, first_raw):
    """Exact short-lead objective at every causal anchor; no stochastic lead draw."""
    weights = model.config.horizon_loss_weights
    levels = first_raw.new_tensor(model.config.quantiles)
    numerator = first_raw.new_zeros(labels.shape[0])
    mass = numerator.clone()
    for lead, weight in enumerate(weights):
        if lead >= labels.shape[2]:
            break
        length = labels.shape[2] - lead
        future = labels[:, :, lead:]
        valid = torch.isfinite(future)
        if lead == 0:
            raw = first_raw
        else:
            raw = model.head(
                model.norm(
                    hidden[:, :, :length] + model.lead_embedding.weight[lead - 1]
                )
            )
            raw = raw.float().reshape(*future.shape, len(levels)).sort(dim=-1).values
        error = (
            (future.float() - loc[:, :, :length]) / scale[:, :, :length]
        ).nan_to_num()[..., None] - raw
        pinball = torch.maximum(levels * error, (levels - 1) * error)
        numerator = numerator + weight * (pinball * valid[..., None]).flatten(1).sum(
            1
        ) / len(levels)
        mass = mass + weight * valid.flatten(1).sum(1)
    per_sample = numerator / mass.clamp_min(torch.finfo(numerator.dtype).tiny)
    return (
        numerator.sum() / mass.sum().clamp_min(torch.finfo(numerator.dtype).tiny),
        per_sample,
        mass,
    )


def weighted_forecast_loss(model, prediction, labels, loc, scale):
    """Same weighted pinball objective for a held-out block from one origin."""
    weights = prediction.new_tensor(
        model.config.horizon_loss_weights
    ).repeat_interleave(model.config.patch_size)[: labels.shape[-1]]
    valid = torch.isfinite(labels)
    levels = prediction.new_tensor(model.config.quantiles)
    error = ((labels.float() - loc) / scale).nan_to_num()[..., None] - (
        prediction - loc[..., None]
    ) / scale[..., None]
    pinball = torch.maximum(levels * error, (levels - 1) * error).mean(-1)
    mass = (valid * weights).flatten(1).sum(1)
    numerator = (pinball * valid * weights).flatten(1).sum(1)
    per_sample = numerator / mass.clamp_min(torch.finfo(numerator.dtype).tiny)
    return (
        numerator.sum() / mass.sum().clamp_min(torch.finfo(numerator.dtype).tiny),
        per_sample,
        mass,
    )


def direct_quantiles(model, hidden, loc, scale, patches):
    if patches < 1 or patches > model.config.output_patches:
        raise ValueError("Requested block exceeds trained output_patches")
    # Lead zero uses the existing head unmodified. Later leads have small learned
    # offsets; all leads share the same projection, preserving ~307M capacity.
    offsets = hidden.new_zeros(patches, hidden.shape[-1])
    if patches > 1:
        offsets[1:] = model.lead_embedding.weight[: patches - 1]
    raw = model.head(model.norm(hidden + offsets[None, None])).float()
    raw = (
        raw.reshape(
            *hidden.shape[:2],
            patches * model.config.patch_size,
            len(model.config.quantiles),
        )
        .sort(dim=-1)
        .values
    )
    return raw * scale[..., None] + loc[..., None]


def auxiliary_loss(model, hidden, labels, loc, scale):
    """Sample one extra future lead per input token; never use labels as features.

    Train label tensors already exclude the final validation patch. Out-of-range,
    missing and padded labels are masked, including shorter tables in a batch.
    """
    b, targets, n, p = labels.shape
    leads = torch.randint(
        1, model.config.output_patches, (b, 1, n), device=labels.device
    )
    indices = torch.arange(n, device=labels.device)[None, None] + leads
    available = indices < n
    future = labels.gather(
        2,
        indices.clamp_max(n - 1)
        .expand(b, targets, n)[..., None]
        .expand(b, targets, n, p),
    )
    valid = torch.isfinite(future) & available[..., None]
    offset = model.lead_embedding(leads - 1)
    raw = (
        model.head(model.norm(hidden + offset))
        .float()
        .reshape(b, targets, n, p, len(model.config.quantiles))
        .sort(dim=-1)
        .values
    )
    error = ((future.float() - loc) / scale).nan_to_num()[..., None] - raw
    levels = error.new_tensor(model.config.quantiles)
    pinball = torch.maximum(levels * error, (levels - 1) * error)
    count = valid.sum()
    return (pinball * valid[..., None]).sum() / (
        count.clamp_min(1) * len(levels)
    ), count

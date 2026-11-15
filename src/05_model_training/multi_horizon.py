"""Multi-patch forecast heads and their losses, without future encoder inputs.

Two heads exist. "lead_offset" (archived 447M/500M models) adds a learned offset
per lead before the final norm and reuses one single-patch projection. "direct"
emits every lead from one projection and is trained on every lead at every
causal anchor.
"""

import torch
from torch.nn import functional as F

# sinh overflows float32 near 89. Twenty is 2.4e8 standard deviations, far past
# any usable forecast, and keeps the inverse finite.
ASINH_LIMIT = 20.0


def to_model_space(model, standardized):
    """Causally standardised values -> the space the model reads and predicts."""
    if model.config.value_transform == "asinh":
        return torch.asinh(standardized)
    return standardized


def from_model_space(model, values):
    """Inverse of to_model_space; monotonic, so quantile order is preserved."""
    if model.config.value_transform == "asinh":
        return torch.sinh(values.clamp(-ASINH_LIMIT, ASINH_LIMIT))
    return values


def lead_quantiles(model, hidden, patches=None):
    """Direct head: hidden [..., width] -> ordered quantiles [..., lead, time, quantile].

    Values are float32 in model space. Sorting guarantees ordered quantiles and
    stays differentiable almost everywhere.
    """
    config = model.config
    raw = model.read_out(hidden).float()
    raw = raw.reshape(
        *hidden.shape[:-1],
        config.output_patches,
        config.patch_size,
        len(config.quantiles),
    )
    if patches is not None:
        raw = raw[..., :patches, :, :]
    return raw.sort(dim=-1).values


def anchors_with_history(features, targets, patch_size):
    """[batch, target, anchor]: has this target been observed at or before the anchor?

    Read from the observation mask inside the patch features, so it needs no
    extra bookkeeping and gives the same answer on CPU and GPU.
    """
    observed = features[:, :targets, :, patch_size : 2 * patch_size].sum(-1)
    return observed.cumsum(-1) > 0


def direct_training_loss(model, hidden, labels, loc, scale, *, supervised=None):
    """Pinball loss of every lead at every causal anchor, in model space.

    hidden [batch, target, anchor, width]; labels [batch, target, anchor, time],
    where labels[..., i, :] is the patch after anchor i. Lead j of anchor i is
    therefore labels[..., i + j, :]; leads that fall outside the chunk, missing
    values and padding are masked. With equal lead weights every observed
    forecast timestep contributes equally.

    supervised [batch, target, anchor]: anchors that take part. An anchor whose
    target has no observation yet (a table that starts with a sensor outage)
    has no location or scale to standardise its labels with, so it is left out.

    Returns the batch loss, per-table loss, per-table loss mass and the
    next-patch quantiles [batch, target, anchor, time, quantile].
    """
    config = model.config
    leads = config.output_patches
    quantiles = lead_quantiles(model, hidden)
    future = F.pad(labels, (0, 0, 0, leads - 1), value=float("nan"))
    future = future.unfold(2, leads, 1).permute(0, 1, 2, 4, 3)
    valid = torch.isfinite(future)
    if supervised is not None:
        valid = valid & supervised[..., None, None]
    target = to_model_space(
        model,
        ((future.float() - loc[..., None]) / scale[..., None]).nan_to_num(),
    )
    error = target[..., None] - quantiles
    levels = quantiles.new_tensor(config.quantiles)
    pinball = torch.maximum(levels * error, (levels - 1) * error).mean(-1)
    weights = quantiles.new_tensor(config.horizon_loss_weights)[:, None]
    numerator = (pinball * valid * weights).flatten(1).sum(1)
    mass = (valid * weights).flatten(1).sum(1)
    tiny = torch.finfo(numerator.dtype).tiny
    return (
        numerator.sum() / mass.sum().clamp_min(tiny),
        numerator / mass.clamp_min(tiny),
        mass,
        quantiles[:, :, :, 0],
    )


def weighted_label_mass(labels, weights, supervised=None):
    """Effective supervised point count; safe to compute on CPU before transfer.

    supervised [batch, target, anchor] restricts the count to those anchors, the
    same way direct_training_loss does. Label i + lead belongs to anchor i.
    """
    anchors = labels.shape[2]
    total = 0
    for lead, weight in enumerate(weights):
        if lead >= anchors:
            break
        valid = torch.isfinite(labels[:, :, lead:])
        if supervised is not None:
            valid = valid & supervised[:, :, : anchors - lead, None]
        total = total + weight * valid.sum()
    return total


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
            raw = model.read_out(
                hidden[:, :, :length] + model.lead_embedding.weight[lead - 1]
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


def weighted_forecast_loss(model, prediction, labels, loc, scale, *, normalized=None):
    """Same weighted pinball objective for a held-out block from one origin.

    normalized: the forecast quantiles in model space, when the caller has them.
    The loss is then measured in that space, exactly as in training. Without
    it, the forecast is standardised back from original units (legacy head).
    """
    weights = prediction.new_tensor(
        model.config.horizon_loss_weights
    ).repeat_interleave(model.config.patch_size)[: labels.shape[-1]]
    valid = torch.isfinite(labels)
    levels = prediction.new_tensor(model.config.quantiles)
    if normalized is None:
        error = ((labels.float() - loc) / scale).nan_to_num()[..., None] - (
            prediction - loc[..., None]
        ) / scale[..., None]
    else:
        error = (
            to_model_space(model, ((labels.float() - loc) / scale).nan_to_num())[
                ..., None
            ]
            - normalized
        )
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
    """Forecast quantiles in original units from one hidden position per request.

    hidden [batch, target, 1, width]; loc and scale [batch, target, 1].
    """
    if patches < 1 or patches > model.config.output_patches:
        raise ValueError("Requested block exceeds trained output_patches")
    if model.config.head == "direct":
        normalized = lead_quantiles(model, hidden, patches).reshape(
            *hidden.shape[:2],
            patches * model.config.patch_size,
            len(model.config.quantiles),
        )
        return from_model_space(model, normalized) * scale[..., None] + loc[..., None]
    # Lead zero uses the existing head unmodified. Later leads have small learned
    # offsets; all leads share the same projection, preserving ~307M capacity.
    offsets = hidden.new_zeros(patches, hidden.shape[-1])
    if patches > 1:
        offsets[1:] = model.lead_embedding.weight[: patches - 1]
    raw = model.read_out(hidden + offsets[None, None]).float()
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
        model.read_out(hidden + offset)
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

"""Shared, dependency-light inference code for the portable Basin forecasters.

Everything here is plain PyTorch. It reproduces the inference path of the
training repository (causal patch tokenisation, patch embedding, cross-variable
attention, direct multi-patch quantile head and rollout) without importing it.
The RWKV-7 time/channel mixing blocks are supplied by a subclass:

* ``modeling_basin_pytorch.BasinPyTorch`` - pure PyTorch, any device.
* ``modeling_basin_fla.BasinFLA`` - ``flash-linear-attention`` (Triton) kernels.

Training is intentionally not supported.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import torch
from torch import nn
from torch.nn import functional as F

WEIGHTS_NAME = "model.safetensors"
CONFIG_NAME = "config.json"


@dataclass(frozen=True)
class BasinConfig:
    patch_size: int = 16
    width: int = 1024
    layers: int = 28
    variate_heads: int = 16
    quantiles: tuple[float, ...] = (0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9)
    context_length: int = 2048
    output_patches: int = 4
    head_size: int = 64
    ffn_width: int = 4096
    decay_low_rank_dim: int = 64
    a_low_rank_dim: int = 64
    v_low_rank_dim: int = 64
    gate_low_rank_dim: int = 160

    @classmethod
    def from_dict(cls, values):
        names = cls.__dataclass_fields__
        selected = {k: v for k, v in values.items() if k in names}
        selected["quantiles"] = tuple(selected["quantiles"])
        return cls(**selected)

    @property
    def median_index(self):
        return self.quantiles.index(0.5)


@dataclass
class BasinForecast:
    point: torch.Tensor  # [batch, target, horizon], median
    quantiles: torch.Tensor  # [batch, target, horizon, quantile]


class ResidualProjection(nn.Module):
    def __init__(self, inputs, outputs, hidden):
        super().__init__()
        self.skip = nn.Linear(inputs, outputs)
        self.mlp = nn.Sequential(
            nn.Linear(inputs, hidden), nn.GELU(), nn.Linear(hidden, outputs)
        )

    def forward(self, x):
        return self.skip(x) + self.mlp(x)


class VariateAttention(nn.Module):
    """Parameter container with the same names as ``nn.MultiheadAttention``."""

    def __init__(self, width, heads):
        super().__init__()
        self.num_heads = heads
        self.in_proj_weight = nn.Parameter(torch.empty(3 * width, width))
        self.in_proj_bias = nn.Parameter(torch.empty(3 * width))
        self.out_proj = nn.Linear(width, width)


class VariateMix(nn.Module):
    """Pre-norm self-attention across variables at each time step (no time mixing)."""

    def __init__(self, width, heads):
        super().__init__()
        self.norm = nn.LayerNorm(width)
        self.attention = VariateAttention(width, heads)

    def forward(self, x):
        b, v, t, d = x.shape
        attention = self.attention
        heads = attention.num_heads
        flat = x.permute(0, 2, 1, 3).reshape(b * t, v, d)
        qkv = F.linear(
            self.norm(flat), attention.in_proj_weight, attention.in_proj_bias
        ).view(b * t, v, 3, heads, d // heads)
        q, k, value = qkv.permute(2, 0, 3, 1, 4).unbind(0)
        # Scaled dot-product attention with a float32 core rounded once on output,
        # which is what PyTorch's "math" attention kernel does for bfloat16 inputs.
        scale = (d // heads) ** -0.25
        weights = torch.softmax(
            (q.float() * scale) @ (k.float() * scale).transpose(-2, -1), dim=-1
        )
        mixed = (weights @ value.float()).to(x.dtype)
        mixed = mixed.transpose(1, 2).reshape(b * t, v, d)
        return (
            (flat + attention.out_proj(mixed)).reshape(b, t, v, d).permute(0, 2, 1, 3)
        )


def tokenize(config, context, past_covariates=None):
    """Causal patch features; identical to the training repository's evaluation path.

    ``context`` is ``[batch, target, time]`` and ``past_covariates`` is
    ``[batch, covariate, time]`` (same time axis). NaN marks a missing value.
    Each patch is normalised with the running mean/std of everything observed up
    to and including that patch, so no future value leaks into a feature.
    """
    p, limit = config.patch_size, config.context_length
    for name, value in (("context", context), ("past_covariates", past_covariates)):
        if value is None:
            continue
        if value.ndim != 3 or min(value.shape) < 1 or not value.is_floating_point():
            raise ValueError(
                f"{name} must be a nonempty floating [batch, variable, time] tensor"
            )
        if torch.isinf(value).any():
            raise ValueError(
                f"{name} contains infinity; use NaN for missing observations"
            )
    b, targets, length = context.shape
    if past_covariates is not None and (
        past_covariates.shape[0] != b or past_covariates.shape[-1] != length
    ):
        raise ValueError("past_covariates must share the context batch and time axes")
    context = context[..., -limit:]
    history = context
    if past_covariates is not None:
        history = torch.cat([context, past_covariates[..., -limit:]], dim=1)
    roles = [0] * targets + [1] * (history.shape[1] - targets)
    length = history.shape[-1]
    history = F.pad(history.float(), ((-length) % p, 0), value=float("nan"))
    patches = history.reshape(b, len(roles), -1, p)
    valid = torch.isfinite(patches)
    clean = patches.nan_to_num().double()
    count = valid.sum(-1, keepdim=True).cumsum(-2).clamp_min(1)
    loc = clean.sum(-1, keepdim=True).cumsum(-2) / count
    variance = clean.square().sum(-1, keepdim=True).cumsum(-2) / count - loc.square()
    scale = variance.clamp_min(0).sqrt()
    scale = torch.where(
        torch.isfinite(scale) & (scale > 0), scale, torch.ones_like(scale)
    ).float()
    loc = loc.float()
    values = ((patches - loc) / scale).nan_to_num().clamp(-1e4, 1e4)
    time = torch.arange(history.shape[-1]).float().reshape(1, 1, -1, p) / limit
    features = torch.cat([values, valid.float(), time.expand_as(values)], dim=-1)
    return {
        "features": features,
        "roles": torch.tensor(roles),
        "loc": loc,
        "scale": scale,
        "targets": targets,
    }


class BasinBase(nn.Module):
    """Everything except the RWKV-7 blocks, which subclasses provide as ``self.blocks``."""

    def __init__(self, config: BasinConfig):
        super().__init__()
        self.config = config
        self.embedding = ResidualProjection(
            3 * config.patch_size, config.width, 4 * config.width
        )
        self.roles = nn.Embedding(3, config.width)
        self.variate = nn.ModuleList(
            VariateMix(config.width, config.variate_heads) for _ in range(config.layers)
        )
        self.norm = nn.LayerNorm(config.width)
        self.head = ResidualProjection(
            config.width, config.patch_size * len(config.quantiles), 4 * config.width
        )
        if config.output_patches > 1:
            self.lead_embedding = nn.Embedding(config.output_patches - 1, config.width)

    # -- supplied by subclasses -------------------------------------------------
    def rwkv_block(self, layer, x, v_first):
        """Return ``(x, v_first)`` for one RWKV-7 block on ``[sequence, time, width]``."""
        raise NotImplementedError

    # -- shared forward ---------------------------------------------------------
    @property
    def device(self):
        return self.norm.weight.device

    @property
    def dtype(self):
        return self.norm.weight.dtype

    def hidden(self, x, observer=None):
        """``x`` is ``[batch, variable, time, width]``; mixes time, then variables, per layer."""
        b, v, t, d = x.shape
        v_first = None
        for layer, mix in enumerate(self.variate):
            flat, v_first = self.rwkv_block(layer, x.reshape(b * v, t, d), v_first)
            x = mix(flat.reshape(b, v, t, d))
            if observer is not None:
                observer(layer, x)
        return x

    def quantiles(self, prepared, patches, observer=None):
        """Tokenised request -> ``[batch, target, patches * patch_size, quantile]`` float32."""
        device = self.device
        features = prepared["features"].to(device=device, dtype=self.dtype)
        x = self.embedding(features)
        x = x + self.roles(prepared["roles"].to(device))[None, :, None, :]
        return self.decode(self.hidden(x, observer), prepared, patches)

    def decode(self, x, prepared, patches):
        """Final hidden state of each target -> de-normalised, ordered quantiles."""
        config = self.config
        if not 1 <= patches <= config.output_patches:
            raise ValueError("Requested block exceeds trained output_patches")
        targets = prepared["targets"]
        last = x[:, :targets, -1:]
        # Lead 0 uses the hidden state as is; later patches add a learned lead offset.
        offsets = last.new_zeros(patches, last.shape[-1])
        if patches > 1:
            offsets[1:] = self.lead_embedding.weight[: patches - 1]
        raw = self.head(self.norm(last + offsets[None, None])).float()
        raw = (
            raw.reshape(
                last.shape[0],
                targets,
                patches * config.patch_size,
                len(config.quantiles),
            )
            .sort(dim=-1)
            .values
        )
        loc = prepared["loc"][:, :targets, -1].to(x.device)
        scale = prepared["scale"][:, :targets, -1].to(x.device)
        return raw * scale[..., None] + loc[..., None]

    @torch.inference_mode()
    def forecast(self, context, horizon=None, past_covariates=None):
        """Forecast ``horizon`` steps after ``context``.

        Args:
            context: ``[target, time]`` or ``[batch, target, time]`` history (NaN = missing).
            horizon: number of future steps; defaults to one direct block
                (``patch_size * output_patches``). Longer horizons are rolled out by
                feeding the median forecast back as history, as in training-repo evaluation.
            past_covariates: optional ``[covariate, time]`` / ``[batch, covariate, time]``
                observed over the same time steps as ``context``.

        Returns:
            ``BasinForecast`` with CPU float32 ``point`` (median) and ``quantiles``.
        """
        config = self.config
        block = config.patch_size * config.output_patches
        horizon = block if horizon is None else horizon
        if isinstance(horizon, bool) or not isinstance(horizon, int) or horizon < 1:
            raise ValueError("horizon must be a positive integer")
        history = torch.as_tensor(context, dtype=torch.float32).cpu()
        past = (
            None
            if past_covariates is None
            else torch.as_tensor(past_covariates, dtype=torch.float32).cpu()
        )
        unbatched = history.ndim == 2
        if unbatched:
            history = history[None]
            past = None if past is None else past[None]
        was_training = self.training
        self.eval()
        try:
            outputs = []
            for offset in range(0, horizon, block):
                patches = min(
                    config.output_patches,
                    (horizon - offset + config.patch_size - 1) // config.patch_size,
                )
                prepared = tokenize(config, history, past)
                part = self.quantiles(prepared, patches).float().cpu()
                if not torch.isfinite(part).all():
                    raise FloatingPointError("Nonfinite forecast quantiles")
                outputs.append(part)
                if offset + block >= horizon:
                    break
                point = part[..., config.median_index]
                history = torch.cat([history, point], dim=-1)[
                    ..., -config.context_length :
                ]
                if past is not None:
                    missing = past.new_full(
                        (*past.shape[:-1], point.shape[-1]), float("nan")
                    )
                    past = torch.cat([past, missing], dim=-1)[
                        ..., -config.context_length :
                    ]
            quantiles = torch.cat(outputs, dim=-2)[..., :horizon, :]
        finally:
            self.train(was_training)
        if unbatched:
            quantiles = quantiles[0]
        return BasinForecast(quantiles[..., config.median_index], quantiles)

    # -- loading ----------------------------------------------------------------
    @classmethod
    def _construct(cls, config):
        # Every tensor is restored from the checkpoint, so skip random initialisation.
        with torch.device("meta"):
            return cls(config)

    @classmethod
    def from_pretrained(
        cls, path, *, device="cpu", dtype=torch.bfloat16, revision=None
    ):
        """Load from a local folder or, with ``huggingface_hub`` installed, a Hub repo id.

        The default ``dtype=torch.bfloat16`` is the precision the model was trained
        and evaluated in: bfloat16 weights and activations, with float32 used inside
        the same steps as the training kernels (state recurrence, gates, norms).
        ``dtype=torch.float32`` widens the stored weights losslessly.
        """
        from safetensors.torch import load_file

        folder = Path(path)
        if not folder.is_dir():
            from huggingface_hub import snapshot_download

            folder = Path(snapshot_download(str(path), revision=revision))
        settings = json.loads((folder / CONFIG_NAME).read_text(encoding="utf-8"))
        config = BasinConfig.from_dict(settings["model"])
        model = cls._construct(config)
        state = load_file(str(folder / WEIGHTS_NAME))
        model.load_state_dict(state, strict=True, assign=True)
        return model.to(device=device, dtype=dtype).eval().requires_grad_(False)

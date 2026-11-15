"""Inference-only stateful execution of existing trained RWKV and variate weights.

No state is shared across requests. Prefix features must remain unchanged to reuse
state: callers must reset when sliding the context window or changing alignment.
"""

from functools import lru_cache
from pathlib import Path

import torch


@lru_cache(maxsize=1)
def load_kernel():
    from torch.utils.cpp_extension import load

    from rwkv7_cuda_paths import configure_cuda_home

    configure_cuda_home()

    load(
        name="basin_recurrent_v1",
        sources=[
            str(Path(__file__).with_name("recurrent_kernel" + suffix))
            for suffix in (".cpp", ".cu")
        ],
        extra_cuda_cflags=["-O3", "--use_fast_math"],
        is_python_module=False,
        verbose=False,
    )
    return torch.ops.basin_recurrent_v1.forward


def empty_state(model, batch, variables):
    # The backbone's dtype: with fp32_io the first parameter is float32.
    weight = model.roles.weight
    width = model.config.width
    return [
        (
            weight.new_zeros(batch * variables, 1, width),
            torch.zeros(
                batch * variables,
                width // 64,
                64,
                64,
                device=weight.device,
                dtype=torch.float32,
            ),
            weight.new_zeros(batch * variables, 1, width),
        )
        for _ in model.blocks
    ]


def recurrent_hidden(model, x, states=None):
    """Consume real tokens only; mutate previous inputs and matrix states in place."""
    # Reuse the backend's fused BF16 rounding, gates and normalization.
    import sys

    backend = sys.modules[type(model.blocks[0]).__module__]
    recurrence = load_kernel()
    b, variables, tokens, width = x.shape
    if states is not None and model.hybrid:
        # Temporal attention reads every earlier patch; a rollout that keeps
        # only RWKV state cannot supply them.
        raise NotImplementedError(
            "Stateful recurrent rollout is unavailable for models with temporal "
            "attention; use the default full-context forecast"
        )
    first = None
    for layer, (block, state) in enumerate(
        zip(model.blocks, states or [None] * len(model.blocks))
    ):
        flat = x.reshape(b * variables, tokens, width).contiguous()
        if block.layer_id == 0:
            flat = block.ln0(flat)
        normalized = block.ln1(flat)
        att = block.att
        # Prepending exactly one previous activation preserves the fused shift
        # kernel's arithmetic and avoids adding another custom mixed-input op.
        joined = (
            normalized if state is None else torch.cat((state[0], normalized), dim=1)
        )
        mixed = backend.tmix_mix6_bf16_v5(
            joined, *(getattr(att, "x_" + name).view(-1) for name in "rwkvag")
        )
        xr, xw, xk, xv, xa, xg = (
            mixed if state is None else (z[:, 1:].contiguous() for z in mixed)
        )
        if state is None:
            matrix = torch.zeros(
                b * variables, width // 64, 64, 64, dtype=torch.float32, device=x.device
            )
        else:
            previous_att, matrix, previous_ffn = state
            previous_att.copy_(normalized[:, -1:])
        r = att.receptance(xr)
        w = att.w0 + torch.tanh(xw @ att.w1) @ att.w2
        k, v = att.key(xk), att.value(xv)
        if block.layer_id == 0:
            first = v
        else:
            v = backend.tmix_vres_gate_bf16_v3(v, first, att.v0, (xv @ att.v1) @ att.v2)
        a = backend.tmix_a_gate_bf16(att.a0, (xa @ att.a1) @ att.a2)
        g = torch.sigmoid(xg @ att.g1) @ att.g2
        k, neg_kk, kka = backend.tmix_kk_pre_bf16_v5(
            k, att.k_k.view(-1), a, att.k_a.view(-1)
        )
        y = recurrence(r, w, k, v, neg_kk, kka, matrix)
        y = backend.tmix_lnx_rkvres_xg_bf16_v1(
            y, r, k, v, att.r_k, att.ln_x.weight, att.ln_x.bias, g
        )
        flat = flat + att.output(y)
        normalized = block.ln2(flat)
        if state is None:
            flat = flat + block.ffn(normalized)
        else:
            flat = flat + block.ffn(torch.cat((previous_ffn, normalized), dim=1))[:, 1:]
            previous_ffn.copy_(normalized[:, -1:])
        x = model.mix(layer, flat.reshape(b, variables, tokens, width))
    return x


def stateless_block(model, prepared):
    """Forecast block for scoring: (quantiles, model-space quantiles, loc, scale).

    Validation measures the loss in model space, exactly as training does, so
    the direct head returns its quantiles before they are mapped back to the
    original units. Heads without a model space return quantiles only.
    """
    x = recurrent_hidden(model, prepared["encoded"])
    b, targets = prepared["b"], prepared["targets"]
    patches = prepared.get("prediction_patches", 1)
    indices = prepared["last_token_indices"].view(b, 1, 1, 1)
    selected = x[:, :targets].gather(2, indices.expand(b, targets, 1, x.shape[-1]))
    loc = prepared["loc"][:, :targets].gather(2, indices.expand(b, targets, 1, 1))
    scale = prepared["scale"][:, :targets].gather(2, indices.expand(b, targets, 1, 1))
    loc, scale = loc[:, :, 0], scale[:, :, 0]
    from multi_horizon import direct_quantiles, from_model_space, lead_quantiles

    if model.config.head != "direct":
        return direct_quantiles(model, selected, loc, scale, patches), loc, scale
    normalized = lead_quantiles(model, selected, patches).reshape(
        b, targets, patches * model.config.patch_size, len(model.config.quantiles)
    )
    prediction = from_model_space(model, normalized) * scale[..., None] + loc[..., None]
    return prediction, normalized, loc, scale


def stateless_quantiles(model, prepared):
    """Same padded full-context graph, with a current-stream-safe recurrence op."""
    x = recurrent_hidden(model, prepared["encoded"])
    b, targets = prepared["b"], prepared["targets"]
    indices = prepared["last_token_indices"].view(b, 1, 1, 1)
    selected = x[:, :targets].gather(2, indices.expand(b, targets, 1, x.shape[-1]))
    loc = prepared["loc"][:, :targets].gather(2, indices.expand(b, targets, 1, 1))
    scale = prepared["scale"][:, :targets].gather(2, indices.expand(b, targets, 1, 1))
    if prepared.get("prediction_patches", 1) > 1 or model.config.head == "direct":
        from multi_horizon import direct_quantiles

        return direct_quantiles(
            model,
            selected,
            loc[:, :, 0],
            scale[:, :, 0],
            prepared.get("prediction_patches", 1),
        )
    raw = (
        model.read_out(selected)
        .float()
        .reshape(b, targets, model.config.patch_size, len(model.config.quantiles))
        .sort(dim=-1)
        .values
    )
    return raw * scale[:, :, 0, :, None] + loc[:, :, 0, :, None]


def recurrent_quantiles(model, prepared, states, *, last_only=False):
    features = prepared["features"]
    if last_only:
        features = features[:, :, -1:]
    x = model.embedding(features.to(model.embedding.skip.weight.dtype))
    roles = model.roles(prepared["roles"])[None, :, None, :]
    x = x.to(roles.dtype) + roles
    x = recurrent_hidden(model, x, states)
    targets = prepared["targets"]
    if model.config.head == "direct":
        from multi_horizon import direct_quantiles

        return direct_quantiles(
            model,
            x[:, :targets, -1:],
            prepared["loc"][:, :targets, -1],
            prepared["scale"][:, :targets, -1],
            1,
        )
    raw = model.read_out(x[:, :targets, -1:]).float()
    raw = (
        raw.reshape(
            prepared["b"], targets, model.config.patch_size, len(model.config.quantiles)
        )
        .sort(dim=-1)
        .values
    )
    return (
        raw * prepared["scale"][:, :targets, -1:, :]
        + prepared["loc"][:, :targets, -1:, :]
    )

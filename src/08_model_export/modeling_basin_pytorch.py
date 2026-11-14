"""Basin forecaster with RWKV-7 blocks written entirely in PyTorch.

No custom CUDA kernels, no Triton, no compiler: runs on CPU or any PyTorch
device. Parameter names match the training checkpoint one-to-one.

Precision. The model was trained in mixed precision: bfloat16 weights and
activations, with fused CUDA kernels that compute in float32 and round once on
output. The functions below marked "fused" keep exactly those float32 islands:

* ``dtype=torch.bfloat16`` (default): the training precision, rounding at the
  same points as the training kernels.
* ``dtype=torch.float32``: ordinary float32 end to end.

Usage::

    from modeling_basin_pytorch import BasinPyTorch
    model = BasinPyTorch.from_pretrained("path/or/hub-id", device="cpu")
    forecast = model.forecast(context, horizon=64, past_covariates=covariates)
    forecast.point, forecast.quantiles
"""

from __future__ import annotations

import torch
from basin_common import BasinBase, BasinConfig, BasinForecast, tokenize  # noqa: F401
from torch import nn
from torch.nn import functional as F

DECAY_SCALE = -0.6065306597126334  # -exp(-0.5): decay = exp(DECAY_SCALE * sigmoid(w))
GROUP_NORM_EPS = 64e-5
NORMALIZE_EPS = 1e-12


def token_mix(x, *coefficients):
    """fused: blend each token with its predecessor (zeros before the first token).

    ``addcmul`` rounds x + delta * c once, like the training kernel's fused multiply-add.
    """
    delta = F.pad(x, (0, 0, 1, 0))[:, :-1] - x
    return tuple(torch.addcmul(x, delta, c) for c in coefficients)


def sigmoid_gate(offset, low_rank):
    """fused: sigmoid(offset + low_rank), rounded once."""
    return torch.sigmoid(offset.float() + low_rank.float()).to(low_rank.dtype)


def value_residual(v, v_first, offset, low_rank):
    """fused: move this layer's value towards the first layer's value."""
    gate = torch.sigmoid(offset.float() + low_rank.float())
    v32 = v.float()
    return (v32 + (v_first.float() - v32) * gate).to(v.dtype)


def key_update(k, k_k, a, k_a, head_size):
    """fused: returns (learning-rate-scaled key, -unit key, unit key * rate)."""
    k32, a32, k_a = k.float(), a.float(), k_a.float()
    shape = (*k.shape[:-1], -1, head_size)
    unit = F.normalize(
        (k32 * k_k.float()).view(shape), dim=-1, p=2.0, eps=NORMALIZE_EPS
    )
    unit = unit.view(k.shape)
    return (
        (k32 * (a32 * k_a + (1.0 - k_a))).to(k.dtype),
        (-unit).to(k.dtype),
        (unit * a32).to(k.dtype),
    )


def wkv7(r, w, k, v, a, b, head_size):
    """fused: RWKV-7 state recurrence with a float32 state, rounded on output.

    Per head, with state ``S[value, key]`` starting at zero:
        S <- S * decay + (S a) b^T + v k^T,      y = S r
    Time steps are sequential; each step is batched over sequences and heads.
    """
    n, t, c = r.shape
    heads = (n, t, c // head_size, head_size)
    r32, w32, k32, v32, a32, b32 = (z.float().view(heads) for z in (r, w, k, v, a, b))
    decay = torch.exp(DECAY_SCALE * torch.sigmoid(w32))
    step = _wkv7_matrix_steps if r.is_cuda else _wkv7_vector_steps
    return step(r32, decay, k32, v32, a32, b32).reshape(n, t, c).to(r.dtype)


def _wkv7_vector_steps(r, decay, k, v, a, b):
    """Five small operations per time step, minimal arithmetic: best on CPU."""
    n, t, h, d = r.shape
    decay = decay.unsqueeze(-2).unbind(1)
    r_t, a_t = r.unsqueeze(-1).unbind(1), a.unsqueeze(-1).unbind(1)
    k_t, b_t = k.unsqueeze(-2).unbind(1), b.unsqueeze(-2).unbind(1)
    v_t = v.unsqueeze(-1).unbind(1)
    state = r.new_zeros(n, h, d, d)
    outputs = []
    for step in range(t):
        update = (state @ a_t[step]) * b_t[step]
        update.addcmul_(v_t[step], k_t[step])
        state = torch.addcmul(update, state, decay[step])
        outputs.append(state @ r_t[step])
    return torch.stack(outputs, dim=1).squeeze(-1)


def _wkv7_matrix_steps(r, decay, k, v, a, b, chunk=32):
    """One operation per time step: best on GPU, where launches cost more than arithmetic.

    The same recurrence written as  S <- S (diag(decay) + a b^T) + v k^T.  The
    per-step matrices and the read-out S r are built for ``chunk`` steps at a time,
    which bounds the extra memory.
    """
    n, t, h, d = r.shape

    def time_major(z):  # [n, steps, h, ...] -> [steps, n * h, ...]
        return z.transpose(0, 1).reshape(z.shape[1], n * h, *z.shape[3:])

    state = r.new_zeros(n * h, d, d)
    outputs = []
    # The state needs full float32 products; TF32 (if the caller enabled it) is too coarse.
    tf32 = torch.backends.cuda.matmul.allow_tf32
    torch.backends.cuda.matmul.allow_tf32 = False
    try:
        _matrix_chunks(r, decay, k, v, a, b, chunk, state, outputs, time_major)
    finally:
        torch.backends.cuda.matmul.allow_tf32 = tf32
    return torch.cat(outputs).view(t, n, h * d).transpose(0, 1)


def _matrix_chunks(r, decay, k, v, a, b, chunk, state, outputs, time_major):
    t = r.shape[1]
    for start in range(0, t, chunk):
        piece = slice(start, start + chunk)
        transition = torch.diag_embed(decay[:, piece])
        transition = transition + a[:, piece].unsqueeze(-1) * b[:, piece].unsqueeze(-2)
        update = v[:, piece].unsqueeze(-1) * k[:, piece].unsqueeze(-2)
        states = []
        for step_update, step_transition in zip(
            time_major(update).unbind(0), time_major(transition).unbind(0)
        ):
            state = torch.baddbmm(step_update, state, step_transition)
            states.append(state)
        read = time_major(r[:, piece]).unsqueeze(-1)
        outputs.append((torch.stack(states) @ read).squeeze(-1))


def normalize_bonus_gate(y, r, k, v, r_k, weight, bias, g, head_size):
    """fused: per-head group norm, plus the (r . k) value bonus, times the output gate."""
    n, t, c = y.shape
    heads = (n, t, c // head_size, head_size)
    y32, v32 = y.float().view(heads), v.float().view(heads)
    centered = y32 - y32.mean(-1, keepdim=True)
    rstd = torch.rsqrt(centered.square().mean(-1, keepdim=True) + GROUP_NORM_EPS)
    bonus = (r.float().view(heads) * k.float().view(heads) * r_k.float()).sum(
        -1, keepdim=True
    )
    out = (centered * rstd).view(n, t, c) * weight.float() + bias.float()
    out = out + (bonus * v32).view(n, t, c)
    return (out * g.float()).to(y.dtype)


class TimeMix(nn.Module):
    def __init__(self, config: BasinConfig, layer_id):
        super().__init__()
        width = config.width
        self.layer_id = layer_id
        self.head_size = config.head_size
        for name in "rwkvag":
            setattr(self, "x_" + name, nn.Parameter(torch.empty(1, 1, width)))
        for name, rank in (
            ("w", config.decay_low_rank_dim),
            ("a", config.a_low_rank_dim),
            ("v", config.v_low_rank_dim),  # unused in layer 0, kept for name parity
        ):
            setattr(self, name + "1", nn.Parameter(torch.empty(width, rank)))
            setattr(self, name + "2", nn.Parameter(torch.empty(rank, width)))
            setattr(self, name + "0", nn.Parameter(torch.empty(1, 1, width)))
        self.g1 = nn.Parameter(torch.empty(width, config.gate_low_rank_dim))
        self.g2 = nn.Parameter(torch.empty(config.gate_low_rank_dim, width))
        self.k_k = nn.Parameter(torch.empty(1, 1, width))
        self.k_a = nn.Parameter(torch.empty(1, 1, width))
        self.r_k = nn.Parameter(
            torch.empty(width // config.head_size, config.head_size)
        )
        self.receptance = nn.Linear(width, width, bias=False)
        self.key = nn.Linear(width, width, bias=False)
        self.value = nn.Linear(width, width, bias=False)
        self.output = nn.Linear(width, width, bias=False)
        self.ln_x = nn.GroupNorm(width // config.head_size, width, eps=GROUP_NORM_EPS)

    def forward(self, x, v_first):
        xr, xw, xk, xv, xa, xg = token_mix(
            x, self.x_r, self.x_w, self.x_k, self.x_v, self.x_a, self.x_g
        )
        r = self.receptance(xr)
        w = self.w0 + torch.tanh(xw @ self.w1) @ self.w2
        k = self.key(xk)
        v = self.value(xv)
        if self.layer_id == 0:
            v_first = v
        else:
            v = value_residual(v, v_first, self.v0, (xv @ self.v1) @ self.v2)
        a = sigmoid_gate(self.a0, (xa @ self.a1) @ self.a2)  # in-context learning rate
        g = torch.sigmoid(xg @ self.g1) @ self.g2
        k, removal, replacement = key_update(k, self.k_k, a, self.k_a, self.head_size)
        y = wkv7(r, w, k, v, removal, replacement, self.head_size)
        y = normalize_bonus_gate(
            y, r, k, v, self.r_k, self.ln_x.weight, self.ln_x.bias, g, self.head_size
        )
        return self.output(y), v_first


class ChannelMix(nn.Module):
    def __init__(self, config: BasinConfig):
        super().__init__()
        self.x_k = nn.Parameter(torch.empty(1, 1, config.width))
        self.key = nn.Linear(config.width, config.ffn_width, bias=False)
        self.value = nn.Linear(config.ffn_width, config.width, bias=False)

    def forward(self, x):
        hidden = torch.relu(self.key(token_mix(x, self.x_k)[0]))
        return self.value(hidden * hidden)


class Block(nn.Module):
    def __init__(self, config: BasinConfig, layer_id):
        super().__init__()
        self.layer_id = layer_id
        self.ln1 = nn.LayerNorm(config.width)
        self.ln2 = nn.LayerNorm(config.width)
        if layer_id == 0:
            self.ln0 = nn.LayerNorm(config.width)
        self.att = TimeMix(config, layer_id)
        self.ffn = ChannelMix(config)

    def forward(self, x, v_first):
        if self.layer_id == 0:
            x = self.ln0(x)
        mixed, v_first = self.att(self.ln1(x), v_first)
        x = x + mixed
        return x + self.ffn(self.ln2(x)), v_first


class BasinPyTorch(BasinBase):
    def __init__(self, config: BasinConfig):
        super().__init__(config)
        self.blocks = nn.ModuleList(Block(config, i) for i in range(config.layers))

    def rwkv_block(self, layer, x, v_first):
        return self.blocks[layer](x, v_first)

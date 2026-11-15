"""Causal temporal attention, interleaved with RWKV blocks in the hybrid backbone.

RWKV compresses a variable's history into a fixed-size recurrent state. This
block lets every patch position read all earlier patch positions of the same
variable directly, so the position a forecast is read from is not limited to
what survived that compression. The formulation follows the attention block of
HART (docs/HART_hybrid_rkwv_transformer_tsfm.pdf, Appendix E.1): RMSNorm,
rotary positions on queries and keys, causal self-attention, a sigmoid gate on
the attention output, and an output projection. Unlike HART it has no
feed-forward network of its own; the RWKV blocks around it supply those.

The output projection starts at zero, so a freshly initialised model computes
exactly what the pure RWKV stack computes and learns how much to rely on it.
Right padding needs no mask: attention is causal, so a real position never
reads a later, padded one.
"""

import torch
from torch import nn
from torch.nn import functional as F


def rotary_angles(positions, head_size, *, base=10000.0):
    """Cosines and sines for rotary position encoding, shape [positions, head_size/2]."""
    if head_size % 2:
        raise ValueError("Rotary positions need an even head size")
    frequency = 1.0 / base ** (torch.arange(0, head_size, 2).float() / head_size)
    angle = torch.arange(positions).float()[:, None] * frequency[None]
    return angle.cos(), angle.sin()


def rotate(x, cosine, sine):
    """Rotate consecutive feature pairs of x [..., time, head_size] by position."""
    even, odd = x[..., 0::2], x[..., 1::2]
    return torch.stack(
        (even * cosine - odd * sine, even * sine + odd * cosine), dim=-1
    ).flatten(-2)


class TemporalAttention(nn.Module):
    def __init__(self, width, max_patches, *, head_size=64):
        super().__init__()
        if width % head_size:
            raise ValueError("width must be a multiple of the attention head size")
        self.heads = width // head_size
        self.norm = nn.RMSNorm(width)
        self.query_key_value = nn.Linear(width, 3 * width, bias=False)
        self.gate = nn.Linear(width, width, bias=False)
        self.output = nn.Linear(width, width, bias=False)
        nn.init.zeros_(self.output.weight)
        self.head_size = head_size
        self.max_patches = max_patches
        # Rotary tables are derived constants, built on first use per length,
        # device and dtype. They are deliberately not buffers: a model that is
        # constructed empty and filled from a checkpoint (RWKVForecaster.load)
        # would keep uninitialised memory in any buffer the checkpoint lacks.
        self._angles = {}

    def angles(self, length, reference):
        key = (length, reference.device, reference.dtype)
        cached = self._angles.get(key)
        if cached is None:
            if length > self.max_patches:
                raise ValueError("Sequence exceeds the configured max_patches")
            # Outside inference mode, so tables first built during validation
            # can be reused by the next training update.
            with torch.inference_mode(False):
                cached = tuple(
                    table.to(device=reference.device, dtype=reference.dtype)
                    for table in rotary_angles(length, self.head_size)
                )
            self._angles[key] = cached
        return cached

    def forward(self, x):
        """x: [batch, variable, time, width]; time is the only attention axis."""
        b, v, t, d = x.shape
        flat = x.reshape(b * v, t, d)
        normalized = self.norm(flat)
        query, key, value = (
            self.query_key_value(normalized)
            .view(b * v, t, 3, self.heads, d // self.heads)
            .permute(2, 0, 3, 1, 4)
        )
        cosine, sine = self.angles(t, query)
        attended = F.scaled_dot_product_attention(
            rotate(query, cosine, sine),
            rotate(key, cosine, sine),
            value,
            is_causal=True,
        )
        attended = attended.transpose(1, 2).reshape(b * v, t, d)
        gated = attended * torch.sigmoid(self.gate(normalized))
        return (flat + self.output(gated)).reshape(b, v, t, d)

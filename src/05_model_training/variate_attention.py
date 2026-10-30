"""Optional compiled execution of the existing cross-variable self-attention."""

import os
from functools import lru_cache
from pathlib import Path

import torch
from loguru import logger
from torch.nn import functional as F
from torch.nn.attention import SDPBackend, sdpa_kernel


def projected_attention(mix, x):
    """Same Q/K/V and output weights as MultiheadAttention, batch-first throughout."""
    b, v, t, d = x.shape
    flat = x.permute(0, 2, 1, 3).reshape(b * t, v, d)
    attention = mix.attention
    qkv = F.linear(
        mix.norm(flat), attention.in_proj_weight, attention.in_proj_bias
    ).view(b * t, v, 3, attention.num_heads, d // attention.num_heads)
    q, k, value = qkv.permute(2, 0, 3, 1, 4).unbind(0)
    # Short variable axes benefit from the compiled math path. The default
    # flash kernel is designed for much longer attention sequences.
    with sdpa_kernel(SDPBackend.MATH):
        mixed = F.scaled_dot_product_attention(q, k, value)
    mixed = mixed.transpose(1, 2).reshape(b * t, v, d)
    return (flat + attention.out_proj(mixed)).reshape(b, t, v, d).permute(0, 2, 1, 3)


@lru_cache(maxsize=1)
def compiled_attention():
    # A function, not an OptimizedModule: checkpoint names and parameter identity
    # stay exactly the same, and all layers share the compiled implementation.
    return torch.compile(projected_attention, dynamic=True, fullgraph=True)


def configure_variate_execution(model, training, root):
    enabled = training.get("compile_variate", True)
    inference = training.get("compile_variate_inference", True)
    cuda = next(model.parameters()).device.type == "cuda"
    if (enabled or inference) and cuda:
        cache = Path(root) / training.get("compile_cache_dir", ".cache/torchinductor")
        cache.mkdir(parents=True, exist_ok=True)
        os.environ.setdefault("TORCHINDUCTOR_CACHE_DIR", str(cache))
        os.environ.setdefault("TRITON_CACHE_DIR", str(cache / "triton"))
        forward = compiled_attention()
        logger.info(
            "Compiled variate attention for up to 8 variables: training={} inference={}; cache={}. "
            "First calls for new shapes include compilation.",
            enabled,
            inference,
            os.environ["TORCHINDUCTOR_CACHE_DIR"],
        )
    else:
        forward = None
        logger.info(
            "Compiled variate attention disabled (requested={} device={})",
            enabled,
            next(model.parameters()).device,
        )
    for mix in model.variate:
        mix.compiled_forward = forward
        mix.compile_training = enabled
        mix.compile_inference = inference
    return forward is not None

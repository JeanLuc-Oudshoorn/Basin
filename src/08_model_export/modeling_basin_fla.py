"""Basin forecaster with RWKV-7 blocks from ``flash-linear-attention`` (FLA).

The time/channel mixing layers are FLA's own ``RWKV7Block`` (Triton kernels);
weights are stored under FLA's parameter names. Requirements:
``pip install flash-linear-attention`` plus a Triton-capable setup, which in
practice means an NVIDIA GPU on Linux (Triton JIT-compiles the kernels on first
use and needs gcc or clang for its launcher stub, but no CUDA toolkit). For a
CPU-only or Windows-native install use the pure PyTorch variant instead.

FLA runs sequences of 64+ tokens with a chunked algorithm that is faster to
train but numerically looser. By default this model keeps every length on FLA's
exact recurrent kernel by feeding at most 63 tokens at a time and carrying
FLA's own state cache between pieces. Set ``model.exact_recurrence = False`` to
use FLA's default dispatch.

Usage::

    from modeling_basin_fla import BasinFLA
    model = BasinFLA.from_pretrained("path/or/hub-id", device="cuda")
    forecast = model.forecast(context, horizon=64, past_covariates=covariates)
"""

from __future__ import annotations

import warnings

import torch
from basin_common import BasinBase, BasinConfig, BasinForecast, tokenize  # noqa: F401
from torch import nn


def fla_config(config: BasinConfig):
    from fla.models.rwkv7.configuration_rwkv7 import RWKV7Config

    return RWKV7Config(
        attn_mode="chunk",
        hidden_size=config.width,
        intermediate_size=config.ffn_width,
        num_hidden_layers=config.layers,
        head_dim=config.head_size,
        decay_low_rank_dim=config.decay_low_rank_dim,
        gate_low_rank_dim=config.gate_low_rank_dim,
        a_low_rank_dim=config.a_low_rank_dim,
        v_low_rank_dim=config.v_low_rank_dim,
        hidden_act="sqrelu",
        norm_first=True,
        norm_bias=True,
        norm_eps=1e-5,  # group-norm eps = head_dim * norm_eps = 64e-5, as trained
        fuse_norm=False,  # torch LayerNorm/GroupNorm, as trained
        vocab_size=1,  # unused: Basin embeds patches, not tokens
        bos_token_id=None,
        eos_token_id=None,
    )


# FLA picks its recurrent kernel below this many tokens (inference, no gradients).
RECURRENT_TOKENS = 63


class BasinFLA(BasinBase):
    def __init__(self, config: BasinConfig):
        super().__init__(config)
        from fla.models.rwkv7.modeling_rwkv7 import RWKV7Block

        settings = fla_config(config)
        with warnings.catch_warnings():
            # FLA warns once per layer that its RWKV-7 should be cross-checked against
            # BlinkDL/RWKV-LM. This port was cross-checked against the BlinkDL-kernel
            # original (see the audit report shipped with the export).
            warnings.filterwarnings("ignore", message="According to Bo")
            self.blocks = nn.ModuleList(
                RWKV7Block(settings, i) for i in range(config.layers)
            )

    exact_recurrence = True

    @classmethod
    def _construct(cls, config):
        # FLA layers randomly initialise themselves (slow orthogonal init) unless this
        # transformers switch is off; every tensor is restored from the checkpoint.
        import transformers.modeling_utils as switches

        previous = getattr(switches, "_init_weights", True)
        switches._init_weights = False
        try:
            with torch.device("meta"):
                return cls(config)
        finally:
            switches._init_weights = previous

    def rwkv_block(self, layer, x, v_first):
        block, tokens = self.blocks[layer], x.shape[1]
        if not self.exact_recurrence or tokens <= RECURRENT_TOKENS:
            x, _, _, v_first = block(x.contiguous(), v_first=v_first)
            return x, v_first
        from fla.models.cache_utils import Cache

        cache, outputs, firsts = Cache(), [], []
        for start in range(0, tokens, RECURRENT_TOKENS):
            piece = slice(start, start + RECURRENT_TOKENS)
            out, _, cache, first = block(
                x[:, piece].contiguous(),
                past_key_values=cache,
                use_cache=True,
                v_first=None if v_first is None else v_first[:, piece],
            )
            outputs.append(out)
            firsts.append(first)
        return torch.cat(outputs, dim=1), torch.cat(firsts, dim=1)


def convert_state_dict(state, config: BasinConfig):
    """Training-checkpoint names/layouts -> FLA ``RWKV7Block`` names/layouts.

    Low-rank matrices are stored transposed (FLA uses ``nn.Linear``), the low-rank
    offsets become the second linear layer's bias, and layer 0's value-residual
    parameters are dropped because RWKV-7 never uses them in the first layer.
    """
    renamed = {
        "ln0": "pre_norm",
        "ln1": "attn_norm",
        "ln2": "ffn_norm",
        "att.receptance": "attn.r_proj",
        "att.key": "attn.k_proj",
        "att.value": "attn.v_proj",
        "att.output": "attn.o_proj",
        "att.ln_x": "attn.g_norm",
    }
    result, dropped = {}, []
    for name, tensor in state.items():
        if not name.startswith("blocks."):
            result[name] = tensor
            continue
        _, layer, rest = name.split(".", 2)
        prefix = f"blocks.{layer}."
        module, _, leaf = rest.rpartition(".")
        if module in renamed:
            result[f"{prefix}{renamed[module]}.{leaf}"] = tensor
        elif module == "ffn" and leaf == "x_k":
            result[prefix + "ffn.x_k"] = tensor.reshape(-1)
        elif module.startswith("ffn"):
            result[prefix + rest] = tensor
        elif leaf in ("x_r", "x_w", "x_k", "x_v", "x_a", "x_g", "r_k"):
            result[prefix + "attn." + leaf] = tensor
        elif leaf in ("k_k", "k_a"):
            result[prefix + "attn." + leaf] = tensor.reshape(-1)
        elif leaf[0] in "wavg" and leaf[1] in "012":
            if leaf[0] == "v" and int(layer) == 0:
                dropped.append(name)
                continue
            lora = f"{prefix}attn.{leaf[0]}_lora.lora."
            if leaf[1] == "1":
                result[lora + "0.weight"] = tensor.t().contiguous()
            elif leaf[1] == "2":
                result[lora + "2.weight"] = tensor.t().contiguous()
            else:
                result[lora + "2.bias"] = tensor.reshape(-1)
        else:
            raise KeyError(f"Unmapped checkpoint tensor: {name}")
    return result, dropped

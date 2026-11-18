"""CPU-only checks of the portable (pure PyTorch / FLA-layout) model code.

The numerical comparison against the original CUDA kernels needs the WSL GPU
runtime and lives in src/08_model_export/verify_blocks.py and verify_ports.py.
"""

import sys
import tempfile
import unittest
from dataclasses import asdict
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
for directory in (
    "src/08_model_export",
    "src/04_patch_indexing",
    "src/03_transforming_input_data",
):
    sys.path.insert(0, str(ROOT / directory))

from basin_common import BasinConfig, tokenize
from export_models import write_variant
from modeling_basin_fla import convert_state_dict
from modeling_basin_pytorch import DECAY_SCALE, BasinPyTorch, wkv7

CONFIG = BasinConfig(
    patch_size=16,
    width=128,
    layers=3,
    variate_heads=4,
    context_length=256,
    output_patches=4,
    head_size=64,
    ffn_width=512,
    decay_low_rank_dim=32,
    a_low_rank_dim=32,
    v_low_rank_dim=32,
    gate_low_rank_dim=32,
)


def tiny_model(seed=0):
    torch.manual_seed(seed)
    model = BasinPyTorch(CONFIG).eval().requires_grad_(False)
    for name, parameter in model.named_parameters():
        if parameter.ndim >= 2 and "embedding" not in name.split(".")[0]:
            torch.nn.init.normal_(parameter, std=parameter.shape[-1] ** -0.5)
        elif name.endswith(
            ("ln_x.weight", "norm.weight", "ln0.weight", "ln1.weight", "ln2.weight")
        ):
            torch.nn.init.ones_(parameter)
        else:
            torch.nn.init.normal_(parameter, std=0.1)
    return model


def series(batch, variables, length, seed):
    generator = torch.Generator().manual_seed(seed)
    return torch.randn(batch, variables, length, generator=generator).cumsum(-1)


class ModelExportTests(unittest.TestCase):
    def test_tokenizer_matches_training_encoder(self):
        from index_patches import ForecastConfig, PatchEncoder

        encoder = PatchEncoder(
            ForecastConfig(
                patch_size=16,
                width=128,
                layers=3,
                variate_heads=4,
                max_patches=16,
                objective="next_patch",
                context_length=256,
                output_patches=4,
            )
        )
        for length in (16, 37, 250, 256, 300):
            context, covariates = series(2, 1, length, 1), series(2, 3, length, 2)
            context[0, 0, 3] = float("nan")
            covariates[1, 2, : length // 2] = float("nan")
            expected = encoder.tokenize_next(context, past_covariates=covariates)
            got = tokenize(CONFIG, context, covariates)
            for name in ("features", "roles", "loc", "scale"):
                assert torch.equal(got[name], expected[name]), (length, name)
            assert got["targets"] == expected["targets"]

    def test_recurrence_matches_scalar_reference(self):
        torch.manual_seed(3)
        n, t, heads, d = 2, 9, 2, 4
        r, w, k, v, a, b = (torch.randn(n, t, heads * d) * 0.5 for _ in range(6))
        out = wkv7(r, w, k, v, a, b, d).double().view(n, t, heads, d)
        r, w, k, v, a, b = (z.double().view(n, t, heads, d) for z in (r, w, k, v, a, b))
        for sequence in range(n):
            for head in range(heads):
                state = torch.zeros(d, d, dtype=torch.float64)  # [value, key]
                for step in range(t):
                    decay = torch.exp(
                        DECAY_SCALE * torch.sigmoid(w[sequence, step, head])
                    )
                    for i in range(d):
                        sa = (state[i] * a[sequence, step, head]).sum()
                        state[i] = (
                            state[i] * decay
                            + sa * b[sequence, step, head]
                            + k[sequence, step, head] * v[sequence, step, head, i]
                        )
                    expected = state @ r[sequence, step, head]
                    assert torch.allclose(
                        out[sequence, step, head], expected, atol=1e-5
                    )

    def test_gpu_and_cpu_recurrence_forms_agree(self):
        from modeling_basin_pytorch import _wkv7_matrix_steps, _wkv7_vector_steps

        torch.manual_seed(5)
        r, k, v, a, b = (torch.randn(3, 70, 2, 8) * 0.5 for _ in range(5))
        decay = torch.rand(3, 70, 2, 8) * 0.4 + 0.55
        slow = _wkv7_vector_steps(r, decay, k, v, a, b)
        fast = _wkv7_matrix_steps(r, decay, k, v, a, b).reshape(slow.shape)
        assert torch.allclose(slow, fast, rtol=1e-4, atol=1e-4)

    def test_hidden_states_are_causal(self):
        """Appending tokens must not change earlier positions, so no kernel padding is needed."""
        model = tiny_model()
        x = torch.randn(1, 3, 12, CONFIG.width)
        full, prefix = model.hidden(x), model.hidden(x[:, :, :7])
        assert torch.allclose(full[:, :, :7], prefix, atol=1e-5)

    def test_forecast_shapes_order_and_rollout(self):
        model = tiny_model()
        context, covariates = series(1, 2, 200, 4), series(1, 3, 200, 5)
        block = model.forecast(context, 64, past_covariates=covariates)
        assert block.quantiles.shape == (1, 2, 64, len(CONFIG.quantiles))
        assert torch.equal(block.point, block.quantiles[..., CONFIG.median_index])
        assert (block.quantiles.diff(dim=-1) >= 0).all()
        # A shorter horizon is the prefix of the same direct block.
        short = model.forecast(context, 20, past_covariates=covariates)
        assert torch.allclose(short.quantiles, block.quantiles[:, :, :20], atol=1e-5)
        # Longer horizons roll the median forward; the first block is unchanged.
        long = model.forecast(context, 100, past_covariates=covariates)
        assert long.quantiles.shape == (1, 2, 100, len(CONFIG.quantiles))
        assert torch.allclose(long.quantiles[:, :, :64], block.quantiles, atol=1e-5)
        # Unbatched input gives the same numbers without the batch axis.
        single = model.forecast(context[0], 64, past_covariates=covariates[0])
        assert torch.allclose(single.quantiles, block.quantiles[0], atol=1e-5)
        try:
            model.forecast(context, 0)
        except ValueError:
            pass
        else:
            raise AssertionError("horizon=0 must be rejected")

    def test_batch_rows_do_not_interact(self):
        model = tiny_model()
        context = series(3, 1, 96, 6)
        together = model.forecast(context, 64).quantiles
        for row in range(3):
            alone = model.forecast(context[row : row + 1], 64).quantiles
            assert torch.allclose(together[row : row + 1], alone, atol=1e-4)

    def test_fla_layout_is_a_relabelling_of_the_same_values(self):
        state = tiny_model().state_dict()
        converted, dropped = convert_state_dict(state, CONFIG)
        assert sorted(dropped) == [
            "blocks.0.att.v0",
            "blocks.0.att.v1",
            "blocks.0.att.v2",
        ]
        assert sum(t.numel() for t in converted.values()) + sum(
            state[k].numel() for k in dropped
        ) == sum(t.numel() for t in state.values())
        att = "blocks.1.att."
        fla = "blocks.1.attn."
        assert torch.equal(
            converted[fla + "w_lora.lora.0.weight"], state[att + "w1"].t()
        )
        assert torch.equal(
            converted[fla + "w_lora.lora.2.weight"], state[att + "w2"].t()
        )
        assert torch.equal(
            converted[fla + "w_lora.lora.2.bias"], state[att + "w0"].flatten()
        )
        assert torch.equal(
            converted[fla + "g_lora.lora.0.weight"], state[att + "g1"].t()
        )
        assert fla + "g_lora.lora.2.bias" not in converted
        assert torch.equal(
            converted[fla + "r_proj.weight"], state[att + "receptance.weight"]
        )
        assert torch.equal(converted[fla + "g_norm.bias"], state[att + "ln_x.bias"])
        assert torch.equal(
            converted["blocks.0.pre_norm.weight"], state["blocks.0.ln0.weight"]
        )
        assert torch.equal(
            converted["blocks.2.ffn.x_k"], state["blocks.2.ffn.x_k"].flatten()
        )
        # Everything outside the RWKV blocks keeps its name.
        for name in state:
            if not name.startswith("blocks."):
                assert torch.equal(converted[name], state[name])

    def test_export_folder_round_trip(self):
        model = tiny_model()
        state = {k: v.to(torch.bfloat16) for k, v in model.state_dict().items()}
        settings = {
            "architecture": "basin-rwkv7",
            "model": asdict(CONFIG),
            "source": {"selected_step": 0, "evaluated_weights_sha256": "test"},
        }
        context = series(1, 1, 120, 7)
        with tempfile.TemporaryDirectory() as directory:
            folder = write_variant("pytorch", state, settings, Path(directory))
            assert {p.name for p in folder.iterdir()} == {
                "README.md",
                "basin_common.py",
                "config.json",
                "model.safetensors",
                "modeling_basin_pytorch.py",
            }
            loaded = BasinPyTorch.from_pretrained(folder, dtype=torch.float32)
            # The default is the training precision.
            low = BasinPyTorch.from_pretrained(folder)
        model.load_state_dict({k: v.float() for k, v in state.items()})
        expected = model.forecast(context, 64).quantiles
        assert next(loaded.parameters()).dtype == torch.float32
        assert torch.equal(loaded.forecast(context, 64).quantiles, expected)
        assert next(low.parameters()).dtype == torch.bfloat16
        difference = (low.forecast(context, 64).quantiles - expected).abs().max()
        assert difference < 0.2 * context.std()


if __name__ == "__main__":
    unittest.main()

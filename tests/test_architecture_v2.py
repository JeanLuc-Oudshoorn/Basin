"""Hybrid backbone, direct multi-lead head, transformed targets and the new recipe.

CPU only: RWKV blocks are replaced by a causal stand-in, so these tests cover
Basin's own layers and arithmetic, not the CUDA kernels.
"""

import copy
import json
import sys
import tempfile
import unittest
from dataclasses import asdict
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import torch
from torch import nn

ROOT = Path(__file__).resolve().parents[1]
for stage in (
    "03_transforming_input_data",
    "04_creating_embeddings",
    "05_model_training",
    "06_holdout_test",
):
    sys.path.insert(0, str(ROOT / "src" / stage))
from checkpoints import LEGACY_MODEL_DEFAULTS
from horizon_policy import evaluation_weights
from master_optimizer import FP32MasterAdamW
from multi_horizon import (
    anchors_with_history,
    direct_training_loss,
    from_model_space,
    lead_quantiles,
    to_model_space,
    weighted_label_mass,
)
from performance import make_optimizer, optimizer_options, parameter_groups
from rolling_validation import (
    robust_table_summary,
    select_validation_origins,
    table_mase_profile,
)
from temporal_attention import TemporalAttention
from test_training_performance import CausalBlock
from token_inventory import parameter_budget
from train import (
    ForecastConfig,
    RWKVForecaster,
    apply_upstream_rwkv_init,
    architecture_report,
)
from transform import load_config
from weight_average import WeightAverage, weight_average_options

LEADS = 7


def build(**overrides):
    settings = dict(
        objective="next_patch",
        patch_size=16,
        width=64,
        layers=4,
        variate_heads=4,
        context_length=512,
        max_patches=32,
        output_patches=LEADS,
        horizon_loss_weights="equal",
        head="direct",
        variate_every=2,
        temporal_every=2,
        gated_mixing=True,
        value_transform="asinh",
    )
    settings.update(overrides)
    with patch("train.load_rwkv", return_value=SimpleNamespace(Block=CausalBlock)):
        return RWKVForecaster(ForecastConfig(**settings))


def open_branches(model):
    """Gated branches start closed; give them weights so tests exercise them."""
    for module in model.temporal:
        nn.init.normal_(module.output.weight, std=0.2)
    for module in model.variate:
        nn.init.normal_(module.attention.out_proj.weight, std=0.2)


class ConfigurationTests(unittest.TestCase):
    def test_defaults_describe_the_archived_models(self):
        defaults = asdict(ForecastConfig(objective="next_patch"))
        for key, value in LEGACY_MODEL_DEFAULTS.items():
            self.assertEqual(defaults[key], value, key)
        with patch("train.load_rwkv", return_value=SimpleNamespace(Block=CausalBlock)):
            legacy = RWKVForecaster(
                ForecastConfig(
                    objective="next_patch",
                    patch_size=16,
                    width=64,
                    layers=3,
                    context_length=256,
                    max_patches=16,
                    output_patches=4,
                    horizon_loss_weights=(0.8, 0.16, 0.03, 0.01),
                )
            )
        names = set(legacy.state_dict())
        self.assertIn("lead_embedding.weight", names)
        self.assertFalse(any(name.startswith("temporal.") for name in names))
        self.assertEqual(len(legacy.variate), 3)
        self.assertEqual(legacy.head.skip.out_features, 16 * 9)
        self.assertFalse(legacy.hybrid)
        self.assertFalse(any(mix.gated for mix in legacy.variate))

    def test_equal_weights_and_invalid_combinations(self):
        config = ForecastConfig(
            objective="next_patch",
            output_patches=LEADS,
            horizon_loss_weights="equal",
            head="direct",
        )
        self.assertEqual(config.horizon_loss_weights, (1 / LEADS,) * LEADS)
        self.assertEqual(
            evaluation_weights(
                {"model": {"output_patches": LEADS, "horizon_loss_weights": "equal"}}
            ),
            [1 / LEADS] * LEADS,
        )
        self.assertEqual(evaluation_weights({"model": {}}), [0.8, 0.16, 0.03, 0.01])
        with self.assertRaises(ValueError):
            evaluation_weights(
                {"model": {"output_patches": 7, "horizon_loss_weights": [0.5, 0.5]}}
            )
        for bad in (
            dict(head="direct", horizon_loss_weights=None),
            dict(value_transform="asinh", head="lead_offset"),
            dict(head="per_lead"),
            dict(variate_every=0),
            dict(temporal_every=-1),
            dict(rwkv_init="language_model"),
            dict(fp32_io=1),
        ):
            with self.assertRaises(ValueError, msg=bad):
                ForecastConfig(
                    **dict(
                        dict(
                            objective="next_patch",
                            output_patches=4,
                            horizon_loss_weights=(0.25,) * 4,
                        ),
                        **bad,
                    )
                )

    def test_default_configuration_builds_the_intended_model(self):
        config = load_config()
        model = config["model"]
        settings = ForecastConfig(**dict(model, quantiles=tuple(model["quantiles"])))
        self.assertEqual(settings.output_patches * settings.patch_size, 112)
        self.assertEqual(config["data"]["horizon"], 112)
        self.assertEqual(len(set(settings.horizon_loss_weights)), 1)
        budget = parameter_budget(config)
        self.assertGreater(budget["parameters"], 210e6)
        self.assertLess(budget["parameters"], 230e6)
        self.assertEqual(
            config["training"]["early_stopping"]["monitor"], "table_mase_profile"
        )

    def test_analytic_parameter_count_matches_a_built_model(self):
        for overrides in (
            {},
            dict(temporal_every=0),
            dict(
                head="lead_offset",
                value_transform="none",
                output_patches=4,
                horizon_loss_weights=(0.25,) * 4,
                variate_every=1,
                temporal_every=0,
                gated_mixing=False,
            ),
        ):
            model = build(**overrides)
            groups = architecture_report(model)["groups"]
            measured = {
                "embedding": groups["embedding"],
                "roles": groups["roles"],
                "norm": groups["norm"],
                "head": groups["head"],
                "variate_mixing": sum(
                    v for k, v in groups.items() if k.startswith("variate.")
                ),
                "temporal_attention": sum(
                    v for k, v in groups.items() if k.startswith("temporal.")
                ),
                "lead_embedding": groups.get("lead_embedding", 0),
            }
            analytic = parameter_budget({"model": asdict(model.config)})["groups"]
            for name, count in measured.items():
                self.assertEqual(analytic[name], count, (name, overrides))


class DirectHeadTests(unittest.TestCase):
    def test_training_loss_equals_the_plain_mean_over_every_observed_lead_timestep(
        self,
    ):
        torch.manual_seed(3)
        model = build().train()
        context = torch.randn(2, 1, 12 * 16) * 5 + 3
        context[0, 0, 37] = float("nan")  # one missing observation
        prepared = model.tokenize_next(context, training=True)
        output = model.forward_encoded(model.encode_tokens(prepared), prepared)
        labels, loc, scale = prepared["labels"], prepared["loc"], prepared["scale"]
        b, targets, anchors, patch_size = labels.shape
        self.assertEqual(anchors, 11)

        hidden = torch.randn(b, targets, anchors, model.config.width)
        with torch.no_grad():
            loss, per_sample, mass, first = direct_training_loss(
                model, hidden, labels, loc, scale
            )
            quantiles = lead_quantiles(model, hidden)
        levels = torch.tensor(model.config.quantiles)
        total, count = 0.0, 0
        for anchor in range(anchors):
            for lead in range(LEADS):
                if anchor + lead >= anchors:
                    continue  # the label lies beyond this chunk
                future = labels[:, :, anchor + lead]
                valid = torch.isfinite(future)
                target = torch.asinh(
                    (future - loc[:, :, anchor]) / scale[:, :, anchor]
                ).nan_to_num()
                error = target[..., None] - quantiles[:, :, anchor, lead]
                pinball = torch.maximum(levels * error, (levels - 1) * error).mean(-1)
                total += float((pinball * valid).sum())
                count += int(valid.sum())
        self.assertAlmostEqual(loss.item(), total / count, places=5)
        # Equal weights: the loss mass is the observed count times one seventh,
        # and matches the mass the trainer computes on CPU before the update.
        self.assertAlmostEqual(mass.sum().item(), count / LEADS, places=4)
        self.assertAlmostEqual(
            float(weighted_label_mass(labels, model.config.horizon_loss_weights)),
            count / LEADS,
            places=4,
        )
        torch.testing.assert_close(first, quantiles[:, :, :, 0])
        self.assertEqual(per_sample.shape, (b,))
        self.assertTrue(torch.isfinite(output.loss))
        output.loss.backward()
        self.assertGreater(model.head.skip.weight.grad.abs().sum().item(), 0)

    def test_every_lead_has_its_own_output_rows(self):
        model = build()
        self.assertEqual(model.head.skip.out_features, LEADS * 16 * 9)
        self.assertFalse(hasattr(model, "lead_embedding"))
        hidden = torch.randn(1, 1, 1, model.config.width)
        with torch.no_grad():
            before = lead_quantiles(model, hidden)
            rows = slice(3 * 16 * 9, 4 * 16 * 9)  # lead four only
            model.head.skip.bias[rows] += 1.0
            after = lead_quantiles(model, hidden)
        changed = (after - before).abs().flatten(-2).amax(-1).flatten()
        self.assertEqual((changed > 0.5).tolist(), [i == 3 for i in range(LEADS)])

    def test_validation_loss_matches_training_objective_at_one_origin(self):
        torch.manual_seed(5)
        model = build().eval()
        history = torch.randn(1, 1, 96).cumsum(-1)
        future = torch.randn(1, 1, 112)
        future[0, 0, 5] = float("nan")
        prepared = model.tokenize_next(history)
        prepared["prediction_patches"] = LEADS
        with torch.no_grad():
            output = model.forward_encoded(
                model.encode_tokens(prepared), prepared, future
            )
        self.assertEqual(output.quantiles.shape, (1, 1, 112, 9))
        self.assertTrue((output.quantiles.diff(dim=-1) >= 0).all())
        loc, scale = prepared["loc"][:, :1, -1], prepared["scale"][:, :1, -1]
        normalized = torch.asinh((output.quantiles - loc[..., None]) / scale[..., None])
        target = torch.asinh((future - loc) / scale).nan_to_num()
        error = target[..., None] - normalized
        levels = torch.tensor(model.config.quantiles)
        pinball = torch.maximum(levels * error, (levels - 1) * error).mean(-1)
        valid = torch.isfinite(future)
        expected = (pinball * valid).sum() / valid.sum()
        self.assertAlmostEqual(output.loss.item(), expected.item(), places=4)
        torch.testing.assert_close(output.point, output.quantiles[..., 4])

    def test_shorter_requests_are_prefixes_and_long_ones_roll_forward(self):
        torch.manual_seed(6)
        model = build().eval()
        history = torch.randn(1, 1, 200)
        full = model.forecast(history, 112).quantiles
        short = model.forecast(history, 40).quantiles
        torch.testing.assert_close(short, full[..., :40, :])
        longer = model.forecast(history, 150).quantiles
        self.assertEqual(longer.shape, (1, 1, 150, 9))
        torch.testing.assert_close(longer[..., :112, :], full)
        self.assertTrue(torch.isfinite(longer).all())

    def test_transform_is_invertible_and_inputs_use_it(self):
        model = build()
        values = torch.tensor([-1e4, -3.0, 0.0, 0.5, 250.0])
        torch.testing.assert_close(
            from_model_space(model, to_model_space(model, values)),
            values,
            rtol=1e-4,
            atol=1e-6,
        )
        # Extreme outputs stay finite instead of overflowing sinh.
        self.assertTrue(
            torch.isfinite(from_model_space(model, torch.tensor([500.0, -500.0]))).all()
        )
        plain = build(value_transform="none")
        context = torch.randn(1, 1, 64) * 100
        context[0, 0, -1] = 1e6
        transformed = model.tokenize_next(context)["features"][..., :16]
        untransformed = plain.tokenize_next(context)["features"][..., :16]
        torch.testing.assert_close(transformed, torch.asinh(untransformed))
        self.assertLess(transformed.abs().max().item(), 10.0)
        torch.testing.assert_close(from_model_space(plain, values), values)


class HybridBackboneTests(unittest.TestCase):
    def test_mixing_layout_follows_the_configured_intervals(self):
        model = build(layers=8, variate_every=4, temporal_every=2)
        report = architecture_report(model)
        self.assertEqual(report["variate_mixing_after_blocks"], [3, 7])
        self.assertEqual(report["temporal_attention_after_blocks"], [1, 3, 5, 7])
        self.assertTrue(model.hybrid)

    def test_a_new_model_starts_as_the_pure_recurrent_stack(self):
        torch.manual_seed(8)
        model = build().eval()
        prepared = model.tokenize_next(
            torch.randn(2, 1, 160), past_covariates=torch.randn(2, 2, 160)
        )
        x = model.encode_tokens(prepared)
        with torch.no_grad():
            hybrid = model.forward_encoded(x, prepared).quantiles
            # Remove the mixing branches entirely: nothing may change at init.
            model.temporal_after = [None] * len(model.blocks)
            model.variate_after = [None] * len(model.blocks)
            recurrent = model.forward_encoded(x, prepared).quantiles
        torch.testing.assert_close(hybrid, recurrent, rtol=0, atol=0)

    def test_temporal_attention_is_causal_and_ignores_right_padding(self):
        torch.manual_seed(9)
        attention = TemporalAttention(64, 32).eval()
        nn.init.normal_(attention.output.weight, std=0.5)
        x = torch.randn(2, 3, 20, 64)
        changed = x.clone()
        changed[:, :, 12:] = torch.randn(2, 3, 8, 64)
        with torch.no_grad():
            a, b = attention(x), attention(changed)
            padded = attention(torch.cat([x, torch.zeros(2, 3, 12, 64)], dim=2))
        torch.testing.assert_close(a[:, :, :12], b[:, :, :12], atol=1e-5, rtol=1e-5)
        self.assertGreater((a[:, :, 12:] - b[:, :, 12:]).abs().max().item(), 1e-3)
        torch.testing.assert_close(a, padded[:, :, :20], atol=1e-5, rtol=1e-5)
        # It really reads history: an early patch changes later positions.
        early = x.clone()
        early[:, :, 0] += 3.0
        with torch.no_grad():
            c = attention(early)
        self.assertGreater((a[:, :, 19] - c[:, :, 19]).abs().max().item(), 1e-3)

    def test_whole_model_stays_causal_with_open_mixing_branches(self):
        torch.manual_seed(10)
        model = build().eval()
        open_branches(model)
        context = torch.randn(1, 1, 160)
        covariates = torch.randn(1, 2, 160)
        later = context.clone()
        later[..., 96:] += 5.0
        with torch.no_grad():
            outputs = []
            for series in (context, later):
                prepared = model.tokenize_next(
                    series, past_covariates=covariates, training=True
                )
                x = model.encode_tokens(prepared)
                outputs.append(model.forward_encoded(x, prepared).quantiles)
        # Anchors 0..4 see only points before 96; the change must not reach them.
        torch.testing.assert_close(
            outputs[0][:, :, :5], outputs[1][:, :, :5], atol=1e-5, rtol=1e-5
        )
        self.assertGreater(
            (outputs[0][:, :, 6:] - outputs[1][:, :, 6:]).abs().max().item(), 1e-3
        )

    def test_gated_variable_mixing_skips_single_variable_tables(self):
        model = build().eval()
        open_branches(model)
        mix = model.variate[0]
        single = torch.randn(2, 1, 6, 64)
        several = torch.randn(2, 3, 6, 64)
        with patch.object(
            mix.attention, "forward", wraps=mix.attention.forward
        ) as forward:
            self.assertIs(mix(single), single)
            self.assertIs(mix.forward_ungraphed(single), single)
            self.assertEqual(forward.call_count, 0)
            mixed = mix(several)
            self.assertEqual(forward.call_count, 1)
        self.assertGreater((mixed - several).abs().max().item(), 1e-3)
        # The archived layout mixes even a single variable, as it was trained.
        legacy = build(gated_mixing=False, variate_every=1, temporal_every=0).variate[0]
        self.assertGreater((legacy(single) - single).abs().max().item(), 1e-4)

    def test_training_with_gradient_checkpointing_matches_plain_execution(self):
        torch.manual_seed(11)
        model = build(gradient_checkpointing=True).train()
        open_branches(model)
        context = torch.randn(2, 1, 128)
        covariates = torch.randn(2, 1, 128)
        losses, gradients = [], []
        for threshold in (0, 10**9):  # every layer checkpointed / none
            model.checkpoint_token_threshold = threshold
            model.zero_grad()
            prepared = model.tokenize_next(
                context, past_covariates=covariates, training=True
            )
            output = model.forward_encoded(model.encode_tokens(prepared), prepared)
            output.loss.backward()
            losses.append(output.loss.item())
            gradients.append(model.temporal[0].query_key_value.weight.grad.clone())
        self.assertAlmostEqual(losses[0], losses[1], places=6)
        torch.testing.assert_close(gradients[0], gradients[1], atol=1e-6, rtol=1e-4)
        self.assertGreater(gradients[0].abs().sum().item(), 0)

    def test_float32_input_and_output_layers_survive_precision_casts(self):
        model = build(fp32_io=True).to(dtype=torch.bfloat16)
        float32 = {
            name
            for name, parameter in model.named_parameters()
            if parameter.dtype == torch.float32
        }
        self.assertTrue(float32)
        self.assertTrue(
            all(name.split(".")[0] in ("embedding", "norm", "head") for name in float32)
        )
        self.assertEqual(model.compute_dtype, torch.bfloat16)
        self.assertEqual(model.blocks[0].projection.weight.dtype, torch.bfloat16)
        self.assertEqual(model.temporal[0].output.weight.dtype, torch.bfloat16)
        model.bfloat16()
        self.assertEqual(model.head.skip.weight.dtype, torch.float32)
        # Without the option the whole model follows the cast, as before.
        plain = build().to(dtype=torch.bfloat16)
        self.assertEqual(plain.head.skip.weight.dtype, torch.bfloat16)

    def test_float32_layers_join_a_low_precision_backbone(self):
        torch.manual_seed(12)
        model = build(fp32_io=True).eval()
        prepared = model.tokenize_next(torch.randn(1, 1, 96))
        with torch.no_grad():
            reference = model.forward_encoded(
                model.encode_tokens(prepared), prepared
            ).quantiles
        model.to(dtype=torch.bfloat16)
        with torch.no_grad():
            x = model.encode_tokens(prepared)
            self.assertEqual(x.dtype, torch.bfloat16)
            mixed = model.forward_encoded(x, prepared).quantiles
        self.assertEqual(mixed.dtype, torch.float32)
        torch.testing.assert_close(mixed, reference, atol=0.15, rtol=0.1)


class FakeTimeMix(nn.Module):
    def __init__(self, width):
        super().__init__()
        self.w0 = nn.Parameter(torch.full((1, 1, width), -4.0))
        self.x_k = nn.Parameter(torch.ones(1, 1, width))
        self.w1 = nn.Parameter(torch.zeros(width, 32))
        self.receptance = nn.Linear(width, width, bias=False)
        self.key = nn.Linear(width, width, bias=False)
        self.value = nn.Linear(width, width, bias=False)
        self.output = nn.Linear(width, width, bias=False)
        self.ln_x = nn.GroupNorm(width // 64, width)


class FakeChannelMix(nn.Module):
    def __init__(self, width):
        super().__init__()
        self.x_k = nn.Parameter(torch.ones(1, 1, width))
        self.key = nn.Linear(width, 4 * width, bias=False)
        self.value = nn.Linear(4 * width, width, bias=False)


class FakeBlock(nn.Module):
    """Parameter names and shapes of the vendored RWKV-7 block, no kernels."""

    def __init__(self, width=64):
        super().__init__()
        self.ln1 = nn.LayerNorm(width)
        self.att = FakeTimeMix(width)
        self.ffn = FakeChannelMix(width)


class FakeModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.blocks = nn.ModuleList([FakeBlock() for _ in range(3)])
        self.roles = nn.Embedding(3, 64)
        self.variate = nn.MultiheadAttention(64, 4, batch_first=True)
        self.temporal = TemporalAttention(64, 16)
        self.norm = nn.LayerNorm(64)
        self.head = nn.Linear(64, 8)


class RecipeTests(unittest.TestCase):
    def test_upstream_initialisation(self):
        torch.manual_seed(13)
        blocks = nn.ModuleList([FakeBlock() for _ in range(4)])
        before = blocks[0].att.w0.detach().clone()
        apply_upstream_rwkv_init(blocks)
        for index, block in enumerate(blocks):
            for projection, gain in (
                (block.att.receptance, 1.0),
                (block.att.value, 1.0),
                (block.att.key, 0.1),
            ):
                weight = projection.weight
                torch.testing.assert_close(
                    weight @ weight.T, gain**2 * torch.eye(64), atol=1e-5, rtol=0
                )
            key = block.ffn.key.weight  # tall: orthonormal columns
            torch.testing.assert_close(key.T @ key, torch.eye(64), atol=1e-5, rtol=0)
            self.assertEqual(block.att.output.weight.abs().sum().item(), 0)
            self.assertEqual(block.ffn.value.weight.abs().sum().item(), 0)
            torch.testing.assert_close(
                block.att.ln_x.weight,
                torch.full((64,), ((index + 1) / 4) ** 0.7),
            )
        # Decay and mixing parameters keep the constructor's values.
        torch.testing.assert_close(blocks[0].att.w0, before)

    def test_parameter_groups_follow_the_upstream_rule(self):
        model = FakeModel()
        groups = {
            group["name"]: group for group in parameter_groups(model, 1e-3, 0.01, 2.0)
        }
        names = {id(p): n for n, p in model.named_parameters()}
        members = {
            key: {names[id(p)] for p in group["params"]}
            for key, group in groups.items()
        }
        self.assertEqual(
            members["rwkv_memory_decay"], {f"blocks.{i}.att.w0" for i in range(3)}
        )
        self.assertEqual(groups["rwkv_memory_decay"]["lr"], 2e-3)
        self.assertEqual(groups["rwkv_memory_decay"]["weight_decay"], 0.0)
        self.assertEqual(groups["no_weight_decay"]["weight_decay"], 0.0)
        self.assertEqual(groups["projection_matrices"]["weight_decay"], 0.01)
        decayed = members["projection_matrices"]
        for name in (
            "blocks.0.att.receptance.weight",
            "blocks.2.ffn.value.weight",
            "variate.in_proj_weight",
            "variate.out_proj.weight",
            "temporal.query_key_value.weight",
            "temporal.gate.weight",
            "head.weight",
        ):
            self.assertIn(name, decayed)
        plain = members["no_weight_decay"]
        for name in (
            "blocks.0.ln1.weight",
            "blocks.0.att.ln_x.weight",
            "blocks.0.att.x_k",
            "blocks.0.att.w1",
            "roles.weight",
            "norm.weight",
            "norm.bias",
            "head.bias",
            "variate.in_proj_bias",
            "temporal.norm.weight",
        ):
            self.assertIn(name, plain)
        total = sum(len(group["params"]) for group in groups.values())
        self.assertEqual(total, len(list(model.parameters())))

    def test_grouped_optimizer_decays_only_matrices_and_scales_memory_rate(self):
        torch.manual_seed(14)
        model = FakeModel()
        training = {
            "learning_rate": 0.1,
            "weight_decay": 0.5,
            "fused_optimizer": False,
            "optimizer": {"parameter_groups": "rwkv7"},
        }
        optimizer = make_optimizer(model, training)
        self.assertIsInstance(optimizer, torch.optim.AdamW)
        gain = model.norm.weight.detach().clone()
        matrix = model.head.weight.detach().clone()
        memory = model.blocks[0].att.w0.detach().clone()
        # No gradient signal except on w0: only decay can move the others.
        for parameter in model.parameters():
            parameter.grad = torch.zeros_like(parameter)
        model.blocks[0].att.w0.grad = torch.ones_like(memory)
        optimizer.step()
        torch.testing.assert_close(model.norm.weight, gain, rtol=0, atol=0)
        torch.testing.assert_close(model.head.weight, matrix * (1 - 0.1 * 0.5))
        # Adam's first step moves by the learning rate: twice the base rate here.
        torch.testing.assert_close(
            model.blocks[0].att.w0, memory - 0.2, atol=1e-6, rtol=0
        )
        # The legacy single group decays everything, as the 500M run did.
        legacy_model = FakeModel()
        legacy = make_optimizer(
            legacy_model,
            {"learning_rate": 0.1, "weight_decay": 0.5, "fused_optimizer": False},
        )
        self.assertEqual(len(legacy.param_groups), 1)
        for parameter in legacy_model.parameters():
            parameter.grad = torch.zeros_like(parameter)
        legacy.step()
        self.assertLess(legacy_model.norm.weight.mean().item(), 1.0)
        for bad in (
            {"parameter_groups": "transformer"},
            {"betas": [0.9]},
            {"eps": 0},
            {"lr_scale": 2},
        ):
            with self.assertRaises(ValueError):
                optimizer_options({"optimizer": bad})

    def test_fp32_master_optimizer_supports_groups_and_resume(self):
        torch.manual_seed(15)
        model = FakeModel().to(dtype=torch.bfloat16)
        groups = parameter_groups(model, 1e-2, 0.01, 2.0)
        optimizer = FP32MasterAdamW(
            groups, lr=1e-2, weight_decay=0.01, fused=False, foreach=False
        )
        self.assertEqual(len(optimizer.param_groups), 3)
        self.assertEqual(
            [id(p) for group in groups for p in group["params"]],
            [id(p) for p in optimizer.model_parameters],
        )
        for parameter in model.parameters():
            parameter.grad = torch.full_like(parameter, 0.01)
        optimizer.step()
        for master, parameter in zip(
            optimizer.master_parameters, optimizer.model_parameters
        ):
            self.assertEqual(master.dtype, torch.float32)
            torch.testing.assert_close(parameter.float(), master.bfloat16().float())
        state = copy.deepcopy(optimizer.state_dict())
        clone = FakeModel().to(dtype=torch.bfloat16)
        restored = FP32MasterAdamW(
            parameter_groups(clone, 1e-2, 0.01, 2.0),
            lr=1e-2,
            weight_decay=0.01,
            fused=False,
            foreach=False,
        )
        restored.load_state_dict(state)
        for a, b in zip(optimizer.master_parameters, restored.master_parameters):
            torch.testing.assert_close(a, b, rtol=0, atol=0)
        self.assertEqual(
            [g["lr"] for g in restored.param_groups],
            [g["lr"] for g in optimizer.param_groups],
        )

    def test_weight_average_follows_slowly_and_restores_training_weights(self):
        torch.manual_seed(16)
        model = nn.Linear(4, 4)
        optimizer = torch.optim.SGD(model.parameters(), lr=1.0)
        average = WeightAverage(
            model, optimizer, decay=0.5, update_every=2, warmup_updates=0
        )
        start = model.weight.detach().clone()
        with torch.no_grad():
            model.weight += 4.0
        average.update()  # first of two: no averaging step yet
        torch.testing.assert_close(average.average[0], start)
        average.update()
        # Two optimizer updates at decay 0.5 each: a quarter of the old value.
        torch.testing.assert_close(average.average[0], start + 4.0 * 0.75)
        current = model.weight.detach().clone()
        with average.applied():
            torch.testing.assert_close(model.weight, start + 3.0)
        torch.testing.assert_close(model.weight, current, rtol=0, atol=0)
        state = copy.deepcopy(average.state_dict())
        other = WeightAverage(
            nn.Linear(4, 4), optimizer, decay=0.5, update_every=2, warmup_updates=0
        )
        other.load_state_dict(state)
        torch.testing.assert_close(other.average[0], average.average[0])
        self.assertEqual(other.updates, 2)
        with self.assertRaises(ValueError):
            WeightAverage(
                nn.Linear(4, 4), optimizer, decay=0.9, update_every=2, warmup_updates=0
            ).load_state_dict(state)
        self.assertIsNone(weight_average_options({}))
        for bad in ({"decay": 1.0}, {"update_every": 0}, {"rate": 0.1}):
            with self.assertRaises(ValueError):
                weight_average_options({"weight_average": bad})

    def test_weight_average_tracks_float32_masters_across_groups(self):
        torch.manual_seed(17)
        model = FakeModel().to(dtype=torch.bfloat16)
        optimizer = FP32MasterAdamW(
            parameter_groups(model, 1e-2, 0.0, 2.0),
            lr=1e-2,
            weight_decay=0.0,
            fused=False,
            foreach=False,
        )
        average = WeightAverage(
            model, optimizer, decay=0.9, update_every=1, warmup_updates=0
        )
        self.assertTrue(all(a.dtype == torch.float32 for a in average.average))
        for _ in range(3):
            for parameter in model.parameters():
                parameter.grad = torch.full_like(parameter, 0.01)
            optimizer.step()
            average.update()
        # Matched by identity, not by position: groups reorder the masters.
        trained = model.blocks[0].att.w0.detach().float().clone()
        index = next(
            i for i, p in enumerate(average.parameters) if p is model.blocks[0].att.w0
        )
        self.assertLess(
            (average.average[index] - (-4.0)).abs().max().item(),
            (trained - (-4.0)).abs().max().item(),
        )
        before = [p.detach().clone() for p in model.parameters()]
        with average.applied():
            self.assertGreater(
                (model.blocks[0].att.w0.float() - trained).abs().max().item(), 0
            )
        for parameter, saved in zip(model.parameters(), before):
            torch.testing.assert_close(parameter, saved, rtol=0, atol=0)

    def test_robust_monitor_ignores_a_few_extreme_forecasts(self):
        rows = [
            {
                "source": f"table_{i}",
                "loss": 0.2 + 0.001 * i,
                "loss_mass": 16.0,
                "mase": 1.0,
            }
            for i in range(100)
            for _ in range(5)
        ]
        calm = robust_table_summary(rows)
        rows[7] = dict(rows[7], loss=5e6)  # one forecast of 500
        wild = robust_table_summary(rows)
        mean = sum(r["loss"] for r in rows) / len(rows)
        self.assertGreater(mean, 1000)
        self.assertAlmostEqual(
            wild["trimmed_table_loss"], calm["trimmed_table_loss"], delta=0.002
        )
        self.assertAlmostEqual(
            wild["median_table_loss"], calm["median_table_loss"], delta=0.002
        )
        self.assertGreater(wild["table_loss_p99"], 100)
        self.assertEqual(calm["median_table_mase"], 1.0)

    def test_validation_reserves_the_full_forecast_block(self):
        config = {
            "seed": 7,
            "data": {"context_length": 2048},
            "model": {"patch_size": 16, "output_patches": LEADS},
            "training": {"validation_sampling": {"tables": 10, "seed": 7}},
        }
        records = [
            {
                "id": f"t{rows}",
                "source": f"table_{rows}",
                "source_rows": rows,
                "source_sha256": "",
                "targets": ["outcome"],
                "roles": {"targets": ["outcome"], "past": [], "known": []},
            }
            for rows in (192, 208, 400, 2056)
        ]
        selected, audit = select_validation_origins(records, config)
        # (7 + 6) * 16 = 208 points are needed for five full-block origins.
        self.assertEqual(
            {r["source"] for r in selected}, {"table_208", "table_400", "table_2056"}
        )
        self.assertEqual(audit["forecast_tokens"], LEADS)
        for record in selected:
            start, stop = record["label_rows"]
            self.assertEqual(stop - start, 112)
            self.assertLessEqual(stop, record["source_rows"])
            self.assertEqual(start, record["validation_origin"])
        self.assertEqual(len(selected), 15)


class MissingLabelTests(unittest.TestCase):
    """Generated tables can lose a sensor for longer than the forecast block."""

    def test_validation_skips_requests_without_observed_history_or_labels(self):
        import numpy as np
        import pandas as pd
        from rolling_validation import validate_origins
        from transform import file_digest

        torch.manual_seed(18)
        model = build(context_length=256, max_patches=16).eval()
        rows = 208  # five origins: 32, 48, 64, 80, 96
        values = np.sin(np.arange(rows) / 5.0) * 3 + 10
        values[:32] = np.nan  # no history at all before the first origin
        values[96:] = np.nan  # the last origin's whole label window
        with tempfile.TemporaryDirectory(dir=ROOT / ".cache") as temporary:
            path = Path(temporary) / "data.parquet"
            pd.DataFrame({"time_index": range(rows), "outcome": values}).to_parquet(
                path
            )
            config = {
                "seed": 7,
                "data": {"context_length": 256},
                "model": {"patch_size": 16, "output_patches": LEADS},
                "training": {
                    "validation_sampling": {"tables": 1, "seed": 7},
                    "validation_batch_size": 2,
                },
            }
            records, _ = select_validation_origins(
                [
                    {
                        "id": "t",
                        "source": path.relative_to(ROOT).as_posix(),
                        "source_rows": rows,
                        "source_sha256": file_digest(path),
                        "targets": ["outcome"],
                        "roles": {"targets": ["outcome"], "past": [], "known": []},
                    }
                ],
                config,
            )
            self.assertEqual([r["validation_origin"] for r in records][0], 32)
            result = validate_origins(model, records, config)
        self.assertEqual(result["unscored_forecasts_without_observed_history"], 1)
        self.assertEqual(result["unscored_forecasts_without_observed_labels"], 1)
        self.assertEqual(result["forecast_count"], 3)
        self.assertEqual([row["origin"] for row in result["tables"]], [48, 64, 80])
        self.assertTrue(np.isfinite(result["loss"]))
        self.assertTrue(np.isfinite(result["trimmed_table_loss"]))


class TableMaseMonitorTests(unittest.TestCase):
    def rows(self, scores):
        return [
            {"source": f"table_{i}", "loss": 0.2, "loss_mass": 16.0, "mase": score}
            for i, score in enumerate(scores)
            for _ in range(5)
        ]

    def test_profile_is_the_average_relative_change_of_the_three_percentiles(self):
        base = table_mase_profile(0.8, 1.4, 3.3)
        self.assertAlmostEqual(base, (0.8 * 1.4 * 3.3) ** (1 / 3), places=9)
        # -3%, -3% and +5% average out to an improvement; -1%, -1%, +5% do not.
        self.assertLess(table_mase_profile(0.8 * 0.97, 1.4 * 0.97, 3.3 * 1.05), base)
        self.assertGreater(table_mase_profile(0.8 * 0.99, 1.4 * 0.99, 3.3 * 1.05), base)
        # The same relative change counts the same at either end.
        self.assertAlmostEqual(
            table_mase_profile(0.8 * 0.9, 1.4, 3.3),
            table_mase_profile(0.8, 1.4, 3.3 * 0.9),
            places=9,
        )
        self.assertIsNone(table_mase_profile(None, None, None))

    def test_validation_reports_the_three_percentiles_and_ignores_degenerate_tables(
        self,
    ):
        import numpy as np

        scores = list(np.linspace(0.5, 4.0, 200))
        summary = robust_table_summary(self.rows(scores))
        self.assertAlmostEqual(summary["table_mase_p10"], np.quantile(scores, 0.1))
        self.assertAlmostEqual(summary["median_table_mase"], np.median(scores))
        self.assertAlmostEqual(summary["table_mase_p90"], np.quantile(scores, 0.9))
        self.assertAlmostEqual(
            summary["table_mase_profile"],
            table_mase_profile(
                summary["table_mase_p10"],
                summary["median_table_mase"],
                summary["table_mase_p90"],
            ),
        )
        # One near-constant table with an astronomically small scale, and one
        # with no defined scale at all: neither moves the monitor noticeably.
        rows = self.rows(scores)
        for row in rows[995:]:  # the worst table becomes astronomically bad
            row["mase"] = 6.7e9
        for row in rows[500:505]:  # a middle table loses its scale
            row["mase"] = None
        wild = robust_table_summary(rows)
        self.assertEqual(wild["tables_with_defined_mase"], 199)
        self.assertAlmostEqual(
            wild["table_mase_profile"], summary["table_mase_profile"], delta=0.01
        )
        from epoch_training import MONITORS

        self.assertIn("table_mase_profile", MONITORS)


class MissingValuesInTrainingTests(unittest.TestCase):
    """Sensor outages in training tables: nothing to learn from, nothing to break."""

    @staticmethod
    def record(model, context, name):
        return {
            "id": name,
            "source": name,
            "split": "train",
            "prepared": model.tokenize_next(context, training=True),
        }

    def test_anchors_before_the_first_observation_are_not_supervised(self):
        torch.manual_seed(20)
        model = build().train()
        context = torch.randn(1, 1, 12 * 16) * 3 + 500  # raw level far from zero
        context[..., :40] = float("nan")  # the sensor starts late
        prepared = model.tokenize_next(context, training=True)
        observed = anchors_with_history(prepared["features"], 1, 16)
        # Patches 0 and 1 hold no observation; the first value is in patch 2.
        self.assertEqual(observed[0, 0].tolist(), [False, False] + [True] * 9)
        output = model.forward_encoded(model.encode_tokens(prepared), prepared)
        labels = prepared["labels"]
        full = weighted_label_mass(labels, model.config.horizon_loss_weights)
        kept = weighted_label_mass(labels, model.config.horizon_loss_weights, observed)
        self.assertLess(float(kept), float(full))
        self.assertAlmostEqual(output.loss_mass.sum().item(), float(kept), places=4)
        # Without the mask those anchors are asked for the raw level (asinh(500)
        # is about 6.9) from no information at all, which inflates the loss.
        with torch.no_grad():
            hidden = torch.randn(1, 1, 11, model.config.width)
            loc, scale = prepared["loc"][:, :1], prepared["scale"][:, :1]
            masked = direct_training_loss(
                model, hidden, labels, loc, scale, supervised=observed
            )[0]
            unmasked = direct_training_loss(model, hidden, labels, loc, scale)[0]
        self.assertGreater(unmasked.item(), masked.item() + 0.5)
        self.assertTrue(torch.isfinite(output.loss))

    def test_tables_without_any_observed_target_do_not_stop_or_change_an_update(self):
        from epoch_training import accumulated_update

        torch.manual_seed(21)
        first = build().train()
        second = copy.deepcopy(first)
        observed = [torch.randn(1, 1, 128), torch.randn(1, 1, 96)]
        observed[0][..., 30:50] = float("nan")  # an outage inside the history
        empty = torch.full((1, 1, 128), float("nan"))  # the target never reports
        left = [self.record(first, c, f"a{i}") for i, c in enumerate(observed)]
        right = [self.record(second, c, f"b{i}") for i, c in enumerate(observed)]
        outage = self.record(second, empty, "outage")
        a = torch.optim.SGD(first.parameters(), lr=0.01)
        b = torch.optim.SGD(second.parameters(), lr=0.01)
        plain = accumulated_update(first, [[r] for r in left], a, 1000)
        # The same tables plus an all-missing one, as its own microbatch.
        mixed = accumulated_update(second, [[right[0]], [outage], [right[1]]], b, 1000)
        self.assertAlmostEqual(plain["loss"], mixed["loss"], places=6)
        self.assertEqual(mixed["microbatches"], 2)
        for x, y in zip(first.parameters(), second.parameters()):
            torch.testing.assert_close(x, y, rtol=0, atol=0)

        # An update made only of such tables is skipped and leaves the weights.
        before = [p.detach().clone() for p in second.parameters()]
        optimizer = torch.optim.AdamW(second.parameters(), lr=0.1, weight_decay=0.5)
        skipped = accumulated_update(second, [[outage]], optimizer, 1.0)
        self.assertFalse(skipped["supervised"])
        self.assertEqual(skipped["target_points"], 0)
        self.assertEqual(skipped["loss"], 0.0)
        for parameter, saved in zip(second.parameters(), before):
            torch.testing.assert_close(parameter, saved, rtol=0, atol=0)

    def test_split_and_unsplit_updates_agree_with_missing_values(self):
        from epoch_training import accumulated_update

        torch.manual_seed(22)
        first = build().train()
        second = copy.deepcopy(first)
        contexts = [torch.randn(1, 1, 128), torch.randn(1, 1, 64)]
        contexts[0][..., :48] = float("nan")  # late start
        contexts[1][..., 20:40] = float("nan")  # outage
        left = [self.record(first, c, str(i)) for i, c in enumerate(contexts)]
        right = [self.record(second, c, str(i)) for i, c in enumerate(contexts)]
        a = torch.optim.SGD(first.parameters(), lr=0.01)
        b = torch.optim.SGD(second.parameters(), lr=0.01)
        together = accumulated_update(first, [left], a, 1000)
        apart = accumulated_update(second, [[r] for r in right], b, 1000)
        self.assertAlmostEqual(together["loss"], apart["loss"], places=6)
        self.assertAlmostEqual(
            together["target_loss_weight"], apart["target_loss_weight"], places=4
        )
        for x, y in zip(first.parameters(), second.parameters()):
            torch.testing.assert_close(x, y, atol=2e-7, rtol=2e-6)

    def test_missing_values_inside_the_history_are_inputs_not_errors(self):
        torch.manual_seed(23)
        model = build().eval()
        history = torch.randn(1, 1, 200).cumsum(-1)
        covariates = torch.randn(1, 2, 200)
        history[..., 60:130] = float("nan")
        covariates[:, 0] = float("nan")  # a covariate that never reports
        quantiles = model.forecast(history, 112, past_covariates=covariates).quantiles
        self.assertEqual(quantiles.shape, (1, 1, 112, 9))
        self.assertTrue(torch.isfinite(quantiles).all())


class HoldoutMissingValueTests(unittest.TestCase):
    def test_missing_labels_are_forecast_but_not_scored(self):
        from unittest.mock import Mock

        import numpy as np
        import pandas as pd
        from holdout import evaluate_holdout
        from transform import file_digest

        config = copy.deepcopy(load_config())
        config["plot"]["max_plots"] = 0
        config["holdout"].update(
            benchmark_models=["naive", "chronos"],
            latency={"warmup_requests": 0, "repeats": 1},
        )
        model = torch.nn.Linear(1, 1)
        model.config = SimpleNamespace(
            patch_size=16, context_length=256, quantiles=(0.1, 0.5, 0.9)
        )
        suite = Mock()
        suite.names = config["holdout"]["benchmark_models"]
        suite.metadata = {name: {"device": "cpu"} for name in suite.names}
        suite.predict.side_effect = lambda contexts, horizon, model_names: [
            {model_names[0]: np.zeros((1, horizon))}
        ]
        calls = []

        def neural(_model, contexts, _batch_size, prediction_patches=1):
            calls.append(contexts[0]["context"].shape[-1])
            return [torch.zeros(1, prediction_patches * 16, 3)]

        rows = 13 * 16  # origins 32, 48, 64, 80, 96
        values = np.sin(np.arange(rows) / 4.0) + 5
        values[:32] = np.nan  # origin 32: the target has no history yet
        values[70:75] = np.nan  # a short outage: scored around it
        values[96:] = np.nan  # origin 96: every label is missing
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            source = root / "data.parquet"
            pd.DataFrame({"outcome": values}).to_parquet(source)
            config["paths"]["holdout_test"] = str(root / "holdout")
            record = {
                "source": str(source),
                "source_rows": rows,
                "source_sha256": file_digest(source),
                "targets": ["outcome"],
                "roles": {"past": []},
            }
            with (
                patch("holdout.FoundationBenchmarks", return_value=suite),
                patch("holdout.forecast_contexts", side_effect=neural),
            ):
                result = evaluate_holdout(model, [record], config)
            predictions = pd.read_parquet(root / "holdout/test_predictions.parquet")
        # Origin 32 is skipped for every model; origin 96 is forecast by all of
        # them and saved, but has nothing to score.
        self.assertEqual(sorted(calls), [48, 64, 80, 96])
        self.assertEqual(result["skipped_origins_without_observed_history"], 1)
        self.assertEqual(result["unscored_origin_targets_without_observed_labels"], 1)
        self.assertEqual(sorted(predictions.origin.unique()), [48, 64, 80, 96])
        unscored = predictions[predictions.origin == 96]
        self.assertEqual(len(unscored), 112)
        self.assertTrue(unscored.actual.isna().all())
        self.assertEqual(unscored.metric_weight.sum(), 0)
        self.assertTrue(unscored.rwkv.notna().all() and unscored.naive.notna().all())
        self.assertEqual(result["origin_target_count"], 3)
        self.assertEqual(result["table_count"], 1)
        # Missing points inside a scored window carry no weight; the rest sum to one.
        scored = predictions[predictions.origin == 48]
        self.assertEqual(scored[scored.actual.isna()].metric_weight.sum(), 0)
        self.assertAlmostEqual(scored.metric_weight.sum(), 1.0, places=8)
        self.assertEqual(
            result["pooled"]["observations"], int(predictions.actual.notna().sum())
        )
        self.assertEqual(result["primary_metric"], "table_mase_median")
        board = result["leaderboard"]
        self.assertEqual(
            [row["table_mase_median"] for row in board],
            sorted(row["table_mase_median"] for row in board),
        )
        self.assertTrue(all("table_mase_profile" in row for row in board))


class TimedRecipeTests(unittest.TestCase):
    """Warm-up/cosine schedule, weight average and its checkpointing in the trainer."""

    @staticmethod
    def parts(directory):
        from test_timed_training import setup

        parts = setup(directory)
        parts[3]["training"].update(
            scheduler={
                "name": "warmup_cosine",
                "warmup_steps": 2,
                "min_learning_rate": 1e-4,
            },
            weight_average={"decay": 0.5, "update_every": 1, "warmup_updates": 0},
        )
        return parts

    def test_averaged_weights_are_validated_exported_and_survive_interruption(self):
        from test_timed_training import Clock, run_training

        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            full = self.parts(root / "full")
            _, summary, schedule = run_training(full, Clock())
            self.assertEqual(summary["completed_updates"], 6)
            self.assertEqual(schedule["plan"]["name"], "warmup_cosine")
            saved = torch.load(root / "full/latest_checkpoint.pt", weights_only=True)
            average = saved["weight_average"]
            self.assertEqual(average["updates"], 6)
            history = json.loads((root / "full/validation_history.json").read_text())
            self.assertEqual(
                {row["validated_weights"] for row in history},
                {"exponential moving average"},
            )
            # The weights that were validated and saved as best are the average,
            # not the training weights of that moment.
            names = [n for n, p in full[0].named_parameters() if p.requires_grad]
            best = torch.load(root / "full/best_weights.pt", weights_only=True)[
                "state_dict"
            ]
            self.assertTrue(
                any(
                    not torch.equal(best[name], saved["state_dict"][name])
                    for name in names
                )
            )
            # Training continued from its own weights after every validation.
            self.assertEqual(
                saved["identity"]["optimization"]["weight_average"]["decay"], 0.5
            )

            interrupted = self.parts(root / "interrupted")
            interrupted[3]["training"]["max_run_hours"] = 1 / 3600
            _, paused, _ = run_training(interrupted, Clock())
            self.assertEqual(paused["reason"], "time_limit")
            run_training(self.parts(root / "interrupted"), Clock())
            resumed = torch.load(
                root / "interrupted/latest_checkpoint.pt", weights_only=True
            )
            self.assertEqual(resumed["step"], 6)
            for name in names:
                torch.testing.assert_close(
                    resumed["state_dict"][name],
                    saved["state_dict"][name],
                    rtol=0,
                    atol=0,
                )
            for a, b in zip(resumed["weight_average"]["average"], average["average"]):
                torch.testing.assert_close(a, b, rtol=0, atol=0)

            # A run without averaging cannot continue an averaged one, or vice versa.
            changed = self.parts(root / "interrupted")
            del changed[3]["training"]["weight_average"]
            with self.assertRaisesRegex(ValueError, "optimization settings differ"):
                run_training(changed, Clock())

    def test_fixed_budget_run_validates_its_final_weights(self):
        from test_timed_training import Clock, run_training, setup

        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            parts = self.parts(root / "cosine")
            parts[3]["training"]["validation_interval_seconds"] = 4
            _, summary, _ = run_training(parts, Clock())
            history = json.loads((root / "cosine/validation_history.json").read_text())
            # The periodic check at update 4, then the final weights at update 6.
            self.assertEqual([row["step"] for row in history], [4, 6])
            self.assertEqual(summary["reason"], "step_limit")
            self.assertEqual(summary["validation_checks"], 2)
            saved = torch.load(root / "cosine/latest_checkpoint.pt", weights_only=True)
            self.assertEqual(saved["loop_state"]["last_validation_step"], 6)
            # An open-ended plateau run with a debug step cap keeps its old behaviour.
            plateau = setup(root / "plateau")
            plateau[3]["training"]["validation_interval_seconds"] = 4
            run_training(plateau, Clock())
            history = json.loads((root / "plateau/validation_history.json").read_text())
            self.assertEqual([row["step"] for row in history], [4])

    def test_legacy_epoch_trainer_refuses_weight_average(self):
        from epoch_training import fit_epochs
        from test_epoch_checkpoints import setup, validation_values

        with tempfile.TemporaryDirectory() as temporary:
            model, records, config, optimizer = setup(Path(temporary), epochs=1)
            config["data"]["validation_policy"] = "reserved_tail"
            config["training"]["weight_average"] = {"decay": 0.99}
            with self.assertRaisesRegex(ValueError, "timed trainer"):
                fit_epochs(
                    model,
                    records,
                    records[:1],
                    config,
                    optimizer,
                    validation_fn=validation_values([1.0]),
                )

    def test_table_monitor_requires_table_statistics(self):
        from epoch_training import monitored_value, training_options

        options = training_options(
            {"early_stopping": {"patience_checks": 3, "monitor": "trimmed_table_loss"}}
        )
        self.assertEqual(options["monitor"], "trimmed_table_loss")
        self.assertEqual(
            monitored_value(
                {"trimmed_table_loss": 0.2, "loss": 9.0}, "trimmed_table_loss"
            ),
            0.2,
        )
        with self.assertRaisesRegex(ValueError, "did not report"):
            monitored_value({"loss": 1.0}, "trimmed_table_loss")
        with self.assertRaises(ValueError):
            training_options({"early_stopping": {"monitor": "mean_absolute_error"}})


class ArchiveTests(unittest.TestCase):
    def test_archived_500m_configuration_is_unchanged_apart_from_paths(self):
        legacy = json.loads((ROOT / "configs/legacy_500m.json").read_text())
        self.assertEqual(legacy["model"]["layers"], 28)
        self.assertEqual(legacy["model"]["width"], 1024)
        self.assertEqual(
            legacy["model"]["horizon_loss_weights"], [0.8, 0.16, 0.03, 0.01]
        )
        self.assertEqual(legacy["paths"]["training"], "results/500m_model/training")
        self.assertNotIn("head", legacy["model"])
        settings = ForecastConfig(
            **dict(legacy["model"], quantiles=tuple(legacy["model"]["quantiles"]))
        )
        self.assertEqual(settings.head, "lead_offset")
        self.assertEqual(parameter_budget(legacy)["parameters"], 500_056_352)

    def test_loaded_model_forecasts_exactly_like_the_saved_one(self):
        # load() builds an empty model and fills it from the file; anything the
        # file does not carry (derived constants) must still be correct.
        torch.manual_seed(19)
        model = build().eval()
        open_branches(model)
        history = torch.randn(1, 1, 200).cumsum(-1)
        covariates = torch.randn(1, 2, 200)
        expected = model.forecast(history, 112, past_covariates=covariates).quantiles
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "model.pt"
            model.save(path)
            with patch(
                "train.load_rwkv", return_value=SimpleNamespace(Block=CausalBlock)
            ):
                loaded = RWKVForecaster.load(
                    path, device="cpu", dtype=torch.float32
                ).eval()
        self.assertEqual(set(loaded.state_dict()), set(model.state_dict()))
        actual = loaded.forecast(history, 112, past_covariates=covariates).quantiles
        torch.testing.assert_close(actual, expected, rtol=0, atol=0)

    def test_rotary_tables_built_in_validation_serve_training(self):
        attention = TemporalAttention(64, 32)
        nn.init.normal_(attention.output.weight, std=0.5)
        x = torch.randn(1, 1, 8, 64)
        with torch.inference_mode():
            attention(x)
        attention(x.requires_grad_()).sum().backward()  # must not raise
        with self.assertRaises(ValueError):
            attention(torch.randn(1, 1, 33, 64))

    def test_checkpoint_round_trip_keeps_new_options(self):
        model = build(fp32_io=True)
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "model.pt"
            model.save(path)
            saved = torch.load(path, weights_only=True)
        self.assertEqual(saved["config"]["head"], "direct")
        self.assertEqual(saved["config"]["temporal_every"], 2)
        restored = ForecastConfig(**saved["config"])
        self.assertEqual(restored, model.config)


if __name__ == "__main__":
    unittest.main()

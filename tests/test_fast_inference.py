"""Direct forecast contracts, gradients and causal label isolation."""

import sys
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src/05_model_training"))
from multi_horizon import auxiliary_loss
from test_training_performance import CausalBlock
from train import ForecastConfig, RWKVForecaster


def small_model():
    with patch("train.load_rwkv", return_value=SimpleNamespace(Block=CausalBlock)):
        return RWKVForecaster(
            ForecastConfig(
                objective="next_patch",
                patch_size=16,
                width=64,
                layers=2,
                context_length=256,
                max_patches=16,
                output_patches=4,
            )
        )


class DirectForecastTests(unittest.TestCase):
    def test_one_backbone_call_for_multi_patch_and_partial_horizon(self):
        model = small_model().eval()
        context = torch.randn(1, 2, 80)
        with patch.object(
            model.blocks[0], "forward", wraps=model.blocks[0].forward
        ) as forward:
            result = model.forecast(context, 49)
        self.assertEqual(forward.call_count, 1)
        self.assertEqual(result.quantiles.shape, (1, 2, 49, 9))
        first = model.forecast(context, 16)
        torch.testing.assert_close(result.quantiles[..., :16, :], first.quantiles)
        with patch.object(
            model.blocks[0], "forward", wraps=model.blocks[0].forward
        ) as forward:
            self.assertEqual(model.forecast(context, 75).point.shape[-1], 75)
        self.assertEqual(forward.call_count, 2)

    def test_training_updates_leads_encoder_and_temporal_model(self):
        torch.manual_seed(13)
        model = small_model().train()
        prepared = model.tokenize_next(torch.randn(2, 1, 256), training=True)
        result = model.forward_encoded(model.encode_tokens(prepared), prepared)
        result.loss.backward()
        for parameter in (
            model.lead_embedding.weight,
            model.embedding.skip.weight,
            model.blocks[0].projection.weight,
        ):
            self.assertIsNotNone(parameter.grad)
            self.assertTrue(torch.isfinite(parameter.grad).all())
            self.assertGreater(parameter.grad.abs().sum().item(), 0)

    def test_auxiliary_targets_never_wrap_or_include_padding(self):
        model = small_model()
        hidden = torch.randn(1, 1, 4, 64)
        labels = torch.full((1, 1, 4, 16), float("nan"))
        labels[:, :, 1] = 2
        loc, scale = torch.zeros(1, 1, 4, 1), torch.ones(1, 1, 4, 1)
        with patch(
            "multi_horizon.torch.randint",
            return_value=torch.ones(1, 1, 4, dtype=torch.long),
        ):
            loss, count = auxiliary_loss(model, hidden, labels, loc, scale)
            self.assertEqual(count.item(), 16)
            loss.backward()
        self.assertTrue(torch.isfinite(loss))
        labels.fill_(float("nan"))
        loss, count = auxiliary_loss(model, hidden, labels, loc, scale)
        self.assertEqual(count.item(), 0)
        self.assertEqual(loss.item(), 0)

    def test_future_labels_do_not_affect_predictions(self):
        model = small_model().train()
        prepared = model.tokenize_next(torch.randn(1, 1, 128), training=True)
        first = model.forward_encoded(model.encode_tokens(prepared), prepared)
        prepared["labels"] = prepared["labels"] + 1000
        second = model.forward_encoded(model.encode_tokens(prepared), prepared)
        torch.testing.assert_close(first.quantiles, second.quantiles)
        self.assertNotEqual(first.loss.item(), second.loss.item())

    def test_invalid_output_settings_and_legacy_default(self):
        self.assertEqual(ForecastConfig().output_patches, 1)
        for kwargs in (
            {"output_patches": 0},
            {"output_patches": True},
            {"multi_horizon_weight": -1},
            {"output_patches": 2},
        ):
            with self.assertRaises(ValueError):
                ForecastConfig(**kwargs)


@unittest.skipUnless(torch.cuda.is_available(), "CUDA required")
class RecurrentKernelTests(unittest.TestCase):
    def test_capture_restores_state_mutated_by_warmup(self):
        from fast_inference import GraphCache

        cache = GraphCache(1)

        def advance(inputs):
            inputs["state"].add_(1)
            return inputs["state"].clone()

        for initial in (0, 2, -1):
            value = torch.full((4,), float(initial), device="cuda")
            actual = cache.run("state", advance, {"state": value})
            torch.testing.assert_close(actual, value + 1)
            torch.testing.assert_close(value, torch.full_like(value, float(initial)))
        self.assertEqual(cache.summary()["captures"], 1)
        self.assertEqual(cache.summary()["replay_requests"], 3)

    def test_graph_forecasts_match_eager_and_do_not_alias(self):
        from fast_inference import FastInference

        torch.manual_seed(11)
        model = (
            RWKVForecaster(
                ForecastConfig(
                    objective="next_patch",
                    width=64,
                    patch_size=16,
                    layers=2,
                    max_patches=16,
                    context_length=256,
                    output_patches=4,
                )
            )
            .to("cuda", dtype=torch.bfloat16)
            .eval()
        )
        with torch.no_grad():
            for block in model.blocks:
                block.att.output.weight.normal_(std=0.01)
                block.ffn.value.weight.normal_(std=0.01)
        engine = FastInference(model, max_graphs=2)
        saved = None
        with torch.inference_mode():
            for length in (32, 48, 127, 256, 307):
                context = torch.randn(1, 2, length)
                covariates = torch.randn(1, 1, length)
                context[..., 3] = float("nan")
                batch = model.tokenize_next(context, past_covariates=covariates)
                batch = {
                    k: v.to("cuda") if isinstance(v, torch.Tensor) else v
                    for k, v in batch.items()
                }
                for patches in (1, 4):
                    batch["prediction_patches"] = patches
                    expected = model.forward_encoded(
                        model.encode_tokens(batch), batch
                    ).quantiles
                    actual = engine.quantiles(batch)
                    torch.testing.assert_close(actual, expected, rtol=0, atol=0)
                    if saved is None:
                        saved = actual, actual.clone()
            torch.testing.assert_close(*saved, rtol=0, atol=0)
        engine.clear()

    def test_state_chunks_and_graph_replay_on_nondefault_stream(self):
        from recurrent import load_kernel

        kernel = load_kernel()
        torch.manual_seed(9)
        tensors = [
            torch.randn(2, 19, 128, device="cuda", dtype=torch.bfloat16) * 0.1
            for _ in range(6)
        ]
        whole_state = torch.zeros(2, 2, 64, 64, device="cuda")
        split_state = torch.zeros_like(whole_state)
        whole = kernel(*tensors, whole_state)
        parts = []
        for start, stop in ((0, 16), (16, 17), (17, 19)):
            parts.append(
                kernel(*(t[:, start:stop].contiguous() for t in tensors), split_state)
            )
        torch.testing.assert_close(whole, torch.cat(parts, dim=1), rtol=0, atol=0)
        torch.testing.assert_close(whole_state, split_state, rtol=0, atol=0)
        from fast_inference import GraphCache

        graph = GraphCache(1)

        def run(inputs):
            state = torch.zeros_like(whole_state)
            return kernel(*(inputs[str(i)] for i in range(6)), state)

        for factor in (1, 2, -1):
            inputs = {str(i): t * factor for i, t in enumerate(tensors)}
            actual = graph.run("kernel", run, inputs)
            torch.testing.assert_close(actual, run(inputs), rtol=0, atol=0)
        graph.clear()


if __name__ == "__main__":
    unittest.main()

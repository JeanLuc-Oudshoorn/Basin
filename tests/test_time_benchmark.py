"""Benchmark adapter contracts: shared quantiles, one target, causal rollouts."""

import sys
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock, patch

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
from time_benchmark import (
    depth_variant,
    foundation_prediction,
    history,
    rwkv_prediction,
)


class TimeBenchmarkTests(unittest.TestCase):
    def test_depth_view_preserves_full_model_and_shares_existing_parameters(self):
        from dataclasses import replace

        from test_training_performance import model_and_records

        model, _ = model_and_records()
        model.blocks.append(model.blocks[0])
        model.variate.append(model.variate[0])
        model.config = replace(model.config, layers=3)
        reduced = depth_variant(model, 2)
        self.assertEqual(len(model.blocks), 3)
        self.assertEqual(len(reduced.blocks), 2)
        self.assertEqual((model.config.layers, reduced.config.layers), (3, 2))
        self.assertIs(model.blocks[0], reduced.blocks[0])
        self.assertIs(model.head, reduced.head)
        self.assertIsNot(model._modules, reduced._modules)

    def test_patch_rollout_caps_context_and_marks_future_covariates_unknown(self):
        levels = [0.1, 0.5, 0.9]
        model = SimpleNamespace(
            config=SimpleNamespace(patch_size=16, context_length=64, quantiles=levels)
        )
        context = history(64, 3, 7)
        original = {k: v.clone() for k, v in context.items()}
        seen = []

        def predict(actual_model, contexts, batch_size):
            self.assertIs(actual_model, model)
            self.assertEqual(batch_size, 1)
            seen.append({k: v.clone() for k, v in contexts[0].items()})
            return [torch.full((1, 16, 3), float(len(seen)))]

        with patch("time_benchmark.forecast_contexts", side_effect=predict):
            result = rwkv_prediction(model, context, 33)
        self.assertEqual(result.shape, (1, 33, 3))
        np.testing.assert_array_equal(result[0, :, 1], [1] * 16 + [2] * 16 + [3])
        self.assertTrue(torch.isnan(seen[1]["past_covariates"][..., -16:]).all())
        self.assertTrue(torch.isnan(seen[2]["past_covariates"][..., -32:]).all())
        self.assertTrue(all(c["context"].shape[-1] == 64 for c in seen))
        for name in context:
            torch.testing.assert_close(context[name], original[name])

    def test_timestral_single_pass_and_default_return_same_requested_levels(self):
        levels = [0.1, 0.5, 0.9]
        trained = (0.01, 0.1, 0.3, 0.5, 0.7, 0.9, 0.99)
        network = Mock()
        network.forecast.return_value = SimpleNamespace(
            quantiles=torch.arange(7).expand(1, 1, 16, 7), quantile_levels=trained
        )
        suite = SimpleNamespace(models={"timestral": network})
        for name, averaging in (("timestral", False), ("timestral_default", True)):
            result = foundation_prediction(
                suite, name, history(32, 0, 7), 16, levels, "fp32"
            )
            self.assertEqual(result.shape, (1, 16, 3))
            np.testing.assert_array_equal(result[0, 0], [1, 3, 5])
            self.assertEqual(
                network.forecast.call_args.kwargs["symmetric_averaging"], averaging
            )

    def test_chronos_receives_only_past_data_and_all_requested_quantiles(self):
        network = Mock()
        network.predict_quantiles.return_value = ([torch.ones(1, 64, 3)], None)
        suite = SimpleNamespace(models={"chronos": network})
        result = foundation_prediction(
            suite, "chronos", history(128, 3, 7), 64, [0.1, 0.5, 0.9], "fp32"
        )
        self.assertEqual(result.shape, (1, 64, 3))
        call = network.predict_quantiles.call_args
        self.assertFalse(call.kwargs["cross_learning"])
        self.assertEqual(set(call.args[0][0]), {"target", "past_covariates"})
        self.assertEqual(call.kwargs["quantile_levels"], [0.1, 0.5, 0.9])


if __name__ == "__main__":
    unittest.main()

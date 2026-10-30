"""Benchmark selection, dataset sampling and covariate-safe provider adapters."""

import copy
import sys
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src/03_transforming_input_data"))
sys.path.insert(0, str(ROOT / "src/05_model_training"))
sys.path.insert(0, str(ROOT / "src/06_holdout_test"))
from benchmarks import FoundationBenchmarks, benchmark_names, sample_holdout
from holdout import baseline_forecasts, summarize
from transform import load_config


class FoundationBenchmarkTests(unittest.TestCase):
    def test_defaults_and_invalid_model_lists(self):
        config = load_config()
        # Test the fallback default, independent of the user's active model list.
        config["holdout"].pop("benchmark_models", None)
        self.assertEqual(benchmark_names(config), ["prophet", "naive", "chronos"])
        for value in (["rwkv"], ["chronos", "chronos"], "chronos", [1]):
            with self.subTest(value=value):
                config["holdout"]["benchmark_models"] = value
                with self.assertRaises(ValueError):
                    benchmark_names(config)

    def test_whole_table_sampling_rounding_order_seed_and_full_coverage(self):
        records = [{"source": str(i)} for i in range(101)]
        config = load_config()
        config["holdout"]["sample_percent"] = 10
        selected, audit = sample_holdout(records, config)
        again, _ = sample_holdout(list(reversed(records)), config)
        self.assertEqual(selected, again)
        self.assertEqual(len(selected), 11)
        self.assertEqual(audit["available_holdout_tables"], 101)
        config["holdout"]["sample_seed"] += 1
        self.assertNotEqual(sample_holdout(records, config)[0], selected)
        config["holdout"]["sample_percent"] = 0.001
        self.assertEqual(len(sample_holdout(records, config)[0]), 1)
        config["holdout"]["sample_percent"] = 100
        self.assertEqual(
            {r["source"] for r in sample_holdout(records, config)[0]},
            {r["source"] for r in records},
        )
        for invalid in (0, -1, 101, True, float("nan")):
            config["holdout"]["sample_percent"] = invalid
            with self.assertRaises(ValueError):
                sample_holdout(records, config)

    def test_naive_only_does_not_fit_prophet_and_dynamic_metrics(self):
        import pandas as pd

        config = load_config()
        config["holdout"]["benchmark_models"] = ["naive"]
        np.testing.assert_array_equal(
            baseline_forecasts([1, 4, np.nan], 3, config)["naive"], [4, 4, 4]
        )
        frame = pd.DataFrame(
            {
                "actual": [1, 2],
                "rwkv": [1, 1],
                "chronos": [2, 2],
                "context_scale": [1, 1],
                "normalized_pinball": [0.1, 0.2],
            }
        )
        metrics = summarize(frame)
        self.assertEqual(set(metrics), {"observations", "rwkv", "chronos"})

    def test_providers_receive_only_independent_historical_targets_and_covariates(self):
        suite = object.__new__(FoundationBenchmarks)
        suite.context_limit = 2048
        chronos = Mock()
        timestral = Mock()
        contexts = [
            {
                "context": torch.arange(2 * n).float().reshape(1, 2, n),
                "past_covariates": torch.arange(n).float().reshape(1, 1, n),
            }
            for n in (32, 64)
        ]
        original = copy.deepcopy(contexts)
        points = [torch.ones(2, 16), torch.full((2, 16), 2.0)]
        chronos.predict_quantiles.return_value = ([], points)
        timestral.forecast.side_effect = [
            SimpleNamespace(median=p[None]) for p in points
        ]
        suite.models = {"chronos": chronos, "timestral": timestral}
        suite.options = {"chronos": {"batch_size": 32}}
        result = suite.predict(contexts, 16)
        kwargs = chronos.predict_quantiles.call_args.kwargs
        self.assertFalse(kwargs["cross_learning"])
        self.assertEqual(kwargs["context_length"], 2048)
        supplied = chronos.predict_quantiles.call_args.args[0]
        self.assertEqual([item["target"].shape[-1] for item in supplied], [32, 64])
        self.assertTrue(
            all(set(item) == {"target", "past_covariates"} for item in supplied)
        )
        for context, saved in zip(contexts, original):
            for key in context:
                torch.testing.assert_close(context[key], saved[key])
        self.assertEqual(set(result[0]), {"chronos", "timestral"})
        self.assertEqual(result[0]["chronos"].shape, (2, 16))
        chronos.predict_quantiles.return_value = ([], [torch.zeros(3, 16)] * 2)
        with self.assertRaisesRegex(ValueError, "Invalid chronos"):
            suite.predict(contexts, 16)


if __name__ == "__main__":
    unittest.main()

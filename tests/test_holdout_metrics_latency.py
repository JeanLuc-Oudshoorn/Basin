"""MASE arithmetic, no-lookahead scaling, and comparable request timing."""

import json
import sys
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock, patch

import numpy as np
import pandas as pd
import torch

ROOT = Path(__file__).resolve().parents[1]
for stage in ("03_transforming_input_data", "05_model_training", "06_holdout_test"):
    sys.path.insert(0, str(ROOT / "src" / stage))
from holdout import evaluate_holdout, mase_scale, summarize
from latency import RequestTimer
from transform import file_digest, load_config


class MaseTests(unittest.TestCase):
    def test_four_token_weights_control_all_scores_and_scaled_quantiles(self):
        # Constant error per token: 1, 2, 3, 4; the last lead must not dominate.
        error = np.repeat([1.0, 2.0, 3.0, 4.0], 16)
        weights = np.repeat(np.array([0.8, 0.16, 0.03, 0.01]) / 16, 16)
        frame = pd.DataFrame(
            {
                "actual": np.zeros(64),
                "rwkv": error,
                "mase_scale": 1.0,
                "context_scale": 1.0,
                "metric_weight": weights,
                "normalized_pinball": error / 2,
            }
        )
        metric = summarize(frame)["rwkv"]
        self.assertAlmostEqual(metric["mase"], 1.25)
        self.assertAlmostEqual(metric["mae"], 1.25)
        self.assertAlmostEqual(metric["rmse"], np.sqrt(1.87))
        self.assertAlmostEqual(metric["normalized_pinball"], 0.625)
        self.assertEqual(metric["median_absolute_scaled_error"], 1.0)
        self.assertEqual(metric["absolute_scaled_error_p10"], 1.0)
        self.assertEqual(metric["absolute_scaled_error_p90"], 2.0)
        # Unobserved futures are retained for plotting but cannot change metrics.
        frame.loc[16:, "actual"] = np.nan
        self.assertEqual(summarize(frame)["observations"], 16)
        self.assertAlmostEqual(summarize(frame)["rwkv"]["mase"], 1.0)

    def test_invalid_mase_seasonality_fails_before_loading_models(self):
        for value in (0, -1, True, 1.5, "1"):
            with (
                self.subTest(value=value),
                self.assertRaisesRegex(ValueError, "mase_seasonality"),
            ):
                evaluate_holdout(None, [], {"holdout": {"mase_seasonality": value}})

    def test_historical_nonseasonal_and_seasonal_scales(self):
        self.assertEqual(mase_scale([1, 3, 6, 10]), 3)
        self.assertEqual(mase_scale([1, 3, 6, 10], 2), 6)
        # Missing pairs are omitted, without interpolating or bridging gaps.
        self.assertEqual(mase_scale([1, 3, np.nan, 100, 104]), 3)
        for history, lag in (([1, 1, 1], 1), ([1], 1), ([1, 2], 2), ([np.nan, 2], 1)):
            self.assertTrue(np.isnan(mase_scale(history, lag)))

    def test_point_weighted_mase_and_explicit_undefined_count(self):
        frame = pd.DataFrame(
            {
                "actual": [0, 0, 0, 0],
                "rwkv": [2, 8, 6, 100],
                "naive": [1, 4, 3, 0],
                "mase_scale": [2, 4, 2, np.nan],
                "context_scale": [1] * 4,
                "normalized_pinball": [0.1] * 4,
            }
        )
        metrics = summarize(frame)
        self.assertEqual(metrics["rwkv"]["mase"], 2)
        self.assertEqual(metrics["naive"]["mase"], 1)
        self.assertEqual(metrics["rwkv"]["mase_observations"], 3)
        self.assertEqual(metrics["rwkv"]["mase_undefined_observations"], 1)
        self.assertEqual(metrics["rwkv"]["mae"], 29)
        frame["mase_scale"] = np.nan
        self.assertIsNone(summarize(frame)["rwkv"]["mase"])
        json.dumps(summarize(frame), allow_nan=False)

    def test_all_adapters_have_single_origin_same_full_horizon_and_timing(self):
        config = load_config("configs/legacy_500m.json")
        config["plot"]["max_plots"] = 0
        config["holdout"].update(
            benchmark_models=["naive", "chronos", "timestral"],
            latency={"warmup_requests": 1, "repeats": 2},
        )
        model = torch.nn.Linear(1, 1)
        model.config = SimpleNamespace(
            patch_size=16, context_length=2048, quantiles=(0.1, 0.5, 0.9)
        )
        suite = Mock()
        suite.names = config["holdout"]["benchmark_models"]
        suite.metadata = {name: {"device": "cpu"} for name in suite.names}

        def neural(actual_model, contexts, batch_size, prediction_patches=1):
            self.assertEqual((len(contexts), batch_size), (1, 1))
            last = contexts[0]["context"][0, 0, -1]
            return [torch.full((1, prediction_patches * 16, 3), last)]

        def foundation(contexts, horizon, *, model_names):
            self.assertEqual((len(contexts), horizon, len(model_names)), (1, 64, 1))
            return [
                {
                    model_names[0]: np.full(
                        (1, horizon), contexts[0]["context"][0, 0, -1]
                    )
                }
            ]

        suite.predict.side_effect = foundation
        with tempfile.TemporaryDirectory() as temporary:
            base = Path(temporary)
            source = base / "data.parquet"
            # Last partial patch has huge changes that MUST NOT enter its scale.
            pd.DataFrame(
                {
                    "outcome": list(range(96))
                    + [10000, 20000, 30000]
                    + list(range(99, 131))
                }
            ).to_parquet(source)
            config["paths"]["holdout_test"] = str(base / "output")
            record = {
                "source": str(source),
                "source_sha256": file_digest(source),
                "targets": ["outcome"],
                "roles": {"past": []},
            }
            with (
                patch("holdout.FoundationBenchmarks", return_value=suite),
                patch("holdout.forecast_contexts", side_effect=neural),
            ):
                result = evaluate_holdout(model, [record], config)
            points = pd.read_parquet(base / "output/test_predictions.parquet")
            np.testing.assert_array_equal(points.mase_scale, 1)
            self.assertTrue((points.groupby("origin").size() == 64).all())
            requests = pd.read_parquet(base / "output/request_latencies.parquet")
            journal = [
                json.loads(line)
                for line in (base / "output/request_latencies.jsonl")
                .read_text()
                .splitlines()
            ]
            self.assertEqual(len(requests), len(journal))
            self.assertEqual(result["primary_metric"], "table_mase_median")
            self.assertEqual(result["pooled"]["observations"], 192)
            self.assertTrue((requests.horizon == 64).all())
            self.assertTrue((requests.scored_horizon == 64).all())
            for name in ["rwkv"] + suite.names:
                self.assertEqual(result["latency"][name]["requests"], 6)
                self.assertEqual(result["latency"][name]["forecast_origins"], 3)
                self.assertEqual(result["latency"][name]["warmup_requests"], 1)
                self.assertGreaterEqual(result["latency"][name]["min_ms"], 0)
            json.dumps(result, allow_nan=False)


class LatencyTests(unittest.TestCase):
    def test_warmup_retained_separately_repeats_keep_first_scored_forecast(self):
        timer = RequestTimer(
            {"holdout": {"latency": {"warmup_requests": 1, "repeats": 2}}}
        )
        function = Mock(side_effect=["cold", "first", "second", "next", "repeat"])
        # Deterministic wall-clock intervals: warmup 10s; scored 1,2,3,4s.
        with patch(
            "latency.perf_counter", side_effect=[0, 10, 10, 11, 11, 13, 13, 16, 16, 20]
        ):
            result = timer.measure("rwkv", function, device="cpu", request={})
            self.assertEqual(result, "first")
            self.assertEqual(
                timer.measure("rwkv", function, device="cpu", request={}), "next"
            )
        summary = timer.summary()["rwkv"]
        self.assertEqual(summary["mean_ms"], 2500)
        self.assertEqual(summary["median_ms"], 2500)
        self.assertAlmostEqual(summary["p95_ms"], 3850)
        self.assertEqual(summary["warmup_seconds"], 10)
        self.assertEqual(summary["total_seconds"], 10)
        self.assertEqual(summary["forecast_origins"], 2)

    def test_latency_ratios_pair_matching_origins_and_context_buckets(self):
        timer = RequestTimer({})
        timer.rows = [
            {
                "source": "table",
                "origin": origin,
                "repeat": 0,
                "phase": "scored",
                "context_points": context,
                "model": method,
                "seconds": seconds,
            }
            for origin, context, method, seconds in (
                (32, 32, "rwkv", 1),
                (1024, 1024, "chronos", 1),
                (32, 32, "chronos", 2),
                (1024, 1024, "rwkv", 10),
            )
        ]
        ratio = timer.paired_comparison()["chronos"]
        self.assertEqual(ratio["paired_requests"], 2)
        self.assertEqual(ratio["median_rwkv_over_baseline_latency"], 5.25)
        self.assertEqual(
            timer.summary()["rwkv"]["by_context_points"]["513-1024"]["median_ms"], 10000
        )

    def test_cuda_is_synchronized_on_both_sides_of_the_request(self):
        timer = RequestTimer({"holdout": {"latency": {"warmup_requests": 0}}})
        events = []
        with patch(
            "latency.torch.cuda.synchronize",
            side_effect=lambda device: events.append(str(device)),
        ):
            timer.measure(
                "rwkv", lambda: events.append("predict"), device="cuda:0", request={}
            )
        self.assertEqual(events, ["cuda:0", "predict", "cuda:0"])

    def test_invalid_timing_settings_fail(self):
        for name, values in (
            ("warmup_requests", [-1, True, 1.5]),
            ("repeats", [0, True, "2"]),
        ):
            for value in values:
                with self.assertRaises(ValueError):
                    RequestTimer({"holdout": {"latency": {name: value}}})


if __name__ == "__main__":
    unittest.main()

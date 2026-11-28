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
from holdout import (
    PredictionPoints,
    by_forecast_token,
    evaluate_holdout,
    mase_scale,
    summarize,
)
from holdout_runs import resume_holdout_run
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

    def test_only_the_latency_sample_is_forecast_one_request_at_a_time(self):
        config = load_config("configs/legacy_500m.json")
        config["plot"]["max_plots"] = 0
        config["holdout"].update(
            benchmark_models=["naive", "chronos"],
            latency={"warmup_requests": 0, "repeats": 1, "tables": 1},
        )
        model = torch.nn.Linear(1, 1)
        model.config = SimpleNamespace(
            patch_size=16, context_length=2048, quantiles=(0.1, 0.5, 0.9)
        )
        suite = Mock()
        suite.names = config["holdout"]["benchmark_models"]
        suite.metadata = {name: {"device": "cpu"} for name in suite.names}
        calls = {"rwkv": [], "chronos": []}

        def last(context):
            return float(context["context"][0, 0, -1])

        def neural(actual_model, contexts, batch_size, prediction_patches=1):
            self.assertEqual(len(contexts), batch_size)
            calls["rwkv"].append(len(contexts))
            return [
                torch.full((1, prediction_patches * 16, 3), last(c)) for c in contexts
            ]

        def foundation(contexts, horizon, *, model_names=None):
            calls["chronos"].append(len(contexts))
            return [{"chronos": np.full((1, horizon), last(c))} for c in contexts]

        suite.predict.side_effect = foundation
        with tempfile.TemporaryDirectory() as temporary:
            base = Path(temporary)
            records = []
            for name, offset in (("first", 0), ("second", 1000)):
                source = base / f"{name}.parquet"
                pd.DataFrame({"outcome": np.arange(131) + offset}).to_parquet(source)
                records.append(
                    {
                        "source": str(source),
                        "source_sha256": file_digest(source),
                        "targets": ["outcome"],
                        "roles": {"past": []},
                    }
                )
            config["paths"]["holdout_test"] = str(base / "output")
            with (
                patch("holdout.FoundationBenchmarks", return_value=suite),
                patch("holdout.forecast_contexts", side_effect=neural),
            ):
                result = evaluate_holdout(model, records, config)
            # One table: three single requests per model. The other: one call.
            self.assertEqual(calls, {"rwkv": [1, 1, 1, 3], "chronos": [1, 1, 1, 3]})
            selection = json.loads((base / "output/latency_selection.json").read_text())
            self.assertEqual(selection["measured_tables"], 1)
            requests = pd.read_parquet(base / "output/request_latencies.parquet")
            self.assertEqual(set(requests.source), set(selection["sources"]))
            self.assertEqual(len(requests), 3 * 3)
            self.assertEqual(result["latency"]["rwkv"]["forecast_origins"], 3)
            self.assertEqual(result["latency_protocol"]["tables"]["measured_tables"], 1)
            # Either way every origin gets the forecast made from its own history.
            self.assertEqual(result["table_count"], 2)
            points = pd.read_parquet(base / "output/test_predictions.parquet")
            self.assertEqual(len(points), 2 * 3 * 64)
            for column in ("rwkv", "chronos", "naive"):
                expected = (
                    points.origin - 1 + 1000 * points.source.str.contains("second")
                )
                np.testing.assert_array_equal(points[column], expected)
            json.dumps(result, allow_nan=False)


class ResumeTests(unittest.TestCase):
    """An interrupted run is continued from the tables it saved."""

    def test_points_go_to_the_file_and_only_score_columns_stay_in_memory(self):
        rng = np.random.default_rng(0)

        def table(name, rows):
            return pd.DataFrame(
                {
                    "source": name,
                    "target": "outcome",
                    "origin": 32,
                    "forecast_token": np.arange(rows) % 3 + 1,
                    "metric_weight": 1 / rows,
                    "actual": rng.normal(size=rows),
                    "rwkv": rng.normal(size=rows),
                    "naive": rng.normal(size=rows),
                    "context_scale": 2.0,
                    "mase_scale": 0.5,
                    "normalized_pinball": rng.random(rows),
                }
            )

        tables = [table(f"table {i}", 6 + i) for i in range(5)]
        tables[1].loc[2:, "actual"] = np.nan  # unobserved futures
        tables[3]["actual"] = np.nan  # a token... and a table without any
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "test_predictions.parquet"
            points = PredictionPoints(path, ["rwkv", "naive"], tables_per_write=2)
            for frame in tables:
                points.add(frame)
            points.close()
            points.close()  # closing twice is harmless
            everything = pd.concat(tables, ignore_index=True)
            pd.testing.assert_frame_equal(
                pd.read_parquet(path), everything, check_dtype=False
            )
            self.assertEqual(points.rows, len(everything))
            columns = points.columns()
        self.assertNotIn("source", columns)
        # Scores from the kept arrays equal those from the whole frame.
        self.assertEqual(summarize(columns), summarize(everything))
        self.assertEqual(
            by_forecast_token(columns),
            {
                str(token): summarize(group)
                for token, group in everything.groupby("forecast_token")
            },
        )

    def test_a_resumed_run_gives_the_results_of_an_uninterrupted_one(self):
        config = load_config("configs/legacy_500m.json")
        config["plot"]["max_plots"] = 1
        config["holdout"].update(
            benchmark_models=["naive", "chronos"],
            latency={"warmup_requests": 0, "repeats": 1, "tables": 1},
        )
        model = torch.nn.Linear(1, 1)
        model.config = SimpleNamespace(
            patch_size=16, context_length=2048, quantiles=(0.1, 0.5, 0.9)
        )
        suite = Mock()
        suite.names = config["holdout"]["benchmark_models"]
        suite.metadata = {name: {"device": "cpu"} for name in suite.names}
        calls = []
        state = {"fail_at_batched_call": None}

        def last(context):
            return float(context["context"][0, 0, -1])

        def neural(actual_model, contexts, batch_size, prediction_patches=1):
            calls.append(len(contexts))
            if calls.count(3) == state["fail_at_batched_call"]:
                raise KeyboardInterrupt("the run is stopped here")
            return [
                torch.full((1, prediction_patches * 16, 3), last(c) + 0.25)
                for c in contexts
            ]

        def foundation(contexts, horizon, *, model_names=None):
            return [{"chronos": np.full((1, horizon), last(c) - 0.5)} for c in contexts]

        suite.predict.side_effect = foundation
        rng = np.random.default_rng(3)
        with tempfile.TemporaryDirectory() as temporary:
            base = Path(temporary)
            records = []
            for index in range(5):
                source = base / f"table{index}.parquet"
                values = rng.normal(size=131).cumsum()
                if index == 2:
                    values[100:] = np.nan  # the last origins have no labels
                pd.DataFrame({"outcome": values}).to_parquet(source)
                records.append(
                    {
                        "source": str(source),
                        "source_sha256": file_digest(source),
                        "targets": ["outcome"],
                        "roles": {"past": []},
                    }
                )

            def evaluate(folder, resume=False):
                run = dict(config, paths=dict(config["paths"], holdout_test=folder))
                (base / folder).mkdir(exist_ok=True)
                if resume:
                    _, run = resume_holdout_run(run, ".")
                with (
                    patch("holdout.FoundationBenchmarks", return_value=suite),
                    patch("holdout.forecast_contexts", side_effect=neural),
                ):
                    return evaluate_holdout(model, records, run)

            whole = evaluate(str(base / "whole"))
            self.assertEqual(calls, [1, 1, 1, 3, 3, 3, 3])
            self.assertNotIn("resumed", whole)
            # The same run, stopped while it forecasts its third batched table.
            calls.clear()
            state["fail_at_batched_call"] = 3
            with self.assertRaises(KeyboardInterrupt):
                evaluate(str(base / "stopped"))
            state["fail_at_batched_call"] = None
            stopped = base / "stopped"
            saved = sorted(stopped.glob("test_table_*.parquet"))
            self.assertEqual(len(saved), 3)
            self.assertFalse((stopped / "test_metrics.json").exists())
            # One saved table was cut off while it was written.
            timed = json.loads((stopped / "latency_selection.json").read_text())
            damaged = next(
                path
                for path in saved
                if records[int(path.stem.rsplit("_", 1)[1])]["source"]
                not in timed["sources"]
            )
            damaged.write_bytes(damaged.read_bytes()[:100])
            calls.clear()
            resumed = evaluate(str(stopped), resume=True)
            # Only the damaged table and the two never reached are forecast.
            self.assertEqual(calls, [3, 3, 3])
            self.assertEqual(
                resumed["resumed"]["tables_read_from_the_interrupted_run"], 2
            )
            for key in (
                "pooled",
                "by_forecast_token",
                "table_summary",
                "table_count",
                "origin_target_count",
                "horizon_policy",
                "skipped_origins_without_observed_history",
                "unscored_origin_targets_without_observed_labels",
                "unscored_tables",
            ):
                self.assertEqual(resumed[key], whole[key], key)
            self.assertEqual(
                [row["model"] for row in resumed["leaderboard"]],
                [row["model"] for row in whole["leaderboard"]],
            )
            self.assertEqual(resumed["latency"]["rwkv"]["forecast_origins"], 3)
            for name in ("table_metrics.json", "origin_metrics.json"):
                self.assertEqual(
                    json.loads((stopped / name).read_text()),
                    json.loads((base / "whole" / name).read_text()),
                    name,
                )
            pd.testing.assert_frame_equal(
                pd.read_parquet(stopped / "test_predictions.parquet"),
                pd.read_parquet(base / "whole/test_predictions.parquet"),
            )
            self.assertEqual(
                len(pd.read_parquet(stopped / "request_latencies.parquet")), 3 * 3
            )
            self.assertEqual(list(stopped.glob("*.partial")), [])
            json.dumps(resumed, allow_nan=False)
            # A finished run is not reopened, and other settings are refused.
            with self.assertRaisesRegex(ValueError, "finished run"):
                evaluate(str(stopped), resume=True)
            (stopped / "test_metrics.json").unlink()
            config["holdout"]["latency"]["tables"] = 2
            with self.assertRaisesRegex(ValueError, "other tables or latency"):
                evaluate(str(stopped), resume=True)


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

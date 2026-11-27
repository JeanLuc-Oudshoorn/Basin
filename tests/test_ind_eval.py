"""Stage 12: IND-Eval reading, long horizons, scoring and a CPU run."""

import json
import sys
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src" / "12_ind_eval"))
HAS_SUBMODULE = (ROOT / "benchmarks/fakebench-ts/src/fakebench_ts").is_dir()


def write_task(path, rows):
    """A task file in the layout of IND-Eval: one row per channel of a window."""
    path.parent.mkdir(parents=True, exist_ok=True)
    columns = {name: [row[name] for row in rows] for name in rows[0]}
    for name in ("context", "target"):
        columns[name] = pa.array(columns[name], pa.list_(pa.float64()))
    pq.write_table(pa.table(columns), path)


def wave(length, period=8, level=10.0):
    return level + np.sin(2 * np.pi * np.arange(length) / period)


def rows_of(instances, horizon, context=64, **extra):
    # A little noise, so that no forecast is exact.
    noise = np.random.default_rng(0).normal(0, 0.01, context + horizon)
    series = wave(context + horizon) + noise
    return [
        {
            "instance_id": f"run-{i}",
            "channel": channel,
            "window": window,
            "context": (series * (1 + i)).tolist()[:context],
            "target": (series * (1 + i)).tolist()[context:],
            "freq": "10s",
            "prediction_length": horizon,
            **extra,
        }
        for i in range(instances)
        for window in (0, 1)
        for channel in ("pressure", "flow")
    ]


@unittest.skipUnless(HAS_SUBMODULE, "the FakeBench-TS submodule is not checked out")
class IndEvalTests(unittest.TestCase):
    def setUp(self):
        import ind_eval
        import ind_eval_data
        import ind_eval_scoring

        self.stage, self.data, self.scoring = ind_eval, ind_eval_data, ind_eval_scoring
        self.folder = tempfile.TemporaryDirectory()
        self.root = Path(self.folder.name)
        self.addCleanup(self.folder.cleanup)

    def options(self, **changes):
        return self.data.settings({"ind_eval": {"path": str(self.root), **changes}})

    def test_a_task_is_read_as_one_series_per_target_channel(self):
        rows = rows_of(2, 16, seasonality=8)
        rows[0]["role"] = "covariate"
        for row in rows[1:]:
            row["role"] = "target"
        # A file may store more history than a model is given.
        rows[1]["context"] = [float("nan")] * 3 + list(range(5000))
        write_task(self.root / "data/pronto/event_short.parquet", rows)
        write_task(
            self.root / "data/electricity/electricity-H_short.parquet", rows_of(1, 16)
        )
        options = self.options()
        tasks = self.data.list_tasks(options)
        self.assertEqual(
            [task.key for task in tasks],
            ["electricity__electricity-H_short", "pronto__event_short"],
        )
        series = [s for group in self.data.read_task(tasks[1], options) for s in group]
        self.assertEqual(len(series), 7)  # the covariate row is not a target
        self.assertEqual(series[0].key, "run-0|0|flow")
        self.assertEqual(len(series[0].context), 2048)
        self.assertEqual({s.horizon for s in series}, {16})
        self.assertEqual(series[0].context[-1], 4999)
        self.assertEqual({s.seasonality for s in series}, {8})
        self.assertEqual(len({s.key for s in series}), 7)
        # Without a stated cycle it follows from the sampling rate.
        hourly = [s for group in self.data.read_task(tasks[0], options) for s in group]
        self.assertEqual({s.seasonality for s in hourly}, {1})
        self.assertEqual(self.data.settings({})["seasonality_by_freq"]["H"], 24)
        # A target longer than the forecast length is cut to its first points.
        short = self.options(horizon=10, context_points=32)
        cut = [s for g in self.data.read_task(tasks[0], short) for s in g]
        self.assertEqual({(len(s.context), s.horizon) for s in cut}, {(32, 10)})
        np.testing.assert_allclose(cut[0].target, hourly[0].target[:10])
        # An even subset is the same for every model.
        few = self.options(max_series_per_task=3)
        picked = [s.key for g in self.data.read_task(tasks[0], few) for s in g]
        self.assertEqual(len(picked), 3)
        self.assertEqual(
            picked, [s.key for g in self.data.read_task(tasks[0], few) for s in g]
        )
        with self.assertRaisesRegex(FileNotFoundError, "not downloaded"):
            self.data.list_tasks(self.options(sources=["neats"]))
        with self.assertRaisesRegex(ValueError, "Unknown ind_eval settings"):
            self.data.settings({"ind_eval": {"max_call_horizon": 5}})
        with self.assertRaisesRegex(ValueError, "positive whole number"):
            self.data.settings({"ind_eval": {"horizon": 0}})

    def test_missing_history_is_filled_and_the_latest_points_are_kept(self):
        context = np.array([np.nan, 1.0, np.nan, 3.0, np.nan], dtype=np.float32)
        np.testing.assert_allclose(
            self.data.model_context(context, 10), [1, 1, 2, 3, 3]
        )
        np.testing.assert_allclose(self.data.model_context(context, 2), [3, 3])
        self.assertIsNone(self.data.model_context(context[:1], 10))

    def test_scores_are_relative_to_seasonal_naive_and_sources_count_equally(self):
        series = self.data.Series(
            "a|0|x", wave(64).astype(np.float32), wave(80)[64:].astype(np.float32), 8
        )
        self.assertAlmostEqual(
            self.scoring.seasonal_scale(series.context, 1),
            float(np.abs(np.diff(series.context)).mean()),
            5,
        )
        self.assertTrue(np.isnan(self.scoring.seasonal_scale(np.ones(20), 1)))
        exact = self.scoring.score_series(
            series, np.repeat(series.target[None], 9, 0), 64
        )
        self.assertEqual((exact["absolute_error"], exact["quantile_loss"]), (0, 0))
        # A point forecast one unit off: absolute error 1 per step, and the
        # quantile loss is the same absolute error.
        off = self.scoring.score_series(
            series, np.repeat(series.target[None] + 1, 9, 0), 64
        )
        self.assertAlmostEqual(off["absolute_error"], 16, 4)
        self.assertAlmostEqual(off["quantile_loss"], 16, 4)
        # A missing target value is not scored.
        series.target[3] = np.nan
        self.assertEqual(
            self.scoring.score_series(series, np.zeros((9, 16)), 64)["scored_points"],
            15,
        )

        def frame(error, scale=2.0):
            return pd.DataFrame(
                {
                    "series": ["s1", "s2"],
                    "horizon": 10,
                    "context_points": 64,
                    "scored_points": 10,
                    "absolute_error": [10.0 * error, 10.0 * error],
                    "scale": scale,
                    "quantile_loss": [10.0 * error, 10.0 * error],
                    "absolute_target": 100.0,
                }
            )

        scores = {
            "seasonal_naive": {
                ("a", "t1"): frame(1.0),
                ("a", "t2"): frame(1.0),
                ("b", "t1"): frame(1.0),
            },
            # Half the error on both tasks of source a, a quarter on source b.
            "model": {
                ("a", "t1"): frame(0.5),
                ("a", "t2"): frame(0.5),
                ("b", "t1"): frame(0.25),
                ("c", "t1"): frame(0.1),
            },
        }
        results = self.scoring.summarise(scores)
        self.assertEqual(results["overall"]["seasonal_naive"]["nmase"], 1.0)
        self.assertAlmostEqual(results["by_source"]["a"]["model"]["nmase"], 0.5)
        self.assertAlmostEqual(results["by_source"]["b"]["model"]["ncrps"], 0.25)
        # The geometric mean of 0.5 and 0.25, not weighted by tasks or series.
        self.assertAlmostEqual(results["overall"]["model"]["nmase"], 0.125**0.5)
        self.assertEqual(results["tasks_not_run_by_every_model"], ["c/t1"])
        self.assertEqual(results["tasks"]["a/t1"]["models"]["model"]["mase"], 0.25)
        board = self.scoring.leaderboard(results)
        self.assertIn("## nMASE by source", board)
        self.assertIn("| model | 0.354 | 0.354 |", board)

    def test_a_run_on_the_cpu_can_be_continued_with_another_model(self):
        write_task(
            self.root / "data/metropt3/short.parquet", rows_of(2, 16, seasonality=8)
        )
        write_task(
            self.root / "data/metropt3/long.parquet", rows_of(2, 40, seasonality=8)
        )
        write_task(self.root / "data/neats/a_short.parquet", rows_of(1, 16))
        results = self.root / "results"
        config = {
            "paths": {"ind_eval": str(results)},
            "ind_eval": {"path": str(self.root), "holt_winters_workers": 1},
        }
        args = SimpleNamespace(models=["naive"], sources=None, limit=None, resume=None)
        output = self.stage.run(config, args)
        first = json.loads((output / "results.json").read_text())
        self.assertEqual(first["models"], ["naive", "seasonal_naive"])
        self.assertEqual(first["series"], 8 + 8 + 4)
        self.assertEqual(first["tasks_per_source"], {"metropt3": 2, "neats": 1})
        # The series repeat every 8 points: seasonal naive is close where the
        # task states that cycle, and equals naive where it states none.
        metro = first["tasks"]["metropt3/short"]["models"]
        # Its error is that of one cycle earlier, which is also the scale.
        self.assertLess(abs(metro["seasonal_naive"]["mase"] - 1.2), 0.6)
        self.assertGreater(metro["naive"]["nmase"], 10)
        self.assertAlmostEqual(first["by_source"]["neats"]["naive"]["nmase"], 1.0)
        written = {
            path: path.stat().st_mtime_ns
            for path in output.rglob("scores/**/*.parquet")
        }
        self.assertEqual(len(written), 6)
        args = SimpleNamespace(
            models=["holtwinters"], sources=None, limit=None, resume=output.name
        )
        self.assertEqual(self.stage.run(config, args), output)
        second = json.loads((output / "results.json").read_text())
        self.assertEqual(
            sorted(second["models"]), ["holtwinters", "naive", "seasonal_naive"]
        )
        # Holt-Winters finds the cycle and beats the last value by far.
        self.assertLess(second["by_source"]["neats"]["holtwinters"]["nmase"], 0.2)
        self.assertEqual(second["speed"]["holtwinters"]["series"], 20)
        # Nothing that was scored is forecast again.
        for path, modified in written.items():
            self.assertEqual(path.stat().st_mtime_ns, modified)
        self.assertIn("Holt-Winters", (output / "leaderboard.md").read_text())
        args.models, config["ind_eval"]["max_series_per_task"] = ["naive"], 2
        with self.assertRaisesRegex(ValueError, "cannot be continued"):
            self.stage.run(config, args)


if __name__ == "__main__":
    unittest.main()

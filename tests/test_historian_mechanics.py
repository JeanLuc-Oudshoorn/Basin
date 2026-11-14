import copy
import json
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src/01_data_generation"))
from generation_runtime import aggregate_frame
from plant_mechanisms import _interlock, _stick_slip
from scm_generator import OperationalSCMGenerator


class MechanismTests(unittest.TestCase):
    def test_aggregate_availability_alignment_and_future_isolation(self):
        frame = pd.DataFrame(
            {
                "time_index": np.arange(23),
                "outcome": np.arange(23.0) ** 2,
                "covariate_00": -np.arange(23.0),
            }
        )
        for w in (2, 3, 4, 5):
            for kind, fn in [("min", np.min), ("max", np.max), ("mean", np.mean)]:
                actual = aggregate_frame(frame, w, kind)
                for i, row in actual.iterrows():
                    t = int(row.time_index)
                    self.assertEqual(t, (i + 1) * w)
                    for col in ["outcome", "covariate_00"]:
                        self.assertEqual(row[col], fn(frame[col].iloc[t - w : t]))
                    changed = frame.copy()
                    changed.loc[t:, ["outcome", "covariate_00"]] = 1e9
                    pd.testing.assert_frame_equal(
                        actual.iloc[: i + 1],
                        aggregate_frame(changed, w, kind).iloc[: i + 1],
                        check_exact=True,
                    )
                    pd.testing.assert_frame_equal(
                        actual.iloc[: i + 1],
                        aggregate_frame(frame.iloc[:t], w, kind),
                        check_exact=True,
                    )
        frame.loc[2, "outcome"] = np.nan
        self.assertTrue(np.isnan(aggregate_frame(frame, 3, "mean").outcome.iloc[0]))

    def test_copy_replays_with_history_and_preserves_rng_and_provenance(self):
        with tempfile.TemporaryDirectory() as folder:
            g = OperationalSCMGenerator(
                folder,
                seed=42,
                min_rows=600,
                max_rows=600,
                min_covariates=3,
                max_covariates=3,
                full_length_probability=1,
                prehistory_probability=1,
                max_prehistory_rows=100,
                aggregate_copy_probability=1,
                save_plot=False,
                save_dag=False,
            )
            model = g.sample_model()
            frame = g.simulate(model)
            before = copy.deepcopy(model)
            state = copy.deepcopy(g.rng.bit_generator.state)
            original = Path(folder) / "original"
            g._write_table(model, frame, original)
            derived = g._write_aggregate_copy(model, frame, original)
            saved = json.loads((derived / "scm.json").read_text())
            pd.testing.assert_frame_equal(
                g.simulate(saved),
                pd.read_parquet(derived / "data.parquet"),
                check_exact=True,
            )
            self.assertEqual(
                json.loads((derived / "provenance.json").read_text())["source_table"],
                "original",
            )
            self.assertEqual(model, before)
            self.assertEqual(g.rng.bit_generator.state, state)
            self.assertIsNone(
                g._write_aggregate_copy(saved, g.simulate(saved), derived)
            )

    def test_aggregate_parallel_replay_and_worker_independence(self):
        with tempfile.TemporaryDirectory() as folder:
            runs = []
            for workers in (2, 3):
                generator = OperationalSCMGenerator(
                    Path(folder) / str(workers),
                    seed=8,
                    min_rows=600,
                    max_rows=600,
                    min_covariates=3,
                    max_covariates=3,
                    generation_workers=workers,
                    generation_batch_size=1,
                    aggregate_copy_probability=1,
                    coarse_copy_probability=0,
                    sign_inversion_probability=0,
                    save_plot=False,
                    save_dag=False,
                )
                paths = generator.generate(4)
                self.assertEqual(len(paths), 8)
                runs.append(paths)
                for path in paths:
                    model = json.loads((path / "scm.json").read_text())
                    pd.testing.assert_frame_equal(
                        generator.simulate(model),
                        pd.read_parquet(path / "data.parquet"),
                        check_exact=True,
                    )
            for left, right in zip(*runs, strict=True):
                self.assertEqual(
                    (left / "data.parquet").read_bytes(),
                    (right / "data.parquet").read_bytes(),
                )
                self.assertEqual(
                    (left / "scm.json").read_bytes(), (right / "scm.json").read_bytes()
                )

    def test_moderate_trends_accepted_without_correction(self):
        for proposal in (True, False):
            for score, strength in [(0.2, 0.1), (0.3, 0.3), (0.1, 0.2)]:
                g = OperationalSCMGenerator(
                    "unused",
                    min_rows=30,
                    max_rows=30,
                    min_covariates=3,
                    max_covariates=3,
                )
                model = g.sample_model(trend_free=proposal)
                frame = g.simulate(model)
                with (
                    patch.object(g, "sample_model", return_value=model),
                    patch.object(g, "_simulate_process", return_value=frame),
                    patch.object(g, "trend_score", return_value=score),
                    patch.object(g, "trend_strength", return_value=strength),
                ):
                    saved, actual = g._sample_accepted_model(proposal)
                self.assertEqual(saved["quality"]["sampling_attempts"], 1)
                if score > 0.15:
                    self.assertFalse(saved["quality"]["screened_trend_free"])
                pd.testing.assert_frame_equal(actual, frame, check_exact=True)

    def test_stick_slip_holds_and_releases(self):
        spec = dict(
            static_friction=1.0,
            kinetic_ratio=0.25,
            drive_rate=0.03,
            release_fraction=0.7,
        )
        result = _stick_slip(spec, np.ones(600), np.zeros(600))
        steps = np.diff(result["position"])
        self.assertGreater(np.sum(steps == 0), 100)
        self.assertGreater(steps.max(), 0.2)
        self.assertGreater(np.sum(np.diff(result["pv"]) < -0.1), 5)
        prefix = _stick_slip(spec, np.ones(300), np.zeros(300))
        for name in result:
            np.testing.assert_array_equal(result[name][:300], prefix[name])

    def test_interlock_latches_until_safe_reset_edge(self):
        spec = dict(
            command={}, trip_level=0.7, safe_level=0.2, tau=5.0, cooling_tau=5.0
        )
        reset = np.zeros(100)
        reset[10:12] = 1
        reset[60:62] = 1
        with patch("plant_mechanisms._command_path", return_value=reset):
            result = _interlock(
                spec, np.ones(100), np.zeros(100), np.random.default_rng(0)
            )
        self.assertEqual(result["status"][59], 1)
        self.assertLess(result["pv"][59], 0.2)
        self.assertEqual(result["status"][60], 0)
        self.assertTrue(np.all(result["actuator"][result["status"] == 1] == 0))

    def test_new_plants_saved_replay_and_causality(self):
        for family in ("stick_slip", "interlock"):
            g = OperationalSCMGenerator(
                "unused",
                seed=7,
                min_rows=600,
                max_rows=600,
                min_covariates=3,
                max_covariates=3,
                plant_probability=1,
                plant_weights={
                    family: 1,
                    **{
                        n: 0
                        for n in ["relay", "pid", "batch", "step_response", "totalizer"]
                    },
                },
            )
            model = g.sample_model(trend_free=True)
            self.assertEqual(model["plant"]["family"], family)
            frame = g.simulate(model)
            pd.testing.assert_frame_equal(
                frame, g.simulate(json.loads(json.dumps(model))), check_exact=True
            )
            self.assertTrue(np.isfinite(frame.to_numpy()).all())


if __name__ == "__main__":
    unittest.main()

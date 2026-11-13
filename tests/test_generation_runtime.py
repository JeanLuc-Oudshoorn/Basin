"""Progress accounting and reproducible bounded multiprocessing generation."""

import io
import json
import sys
import tempfile
import unittest
from contextlib import redirect_stdout
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src/01_data_generation"))
from generation_runtime import GenerationProgress, duration
from scm_generator import OperationalSCMGenerator


class GenerationRuntimeTests(unittest.TestCase):
    def test_eta_and_throttled_progress(self):
        time = [0.0]
        output = io.StringIO()
        progress = GenerationProgress(100, 4, interval=1, clock=lambda: time[0])
        with redirect_stdout(output):
            progress.update(0, force=True)
            time[0] = 10
            progress.update(5)
            time[0] = 10.1
            progress.update(6)  # Suppressed, rather than logging every table.
            time[0] = 20
            progress.update(100, force=True)
        lines = output.getvalue().splitlines()
        self.assertEqual(len(lines), 3)
        self.assertIn("ETA estimating", lines[0])
        self.assertIn("0.50 tables/s | ETA 00:03:10", lines[1])
        self.assertIn("[100/100 100.0%]", lines[2])
        self.assertIn("ETA 00:00:00", lines[2])
        self.assertEqual(duration(90061), "1d 01:01:01")

    def test_worker_settings_validation(self):
        for name in ("generation_workers", "generation_batch_size"):
            for value in (0, -1, True, 1.5):
                with (
                    self.subTest(name=name, value=value),
                    self.assertRaises(ValueError),
                ):
                    OperationalSCMGenerator("unused", **{name: value})
        for value in (0, -1, float("inf"), float("nan")):
            with self.assertRaises(ValueError):
                OperationalSCMGenerator("unused", progress_interval=value)

    def test_parallel_seed_stream_independent_of_worker_and_batch_counts(self):
        with tempfile.TemporaryDirectory() as directory:
            options = dict(
                seed=915,
                sign_inversion_probability=0,
                process_episode_probability=1,
                formula_covariate_probability=1,
                min_rows=48,
                max_rows=48,
                min_covariates=3,
                max_covariates=4,
                save_plot=True,
                save_dag=True,
            )
            left = OperationalSCMGenerator(
                Path(directory) / "two",
                generation_workers=2,
                generation_batch_size=1,
                **options,
            )
            right = OperationalSCMGenerator(
                Path(directory) / "three",
                generation_workers=3,
                generation_batch_size=2,
                **options,
            )
            left_paths, right_paths = left.generate(6), right.generate(6)
            trend_free = 0
            for source, candidate in zip(left_paths, right_paths, strict=True):
                expected = pd.read_parquet(source / "data.parquet")
                pd.testing.assert_frame_equal(
                    expected,
                    pd.read_parquet(candidate / "data.parquet"),
                    check_exact=True,
                )
                self.assertEqual(
                    (source / "scm.json").read_bytes(),
                    (candidate / "scm.json").read_bytes(),
                )
                model = json.loads((source / "scm.json").read_text())
                pd.testing.assert_frame_equal(
                    expected, left.simulate(model), check_exact=True
                )
                trend_free += model["quality"]["screened_trend_free"]
                for name in ("series.png", "dag.png"):
                    self.assertEqual(
                        (source / name).read_bytes(), (candidate / name).read_bytes()
                    )
            self.assertEqual(trend_free, 3)
            # The parent stream advances between calls; no duplicate datasets.
            more = left.generate(1)[0]
            self.assertNotEqual(
                (more / "scm.json").read_bytes(),
                (left_paths[0] / "scm.json").read_bytes(),
            )


if __name__ == "__main__":
    unittest.main()

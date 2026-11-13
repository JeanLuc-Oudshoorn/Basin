"""Synchronized thinning, exact replay and original RNG isolation."""

import copy
import io
import json
import sys
import tempfile
import unittest
from contextlib import redirect_stdout
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src/01_data_generation"))
from generation_runtime import coarsen_frame
from scm_generator import OperationalSCMGenerator


class CoarseCopyTests(unittest.TestCase):
    def test_fixed_offset_all_windows_and_all_columns(self):
        index = np.arange(517)
        source = pd.DataFrame(
            {
                "time_index": index,
                "outcome": index * 3.0,
                "covariate_00": -index,
                "covariate_01": index % 2,
            }
        )
        original = source.copy()
        for window in (2, 3, 4, 5):
            for offset in range(window):
                indices = offset + window * np.arange(len(source) // window)
                result = coarsen_frame(source, window, offset)
                expected = source.iloc[indices].reset_index(drop=True)
                expected["time_index"] = np.arange(len(expected))
                pd.testing.assert_frame_equal(result, expected, check_exact=True)
        pd.testing.assert_frame_equal(source, original)
        for window, offset in ((1, 0), (6, 0), (2, -1), (3, 3), (2, True)):
            with self.assertRaises(ValueError):
                coarsen_frame(source, window, offset)

    def test_threshold_lineage_and_saved_scm_replay(self):
        with tempfile.TemporaryDirectory() as directory:
            generator = OperationalSCMGenerator(
                directory,
                seed=32,
                min_rows=513,
                max_rows=513,
                min_covariates=3,
                max_covariates=4,
                coarse_copy_probability=1,
                save_plot=True,
                save_dag=True,
            )
            model = generator.sample_model()
            untouched = copy.deepcopy(model)
            frame = generator.simulate(model)
            target = Path(directory) / "original"
            generator._write_table(model, frame, target)
            state = copy.deepcopy(generator.rng.bit_generator.state)
            self.assertIsNone(
                generator._write_coarse_copy(model, frame.iloc[:512], target)
            )
            derived_dir = generator._write_coarse_copy(model, frame, target)
            derived = json.loads((derived_dir / "scm.json").read_text())
            actual = pd.read_parquet(derived_dir / "data.parquet")
            pd.testing.assert_frame_equal(
                actual, generator.simulate(derived), check_exact=True
            )
            self.assertEqual(derived["rows"], len(actual))
            self.assertFalse(derived["quality"]["screened"])
            self.assertEqual(generator.rng.bit_generator.state, state)
            self.assertEqual(model, untouched)
            self.assertIsNone(
                generator._write_coarse_copy(derived, actual, derived_dir)
            )
            provenance = json.loads((derived_dir / "provenance.json").read_text())
            self.assertEqual(provenance["source_table"], target.name)
            for name in ("series.png", "dag.png"):
                self.assertGreater((derived_dir / name).stat().st_size, 1000)

    def test_serial_augmentation_preserves_original_tables_and_progress(self):
        with tempfile.TemporaryDirectory() as directory:
            options = dict(
                seed=816,
                sign_inversion_probability=0,
                min_rows=513,
                max_rows=513,
                min_covariates=3,
                max_covariates=4,
                save_plot=False,
                save_dag=False,
            )
            disabled = OperationalSCMGenerator(
                Path(directory) / "off", coarse_copy_probability=0, **options
            )
            enabled = OperationalSCMGenerator(
                Path(directory) / "on", coarse_copy_probability=1, **options
            )
            output = io.StringIO()
            with redirect_stdout(output):
                before, after = disabled.generate(3), enabled.generate(3)
            self.assertEqual(len(after), 6)
            self.assertIn("[3/3 100.0%]", output.getvalue())
            self.assertIn("+3 coarse copies", output.getvalue())
            for source, candidate in zip(before, after[::2], strict=True):
                pd.testing.assert_frame_equal(
                    pd.read_parquet(source / "data.parquet"),
                    pd.read_parquet(candidate / "data.parquet"),
                    check_exact=True,
                )
                self.assertEqual(
                    (source / "scm.json").read_bytes(),
                    (candidate / "scm.json").read_bytes(),
                )

    def test_parallel_copies_are_worker_and_batch_independent_without_saved_scm(self):
        with tempfile.TemporaryDirectory() as directory:
            options = dict(
                seed=29,
                sign_inversion_probability=0,
                min_rows=513,
                max_rows=513,
                min_covariates=3,
                max_covariates=3,
                coarse_copy_probability=1,
                save_scm=False,
                save_plot=False,
                save_dag=False,
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
            before, after = left.generate(6), right.generate(6)
            self.assertEqual(len(before), 12)
            for source, candidate in zip(before, after, strict=True):
                pd.testing.assert_frame_equal(
                    pd.read_parquet(source / "data.parquet"),
                    pd.read_parquet(candidate / "data.parquet"),
                    check_exact=True,
                )
                self.assertFalse((source / "scm.json").exists())
                if "_coarse_" in source.name:
                    provenance = json.loads((source / "provenance.json").read_text())
                    full = pd.read_parquet(
                        source.parent / provenance["source_table"] / "data.parquet"
                    )
                    selection = provenance["observation_selection"]
                    expected = coarsen_frame(
                        full, selection["window_size"], selection["offset"]
                    )
                    pd.testing.assert_frame_equal(
                        expected, pd.read_parquet(source / "data.parquet")
                    )


if __name__ == "__main__":
    unittest.main()

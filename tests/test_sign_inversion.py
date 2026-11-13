"""Observation-only sign inversion, deterministic augmentation and provenance."""

import io
import json
import sys
import tempfile
import unittest
from contextlib import redirect_stdout
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src/01_data_generation"))
from generation_runtime import invert_frame
from scm_generator import OperationalSCMGenerator


class SignInversionTests(unittest.TestCase):
    def test_all_values_negated_with_clock_and_source_unchanged(self):
        source = pd.DataFrame(
            {
                "time_index": [10, 11, 12],
                "outcome": [-2.5, 0.0, 7.0],
                "covariate_00": [0, 1, 0],
                "covariate_01": [3, -4, 5],
            }
        )
        untouched = source.copy()
        actual = invert_frame(source)
        pd.testing.assert_series_equal(actual.time_index, source.time_index)
        for name in ("outcome", "covariate_00", "covariate_01"):
            pd.testing.assert_series_equal(actual[name], -source[name])
        pd.testing.assert_frame_equal(source, untouched)
        pd.testing.assert_frame_equal(invert_frame(actual), source)

    def test_original_and_coarse_copies_unchanged_and_mirror_replays(self):
        with tempfile.TemporaryDirectory() as directory:
            options = dict(
                seed=716,
                min_rows=513,
                max_rows=513,
                min_covariates=3,
                max_covariates=3,
                coarse_copy_probability=1,
                save_plot=False,
                save_dag=False,
            )
            off = OperationalSCMGenerator(
                Path(directory) / "off", sign_inversion_probability=0, **options
            )
            on = OperationalSCMGenerator(
                Path(directory) / "on", sign_inversion_probability=1, **options
            )
            output = io.StringIO()
            with redirect_stdout(output):
                before, after = off.generate(2), on.generate(2)
            self.assertEqual(len(before), 4)
            self.assertEqual(len(after), 6)
            self.assertIn("+2 coarse copies", output.getvalue())
            self.assertIn("+2 sign-inverted copies", output.getvalue())
            for index in range(2):
                for variant in range(2):
                    expected, actual = (
                        before[2 * index + variant],
                        after[3 * index + variant],
                    )
                    pd.testing.assert_frame_equal(
                        pd.read_parquet(expected / "data.parquet"),
                        pd.read_parquet(actual / "data.parquet"),
                        check_exact=True,
                    )
                    self.assertEqual(
                        (expected / "scm.json").read_bytes(),
                        (actual / "scm.json").read_bytes(),
                    )
                source, mirrored = after[3 * index], after[3 * index + 2]
                expected = invert_frame(pd.read_parquet(source / "data.parquet"))
                actual = pd.read_parquet(mirrored / "data.parquet")
                pd.testing.assert_frame_equal(expected, actual, check_exact=True)
                model = json.loads((mirrored / "scm.json").read_text())
                pd.testing.assert_frame_equal(
                    actual, on.simulate(model), check_exact=True
                )
                self.assertFalse(model["quality"]["screened"])
                self.assertEqual(on._write_extra_copies(model, actual, mirrored), [])
                provenance = json.loads((mirrored / "provenance.json").read_text())
                self.assertEqual(provenance["source_table"], source.name)

    def test_parallel_mirrors_and_lineage_without_scm_export(self):
        with tempfile.TemporaryDirectory() as directory:
            options = dict(
                seed=46,
                min_rows=48,
                max_rows=48,
                min_covariates=3,
                max_covariates=4,
                coarse_copy_probability=0,
                sign_inversion_probability=1,
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
            for expected, actual in zip(before, after, strict=True):
                pd.testing.assert_frame_equal(
                    pd.read_parquet(expected / "data.parquet"),
                    pd.read_parquet(actual / "data.parquet"),
                    check_exact=True,
                )
                if actual.name.endswith("_sign_inverted"):
                    self.assertFalse((actual / "scm.json").exists())
                    lineage = json.loads((actual / "provenance.json").read_text())
                    source = pd.read_parquet(
                        actual.parent / lineage["source_table"] / "data.parquet"
                    )
                    pd.testing.assert_frame_equal(
                        invert_frame(source),
                        pd.read_parquet(actual / "data.parquet"),
                        check_exact=True,
                    )

    def test_defaults_validation_and_optional_plots(self):
        self.assertEqual(
            OperationalSCMGenerator("unused").sign_inversion_probability, 0.03
        )
        for value in (-0.1, 1.1, float("nan")):
            with self.assertRaises(ValueError):
                OperationalSCMGenerator("unused", sign_inversion_probability=value)
        with tempfile.TemporaryDirectory() as directory:
            generator = OperationalSCMGenerator(
                directory,
                min_rows=48,
                max_rows=48,
                sign_inversion_probability=1,
                coarse_copy_probability=0,
            )
            paths = generator.generate(1)
            self.assertEqual(len(paths), 2)
            for name in ("scm.json", "series.png", "dag.png", "provenance.json"):
                self.assertGreater((paths[1] / name).stat().st_size, 0)


if __name__ == "__main__":
    unittest.main()

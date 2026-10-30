import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import MagicMock, patch

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src/02_data_validation"))
from prophet_validator import (
    ProphetValidator,
    _worker_environment,
    multivariate_admitted,
)


class ProphetValidationTests(unittest.TestCase):
    def test_fit_outputs_are_on_project_drive_and_removed_each_fit(self):
        import os
        import tempfile as temporary

        with tempfile.TemporaryDirectory() as root:
            directory = (Path(root) / "stan-temp").resolve()
            validator = ProphetValidator(root, root, temp_dir=directory)
            model = MagicMock()
            model.predict.return_value = pd.DataFrame({"yhat": [1.0]})
            before, default = dict(os.environ), temporary.tempdir

            def fit(train, **kwargs):
                self.assertEqual(Path(kwargs["output_dir"]).parent, directory)
                self.assertEqual(Path(temporary.gettempdir()), directory)
                (Path(kwargs["output_dir"]) / "large-output.csv").write_text("output")

            model.fit.side_effect = fit
            train = pd.DataFrame(
                {"ds": pd.date_range("2000-01-01", periods=3), "y": [0.0, 1.0, 2.0]}
            )
            future = pd.DataFrame({"ds": [pd.Timestamp("2000-01-04")]})
            with patch("prophet.Prophet", return_value=model):
                np.testing.assert_equal(validator._forecast(train, future), [1.0])
                self.assertEqual(list(directory.iterdir()), [])

                def failed_fit(train, **kwargs):
                    fit(train, **kwargs)
                    raise RuntimeError("fit failed")

                model.fit.side_effect = failed_fit
                with self.assertRaisesRegex(RuntimeError, "fit failed"):
                    validator._forecast(train, future)
            self.assertEqual(list(directory.iterdir()), [])
            self.assertEqual(dict(os.environ), before)
            self.assertEqual(temporary.tempdir, default)

    def test_disk_full_aborts_instead_of_rejecting_remaining_tables(self):
        with tempfile.TemporaryDirectory() as root:
            path, _ = self.fixture(root)
            validator = ProphetValidator(root, root)
            with patch.object(
                validator,
                "_forecast",
                side_effect=OSError(28, "No space left on device"),
            ):
                with self.assertRaisesRegex(OSError, "stopping screening"):
                    validator.evaluate_table(path)

    def test_reserved_test_tail_never_enters_screening(self):
        with tempfile.TemporaryDirectory() as root:
            path, frame = self.fixture(root)
            validator = ProphetValidator(path.parent.parent, root, reserved_tail_rows=4)
            calls = []

            def forecast(train, future, covariates=()):
                calls.append((train.copy(), future.copy()))
                return np.full(len(future), 12.0)

            with patch.object(validator, "_forecast", side_effect=forecast):
                first = validator.evaluate_table(path)
                frame.loc[16:, "outcome"] = np.nan
                frame.loc[16:, "covariate_00"] = 999999
                frame.to_parquet(path)
                second = validator.evaluate_table(path)
            self.assertEqual(first["status"], "ok")
            self.assertEqual(first["n_rows"], 16)
            self.assertEqual(first["n_train"], 12)
            for key in ["baseline_rmse", "multivariate_rmse"]:
                self.assertEqual(first[key], second[key])
            for train, future in calls:
                self.assertLessEqual(train.y.max(), 11)
                self.assertLess(future.ds.max(), pd.Timestamp("2000-01-17"))

    def fixture(self, root):
        path = Path(root) / "day" / "table" / "data.parquet"
        path.parent.mkdir(parents=True)
        frame = pd.DataFrame(
            {
                "time_index": np.arange(20),
                "outcome": np.arange(20, dtype=float),
                "covariate_00": np.arange(20, dtype=float),
                "covariate_01": 1.0,
                "covariate_00_lag_1": np.arange(20) - 1,
                "outcome_lag_1": np.arange(20) - 1,
            }
        )
        frame.to_parquet(path, index=False)
        return path, frame

    def test_split_metrics_and_no_target_leakage(self):
        with tempfile.TemporaryDirectory() as root:
            path, frame = self.fixture(root)
            validator = ProphetValidator(path.parent.parent, root)
            calls = []

            def forecast(train, future, covariates):
                calls.append((train.copy(), future.copy(), list(covariates)))
                return np.full(len(future), 15 if not covariates else 17)

            with patch.object(validator, "_forecast", side_effect=forecast):
                result = validator.evaluate_table(path)
            baseline = np.sqrt(7.5)
            multi = np.sqrt(1.5)
            self.assertEqual(result["n_train"], 16)
            self.assertEqual(result["n_test"], 4)
            self.assertAlmostEqual(result["baseline_rmse"], baseline)
            self.assertEqual(result["baseline_mae"], 2.5)
            self.assertEqual(result["multivariate_mae"], 1.0)
            self.assertAlmostEqual(result["multivariate_rmse"], multi)
            self.assertNotIn("univariate_rmse", result)
            self.assertEqual(result["multivariate_beats_baseline"], 1)
            self.assertAlmostEqual(
                result["multivariate_improvement_pct"],
                60.0,
            )
            self.assertEqual(len(calls), 1)
            self.assertEqual(calls[0][2], ["covariate_00"])
            for train, future, _ in calls:
                self.assertEqual(train.y.tolist(), list(range(16)))
                self.assertNotIn("y", future)
                self.assertNotIn("outcome_lag_1", train)
                self.assertLess(train.ds.max(), future.ds.min())
            self.assertEqual(calls[0][1].covariate_00.tolist(), [16, 17, 18, 19])

    def test_forecast_covariates_never_reads_future_covariates(self):
        with tempfile.TemporaryDirectory() as root:
            path, frame = self.fixture(root)
            frame.loc[16:, "covariate_00"] = np.nan
            frame.to_parquet(path)
            validator = ProphetValidator(
                path.parent.parent, root, covariate_mode="forecast"
            )
            calls = []

            def forecast(train, future, covariates=()):
                calls.append(future.copy())
                return np.full(len(future), 12.0)

            with patch.object(validator, "_forecast", side_effect=forecast):
                result = validator.evaluate_table(path)
            self.assertEqual(result["status"], "ok")
            self.assertEqual(len(calls), 2)
            self.assertEqual(calls[-1].covariate_00.tolist(), [12.0] * 4)

    def test_zero_baseline_tie_and_failed_fit(self):
        with tempfile.TemporaryDirectory() as root:
            path, frame = self.fixture(root)
            frame["outcome"] = 5.0
            frame.to_parquet(path)
            validator = ProphetValidator(path.parent.parent, root)
            with patch.object(
                validator,
                "_forecast",
                return_value=np.full(4, 5.0),
            ):
                result = validator.evaluate_table(path)
            self.assertEqual(result["baseline_rmse"], 0)
            self.assertEqual(result["baseline_mae"], 0)
            self.assertEqual(result["multivariate_beats_baseline"], 0)
            self.assertEqual(result["multivariate_not_worse_than_baseline"], 1)
            self.assertTrue(np.isnan(result["multivariate_improvement_pct"]))
            self.assertTrue(multivariate_admitted(pd.DataFrame([result])).iloc[0])
            with patch.object(
                validator, "_forecast", side_effect=RuntimeError("fit failed")
            ):
                result = validator.evaluate_table(path)
            self.assertTrue(np.isnan(result["multivariate_rmse"]))
            self.assertTrue(np.isnan(result["multivariate_mae"]))
            self.assertTrue(np.isnan(result["multivariate_not_worse_than_baseline"]))
            self.assertEqual(result["status"], "error")
            self.assertIn("fit failed", result["multivariate_error"])
            self.assertFalse(multivariate_admitted(pd.DataFrame([result])).iloc[0])

    def test_real_prophet_batch_and_artifacts(self):
        with tempfile.TemporaryDirectory() as root:
            path, _ = self.fixture(root)
            (path.parent.parent / "missing").mkdir()
            batches = []
            for workers in (1, 2):
                validator = ProphetValidator(
                    path.parent.parent,
                    Path(root) / "results",
                    workers=workers,
                    prophet_options={
                        "weekly_seasonality": False,
                        "yearly_seasonality": False,
                    },
                )
                results, directory = validator.run()
                batches.append(results)
                self.assertEqual(
                    pd.read_csv(directory / "metrics.csv").table.tolist(),
                    results.table.tolist(),
                )
            pd.testing.assert_frame_equal(*batches)
            self.assertEqual(len(results), 2)
            valid = results.loc[results.table == "table"].iloc[0]
            self.assertEqual(valid.status, "ok", valid.to_dict())
            self.assertTrue(np.isfinite(valid.multivariate_rmse))
            self.assertEqual(
                results.loc[results.table == "missing", "status"].iloc[0], "error"
            )
            pd.testing.assert_frame_equal(
                results, pd.read_parquet(directory / "metrics.parquet")
            )
            self.assertTrue((directory / "metrics.csv").exists())
            self.assertTrue((directory / "settings.json").exists())

    def test_mae_admits_a_forecast_that_loses_on_rmse(self):
        with tempfile.TemporaryDirectory() as root:
            path, _ = self.fixture(root)
            validator = ProphetValidator(root, root)
            with patch.object(
                validator, "_forecast", return_value=np.array([16.0, 17.0, 18.0, 26.0])
            ):
                result = validator.evaluate_table(path)
            self.assertEqual(result["multivariate_mae"], 1.75)
            self.assertLess(result["multivariate_mae"], result["baseline_mae"])
            self.assertGreater(result["multivariate_rmse"], result["baseline_rmse"])
            self.assertTrue(multivariate_admitted(pd.DataFrame([result])).iloc[0])

    def test_no_usable_covariates_still_fits_target_once(self):
        with tempfile.TemporaryDirectory() as root:
            path, frame = self.fixture(root)
            frame["covariate_00"] = 1.0
            frame.to_parquet(path)
            validator = ProphetValidator(path.parent.parent, root)
            with patch.object(
                validator, "_forecast", return_value=np.full(4, 15.0)
            ) as forecast:
                result = validator.evaluate_table(path)
            self.assertEqual(forecast.call_count, 1)
            self.assertEqual(result["n_covariates"], 0)
            self.assertEqual(result["multivariate_not_worse_than_baseline"], 1)

    def test_worker_limits_and_environment_restoration(self):
        import os

        for value in (0, -1, True, 1.5):
            with self.assertRaisesRegex(ValueError, "workers"):
                ProphetValidator("input", "output", workers=value)
        validator = ProphetValidator("input", "output", workers=8)
        self.assertEqual(validator.worker_count(2), 2)
        with patch.dict(os.environ, {"OMP_NUM_THREADS": "7"}):
            before = dict(os.environ)
            with self.assertRaises(RuntimeError), _worker_environment():
                self.assertEqual(os.environ["OMP_NUM_THREADS"], "1")
                raise RuntimeError("worker failed")
            self.assertEqual(dict(os.environ), before)


if __name__ == "__main__":
    unittest.main()

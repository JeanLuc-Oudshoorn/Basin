import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from data_validation.prophet_validator import ProphetValidator


class ProphetValidationTests(unittest.TestCase):
    def fixture(self, root):
        path = Path(root) / "day" / "table" / "data.parquet"
        path.parent.mkdir(parents=True)
        frame = pd.DataFrame(
            {
                "time_index": np.arange(20),
                "outcome": np.arange(20, dtype=float),
                "covariate_00": np.arange(20, dtype=float),
                "covariate_01": 1.0,
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
            self.assertEqual(result["univariate_beats_baseline"], 0)
            self.assertEqual(result["multivariate_beats_baseline"], 1)
            self.assertAlmostEqual(
                result["multivariate_improvement_pct"],
                100 * (baseline - multi) / baseline,
            )
            self.assertAlmostEqual(
                result["rmse_reduction_multivariate_vs_univariate"], baseline - multi
            )
            self.assertEqual(calls[1][2], ["covariate_00"])
            for train, future, _ in calls:
                self.assertEqual(train.y.tolist(), list(range(16)))
                self.assertNotIn("y", future)
                self.assertNotIn("outcome_lag_1", train)
                self.assertLess(train.ds.max(), future.ds.min())
            self.assertEqual(calls[1][1].covariate_00.tolist(), [16, 17, 18, 19])

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
            self.assertEqual(len(calls), 3)
            self.assertEqual(calls[-1].covariate_00.tolist(), [12.0] * 4)

    def test_zero_baseline_and_failed_variant(self):
        with tempfile.TemporaryDirectory() as root:
            path, frame = self.fixture(root)
            frame["outcome"] = 5.0
            frame.to_parquet(path)
            validator = ProphetValidator(path.parent.parent, root)
            with patch.object(
                validator,
                "_forecast",
                side_effect=[np.full(4, 5.0), RuntimeError("fit failed")],
            ):
                result = validator.evaluate_table(path)
            self.assertEqual(result["baseline_rmse"], 0)
            self.assertEqual(result["univariate_beats_baseline"], 0)
            self.assertTrue(np.isnan(result["univariate_improvement_pct"]))
            self.assertTrue(np.isnan(result["multivariate_beats_baseline"]))
            self.assertEqual(result["status"], "partial_error")
            self.assertIn("fit failed", result["multivariate_error"])

    def test_real_prophet_batch_and_artifacts(self):
        with tempfile.TemporaryDirectory() as root:
            path, _ = self.fixture(root)
            (path.parent.parent / "missing").mkdir()
            validator = ProphetValidator(
                path.parent.parent,
                Path(root) / "results",
                prophet_options={
                    "weekly_seasonality": False,
                    "yearly_seasonality": False,
                },
            )
            results, directory = validator.run()
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


if __name__ == "__main__":
    unittest.main()

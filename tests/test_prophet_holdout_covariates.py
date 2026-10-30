"""Prophet holdout regressors use history-derived future values without leakage."""

import sys
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
for stage in ("03_transforming_input_data", "05_model_training", "06_holdout_test"):
    sys.path.insert(0, str(ROOT / "src" / stage))
from holdout import baseline_forecasts, prophet_forecast, prophet_regressor_inputs
from transform import load_config


class ProphetHoldoutCovariateTests(unittest.TestCase):
    def config(self):
        config = load_config()
        config["holdout"]["benchmark_models"] = ["prophet", "naive"]
        return config

    def test_forecast_regressors_then_target_and_skip_unusable_columns(self):
        config = self.config()
        covariates = pd.DataFrame(
            {
                "x": np.arange(32.0),
                "constant": np.ones(32),
                "missing": np.full(32, np.nan),
            }
        )
        before = covariates.copy(deep=True)
        with patch(
            "holdout.prophet_forecast",
            side_effect=[np.full(16, 123.0), np.full(16, 42.0)],
        ) as fit:
            result = baseline_forecasts(
                np.arange(32.0), 16, config, past_covariates=covariates
            )
        self.assertEqual(fit.call_count, 2)
        bundle = fit.call_args.args[3]
        self.assertEqual(bundle["columns"], ["x"])
        self.assertEqual(bundle["skipped"], ["constant", "missing"])
        np.testing.assert_array_equal(
            bundle["future"]["regressor_000"], np.full(16, 123.0)
        )
        np.testing.assert_array_equal(result["naive"], np.full(16, 31.0))
        np.testing.assert_array_equal(result["prophet"], np.full(16, 42.0))
        pd.testing.assert_frame_equal(covariates, before)

    def test_naive_covariate_future_uses_last_imputed_past_value(self):
        config = self.config()
        config["prophet"]["covariate_forecast"] = "naive"
        covariates = pd.DataFrame({"x": [1.0, np.nan, 4.0, np.nan]})
        with patch("holdout.prophet_forecast") as fit:
            bundle = prophet_regressor_inputs(covariates, 3, config)
        fit.assert_not_called()
        np.testing.assert_array_equal(
            bundle["future"]["regressor_000"], [4.0, 4.0, 4.0]
        )
        self.assertTrue(np.isfinite(bundle["training"]["regressor_000"]).all())

    def test_univariate_override_and_empty_covariates(self):
        config = self.config()
        config["prophet"]["use_covariates"] = False
        with patch("holdout.prophet_forecast", return_value=np.ones(16)) as fit:
            baseline_forecasts(
                np.arange(32.0),
                16,
                config,
                past_covariates=pd.DataFrame({"x": np.arange(32.0)}),
            )
        self.assertEqual(fit.call_count, 1)
        self.assertFalse(fit.call_args.args[3]["training"])
        self.assertFalse(
            prophet_regressor_inputs(pd.DataFrame(), 16, config)["columns"]
        )

    def test_prophet_fit_and_predict_include_matching_regressors_and_clean_temp_files(
        self,
    ):
        config = self.config()
        config["prophet"]["covariate_forecast"] = "naive"
        bundle = prophet_regressor_inputs(
            pd.DataFrame({"y": np.arange(32.0)}), 16, config
        )
        with patch("prophet.Prophet") as prophet:
            prophet.return_value.predict.return_value = pd.DataFrame(
                {"yhat": np.ones(16)}
            )
            prophet_forecast(np.arange(32.0), 16, config, bundle)
            prophet.return_value.add_regressor.assert_called_once_with("regressor_000")
            train = prophet.return_value.fit.call_args.args[0]
            future = prophet.return_value.predict.call_args.args[0]
            self.assertEqual(set(train), {"ds", "y", "regressor_000"})
            self.assertEqual(set(future), {"ds", "regressor_000"})
            self.assertEqual((len(train), len(future)), (32, 16))
            self.assertTrue((future.regressor_000 == 31).all())
            self.assertFalse(
                Path(prophet.return_value.fit.call_args.kwargs["output_dir"]).exists()
            )

    def test_invalid_configuration_and_infinite_covariates_fail(self):
        for field, value in (
            ("use_covariates", "yes"),
            ("covariate_forecast", "observed_future"),
        ):
            config = self.config()
            config["prophet"][field] = value
            with self.assertRaises(ValueError):
                prophet_regressor_inputs(None, 16, config)
        config = self.config()
        with self.assertRaisesRegex(ValueError, "Infinite"):
            prophet_regressor_inputs(pd.DataFrame({"x": [1.0, np.inf]}), 16, config)


if __name__ == "__main__":
    unittest.main()

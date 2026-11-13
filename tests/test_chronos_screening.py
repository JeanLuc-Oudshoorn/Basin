"""Chronos-2 covariate inputs, leakage boundaries and report integration."""

import copy
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import Mock, patch

import numpy as np
import pandas as pd
import torch

ROOT = Path(__file__).resolve().parents[1]
for stage in ("02_data_validation", "03_transforming_input_data"):
    sys.path.insert(0, str(ROOT / "src" / stage))
from chronos_validator import ChronosValidator, chronos_admitted
from transform import load_config, screening_signature, validated_sources
from validate_generated_data import validate


class ChronosScreeningTests(unittest.TestCase):
    def config(self, directory):
        config = load_config()
        config["data"]["horizon"] = 16  # Explicit legacy screening fixture.
        config["data"].pop("generated_dates", None)
        config["paths"].update(
            raw=str(Path(directory) / "raw"),
            processed=str(Path(directory) / "processed"),
        )
        config["validation"].update(
            enabled=True,
            metrics_path=str(Path(directory) / "metrics.parquet"),
            run_folders=False,
        )
        config["validation"]["chronos2"].update(context_length=32, batch_size=4)
        return config

    def table(self, config, name="table", extra=False):
        path = Path(config["paths"]["raw"]) / name / "data.parquet"
        path.parent.mkdir(parents=True, exist_ok=True)
        values = np.arange(96, dtype=float)
        frame = pd.DataFrame(
            {
                "time_index": values,
                "outcome": values,
                "covariate_0": values * 2,
                "outcome_lag_1": values,
            }
        )
        if extra:
            frame["covariate_1"] = values * 3
        frame.to_parquet(path)
        return path

    def pipeline(self):
        def predict(inputs, prediction_length, quantile_levels, **kwargs):
            self.assertEqual(quantile_levels, [0.5])
            self.assertFalse(kwargs["cross_learning"])
            self.assertTrue(all(isinstance(item, dict) for item in inputs))
            quantiles = [
                torch.full((1, prediction_length, 1), float(item["target"][-1]))
                for item in inputs
            ]
            return quantiles, [value[..., 0] for value in quantiles]

        return Mock(predict_quantiles=Mock(side_effect=predict))

    def test_fifteen_percent_boundary_zero_baseline_and_constant_history(self):
        scores = pd.DataFrame(
            {
                "status": ["ok"] * 8 + ["error"],
                "baseline_mae": [1, 1, 1, 0, 0, 1, 1, 1, 1],
                "chronos2_mae": [
                    1,
                    1.15,
                    np.nextafter(1.15, np.inf),
                    0,
                    0.01,
                    np.nan,
                    np.inf,
                    -1,
                    1,
                ],
                "mase_scale": [0] * 9,
            }
        )
        self.assertEqual(
            chronos_admitted(scores).tolist(),
            [True, True, False, True, False, False, False, False, False],
        )

    def test_covariates_batched_by_schema_and_future_values_do_not_leak(self):
        with tempfile.TemporaryDirectory() as directory:
            config = self.config(directory)
            paths = [
                self.table(config, "a"),
                self.table(config, "b"),
                self.table(config, "c", extra=True),
            ]
            validator = ChronosValidator(config)
            validator.pipeline = self.pipeline()
            different_names = pd.read_parquet(paths[1]).rename(
                columns={"covariate_0": "covariate_99"}
            )
            different_names.to_parquet(paths[1])
            rows = list(validator.evaluate_tables(paths))
            self.assertEqual(validator.pipeline.predict_quantiles.call_count, 2)
            self.assertTrue(all(row["admitted"] for _, row in rows))
            before = validator._prepare(paths[0])
            record, inputs, actual = before
            self.assertEqual(set(inputs["past_covariates"]), {"covariate_0"})
            self.assertNotIn("future_covariates", inputs)
            np.testing.assert_array_equal(
                inputs["past_covariates"]["covariate_0"], np.arange(32, 64) * 2
            )
            self.assertEqual(record["forecast_stop"], 80)
            frame = pd.read_parquet(paths[0])
            frame.loc[64:, "covariate_0"] = np.nan
            frame.loc[80:, "outcome"] = 1e9
            frame.to_parquet(paths[0])
            after = validator._prepare(paths[0])
            self.assertEqual(after[0]["status"], "ok")
            np.testing.assert_array_equal(
                inputs["past_covariates"]["covariate_0"],
                after[1]["past_covariates"]["covariate_0"],
            )
            np.testing.assert_array_equal(actual, after[2])

    def test_known_future_covariates_and_explicit_past_roles(self):
        with tempfile.TemporaryDirectory() as directory:
            config = self.config(directory)
            path = self.table(config, extra=True)
            config["data"].update(
                past_covariates=["covariate_1"], known_covariates=["covariate_0"]
            )
            validator = ChronosValidator(config)
            record, inputs, _ = validator._prepare(path)
            self.assertEqual(record["status"], "ok")
            self.assertEqual(
                set(inputs["past_covariates"]), {"covariate_0", "covariate_1"}
            )
            np.testing.assert_array_equal(
                inputs["future_covariates"]["covariate_0"], np.arange(64, 80) * 2
            )
            canonical = validator._canonical_input(inputs)
            self.assertEqual(set(canonical["past_covariates"]), {"past_0", "known_0"})
            np.testing.assert_array_equal(
                canonical["future_covariates"]["known_0"],
                inputs["future_covariates"]["covariate_0"],
            )
            config["data"]["past_covariates"] = ["outcome_lag_1"]
            self.assertEqual(
                ChronosValidator(config)._prepare(path)[0]["status"], "error"
            )

    def test_report_retry_transform_and_old_bolt_scores_are_not_reused(self):
        with tempfile.TemporaryDirectory() as directory:
            config = self.config(directory)
            paths = [self.table(config)]
            pipeline = self.pipeline()
            with patch.object(
                ChronosValidator,
                "_load",
                lambda obj: setattr(obj, "pipeline", pipeline),
            ):
                scores = validate(config)
            self.assertEqual(scores.status.tolist(), ["ok"])
            self.assertEqual(scores.admission_metric.tolist(), ["mae"])
            self.assertEqual(validated_sources(config, paths)[0], paths)
            with patch.object(
                ChronosValidator, "_load", side_effect=AssertionError("must reuse")
            ):
                validate(config, retry_errors=True)
            report = Path(config["validation"]["metrics_path"])
            old = pd.read_parquet(report).rename(
                columns={
                    "chronos2_mae": "chronos_bolt_mae",
                    "chronos2_mase": "chronos_bolt_mase",
                }
            )
            old["screening_signature"] = "old_bolt_signature"
            old.to_parquet(report)
            config["validation"]["auto_run"] = False
            with self.assertRaisesRegex(ValueError, "stale Chronos-2"):
                validated_sources(config, paths)
            with patch.object(
                ChronosValidator,
                "_load",
                lambda obj: setattr(obj, "pipeline", pipeline),
            ):
                retried = validate(config, retry_errors=True)
            self.assertTrue(retried.chronos2_mae.notna().all())

    def test_failed_covariate_inputs_do_not_call_model(self):
        with tempfile.TemporaryDirectory() as directory:
            config = self.config(directory)
            path = self.table(config)
            frame = pd.read_parquet(path)
            frame.loc[40, "covariate_0"] = np.inf
            frame.to_parquet(path)
            validator = ChronosValidator(config)
            with patch.object(
                validator, "_load", side_effect=AssertionError("invalid input")
            ):
                result = list(validator.evaluate_tables([path]))[0][1]
            self.assertEqual(result["status"], "error")
            self.assertIn("Historical covariate", result["error"])

    def test_signature_binds_model_threshold_and_input_roles(self):
        config = load_config()
        expected = screening_signature(config)
        for section, key, value in [
            ("data", "past_covariates", []),
            ("data", "targets", ["different_target"]),
            ("data", "excluded_column_patterns", []),
            ("validation", "max_relative_degradation", 0.1),
            ("validation", "engine", "chronos_bolt"),
        ]:
            changed = copy.deepcopy(config)
            changed[section][key] = value
            self.assertNotEqual(screening_signature(changed), expected)
        config["validation"]["chronos2"]["revision"] = "different_revision"
        self.assertNotEqual(screening_signature(config), expected)


if __name__ == "__main__":
    unittest.main()

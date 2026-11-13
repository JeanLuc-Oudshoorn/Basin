"""Chronos-Bolt admission boundaries, no leakage, batching and gate integration."""

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
from bolt_validator import BoltValidator, bolt_admitted
from transform import load_config, screening_signature, validated_sources
from validate_generated_data import validate


class BoltScreeningTests(unittest.TestCase):
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
            engine="chronos_bolt",
            max_relative_degradation=0.1,
            metrics_path=str(Path(directory) / "metrics.parquet"),
            run_folders=False,
        )
        config["validation"]["chronos_bolt"] = dict(
            config["validation"]["chronos2"],
            model_id="amazon/chronos-bolt-small",
            revision="772f3d25d38aec6d914c8949dab4462e2d46f5d8",
        )
        return config

    def table(self, config, name="table", length=80):
        path = Path(config["paths"]["raw"]) / name / "data.parquet"
        path.parent.mkdir(parents=True, exist_ok=True)
        pd.DataFrame(
            {
                "time_index": np.arange(length),
                "outcome": np.arange(length, dtype=float),
                "covariate_0": np.arange(length) * 2,
                "outcome_lag_1": np.arange(length),
            }
        ).to_parquet(path)
        return path

    def pipeline(self):
        def predict(inputs, prediction_length, quantile_levels):
            self.assertEqual(quantile_levels, [0.5])
            self.assertTrue(all(item.ndim == 1 for item in inputs))
            points = torch.stack(
                [item[-1].expand(prediction_length) for item in inputs]
            )
            return points[..., None], points

        return Mock(predict_quantiles=Mock(side_effect=predict))

    def test_strict_ten_percent_boundary_zero_baseline_and_undefined_scale(self):
        scores = [1.0, 1.1, np.nextafter(1.1, np.inf), 0.0, 0.01, np.nan, np.inf, -1, 1]
        baseline = [1, 1, 1, 0, 0, np.nan, 1, 1, 1]
        frame = pd.DataFrame(
            {
                "chronos_bolt_mase": scores,
                "baseline_mase": baseline,
                "mase_scale": [1, 1, 1, 1, 1, 0, 1, 1, 1],
                "baseline_mae": [1] * 9,
                "chronos_bolt_mae": [1] * 9,
                "status": ["ok"] * 8 + ["error"],
            }
        )
        self.assertEqual(
            bolt_admitted(frame).tolist(),
            [True, True, False, True, False, True, False, False, False],
        )

    def test_batches_historical_targets_only_and_reserves_final_token(self):
        with tempfile.TemporaryDirectory() as directory:
            config = self.config(directory)
            config["validation"]["chronos_bolt"].update(batch_size=2, context_length=32)
            paths = [
                self.table(config, str(i), length)
                for i, length in enumerate((80, 80, 48))
            ]
            validator = BoltValidator(config)
            validator.pipeline = self.pipeline()
            results = list(validator.evaluate_tables(paths))
            self.assertEqual(len(results), 3)
            self.assertEqual(validator.pipeline.predict_quantiles.call_count, 2)
            self.assertEqual(
                len(validator.pipeline.predict_quantiles.call_args_list[0].args[0]), 2
            )
            for _, record in results:
                self.assertTrue(record["admitted"])
                self.assertLessEqual(
                    record["forecast_stop"], record["source_rows"] - 16
                )
                self.assertLessEqual(record["context_rows"], 32)
            before = validator._prepare(paths[0])
            frame = pd.read_parquet(paths[0])
            frame.loc[len(frame) - 16 :, "outcome"] = 1e9
            frame.to_parquet(paths[0])
            after = validator._prepare(paths[0])
            torch.testing.assert_close(before[1], after[1])
            np.testing.assert_array_equal(before[2], after[2])
            self.assertEqual(before[0]["mase_scale"], after[0]["mase_scale"])

    def test_validation_report_transform_gate_and_retry_reuse(self):
        with tempfile.TemporaryDirectory() as directory:
            config = self.config(directory)
            paths = [self.table(config, str(i)) for i in range(2)]
            pipeline = self.pipeline()
            with patch.object(
                BoltValidator, "_load", lambda obj: setattr(obj, "pipeline", pipeline)
            ):
                results = validate(config)
            self.assertEqual(results.status.tolist(), ["ok", "ok"])
            self.assertEqual(validated_sources(config, paths)[0], paths)
            with patch.object(
                BoltValidator, "_load", side_effect=AssertionError("must reuse")
            ):
                again = validate(config, retry_errors=True)
            self.assertEqual(again.status.tolist(), ["ok", "ok"])
            config["validation"]["max_relative_degradation"] = 0.2
            with (
                patch("validate_generated_data.validate", wraps=validate) as run,
                patch.object(
                    BoltValidator,
                    "_load",
                    lambda obj: setattr(obj, "pipeline", pipeline),
                ),
            ):
                validated_sources(config, paths)
                run.assert_called_once()

    def test_transform_automatically_screens_missing_report_but_rejects_stale_when_disabled(
        self,
    ):
        with tempfile.TemporaryDirectory() as directory:
            config = self.config(directory)
            paths = [self.table(config)]
            with patch.object(
                BoltValidator,
                "_load",
                lambda obj: setattr(obj, "pipeline", self.pipeline()),
            ):
                admitted, audit = validated_sources(config, paths)
            self.assertEqual(admitted, paths)
            self.assertEqual(audit["engine"], "chronos_bolt")
            report = Path(config["validation"]["metrics_path"])
            scores = pd.read_parquet(report)
            scores["screening_signature"] = "old_prophet_policy"
            scores.to_parquet(report)
            config["validation"]["auto_run"] = False
            with self.assertRaisesRegex(ValueError, "stale Chronos-Bolt"):
                validated_sources(config, paths)

    def test_invalid_configuration_and_failed_tables_do_not_call_model(self):
        with tempfile.TemporaryDirectory() as directory:
            config = self.config(directory)
            for key, value in (
                ("batch_size", 0),
                ("context_length", True),
                ("cpu_threads", -1),
            ):
                bad = copy.deepcopy(config)
                bad["validation"]["chronos_bolt"][key] = value
                with self.assertRaises(ValueError):
                    BoltValidator(bad)
            path = self.table(config)
            pd.DataFrame({"time_index": [1, 0], "outcome": [1, 2]}).to_parquet(path)
            validator = BoltValidator(config)
            with patch.object(
                validator, "_load", side_effect=AssertionError("bad table")
            ):
                result = list(validator.evaluate_tables([path]))[0][1]
            self.assertEqual(result["status"], "error")
            self.assertIn("increasing", result["error"])

    def test_signature_separates_prophet_bolt_and_threshold_policies(self):
        config = load_config()
        original = screening_signature(config)
        config["validation"]["auto_run"] = False
        self.assertEqual(screening_signature(config), original)
        config["validation"]["max_relative_degradation"] = 0.2
        self.assertNotEqual(screening_signature(config), original)
        config["validation"]["engine"] = "prophet"
        self.assertNotEqual(screening_signature(config), original)


if __name__ == "__main__":
    unittest.main()

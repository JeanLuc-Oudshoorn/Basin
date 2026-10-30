"""Date cohorts flow through validation, processing, embedding and training guards."""

import copy
import json
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
for folder in [
    "src/02_data_validation",
    "src/03_transforming_input_data",
    "src/04_creating_embeddings",
    "src/05_model_training",
]:
    sys.path.insert(0, str(ROOT / folder))
from embed import embed
from prophet_validator import ProphetValidator
from train import train
from transform import (
    discover_source_tables,
    file_digest,
    generated_dates,
    load_config,
    read_artifact,
    signature,
    transform,
    validated_sources,
    validation_report_path,
)
from validate_generated_data import validate


class GeneratedDateTests(unittest.TestCase):
    def test_smoke_alias_provenance_survives_transform_but_is_not_reused_as_real_mae(
        self,
    ):
        with tempfile.TemporaryDirectory() as directory:
            config = self.fixture(directory)

            def score(path):
                return {
                    "source_sha256": file_digest(path),
                    "status": "ok",
                    "baseline_mae": 1.0,
                    "multivariate_mae": 0.5,
                    "multivariate_error": "",
                    "error": "",
                    "reserved_tail_rows": 16,
                    "covariate_mode": "forecast",
                }

            with patch.object(ProphetValidator, "evaluate_table", side_effect=score):
                results = validate(config)
            report = Path(results.attrs["report_path"])
            results["smoke_test_only"] = True
            results["actual_scoring_metric"] = "rmse"
            results.to_parquet(report, index=False)
            processed = transform(config)
            self.assertTrue(
                all(r["screening_smoke_test_only"] for r in processed["records"])
            )
            audit = json.loads(
                (
                    Path(config["paths"]["processed"]) / "validation_selection.json"
                ).read_text()
            )
            self.assertTrue(audit["smoke_test_only"])
            self.assertEqual(audit["actual_scoring_metric"], "rmse")
            with patch.object(
                ProphetValidator, "evaluate_table", side_effect=score
            ) as evaluate:
                validate(config, retry_errors=True)
                self.assertEqual(evaluate.call_count, 2)
            summary = json.loads(report.with_name("summary.json").read_text())
            self.assertEqual(summary["reused"], 0)
            self.assertEqual(summary["reevaluated"], 2)

    def test_dated_runs_preserve_history_and_transform_follows_selected_cohort(self):
        with tempfile.TemporaryDirectory() as directory:
            config = self.fixture(directory)
            config["validation"]["run_folders"] = True

            def score(path):
                return {
                    "source_sha256": file_digest(path),
                    "status": "ok",
                    "baseline_mae": 1.0,
                    "multivariate_mae": 0.5,
                    "multivariate_error": "",
                    "error": "",
                    "reserved_tail_rows": 16,
                    "covariate_mode": "forecast",
                }

            with patch.object(ProphetValidator, "evaluate_table", side_effect=score):
                first = validate(config)
                first_path = Path(first.attrs["report_path"])
                first_hash = file_digest(first_path)
                second = validate(config)
            second_path = Path(second.attrs["report_path"])
            self.assertNotEqual(first_path, second_path)
            self.assertEqual(first_path.parent.parent, second_path.parent.parent)
            self.assertEqual(validation_report_path(config), second_path)
            settings = json.loads(first_path.with_name("settings.json").read_text())
            self.assertEqual(settings["dataset_dates"], ["2026-10-23", "2026-10-25"])
            self.assertEqual(settings["admission_metric"], "mae")
            self.assertIn("started_at_utc", settings)
            other = copy.deepcopy(config)
            other["data"]["generated_dates"] = ["2026-10-24"]
            with self.assertRaises(FileNotFoundError):
                validation_report_path(other)
            with patch.object(ProphetValidator, "evaluate_table", side_effect=score):
                third = validate(other)
            third_path = Path(third.attrs["report_path"])
            self.assertNotEqual(third_path.parent.parent, second_path.parent.parent)
            self.assertEqual(validation_report_path(config), second_path)
            self.assertEqual(validation_report_path(other), third_path)
            self.assertEqual(file_digest(first_path), first_hash)
            transformed = transform(config)
            self.assertEqual(
                {r["source_key"].split("/")[0] for r in transformed["records"]},
                {"2026-10-23", "2026-10-25"},
            )
            with patch.object(ProphetValidator, "evaluate_table") as evaluate:
                retried = validate(config, retry_errors=True)
                evaluate.assert_not_called()
            retried_path = Path(retried.attrs["report_path"])
            self.assertNotEqual(retried_path, second_path)
            self.assertTrue(second_path.exists())
            self.assertEqual(validation_report_path(config), retried_path)
            summary = json.loads(retried_path.with_name("summary.json").read_text())
            self.assertEqual((summary["reused"], summary["reevaluated"]), (2, 0))
            # A failed new run must never replace the cohort's usable pointer.
            with patch.object(
                ProphetValidator,
                "evaluate_table",
                side_effect=RuntimeError("interrupted"),
            ):
                with self.assertRaisesRegex(RuntimeError, "interrupted"):
                    validate(config)
            self.assertEqual(validation_report_path(config), retried_path)
            statuses = [
                json.loads(p.read_text())["status"]
                for p in retried_path.parent.parent.glob("*/status.json")
            ]
            self.assertIn("failed", statuses)

    def test_retry_reuses_only_current_successes_and_archives_previous_report(self):
        with tempfile.TemporaryDirectory() as directory:
            config = self.fixture(directory)
            paths = discover_source_tables(config)

            def score(path):
                return {
                    "source_key": path.relative_to(
                        Path(config["paths"]["raw"])
                    ).as_posix(),
                    "source_sha256": file_digest(path),
                    "status": "ok",
                    "baseline_mae": 1.0,
                    "multivariate_mae": 0.5,
                    "multivariate_error": "",
                    "error": "",
                    "reserved_tail_rows": 16,
                    "covariate_mode": "forecast",
                }

            failed = score(paths[0])
            failed.update(
                status="error",
                multivariate_mae=np.nan,
                multivariate_error="No space left",
            )
            with patch.object(
                ProphetValidator,
                "evaluate_table",
                side_effect=[failed, score(paths[1])],
            ):
                validate(config)
            report = Path(config["validation"]["metrics_path"])
            before = file_digest(report)
            with patch.object(
                ProphetValidator, "evaluate_table", side_effect=score
            ) as evaluate:
                results = validate(config, retry_errors=True)
                self.assertEqual(evaluate.call_count, 1)
                self.assertEqual(evaluate.call_args.args[0], paths[0])
            self.assertTrue(results.status.eq("ok").all())
            summary = json.loads(report.with_name("summary.json").read_text())
            self.assertEqual(
                (summary["reused"], summary["reevaluated"], summary["failed"]),
                (1, 1, 0),
            )
            archives = list((report.parent / "archive").glob("*/metrics.parquet"))
            self.assertEqual(len(archives), 1)
            self.assertEqual(file_digest(archives[0]), before)
            frame = pd.read_parquet(paths[1])
            frame.loc[0, "outcome"] += 1
            frame.to_parquet(paths[1])
            with patch.object(
                ProphetValidator, "evaluate_table", side_effect=score
            ) as evaluate:
                validate(config, retry_errors=True)
                self.assertEqual(evaluate.call_count, 1)
                self.assertEqual(evaluate.call_args.args[0], paths[1])

    def fixture(self, directory):
        base = Path(directory)
        config = load_config()
        for key in ["raw", "processed", "embedded", "logs"]:
            config["paths"][key] = str(base / key)
        config["data"].update(
            generated_dates=["2026-10-25", "2026-10-23"],
            context_length=64,
            horizon=16,
            max_tables=None,
        )
        config["model"].update(
            width=128, layers=2, variate_heads=4, context_length=64, max_patches=16
        )
        config["validation"]["metrics_path"] = str(base / "report" / "metrics.parquet")
        config["validation"]["workers"] = 1  # Evaluate mocks in the parent process.
        config["validation"]["run_folders"] = False  # Explicit-path legacy fixtures.
        for i, day in enumerate(["2026-10-23", "2026-10-24", "2026-10-25"]):
            path = base / "raw" / day / "shared_table_name" / "data.parquet"
            path.parent.mkdir(parents=True)
            pd.DataFrame(
                {
                    "time_index": np.arange(64),
                    "outcome": np.arange(64, dtype=float) + 100 * i,
                    "covariate_00": np.arange(64, dtype=float),
                    "outcome_lag_1": np.arange(64) - 1,
                }
            ).to_parquet(path, index=False)
        return config

    def test_real_parallel_screening_report_and_summary_match_selection(self):
        with tempfile.TemporaryDirectory() as directory:
            config = self.fixture(directory)
            config["validation"]["workers"] = 2
            paths = discover_source_tables(config)
            # Keep a broken table in the denominator and the progress journal.
            pd.DataFrame({"time_index": [0, 1]}).to_parquet(paths[0])
            results = validate(config)
            self.assertEqual(results.status.tolist(), ["error", "ok"])
            self.assertNotIn("univariate_rmse", results)
            report = Path(config["validation"]["metrics_path"])
            pd.testing.assert_frame_equal(results, pd.read_parquet(report))
            progress = pd.read_csv(report.with_name("metrics_progress.csv"))
            self.assertEqual(set(progress.source_key), set(results.source_key))
            self.assertEqual(len(progress), 2)
            summary = json.loads(report.with_name("summary.json").read_text())
            admitted, _ = validated_sources(config, paths)
            self.assertEqual(summary["admitted"], len(admitted))
            self.assertEqual(summary["rejected"], 1)
            self.assertEqual(summary["effective_workers"], 2)

    def test_two_dates_reach_embeddings_and_stale_selection_cannot_train(self):
        with tempfile.TemporaryDirectory() as directory:
            config = self.fixture(directory)

            def score(path):
                return {
                    "table": path.parent.name,
                    "source": str(path),
                    "source_sha256": file_digest(path),
                    "status": "ok",
                    "baseline_mae": 1.0,
                    "multivariate_mae": 1.0,
                    "multivariate_error": "",
                    "reserved_tail_rows": 16,
                    "covariate_mode": "forecast",
                }

            with patch.object(
                ProphetValidator, "evaluate_table", side_effect=score
            ) as evaluate:
                results = validate(config)
            self.assertEqual(evaluate.call_count, 2)
            self.assertEqual(
                results.source_key.tolist(),
                [
                    "2026-10-23/shared_table_name/data.parquet",
                    "2026-10-25/shared_table_name/data.parquet",
                ],
            )
            self.assertFalse(results.source_key.duplicated().any())
            transformed = transform(config)
            self.assertEqual(
                {Path(r["source"]).parents[1].name for r in transformed["records"]},
                {"2026-10-23", "2026-10-25"},
            )
            embed(config)
            artifact = read_artifact(
                Path(config["paths"]["embedded"]) / "embeddings.pt", config
            )
            training = [r for r in artifact["records"] if r["split"] == "train"]
            self.assertEqual(len(training), 2)
            self.assertEqual(
                artifact["source_selection"]["generated_dates"],
                ["2026-10-23", "2026-10-25"],
            )
            self.assertEqual(
                {Path(r["source"]).parents[1].name for r in training},
                {"2026-10-23", "2026-10-25"},
            )
            config["data"]["generated_dates"] = ["2026-10-23"]
            with self.assertRaisesRegex(ValueError, "Stale artifact"):
                train(config)  # Guard executes before any CUDA/model initialization.

    def test_explicit_selection_errors_order_and_source_root_identity(self):
        with tempfile.TemporaryDirectory() as directory:
            config = self.fixture(directory)
            paths = discover_source_tables(config)
            self.assertEqual(
                [p.parents[1].name for p in paths], ["2026-10-23", "2026-10-25"]
            )
            original = signature(config)
            config["data"]["generated_dates"].reverse()
            self.assertEqual(signature(config), original)
            config["paths"]["raw"] = str(Path(directory) / "another_root")
            self.assertNotEqual(signature(config), original)
            config = self.fixture(Path(directory) / "second")
            config["data"]["generated_dates"] = ["2026-10-22"]
            with self.assertRaisesRegex(FileNotFoundError, "Selected generated date"):
                discover_source_tables(config)
            config["data"]["generated_dates"] = ["2026-10-26"]
            (Path(config["paths"]["raw"]) / "2026-10-26").mkdir()
            with self.assertRaisesRegex(ValueError, "No data.parquet"):
                discover_source_tables(config)

    def test_invalid_lists_and_legacy_single_folder(self):
        config = load_config()
        for dates in [
            [],
            None,
            "2026-10-23",
            ["2026-10-23", "2026-10-23"],
            ["2026-02-30"],
            ["../2026-10-23"],
            ["20261023"],
            [23],
        ]:
            with self.subTest(dates=dates):
                candidate = copy.deepcopy(config)
                candidate["data"]["generated_dates"] = dates
                with self.assertRaises(ValueError):
                    generated_dates(candidate)
        with tempfile.TemporaryDirectory() as directory:
            config = self.fixture(directory)
            config["data"].pop("generated_dates")
            config["paths"]["raw"] = str(Path(config["paths"]["raw"]) / "2026-10-23")
            self.assertEqual(len(discover_source_tables(config)), 1)
            content = {k: config[k] for k in ["seed", "data", "model", "validation"]}
            import hashlib

            expected = hashlib.sha256(
                json.dumps(content, sort_keys=True).encode()
            ).hexdigest()
            self.assertEqual(signature(config), expected)

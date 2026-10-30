"""Admission policy and persisted transform integration, without expensive fits."""

import json
import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(
    0, str(Path(__file__).resolve().parents[1] / "src/03_transforming_input_data")
)
from transform import (
    file_digest,
    load_config,
    screening_signature,
    transform,
    validated_sources,
)


class LearnabilityGateTests(unittest.TestCase):
    def fixture(self, root):
        config = load_config()
        config["data"].pop("generated_dates", None)  # Legacy flat-table fixture.
        base = Path(root)
        config["paths"]["raw"] = str(base / "raw")
        config["paths"]["processed"] = str(base / "processed")
        config["validation"]["metrics_path"] = str(base / "metrics.parquet")
        config["validation"]["run_folders"] = False
        paths, rows = [], []
        cases = [
            ("uni", 1.0, 3.0, "ok"),
            ("multi", 3.0, 1.0, "ok"),
            ("both", 1.0, 1.5, "ok"),
            ("tie", 2.0, 2.0, "ok"),
            ("nan", np.nan, np.nan, "error"),
            ("missing", 1.0, 1.0, "ok"),
            ("stale", 1.0, 1.0, "ok"),
            ("wrong_policy", 1.0, 1.0, "ok"),
            ("failed_fit", 1.0, 3.0, "partial_error"),
            ("perfect_naive", 0.0, 0.0, "ok"),
            ("multi_error", 1.0, 1.0, "ok"),
            ("infinite", 1.0, np.inf, "ok"),
            ("negative", 1.0, -1.0, "ok"),
        ]
        for name, uni, multi, status in cases:
            path = base / "raw" / name / "data.parquet"
            path.parent.mkdir(parents=True)
            pd.DataFrame(
                {
                    "time_index": np.arange(64),
                    "outcome": np.arange(64, dtype=float),
                    "covariate_00": np.arange(64),
                    "outcome_lag_1": np.arange(64),
                }
            ).to_parquet(path)
            paths.append(path)
            if name == "missing":
                continue
            rows.append(
                {
                    "source_key": f"{name}/data.parquet",
                    "source_sha256": "obsolete"
                    if name == "stale"
                    else file_digest(path),
                    "status": status,
                    "baseline_mae": 0.0 if name == "perfect_naive" else 2.0,
                    "univariate_rmse": uni,
                    "multivariate_mae": multi,
                    "baseline_rmse": 1.0,
                    "multivariate_rmse": 99.0,  # MAE alone determines admission.
                    "univariate_error": "fit failed"
                    if name in ("multi", "failed_fit")
                    else "",
                    "multivariate_error": "fit failed" if name == "multi_error" else "",
                    "reserved_tail_rows": 16,
                    "covariate_mode": config["validation"]["covariate_mode"],
                    "screening_signature": "obsolete"
                    if name == "wrong_policy"
                    else screening_signature(config),
                }
            )
        pd.DataFrame(rows).to_parquet(config["validation"]["metrics_path"])
        return config, paths

    def test_multivariate_gate_includes_ties_and_ignores_univariate_wins(self):
        with tempfile.TemporaryDirectory() as root:
            config, paths = self.fixture(root)
            admitted, audit = validated_sources(config, paths)
            self.assertEqual(
                [p.parent.name for p in admitted],
                ["multi", "both", "tie", "perfect_naive"],
            )
            self.assertEqual(len(audit["excluded"]), 9)
            artifact = transform(config)
            self.assertEqual(
                {Path(r["source"]).parent.name for r in artifact["records"]},
                {"multi", "both", "tie", "perfect_naive"},
            )
            manifest = json.loads(
                (Path(config["paths"]["processed"]) / "manifest.json").read_text()
            )
            self.assertEqual(manifest["validation"]["admitted_count"], 4)
            self.assertEqual(len(manifest["excluded_tables"]), 9)
            for record in artifact["records"]:
                self.assertNotIn("outcome_lag_1", record["variable_names"])
                self.assertEqual(record["evaluation_cutoff"], 48)

    def test_multivariate_only_report_and_worker_count_do_not_change_selection(self):
        with tempfile.TemporaryDirectory() as root:
            config, paths = self.fixture(root)
            report = Path(config["validation"]["metrics_path"])
            scores = pd.read_parquet(report).drop(
                columns=["univariate_rmse", "univariate_error"]
            )
            scores.to_parquet(report)
            signature = screening_signature(config)
            config["validation"]["workers"] = 8
            self.assertEqual(screening_signature(config), signature)
            admitted, _ = validated_sources(config, paths)
            self.assertEqual(
                [p.parent.name for p in admitted],
                ["multi", "both", "tie", "perfect_naive"],
            )

    def test_missing_duplicate_all_rejected_and_policy_change(self):
        with tempfile.TemporaryDirectory() as root:
            config, paths = self.fixture(root)
            report = Path(config["validation"]["metrics_path"])
            rows = pd.read_parquet(report)
            pd.concat([rows, rows.iloc[:1]]).to_parquet(report)
            with self.assertRaisesRegex(ValueError, "Duplicate"):
                validated_sources(config, paths)
            rows.to_parquet(report)
            config["seed"] += 1
            with self.assertRaisesRegex(ValueError, "No datasets passed"):
                validated_sources(config, paths)
            report.unlink()
            with self.assertRaisesRegex(
                FileNotFoundError, "Run src/02_data_validation"
            ):
                validated_sources(config, paths)

    def test_rmse_only_scores_cannot_be_used_as_mae_scores(self):
        with tempfile.TemporaryDirectory() as root:
            config, paths = self.fixture(root)
            report = Path(config["validation"]["metrics_path"])
            scores = pd.read_parquet(report).drop(
                columns=["baseline_mae", "multivariate_mae"]
            )
            scores.to_parquet(report)
            with self.assertRaisesRegex(ValueError, "Validation report lacks columns"):
                validated_sources(config, paths)


if __name__ == "__main__":
    unittest.main()

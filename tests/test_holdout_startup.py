"""Holdout startup must not depend on training tensors or today's gate policy."""

import json
import os
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import Mock, patch

import pandas as pd
import torch

ROOT = Path(__file__).resolve().parents[1]
for stage in ("03_transforming_input_data", "05_model_training", "06_holdout_test"):
    sys.path.insert(0, str(ROOT / "src" / stage))
from holdout_test import (
    checkpoint_test_records,
    holdout_checkpoint,
    load_holdout_metadata,
    populate_test_schema,
    run,
)
from transform import file_digest, load_config, signature


class HoldoutStartupTests(unittest.TestCase):
    def test_new_training_snapshot_wins_over_stale_export_and_can_be_pinned(self):
        with tempfile.TemporaryDirectory() as directory:
            base = Path(directory)
            config = load_config()
            config["paths"]["training"] = str(base)
            exported, recovery = base / "model.pt", base / "latest_checkpoint.pt"
            self.assertEqual(holdout_checkpoint(config), exported)
            recovery.touch()
            self.assertEqual(holdout_checkpoint(config), recovery)
            exported.touch()
            os.utime(exported, (100, 100))
            os.utime(recovery, (200, 200))
            self.assertEqual(holdout_checkpoint(config), recovery)
            config["holdout"]["checkpoint"] = str(exported)
            self.assertEqual(holdout_checkpoint(config), exported)
            del config["holdout"]["checkpoint"]
            os.utime(exported, (300, 300))
            self.assertEqual(holdout_checkpoint(config), exported)

    def test_recovery_uses_best_weights_and_its_saved_split(self):
        with tempfile.TemporaryDirectory() as directory:
            base = Path(directory)
            config = load_config()
            source = "data/01_generated/2026-11-06/table/data.parquet"
            split = {
                "seed": config["seed"],
                "requested_test_fraction": config["training"]["holdout_fraction"],
                "input_signature": signature(config),
                "train_sources": ["train"],
                "test_sources": [source],
                "source_sha256": {"train": "unused", source: "digest"},
            }
            split_path = base / "dataset_split.json"
            split_path.write_text(json.dumps(split), encoding="utf-8")
            checkpoint = base / "latest_checkpoint.pt"
            torch.save(
                {
                    "identity": {
                        "input_signature": signature(config),
                        "model": config["model"],
                    },
                    "state_dict": {"weight": torch.tensor(9)},
                    "best_state_dict": {"weight": torch.tensor(3)},
                    "best_step": 42,
                    "early_stopping": {"best_epoch": 1},
                },
                checkpoint,
            )
            metadata = load_holdout_metadata(checkpoint, config)
            self.assertEqual(metadata["state_dict"]["weight"].item(), 3)
            self.assertEqual(metadata["training_summary"]["exported_step"], 42)
            self.assertEqual(checkpoint_test_records(metadata)[0]["source"], source)
            self.assertEqual(
                metadata["dataset_split"]["source_sha256"], {source: "digest"}
            )
            config["seed"] += 1
            with self.assertRaisesRegex(ValueError, "does not match"):
                load_holdout_metadata(checkpoint, config)
            config["seed"] -= 1
            split["train_sources"] = [source]
            split_path.write_text(json.dumps(split), encoding="utf-8")
            os.utime(split_path, (100, 100))
            with self.assertRaisesRegex(ValueError, "overlap"):
                load_holdout_metadata(checkpoint, config)
            split_path.unlink()
            with self.assertRaisesRegex(FileNotFoundError, "saved split"):
                load_holdout_metadata(checkpoint, config)

    def test_missing_source_explains_checkpoint_owned_dates(self):
        with tempfile.TemporaryDirectory() as directory:
            with self.assertRaisesRegex(FileNotFoundError, "saved checkpoint split"):
                populate_test_schema(
                    [{"source": str(Path(directory) / "missing.parquet")}],
                    {"pipeline_config": load_config()},
                )

    def test_checkpoint_loader_restores_all_weights_and_identical_predictions(
        self,
    ):
        from types import SimpleNamespace

        from test_training_performance import CausalBlock, model_and_records
        from train import RWKVForecaster

        model, records = model_and_records()
        model.eval()
        batch = records[0]["batch"]
        expected = model(**batch).point
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "model.pt"
            model.save(path)
            with patch(
                "train.load_rwkv", return_value=SimpleNamespace(Block=CausalBlock)
            ):
                restored = RWKVForecaster.load(
                    path, device="cpu", dtype=torch.float32
                ).eval()
                torch.testing.assert_close(
                    restored(**batch).point, expected, rtol=0, atol=0
                )
                self.assertTrue(all(not p.is_meta for p in restored.parameters()))
                broken = torch.load(path, weights_only=True)
                broken["state_dict"].pop("head.skip.weight")
                with self.assertRaisesRegex(RuntimeError, "Missing key"):
                    RWKVForecaster.load(
                        path, checkpoint=broken, device="cpu", dtype=torch.float32
                    )

    def test_saved_split_validates_hashes_uniqueness_and_disjointness(self):
        metadata = {
            "dataset_split": {
                "test_sources": ["table"],
                "train_sources": ["other"],
                "source_keys": {"table": "date/table"},
                "source_sha256": {"table": "digest"},
            }
        }
        records = checkpoint_test_records(metadata)
        self.assertEqual(records[0]["source_key"], "date/table")
        for change in (
            {"test_sources": ["table", "table"]},
            {"train_sources": ["table"]},
            {"source_sha256": {}},
        ):
            with self.assertRaises(ValueError):
                checkpoint_test_records(
                    {"dataset_split": dict(metadata["dataset_split"], **change)}
                )

    def test_real_checkpoint_schema_startup_without_training_artifacts(self):
        with tempfile.TemporaryDirectory() as directory:
            base = Path(directory)
            config = load_config()
            config["paths"].update(
                training=str(base / "training"),
                holdout_test=str(base / "output"),
                embedded=str(base / "missing_embeddings"),
                processed=str(base / "missing_windows"),
                logs=str(base / "logs"),
            )
            source = base / "data.parquet"
            pd.DataFrame(
                {
                    "time_index": range(64),
                    "outcome": range(64),
                    "past": range(64),
                    "outcome_lag_1": range(64),
                }
            ).to_parquet(source)
            metadata = {
                "pipeline_config": {
                    "data": dict(
                        config["data"], targets=["outcome"], past_covariates="auto"
                    )
                },
                "dataset_split": {
                    "test_sources": [str(source)],
                    "source_sha256": {str(source): file_digest(source)},
                },
            }
            # Today's filter/model configuration may differ from this saved export.
            config["data"]["targets"] = ["not_a_saved_target"]
            checkpoint = base / "training/model.pt"
            checkpoint.parent.mkdir()
            torch.save(metadata, checkpoint)
            model = Mock()
            model.eval.return_value = model
            with (
                patch("holdout_test.RWKVForecaster.load", return_value=model),
                patch("holdout_test.configure_variate_execution"),
                patch("holdout_test.configure_cuda_memory"),
                patch(
                    "holdout_test.evaluate_holdout",
                    return_value={"table_count": 1, "pooled": {}},
                ) as evaluate,
                patch(
                    "transform.read_artifact",
                    side_effect=AssertionError("must not open training artifacts"),
                ),
            ):
                run(config)
            record = evaluate.call_args.args[1][0]
            self.assertEqual(record["targets"], ["outcome"])
            self.assertEqual(record["roles"]["past"], ["past"])
            self.assertEqual(record["source_rows"], 64)
            self.assertFalse(Path(config["paths"]["embedded"]).exists())
            self.assertFalse(Path(config["paths"]["processed"]).exists())

    def test_schema_requires_saved_policy_columns(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "data.parquet"
            pd.DataFrame({"wrong": [1, 2]}).to_parquet(path)
            config = load_config()
            with self.assertRaisesRegex(ValueError, "columns"):
                populate_test_schema(
                    [{"source": str(path)}], {"pipeline_config": config}
                )


if __name__ == "__main__":
    unittest.main()

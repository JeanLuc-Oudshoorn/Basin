"""Independent holdout stage loads the exported model and preserves its split."""

import copy
import json
import sys
import tempfile
import unittest
from dataclasses import asdict
from pathlib import Path
from unittest.mock import Mock, patch

ROOT = Path(__file__).resolve().parents[1]
for stage in ("03_transforming_input_data", "05_model_training", "06_holdout_test"):
    sys.path.insert(0, str(ROOT / "src" / stage))
from holdout import holdout_directory
from holdout_test import run, select_test_records
from transform import load_config


class HoldoutStageTests(unittest.TestCase):
    def test_training_returns_after_export_without_calling_holdout(self):
        from test_training_performance import model_and_records
        from train import train

        model, records = model_and_records("train")
        with tempfile.TemporaryDirectory() as temporary:
            base = Path(temporary)
            config = load_config()
            config["paths"].update(
                training=str(base / "training"), logs=str(base / "logs")
            )
            config["model"] = asdict(model.config)
            config["training"]["train_embeddings"] = False
            data = {
                "format": "lazy_patch_index_v1",
                "records": records,
                "encoder_state": model.state_dict(),
            }
            summary = {"completed_updates": 1, "best_step": 1, "best_epoch": 1}
            history = [{"embedding_skip_gradient_norm": 0.0}]
            with (
                patch("train.read_artifact", return_value=data),
                patch("train.split_datasets", return_value=(records, records, [], {})),
                patch("train.prepare_token_inventory", return_value=None),
                patch("train.file_digest", return_value="digest"),
                patch("train.RWKVForecaster", return_value=model),
                patch.object(model, "to", return_value=model),
                patch("train.torch.cuda.synchronize"),
                patch("train.torch.cuda.get_device_name", return_value="CPU test"),
                patch("train.fit_epochs", return_value=(history, summary, {})),
                patch(
                    "holdout.evaluate_holdout",
                    side_effect=AssertionError("training must not test"),
                ) as evaluate,
            ):
                train(config)
            evaluate.assert_not_called()
            self.assertTrue((base / "training/model.pt").is_file())
            self.assertFalse((base / "training/test_metrics.json").exists())
            journal = next((base / "logs").glob("*.jsonl"))
            rows = [json.loads(line) for line in journal.read_text().splitlines()]
            self.assertEqual(rows[-1]["event"], "run_complete")
            self.assertIn("training_complete", [r["event"] for r in rows])

    def fixture(self):
        metadata = {
            "embedding_sha256": "digest",
            "dataset_split": {
                "test_sources": ["test"],
                "source_sha256": {"test": "digest"},
            },
            "training_summary": {"best_epoch": 7, "best_step": 85127},
        }
        data = {
            "records": [
                {"source": "train", "split": "evaluation", "source_sha256": "digest"},
                {"source": "test", "split": "train", "source_sha256": "digest"},
                {"source": "test", "split": "evaluation", "source_sha256": "digest"},
            ]
        }
        return metadata, data

    def test_saved_split_requires_exactly_one_unchanged_evaluation_record(self):
        metadata, data = self.fixture()
        selected = select_test_records(metadata, data)
        self.assertEqual(selected, [data["records"][-1]])
        with self.assertRaisesRegex(ValueError, "missing or duplicated"):
            select_test_records(metadata, {"records": data["records"][:-1]})
        with self.assertRaisesRegex(ValueError, "missing or duplicated"):
            select_test_records(metadata, {"records": data["records"] + selected})
        changed = copy.deepcopy(data)
        changed["records"][-1]["source_sha256"] = "changed"
        with self.assertRaisesRegex(ValueError, "checksum"):
            select_test_records(metadata, changed)

    def test_entrypoint_loads_latest_export_and_writes_dedicated_model_provenance(self):
        metadata, data = self.fixture()
        with tempfile.TemporaryDirectory() as temporary:
            base = Path(temporary)
            config = load_config()
            config["paths"].update(
                training=str(base / "training"),
                holdout_test=str(base / "holdout_test"),
                logs=str(base / "logs"),
            )
            checkpoint = base / "training/model.pt"
            checkpoint.parent.mkdir()
            checkpoint.touch()
            model = Mock()
            model.eval.return_value = model
            metrics = {"table_count": 1, "pooled": {"rwkv": {"rmse": 0.2}}}
            with (
                patch("holdout_test.read_artifact", side_effect=[data, metadata]),
                patch("holdout_test.file_digest", return_value="digest"),
                patch("holdout_test.RWKVForecaster.load", return_value=model) as load,
                patch("holdout_test.configure_cuda_memory"),
                patch("holdout_test.configure_variate_execution"),
                patch(
                    "holdout_test.evaluate_holdout", return_value=metrics
                ) as evaluate,
            ):
                self.assertEqual(run(config), metrics)
            load.assert_called_once_with(checkpoint)
            self.assertEqual(evaluate.call_args.args[1], [data["records"][-1]])
            provenance = json.loads((base / "holdout_test/model_used.json").read_text())
            self.assertEqual(provenance["selected_epoch"], 7)
            self.assertEqual(provenance["checkpoint_path"], str(checkpoint))
            journal = next((base / "logs").glob("rwkv7_holdout_test*.jsonl"))
            events = [
                json.loads(line)["event"] for line in journal.read_text().splitlines()
            ]
            self.assertEqual(events[-1], "run_complete")
            self.assertIn("out_of_sample_evaluation", events)

    def test_older_configuration_defaults_to_training_sibling(self):
        config = load_config()
        config["paths"].pop("holdout_test")
        config["paths"]["training"] = "results/example/training"
        self.assertEqual(
            holdout_directory(config), ROOT / "results/example/holdout_test"
        )


if __name__ == "__main__":
    unittest.main()

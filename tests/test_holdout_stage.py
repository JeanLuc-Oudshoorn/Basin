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
from holdout_runs import holdout_runs, latest_holdout_run, new_holdout_run
from holdout_test import evaluation_records, run, select_test_records
from transform import load_config


class HoldoutStageTests(unittest.TestCase):
    def test_separate_test_split_cannot_include_model_validation_tables(self):
        model = {
            "dataset_split": {
                "train_sources": ["train"],
                "validation_sources": ["validation"],
            }
        }
        split = {
            "dataset_split": {
                "train_sources": [],
                "test_sources": ["validation"],
                "source_sha256": {"validation": "digest"},
            }
        }
        with self.assertRaisesRegex(ValueError, "evaluated model validation"):
            evaluation_records(model, split)
        split["dataset_split"]["validation_sources"] = ["validation"]
        with self.assertRaisesRegex(ValueError, "validation and holdout"):
            evaluation_records(split, split)

    def test_separate_split_checks_evaluated_model_training_before_sampling(self):
        model = {"dataset_split": {"train_sources": ["old_train"]}}
        split = {
            "dataset_split": {
                "train_sources": ["current_train"],
                "test_sources": ["current_test"],
                "source_sha256": {"current_test": "digest"},
            }
        }
        self.assertEqual(evaluation_records(model, split)[0]["source"], "current_test")
        model["dataset_split"]["train_sources"] = ["current_test"]
        with self.assertRaisesRegex(ValueError, "evaluated model training"):
            evaluation_records(model, split)
        with self.assertRaisesRegex(ValueError, "record its training"):
            evaluation_records({}, split)
        split["dataset_split"]["train_sources"] = ["current_test"]
        with self.assertRaisesRegex(ValueError, "training and holdout"):
            evaluation_records(model, split)

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
            "patch_index_sha256": "digest",
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
            output = base / "holdout_test"
            output.mkdir()
            earlier_plot = output / "test_table_999_target_0.html"
            earlier_plot.write_text("previous model")
            folders = []

            def evaluate_new_model(model, records, run_config):
                # The run writes into its own new folder and is told so
                # through the config; nothing of an earlier run is removed.
                directory = Path(run_config["paths"]["holdout_test"])
                folders.append(directory)
                self.assertEqual(directory.parent, output)
                self.assertRegex(directory.name, r"^\d{8}T\d{6}Z_epoch7_step85127")
                self.assertTrue((directory / "model_used.json").is_file())
                self.assertEqual(earlier_plot.read_text(), "previous model")
                return metrics

            with (
                patch("holdout_test.torch.load", return_value=metadata) as read,
                patch("holdout_test.populate_test_schema"),
                patch("holdout_test.artifact_digest", return_value="digest"),
                patch("holdout_test.RWKVForecaster.load", return_value=model) as load,
                patch("holdout_test.configure_cuda_memory"),
                patch("holdout_test.configure_variate_execution"),
                patch(
                    "holdout_test.evaluate_holdout", side_effect=evaluate_new_model
                ) as evaluate,
            ):
                self.assertEqual(run(config), metrics)
            load.assert_called_once_with(checkpoint, checkpoint=metadata)
            read.assert_called_once_with(
                checkpoint, map_location="cpu", weights_only=True, mmap=True
            )
            self.assertEqual(
                [r["source"] for r in evaluate.call_args.args[1]], ["test"]
            )
            provenance = json.loads((folders[0] / "model_used.json").read_text())
            self.assertEqual(provenance["selected_epoch"], 7)
            self.assertEqual(provenance["checkpoint_path"], str(checkpoint))
            self.assertTrue((folders[0] / "test_metrics.json").is_file())
            journal = next((base / "logs").glob("rwkv7_holdout_test*.jsonl"))
            events = [
                json.loads(line)["event"] for line in journal.read_text().splitlines()
            ]
            self.assertEqual(events[-1], "run_complete")
            self.assertIn("out_of_sample_evaluation", events)
            # A second run, even within the same second, gets another folder
            # and leaves the first run's files exactly as they were.
            first = {
                path.name: path.read_bytes() for path in sorted(folders[0].iterdir())
            }
            with (
                patch("holdout_test.torch.load", return_value=metadata),
                patch("holdout_test.populate_test_schema"),
                patch("holdout_test.artifact_digest", return_value="digest"),
                patch("holdout_test.RWKVForecaster.load", return_value=model),
                patch("holdout_test.configure_cuda_memory"),
                patch("holdout_test.configure_variate_execution"),
                patch("holdout_test.evaluate_holdout", side_effect=evaluate_new_model),
            ):
                run(config)
            self.assertNotEqual(folders[0], folders[1])
            self.assertEqual(
                first,
                {path.name: path.read_bytes() for path in sorted(folders[0].iterdir())},
            )
            self.assertEqual(holdout_runs(config), sorted(folders))
            self.assertEqual(latest_holdout_run(config), max(folders))
            self.assertEqual(latest_holdout_run(config, folders[0].name), folders[0])

    def test_results_saved_before_run_folders_count_as_the_oldest_run(self):
        with tempfile.TemporaryDirectory() as temporary:
            base = Path(temporary)
            config = load_config()
            config["paths"]["holdout_test"] = str(base / "holdout_test")
            self.assertEqual(holdout_runs(config), [])
            with self.assertRaisesRegex(FileNotFoundError, "No completed"):
                latest_holdout_run(config)
            root = base / "holdout_test"
            root.mkdir()
            (root / "test_metrics.json").write_text("{}")
            self.assertEqual(latest_holdout_run(config), root)
            unfinished, _ = new_holdout_run(config, epoch=2, step=10)
            self.assertEqual(latest_holdout_run(config), root)
            (unfinished / "test_metrics.json").write_text("{}")
            self.assertEqual(holdout_runs(config), [root, unfinished])
            self.assertEqual(latest_holdout_run(config), unfinished)
            with self.assertRaisesRegex(FileNotFoundError, "No completed"):
                latest_holdout_run(config, "missing")

    def test_older_configuration_defaults_to_training_sibling(self):
        config = load_config()
        config["paths"].pop("holdout_test")
        config["paths"]["training"] = "results/example/training"
        self.assertEqual(
            holdout_directory(config), ROOT / "results/example/holdout_test"
        )

    def test_entrypoint_uses_external_split_but_loads_archived_weights(self):
        split, _ = self.fixture()
        archived = {
            "dataset_split": {"train_sources": ["old_train"]},
            "training_summary": {"best_epoch": 7},
        }
        with tempfile.TemporaryDirectory() as temporary:
            base = Path(temporary)
            config = load_config()
            config["paths"].update(
                training=str(base / "training"),
                holdout_test=str(base / "holdout"),
                logs=str(base / "logs"),
            )
            config["holdout"]["split_checkpoint"] = str(base / "current.pt")
            checkpoint = base / "training/model.pt"
            checkpoint.parent.mkdir()
            checkpoint.touch()
            model = Mock()
            model.eval.return_value = model
            with (
                patch("holdout_test.torch.load", side_effect=[archived, split]),
                patch("holdout_test.populate_test_schema") as schema,
                patch(
                    "holdout_test.artifact_digest", side_effect=["old_hash", "new_hash"]
                ),
                patch("holdout_test.RWKVForecaster.load", return_value=model) as load,
                patch("holdout_test.configure_cuda_memory"),
                patch("holdout_test.configure_variate_execution"),
                patch("holdout_test.configure_fast_inference") as fast,
                patch(
                    "holdout_test.evaluate_holdout",
                    return_value={"table_count": 1, "pooled": {}},
                ) as evaluate,
            ):
                run(config)
            load.assert_called_once_with(checkpoint, checkpoint=archived)
            fast.assert_called_once_with(model, config["training"])
            self.assertIs(schema.call_args.args[1], archived)
            self.assertEqual(evaluate.call_args.args[1][0]["source"], "test")
            (used,) = (base / "holdout").glob("*/model_used.json")
            provenance = json.loads(used.read_text())
            self.assertEqual(provenance["checkpoint_sha256"], "old_hash")
            self.assertEqual(provenance["split_checkpoint_sha256"], "new_hash")

    def test_split_of_another_run_is_checked_against_that_runs_configuration(self):
        # An annealing experiment's model, tested on the main run's holdout.
        split, _ = self.fixture()
        archived = {
            "dataset_split": {"train_sources": ["annealing_train"]},
            "training_summary": {"best_epoch": 1},
        }
        with tempfile.TemporaryDirectory() as temporary:
            base = Path(temporary)
            config = load_config()
            config["paths"].update(
                training=str(base / "training"),
                holdout_test=str(base / "holdout"),
                logs=str(base / "logs"),
            )
            main = copy.deepcopy(config)
            main["seed"] = config["seed"] + 1
            (base / "main.json").write_text(json.dumps(main), encoding="utf-8")
            config["holdout"]["split_checkpoint"] = str(base / "main.pt")
            config["holdout"]["split_config"] = str(base / "main.json")
            checkpoint = base / "training/model.pt"
            checkpoint.parent.mkdir()
            checkpoint.touch()
            model = Mock()
            model.eval.return_value = model
            import holdout_test

            seen = []
            real = holdout_test.load_holdout_metadata

            def load(path, used):
                seen.append((Path(path).name, used["seed"]))
                return real(path, used)

            with (
                patch("holdout_test.torch.load", side_effect=[archived, split]),
                patch("holdout_test.load_holdout_metadata", side_effect=load),
                patch("holdout_test.populate_test_schema"),
                patch("holdout_test.artifact_digest", return_value="hash"),
                patch("holdout_test.RWKVForecaster.load", return_value=model),
                patch("holdout_test.configure_cuda_memory"),
                patch("holdout_test.configure_variate_execution"),
                patch("holdout_test.configure_fast_inference"),
                patch(
                    "holdout_test.evaluate_holdout",
                    return_value={"table_count": 1, "pooled": {}},
                ) as evaluate,
            ):
                run(config)
            self.assertEqual(
                seen, [("model.pt", config["seed"]), ("main.pt", config["seed"] + 1)]
            )
            self.assertEqual(evaluate.call_args.args[1][0]["source"], "test")
            # Results go to the folder of the configuration that was run.
            self.assertEqual(len(list((base / "holdout").rglob("model_used.json"))), 1)


if __name__ == "__main__":
    unittest.main()

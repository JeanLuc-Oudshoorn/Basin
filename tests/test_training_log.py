"""Automatic, flushed, strict JSONL journals across direct calls and workers."""

import json
import subprocess  # nosec B404 - Test worker logging in a child Python process.
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src/03_transforming_input_data"))
sys.path.insert(0, str(ROOT / "src/05_model_training"))
from training_log import (
    ENV_KEY,
    emit_training_event,
    logged_training,
    training_run,
    verify_worker_completed,
    worker_environment,
)
from transform import load_config


def records(path):
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]


class JournalTests(unittest.TestCase):
    def test_epoch_metrics_and_checkpoint_events_sync_while_run_is_open(self):
        with tempfile.TemporaryDirectory() as directory:
            with training_run({"paths": {"logs": directory}}, "train") as journal:
                with patch("training_log.os.fsync") as sync:
                    emit_training_event("metrics", global_step=10, train_loss=0.3)
                    sync.assert_not_called()
                    emit_training_event(
                        "epoch_end",
                        global_step=10,
                        epoch=1,
                        validation_loss=0.2,
                        validation_rmse=0.5,
                    )
                    sync.assert_called_once_with(journal.stream.fileno())
                    last = records(journal.path)[-1]
                    self.assertEqual(last["epoch"], 1)
                    self.assertEqual(last["validation_loss"], 0.2)
                    emit_training_event(
                        "epoch_checkpoint_saved",
                        epoch=1,
                        checkpoint_path="latest_checkpoint.pt",
                    )
                    self.assertEqual(sync.call_count, 2)
                    self.assertEqual(
                        records(journal.path)[-1]["event"], "epoch_checkpoint_saved"
                    )

    def test_missing_worker_journal_is_a_failure(self):
        with tempfile.TemporaryDirectory() as directory:
            config = {"paths": {"logs": directory}}
            with self.assertRaisesRegex(RuntimeError, "shared training JSONL"):
                with training_run(config, "train"):
                    worker_environment()
                    verify_worker_completed()
            rows = records(next(Path(directory).glob("*.jsonl")))
            self.assertEqual(rows[-1]["event"], "run_error")

    def test_metrics_best_values_flush_and_unique_names(self):
        with tempfile.TemporaryDirectory() as directory:
            config = {"paths": {"logs": directory}}
            with training_run(config, "train") as journal:
                emit_training_event(
                    "metrics",
                    global_step=1,
                    train_loss=0.5,
                    validation_loss=0.4,
                    diagnostic=float("nan"),
                )
                self.assertEqual(records(journal.path)[-1]["diagnostic"], None)
                emit_training_event(
                    "metrics", global_step=2, train_loss=0.3, validation_loss=0.2
                )
                emit_training_event(
                    "metrics", global_step=3, train_loss=0.2, validation_loss=0.3
                )
                emit_training_event("training_complete", checkpoint_path="example.pt")
            rows = records(journal.path)
            self.assertEqual(rows[0]["event"], "run_start")
            self.assertEqual(rows[-1]["event"], "run_complete")
            self.assertEqual(rows[-1]["best_validation_loss"], 0.2)
            self.assertEqual(rows[-1]["best_step"], 2)
            self.assertEqual(rows[-1]["final_validation_loss"], 0.3)
            self.assertEqual(rows[-1]["completed_run_step"], 3)
            self.assertIsNone(rows[1]["perplexity"])
            self.assertIsNone(rows[1]["training_energy_joules"])
            self.assertTrue(all(r["run_id"] == journal.run_id for r in rows))
            self.assertTrue(all(r["timestamp_utc"].endswith("+00:00") for r in rows))
            self.assertNotIn("NaN", journal.path.read_text())
            with training_run(config, "train") as second:
                pass
            self.assertNotEqual(journal.path, second.path)
            self.assertRegex(
                journal.path.name,
                r"rwkv7_train_\d{8}_\d{6}_\d{6}Z_pid\d+_[0-9a-f]+\.jsonl",
            )

    def test_failure_and_nested_decorator_do_not_duplicate_run(self):
        with tempfile.TemporaryDirectory() as directory:
            config = {"paths": {"logs": directory}}

            @logged_training("train")
            def fail(config):
                emit_training_event("metrics", global_step=2)
                raise RuntimeError("intentional failure")

            with self.assertRaisesRegex(RuntimeError, "intentional failure"):
                with training_run(config, "train"):
                    fail(config)
            files = list(Path(directory).glob("*.jsonl"))
            self.assertEqual(len(files), 1)
            rows = records(files[0])
            self.assertEqual(
                [r["event"] for r in rows], ["run_start", "metrics", "run_error"]
            )
            self.assertEqual(rows[-1]["error"], "intentional failure")
            self.assertEqual(rows[-1]["global_step"], 2)

    def test_final_test_summary_does_not_change_validation_selection(self):
        with tempfile.TemporaryDirectory() as directory:
            with training_run({"paths": {"logs": directory}}, "train") as journal:
                emit_training_event("metrics", global_step=1, validation_loss=0.4)
                emit_training_event("training_complete", completed_run_step=1)
                emit_training_event(
                    "out_of_sample_evaluation",
                    out_of_sample_metrics={
                        "table_count": 2,
                        "pooled": {"rwkv": {"rmse": 0.1}},
                    },
                )
            final = records(journal.path)[-1]
            self.assertEqual(final["best_validation_loss"], 0.4)
            self.assertEqual(final["final_validation_loss"], 0.4)
            self.assertEqual(final["out_of_sample_metrics"]["table_count"], 2)

    def test_worker_appends_to_launcher_file_with_one_terminal_record(self):
        with tempfile.TemporaryDirectory() as directory:
            config = {"paths": {"logs": directory}}
            child_code = """
import json,os,sys
sys.path.insert(0,'src/05_model_training')
from training_log import training_run,emit_training_event
with training_run(json.loads(os.environ['BASIN_TEST_CONFIG']),'train'):
    emit_training_event('metrics',global_step=4,train_loss=.25,validation_loss=.2)
    emit_training_event('training_complete',checkpoint_path='worker.pt')
"""
            with training_run(config, "train") as journal:
                env = worker_environment()
                env["BASIN_TEST_CONFIG"] = json.dumps(config)
                self.assertIn(ENV_KEY, env)
                self.assertIn(ENV_KEY, env["WSLENV"].split(":"))
                # Fixed test code via this interpreter; no shell or untrusted command.
                result = subprocess.run(  # nosec B603
                    [sys.executable, "-c", child_code],
                    env=env,
                    cwd=ROOT,
                    capture_output=True,
                    text=True,
                )
                self.assertEqual(result.returncode, 0, result.stderr)
            self.assertEqual(len(list(Path(directory).glob("*.jsonl"))), 1)
            rows = records(journal.path)
            self.assertEqual(
                [r["event"] for r in rows],
                [
                    "run_start",
                    "worker_launch",
                    "worker_start",
                    "metrics",
                    "training_complete",
                    "run_complete",
                ],
            )
            self.assertEqual(rows[-1]["completed_run_step"], 4)
            self.assertEqual(rows[-1]["checkpoint_path"], "worker.pt")

    def test_actual_python_entrypoints_create_journal_before_data_loading(self):
        from capacity_test import run
        from train import train

        with tempfile.TemporaryDirectory() as directory:
            config = load_config()
            config["paths"]["logs"] = directory
            config["paths"]["embedded"] = str(Path(directory) / "missing")
            for entry in [train, run]:
                with self.assertRaises(FileNotFoundError):
                    entry(config)
            files = list(Path(directory).glob("*.jsonl"))
            self.assertEqual(len(files), 2)
            for path in files:
                rows = records(path)
                self.assertEqual(rows[0]["event"], "run_start")
                self.assertEqual(rows[-1]["event"], "run_error")
                self.assertEqual(rows[-1]["error_type"], "FileNotFoundError")

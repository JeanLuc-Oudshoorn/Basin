"""Progress accounts for accumulation and tails live journals without restarting."""

import json
import sys
import tempfile
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src/05_model_training"))
from training_progress import duration, epoch_progress, journal_progress


class TrainingProgressTests(unittest.TestCase):
    def test_epoch_fraction_rate_eta_and_complete_pass(self):
        value = epoch_progress(25, 100, 200, 800, 50.0)
        self.assertEqual(value["epoch_progress_percent"], 25.0)
        self.assertEqual(value["epoch_updates_per_second"], 0.5)
        self.assertEqual(value["epoch_eta_seconds"], 150.0)
        self.assertEqual(
            epoch_progress(100, 100, 800, 800, 200.0)["epoch_eta_seconds"], 0.0
        )
        self.assertEqual(duration(3661), "01:01:01")
        self.assertEqual(duration(None), "unknown")

    def test_legacy_active_journal_epoch_offset_and_incomplete_lines(self):
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "training.jsonl"
            first = {
                "event": "metrics",
                "epoch": 2,
                "global_step": 105,
                "elapsed_seconds": 300.0,
            }
            last = {
                "event": "metrics",
                "epoch": 2,
                "global_step": 110,
                "elapsed_seconds": 310.0,
            }
            path.write_text(
                "truncated prefix\n"
                + json.dumps(first)
                + "\n"
                + json.dumps(last)
                + '\n{"partial":'
            )
            progress = journal_progress(path, {"updates_per_epoch": 100})
            self.assertEqual(progress["epoch_update"], 10)
            self.assertEqual(progress["epoch_progress_percent"], 10.0)
            self.assertEqual(progress["epoch_eta_seconds"], 180.0)
            self.assertIn("10/100", progress["message"])
            self.assertIn("training ETA", progress["message"])

    def test_new_progress_fields_and_terminal_status(self):
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "training.jsonl"
            event = {
                "event": "metrics",
                "epoch": 1,
                "global_step": 5,
                "epoch_update": 5,
                "epoch_total_updates": 5,
                "epoch_updates_per_second": 2.0,
            }
            path.write_text(
                json.dumps(event) + "\n" + json.dumps({"event": "run_complete"}) + "\n"
            )
            progress = journal_progress(path, {"updates_per_epoch": 5})
            self.assertIsNone(progress["epoch_eta_seconds"])
            self.assertEqual(progress["latest_event"], "run_complete")
            self.assertIn("Run complete", progress["message"])

    def test_startup_has_no_fabricated_eta(self):
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "training.jsonl"
            path.write_text(json.dumps({"event": "model_initialized"}) + "\n")
            progress = journal_progress(path, {"updates_per_epoch": 5})
            self.assertNotIn("epoch_eta_seconds", progress)
            self.assertIn("first optimizer update", progress["message"])

    def test_validation_phase_has_its_own_eta(self):
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "training.jsonl"
            path.write_text(
                json.dumps(
                    {
                        "event": "validation_progress",
                        "validation_tables_completed": 1000,
                        "validation_total_tables": 2000,
                        "validation_eta_seconds": 60.0,
                    }
                )
                + "\n"
            )
            progress = journal_progress(path, {"updates_per_epoch": 5})
            self.assertIn("Full validation", progress["message"])
            self.assertIn("1,000/2,000", progress["message"])
            self.assertIn("00:01:00", progress["message"])

    def test_interrupted_run_is_not_reported_as_still_training(self):
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "training.jsonl"
            event = {
                "event": "metrics",
                "epoch": 1,
                "global_step": 10,
                "epoch_updates_per_second": 2.0,
            }
            path.write_text(
                json.dumps(event)
                + "\n"
                + json.dumps({"event": "run_error", "error_type": "KeyboardInterrupt"})
                + "\n"
            )
            progress = journal_progress(path, {"updates_per_epoch": 100})
            self.assertIsNone(progress["epoch_eta_seconds"])
            self.assertIn("Run stopped: KeyboardInterrupt", progress["message"])
            self.assertNotIn("ETA", progress["message"])


if __name__ == "__main__":
    unittest.main()

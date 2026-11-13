"""New holdout runs clear generated root files while preserving user notes."""

import sys
import tempfile
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src/06_holdout_test"))
from output_cleanup import clear_holdout_outputs


class HoldoutOutputCleanupTests(unittest.TestCase):
    def test_stale_metrics_plots_and_unselected_table_outputs_are_removed(self):
        with tempfile.TemporaryDirectory() as temporary:
            directory = Path(temporary)
            generated = [
                "test_metrics.json",
                "test_metrics_progress.json",
                "test_table_999.parquet",
                "test_table_999_target_0.html",
                "test_table_000_target_2.png",
                "error_distributions.html",
                "latency_distributions.html",
                "model_used.json",
                "plot_selection.json",
                "table_metrics.json",
                "origin_metrics.json",
                "request_latencies.jsonl",
                "sample_selection.json",
            ]
            for name in generated:
                (directory / name).write_text("old")
            (directory / "previous_holdout.txt").write_text("user reference")
            (directory / "notes.md").write_text("notes")
            nested = directory / "nested"
            nested.mkdir()
            (nested / "test_metrics.json").write_text("independent subdirectory")
            self.assertEqual(clear_holdout_outputs(directory), len(generated))
            self.assertTrue(all(not (directory / name).exists() for name in generated))
            self.assertTrue((directory / "previous_holdout.txt").is_file())
            self.assertTrue((directory / "notes.md").is_file())
            self.assertTrue((nested / "test_metrics.json").is_file())
            self.assertEqual(clear_holdout_outputs(directory), 0)

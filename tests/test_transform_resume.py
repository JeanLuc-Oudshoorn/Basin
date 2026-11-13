"""Interruption, source identity, checkpoint integrity, and output compatibility."""

import copy
import json
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import pandas as pd
import torch
from loguru import logger

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src/03_transforming_input_data"))
from transform import load_config, read_artifact, transform_next_patch
from transform_progress import TransformCheckpoint, TransformProgress, atomic_write


class TransformResumeTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.base = Path(self.temp.name)
        self.config = copy.deepcopy(load_config())
        self.config["transform"] = {
            "checkpoint_every_tables": 100,
            "checkpoint_interval_seconds": 60,
        }
        self.config["data"].pop("generated_dates", None)
        self.config["data"].update(
            horizon=16,
            context_length=64,
            validation_policy="reserved_tail",
            max_tables=None,
        )
        self.config["model"].update(context_length=64, max_patches=16)
        self.config["validation"]["enabled"] = False
        self.config["paths"].update(
            raw=str(self.base / "raw"),
            processed=str(self.base / "processed"),
            embedded=str(self.base / "embedded"),
        )
        logger.disable("transform")
        self.addCleanup(logger.enable, "transform")
        self.frame = pd.DataFrame({"time_index": range(80), "outcome": range(80)})

    def sources(self, count):
        paths = []
        for index in range(count):
            path = self.base / "raw" / f"{index:04d}" / "data.parquet"
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_bytes(b"source fixture")
            paths.append(path)
        return paths

    def states(self):
        return list(
            (self.base / "processed/.transform_checkpoints").glob("*/progress.json")
        )

    def test_killed_run_resumes_without_duplicates_and_matches_clean_run(self):
        paths = self.sources(205)
        for policy in ("reserved_tail", "sampled_training_origins"):
            with self.subTest(policy=policy):
                self.config["data"]["validation_policy"] = policy
                read_count = 0

                def interrupt(path):
                    nonlocal read_count
                    read_count += 1
                    if path == paths[137]:
                        raise KeyboardInterrupt()
                    return self.frame

                with (
                    patch("transform.discover_source_tables", return_value=paths),
                    patch("transform.pd.read_parquet", side_effect=interrupt),
                    self.assertRaises(KeyboardInterrupt),
                ):
                    transform_next_patch(self.config)
                self.assertEqual(read_count, 138)
                state = [json.loads(p.read_text()) for p in self.states()]
                self.assertTrue(any(s["completed_tables"] == 100 for s in state))
                with (
                    patch("transform.discover_source_tables", return_value=paths),
                    patch(
                        "transform.pd.read_parquet", return_value=self.frame
                    ) as reader,
                ):
                    resumed = transform_next_patch(self.config)
                self.assertEqual(reader.call_count, 105)
                self.assertEqual(reader.call_args_list[0].args[0], paths[100])
                ids = [r["id"] for r in resumed["records"]]
                self.assertEqual(len(ids), len(set(ids)))
                clean_config = copy.deepcopy(self.config)
                clean_config["paths"]["processed"] = str(self.base / f"clean_{policy}")
                with (
                    patch("transform.discover_source_tables", return_value=paths),
                    patch("transform.pd.read_parquet", return_value=self.frame),
                ):
                    clean = transform_next_patch(clean_config)
                self.assertEqual(ids, [r["id"] for r in clean["records"]])
                for actual, expected in zip(resumed["records"], clean["records"]):
                    self.assertEqual(actual.keys(), expected.keys())
                    for key in actual:
                        if key == "batch":
                            for name, tensor in actual[key].items():
                                if isinstance(tensor, torch.Tensor):
                                    torch.testing.assert_close(
                                        tensor, expected[key][name]
                                    )
                                else:
                                    self.assertEqual(tensor, expected[key][name])
                        else:
                            self.assertEqual(actual[key], expected[key])
                output = read_artifact(self.base / "processed/windows.pt", self.config)
                self.assertEqual(len(output["records"]), len(resumed["records"]))

    def test_final_write_failure_retries_without_transforming_any_table(self):
        paths = self.sources(3)
        with (
            patch("transform.discover_source_tables", return_value=paths),
            patch("transform.pd.read_parquet", return_value=self.frame),
            patch("transform.atomic_write", side_effect=OSError("disk full")),
            self.assertRaises(OSError),
        ):
            transform_next_patch(self.config)
        state = json.loads(self.states()[0].read_text())
        self.assertEqual(state["completed_tables"], 3)
        self.assertEqual(state["status"], "finalizing")
        with (
            patch("transform.discover_source_tables", return_value=paths),
            patch(
                "transform.pd.read_parquet",
                side_effect=AssertionError("reprocessed saved table"),
            ),
        ):
            transform_next_patch(self.config)
        self.assertEqual(json.loads(self.states()[0].read_text())["status"], "complete")

    def test_source_and_config_changes_start_separate_checkpoints(self):
        paths = self.sources(2)
        for change in (None, "source", "config"):
            if change == "source":
                paths[0].write_bytes(b"changed source contents")
            if change == "config":
                self.config["seed"] += 1
            with (
                patch("transform.discover_source_tables", return_value=paths),
                patch("transform.pd.read_parquet", return_value=self.frame) as reader,
            ):
                transform_next_patch(self.config)
            self.assertEqual(reader.call_count, 2)
        self.assertEqual(len(self.states()), 3)

    def test_corrupt_checkpoint_is_rejected(self):
        paths = self.sources(1)
        with (
            patch("transform.discover_source_tables", return_value=paths),
            patch("transform.pd.read_parquet", return_value=self.frame),
        ):
            transform_next_patch(self.config)
        shard = next(self.states()[0].parent.glob("batch_*.pt"))
        shard.write_bytes(b"truncated")
        with (
            patch("transform.discover_source_tables", return_value=paths),
            self.assertRaisesRegex(ValueError, "Corrupt transform checkpoint"),
        ):
            transform_next_patch(self.config)

    def test_time_based_checkpoint_and_skipped_tables(self):
        paths = self.sources(3)
        with TransformCheckpoint(
            self.base,
            "signature",
            paths,
            self.base / "raw",
            "test",
            options={"checkpoint_interval_seconds": 60},
        ) as checkpoint:
            checkpoint.restore()
            checkpoint.last_save -= 61
            checkpoint.save(1, [], [], [{"source": str(paths[0]), "reason": "short"}])
            self.assertEqual(checkpoint.state["completed_tables"], 1)
        with TransformCheckpoint(
            self.base, "signature", paths, self.base / "raw", "test"
        ) as checkpoint:
            records, tables, excluded = checkpoint.restore()
            self.assertEqual((records, tables), ([], []))
            self.assertEqual(len(excluded), 1)

    def test_progress_shows_bar_and_eta_at_100_and_final_partial(self):
        messages = []
        sink = logger.add(lambda message: messages.append(str(message)), level="INFO")
        try:
            progress = TransformProgress("Transform", 205)
            for completed in range(1, 206):
                progress.update(completed)
        finally:
            logger.remove(sink)
        self.assertEqual(len(messages), 4)
        self.assertIn("100/205", messages[1])
        self.assertIn("200/205", messages[2])
        self.assertIn("205/205 (100.0%)", messages[3])
        self.assertIn("[########################]", messages[3])
        self.assertIn("ETA=", messages[3])

    def test_configurable_table_interval_and_disabled_time_trigger(self):
        paths = self.sources(3)
        checkpoint = TransformCheckpoint(
            self.base, "signature", paths, self.base / "raw", "test"
        )
        self.assertEqual(checkpoint.every_tables, 100_000)
        self.assertIsNone(checkpoint.interval_seconds)
        with TransformCheckpoint(
            self.base,
            "signature",
            paths,
            self.base / "raw",
            "test",
            options={"checkpoint_every_tables": 2, "checkpoint_interval_seconds": None},
        ) as checkpoint:
            checkpoint.restore()
            checkpoint.last_save -= 100_000
            checkpoint.save(1, [], [], [])
            self.assertEqual(checkpoint.state["completed_tables"], 0)
            checkpoint.save(2, [], [], [])
            self.assertEqual(checkpoint.state["completed_tables"], 2)
        # Operational interval changes must reuse the same checkpoint.
        with TransformCheckpoint(
            self.base,
            "signature",
            paths,
            self.base / "raw",
            "test",
            options={
                "checkpoint_every_tables": None,
                "checkpoint_interval_seconds": None,
            },
        ) as checkpoint:
            checkpoint.restore()
            self.assertEqual(checkpoint.state["completed_tables"], 2)
            checkpoint.save(3, [], [], [])
            self.assertEqual(checkpoint.state["completed_tables"], 2)
            checkpoint.save(3, [], [], [], force=True)
            self.assertEqual(checkpoint.state["completed_tables"], 3)

    def test_invalid_checkpoint_intervals_fail_before_source_scan(self):
        for name, values in (
            ("checkpoint_every_tables", [0, -1, True, 1.5, "100"]),
            (
                "checkpoint_interval_seconds",
                [0, -1, True, float("inf"), float("nan"), "60"],
            ),
        ):
            for value in values:
                with (
                    self.subTest(name=name, value=value),
                    self.assertRaisesRegex(ValueError, name),
                ):
                    TransformCheckpoint(
                        self.base,
                        "signature",
                        [],
                        self.base / "raw",
                        "test",
                        options={name: value},
                    )

    def test_atomic_write_keeps_previous_output_on_failure(self):
        path = self.base / "windows.pt"
        path.write_bytes(b"previous completed artifact")

        def interrupted(stream):
            stream.write(b"partial replacement")
            raise OSError("interrupted write")

        with self.assertRaises(OSError):
            atomic_write(path, interrupted)
        self.assertEqual(path.read_bytes(), b"previous completed artifact")
        self.assertEqual(list(self.base.glob(".*.tmp")), [])

    def test_uncommitted_batch_is_not_restored_and_output_lock_is_exclusive(self):
        paths = self.sources(2)
        with TransformCheckpoint(
            self.base, "signature", paths, self.base / "raw", "test"
        ) as checkpoint:
            checkpoint.restore()
            checkpoint.save(1, [], [], [], force=True)
            with self.assertRaisesRegex(RuntimeError, "Another transform"):
                with TransformCheckpoint(
                    self.base, "different", paths, self.base / "raw", "test"
                ):
                    pass
            with (
                patch(
                    "transform_progress.atomic_json",
                    side_effect=OSError("interrupted commit"),
                ),
                self.assertRaises(OSError),
            ):
                checkpoint.save(2, [], [], [], force=True)
        with TransformCheckpoint(
            self.base, "signature", paths, self.base / "raw", "test"
        ) as checkpoint:
            checkpoint.restore()
            self.assertEqual(checkpoint.state["completed_tables"], 1)
            checkpoint.save(2, [], [], [], force=True)
            self.assertEqual(checkpoint.state["completed_tables"], 2)


if __name__ == "__main__":
    unittest.main()

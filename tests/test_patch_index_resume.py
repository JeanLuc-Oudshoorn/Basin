"""Patch index recovery must skip committed windows and keep metadata output bounded."""

import copy
import gc
import json
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import torch
from loguru import logger

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src/04_patch_indexing"))
import index_patches
import patch_index_progress
import test_lazy_patch_index as fixtures
from patch_index_progress import (
    PatchIndexCheckpoint,
    PatchIndexProgress,
    patch_index_options,
)
from transform import file_digest, load_config, read_artifact


class PatchIndexResumeTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.base = Path(self.temp.name)
        self.config = fixtures.LazyPatchIndexTests().fixture(self.base, count=205)
        self.config["patch_index"].update(
            batch_size_records=17,
            checkpoint_every_records=100,
            checkpoint_interval_seconds=None,
            progress_every=100,
        )

    def checkpoint_path(self):
        return next(
            (self.base / "patch_index/.patch_index_checkpoints").glob("*/progress.json")
        )

    def interrupt(self, position=137):
        original = index_patches.record_summary

        def summary(record, settings):
            if record["id"] == f"window_{position}":
                raise KeyboardInterrupt()
            return original(record, settings)

        with (
            patch.object(index_patches, "record_summary", side_effect=summary),
            self.assertRaises(KeyboardInterrupt),
        ):
            index_patches.build_patch_index(self.config)

    def test_resume_skips_saved_records_truncates_partial_tail_and_matches_clean(self):
        previous_threads = torch.get_num_threads()
        self.interrupt()
        self.assertEqual(torch.get_num_threads(), previous_threads)
        state = json.loads(self.checkpoint_path().read_text())
        self.assertEqual(state["completed_records"], 100)
        details = self.checkpoint_path().with_name("windows.jsonl")
        with details.open("ab") as stream:
            stream.write(b'{"partially-written')
        self.config["patch_index"].update(
            batch_size_records=31, checkpoint_every_records=75
        )
        with patch.object(
            index_patches, "record_summary", wraps=index_patches.record_summary
        ) as summary:
            report = index_patches.build_patch_index(self.config)
        self.assertEqual(summary.call_count, 105)
        self.assertEqual(summary.call_args_list[0].args[0]["id"], "window_100")
        self.assertEqual(report["resumed_records"], 100)
        rows = (self.base / "patch_index/windows.jsonl").read_text().splitlines()
        self.assertEqual(
            [json.loads(row)["id"] for row in rows], [f"window_{i}" for i in range(205)]
        )
        clean = copy.deepcopy(self.config)
        clean["paths"]["patch_index"] = str(self.base / "clean")
        clean_report = index_patches.build_patch_index(clean)
        self.assertEqual(
            (self.base / "clean/windows.jsonl").read_bytes(),
            (self.base / "patch_index/windows.jsonl").read_bytes(),
        )
        self.assertEqual(
            report["avoided_float32_snapshot_bytes"],
            clean_report["avoided_float32_snapshot_bytes"],
        )
        actual = torch.load(self.base / "patch_index/patch_index.pt", weights_only=True)
        expected = torch.load(self.base / "clean/patch_index.pt", weights_only=True)
        for key in actual["encoder_state"]:
            torch.testing.assert_close(
                actual["encoder_state"][key], expected["encoder_state"][key]
            )

    def test_finalization_failure_retries_without_indexing_again(self):
        with (
            patch.object(
                index_patches, "save_artifact", side_effect=OSError("disk full")
            ),
            self.assertRaises(OSError),
        ):
            index_patches.build_patch_index(self.config)
        state = json.loads(self.checkpoint_path().read_text())
        self.assertEqual(state["completed_records"], 205)
        self.assertEqual(state["status"], "finalizing")
        with patch.object(
            index_patches, "record_summary", side_effect=AssertionError("reindexed")
        ):
            report = index_patches.build_patch_index(self.config)
        self.assertEqual(report["resumed_records"], 205)
        self.assertEqual(
            json.loads(self.checkpoint_path().read_text())["status"], "complete"
        )

    def test_corrupt_or_truncated_saved_metadata_is_rejected(self):
        self.interrupt()
        details = self.checkpoint_path().with_name("windows.jsonl")
        original = details.read_bytes()
        details.write_bytes(b"X" + original[1:])
        with self.assertRaisesRegex(ValueError, "Corrupt patch index checkpoint"):
            index_patches.build_patch_index(self.config)
        details.write_bytes(original[:10])
        with self.assertRaisesRegex(ValueError, "Truncated patch index checkpoint"):
            index_patches.build_patch_index(self.config)

    def test_interrupted_checkpoint_commit_replays_uncommitted_output(self):
        self.config["patch_index"]["checkpoint_every_records"] = 50
        original = patch_index_progress.atomic_json
        writes = 0

        def commit(path, state):
            nonlocal writes
            writes += 1
            if writes == 2:
                raise OSError("interrupted checkpoint")
            return original(path, state)

        with (
            patch.object(patch_index_progress, "atomic_json", side_effect=commit),
            self.assertRaises(OSError),
        ):
            index_patches.build_patch_index(self.config)
        self.assertEqual(
            json.loads(self.checkpoint_path().read_text())["completed_records"], 50
        )
        report = index_patches.build_patch_index(self.config)
        self.assertEqual(report["resumed_records"], 50)
        self.assertEqual(
            len((self.base / "patch_index/windows.jsonl").read_text().splitlines()), 205
        )

    def test_changed_input_starts_new_checkpoint(self):
        self.interrupt()
        # Mock call/exception cycles can retain mapped fixture tensors on Windows.
        gc.collect()
        # Rebuild the processed fixture with a different record count/content hash.
        fixtures.LazyPatchIndexTests().fixture(self.base, count=206)
        report = index_patches.build_patch_index(self.config)
        self.assertEqual(report["resumed_records"], 0)
        self.assertEqual(
            len(list(self.checkpoint_path().parent.parent.glob("*/progress.json"))), 2
        )

    def test_shards_and_metadata_only_records_work_without_reloading_each_batch(self):
        from sharded_records import load_shard

        path = self.base / "processed/windows.pt"
        raw = torch.load(path, weights_only=True)
        records = raw["records"][:10]
        for record in records[1::2]:
            record["split"] = "evaluation"
            record.pop("batch")
        shards = []
        for i in range(2):
            shard = path.with_name(f"shard_{i}.pt")
            torch.save({"records": records[i * 5 : (i + 1) * 5]}, shard)
            shards.append(
                {"path": shard.name, "record_count": 5, "sha256": file_digest(shard)}
            )
        torch.save(
            {
                "format": "sharded_windows_v1",
                "signature": raw["signature"],
                "record_count": 10,
                "shards": shards,
            },
            path,
        )
        self.config["patch_index"].update(
            batch_size_records=2, checkpoint_every_records=3
        )
        with patch("sharded_records.load_shard", wraps=load_shard) as loader:
            report = index_patches.build_patch_index(self.config)
        self.assertEqual(loader.call_count, 2)
        self.assertEqual(report["window_count"], 10)
        rows = [
            json.loads(line)
            for line in (self.base / "patch_index/windows.jsonl")
            .read_text()
            .splitlines()
        ]
        self.assertEqual(sum(row.get("metadata_only", False) for row in rows), 5)
        self.assertEqual(
            len(
                read_artifact(self.base / "patch_index/patch_index.pt", self.config)[
                    "records"
                ]
            ),
            10,
        )
        self.config["paths"]["patch_index"] = str(self.base / "interrupted")
        original = index_patches.record_summary

        def interrupt(record, settings):
            if record["id"] == "window_7":
                raise KeyboardInterrupt()
            return original(record, settings)

        with (
            patch.object(index_patches, "record_summary", side_effect=interrupt),
            self.assertRaises(KeyboardInterrupt),
        ):
            index_patches.build_patch_index(self.config)
        with (
            patch("sharded_records.load_shard", wraps=load_shard) as loader,
            patch.object(index_patches, "record_summary", wraps=original) as summary,
        ):
            resumed = index_patches.build_patch_index(self.config)
        self.assertEqual(resumed["resumed_records"], 6)
        self.assertEqual(loader.call_count, 1)  # The completed first shard is skipped.
        self.assertEqual(summary.call_count, 4)
        self.assertEqual(summary.call_args_list[0].args[0]["id"], "window_6")


class WorkerProcessTests(unittest.TestCase):
    """Inspecting shards in worker processes changes the speed, never the output."""

    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.base = Path(self.temp.name)
        self.config = fixtures.LazyPatchIndexTests().fixture(self.base, count=45)
        path = self.base / "processed/windows.pt"
        raw = torch.load(path, weights_only=True)
        for record in raw["records"][1::2]:
            record["split"] = "evaluation"
            record.pop("batch")
        shards = []
        for start in range(0, 45, 10):
            shard = path.with_name(f"shard_{start // 10}.pt")
            records = raw["records"][start : start + 10]
            torch.save({"records": records}, shard)
            shards.append(
                {
                    "path": shard.name,
                    "record_count": len(records),
                    "sha256": file_digest(shard),
                }
            )
        torch.save(
            {
                "format": "sharded_windows_v1",
                "signature": raw["signature"],
                "record_count": 45,
                "shards": shards,
            },
            path,
        )

    def run_stage(self, name, **options):
        config = copy.deepcopy(self.config)
        config["paths"]["patch_index"] = str(self.base / name)
        config["patch_index"].update(options)
        return index_patches.build_patch_index(config)

    def details(self, name):
        return (self.base / name / "windows.jsonl").read_bytes()

    def test_workers_write_the_same_output_as_in_place_inspection(self):
        expected = self.run_stage("in_place", workers=0)
        with patch.object(
            index_patches, "record_summary", side_effect=AssertionError("in place")
        ):
            report = self.run_stage("workers", workers=2)
        self.assertEqual(self.details("workers"), self.details("in_place"))
        self.assertEqual(len(self.details("workers").splitlines()), 45)
        for key in ("window_count", "avoided_float32_snapshot_bytes"):
            self.assertEqual(report[key], expected[key])

    def test_interrupted_worker_run_resumes_inside_a_shard(self):
        original = PatchIndexProgress.update

        def update(progress, position, **options):
            if position == 26:
                raise KeyboardInterrupt()
            return original(progress, position, **options)

        with (
            patch.object(PatchIndexProgress, "update", update),
            self.assertRaises(KeyboardInterrupt),
        ):
            self.run_stage("resumed", workers=2, checkpoint_every_records=7)
        state = json.loads(
            next(
                (self.base / "resumed/.patch_index_checkpoints").glob("*/progress.json")
            ).read_text()
        )
        self.assertEqual(state["completed_records"], 21)
        report = self.run_stage("resumed", workers=2, checkpoint_every_records=7)
        self.assertEqual(report["resumed_records"], 21)
        self.run_stage("in_place", workers=0)
        self.assertEqual(self.details("resumed"), self.details("in_place"))

    def test_workers_reject_a_corrupt_shard(self):
        shard = self.base / "processed/shard_3.pt"
        content = bytearray(shard.read_bytes())
        content[-1] ^= 1
        shard.write_bytes(content)
        with self.assertRaisesRegex(ValueError, "Corrupt processed shard"):
            self.run_stage("corrupt", workers=2)

    def test_workers_start_only_for_enough_remaining_shards(self):
        records = read_artifact(self.base / "processed/windows.pt", self.config)[
            "records"
        ]
        automatic, in_place = (
            patch_index_options({}),
            patch_index_options({"workers": 0}),
        )
        with (
            patch.object(index_patches, "WORKER_MINIMUM_SHARDS", 4),
            patch.object(index_patches.os, "cpu_count", return_value=12),
        ):
            self.assertEqual(index_patches.worker_count(automatic, records, 0), 6)
            # From record 20 on, only three of the five shards remain.
            self.assertEqual(index_patches.worker_count(automatic, records, 20), 0)
            self.assertEqual(index_patches.worker_count(in_place, records, 0), 0)
        explicit = patch_index_options({"workers": 3})
        self.assertEqual(index_patches.worker_count(explicit, records, 20), 3)
        self.assertEqual(index_patches.worker_count(explicit, records, 45), 0)
        self.assertEqual(index_patches.worker_count(explicit, list(records), 0), 0)
        self.assertEqual(index_patches.worker_count(automatic, records, 0), 0)


class PatchIndexPolicyTests(unittest.TestCase):
    def test_time_trigger_and_disabled_record_trigger(self):
        with tempfile.TemporaryDirectory() as temporary:
            options = patch_index_options(
                {"checkpoint_every_records": None, "checkpoint_interval_seconds": 30}
            )
            with PatchIndexCheckpoint(
                Path(temporary), "signature", "digest", 3, "code", options
            ) as checkpoint:
                checkpoint.restore()
                checkpoint.append({"id": "0"}, 12)
                checkpoint.save()
                self.assertFalse(checkpoint.path.exists())
                checkpoint.last_save -= 31
                checkpoint.save()
                self.assertEqual(checkpoint.state["completed_records"], 1)
            with PatchIndexCheckpoint(
                Path(temporary),
                "signature",
                "digest",
                3,
                "code",
                patch_index_options({}),
            ) as checkpoint:
                checkpoint.restore()
                self.assertEqual(checkpoint.position, 1)
                self.assertEqual(checkpoint.estimated_old_bytes, 12)

    def test_options_validate_without_touching_inputs(self):
        defaults = patch_index_options({})
        self.assertEqual(defaults["checkpoint_every_records"], 100_000)
        self.assertIsNone(defaults["checkpoint_interval_seconds"])
        self.assertIsNone(defaults["workers"])
        self.assertEqual(patch_index_options({"workers": 0})["workers"], 0)
        for name in defaults:
            for value in (0, -1, True, "100", float("nan"), float("inf")):
                if name == "workers" and value == 0:
                    continue  # Zero workers means inspecting in this process.
                with (
                    self.subTest(name=name, value=value),
                    self.assertRaisesRegex(ValueError, name),
                ):
                    patch_index_options({name: value})
        with self.assertRaisesRegex(ValueError, "batch_size_records"):
            patch_index_options({"batch_size_records": 1.5})

    def test_configuration_with_the_former_names_is_rejected(self):
        with tempfile.TemporaryDirectory() as temporary:
            source = ROOT / "configs/config.json"
            config = json.loads(source.read_text(encoding="utf-8"))
            config["paths"]["embedded"] = config["paths"].pop("patch_index")
            config["embedding"] = config.pop("patch_index")
            path = Path(temporary) / "config.json"
            path.write_text(json.dumps(config), encoding="utf-8")
            with self.assertRaisesRegex(
                ValueError, "paths.embedded is now paths.patch_index; embedding is now"
            ):
                load_config(path)

    def test_progress_bar_reports_records_and_final_partial_interval(self):
        messages = []
        sink = logger.add(lambda message: messages.append(str(message)), level="INFO")
        try:
            progress = PatchIndexProgress(205, 0, 100)
            for index in range(1, 206):
                progress.update(index)
        finally:
            logger.remove(sink)
        self.assertEqual(len(messages), 4)
        self.assertIn("100/205 records", messages[1])
        self.assertIn("[########################]", messages[-1])
        self.assertIn("records/s", messages[-1])
        self.assertIn("ETA=", messages[-1])


if __name__ == "__main__":
    unittest.main()

"""Bounded buffers, lazy shard lifetimes, and interrupted spill recovery."""

import gc
import json
import sys
import tempfile
import unittest
import weakref
from pathlib import Path
from unittest.mock import patch

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src/03_transforming_input_data"))
import test_transform_resume as resume_fixtures
from sharded_records import ShardBatch, ShardedRecords
from transform import read_artifact, transform_next_patch
from transform_progress import ShardedTransformCheckpoint


class ShardedStorageTests(unittest.TestCase):
    def test_spill_releases_tensors_and_restore_does_not_load_them(self):
        with tempfile.TemporaryDirectory() as temporary:
            base = Path(temporary)
            paths = [base / "source"]
            paths[0].write_bytes(b"source")
            options = {
                "batch_memory_limit_mb": 1,
                "batch_size_tables": 2,
                "checkpoint_every_tables": 100_000,
            }
            with ShardedTransformCheckpoint(
                base, "test", paths, base, "code", options=options
            ) as checkpoint:
                checkpoint.restore()
                tensor = torch.ones(256, 1024)  # One MiB: force a spill mid-table.
                reference = weakref.ref(tensor)
                checkpoint.append_record({"id": "first", "batch": {"context": tensor}})
                del tensor
                gc.collect()
                self.assertIsNone(reference())
                self.assertEqual(checkpoint.records, [])
                self.assertEqual(checkpoint.buffer_bytes, 0)
                self.assertFalse(checkpoint.path.exists())  # Spill is not a checkpoint.
                checkpoint.append_table({"training_transitions": 1})
                checkpoint.save_progress(1, force=True)
                self.assertEqual(checkpoint.state["record_count"], 1)
                index = checkpoint.artifact_index()
            with ShardedTransformCheckpoint(
                base, "test", paths, base, "code", options=options
            ) as checkpoint:
                with patch(
                    "sharded_records.load_shard",
                    side_effect=AssertionError("eager restore"),
                ):
                    checkpoint.restore()
                self.assertEqual(checkpoint.records, [])
                self.assertEqual(checkpoint.state["completed_tables"], 1)
            self.assertEqual(len(ShardedRecords(index, base)), 1)

    def test_orphan_spill_replayed_after_last_checkpoint(self):
        with tempfile.TemporaryDirectory() as temporary:
            base = Path(temporary)
            paths = [base / f"source_{i}" for i in range(3)]
            for path in paths:
                path.write_bytes(b"source")
            options = {"batch_size_tables": 1, "checkpoint_every_tables": 2}
            with ShardedTransformCheckpoint(
                base, "test", paths, base, "code", options=options
            ) as checkpoint:
                checkpoint.restore()
                for i in range(3):
                    checkpoint.append_record({"id": str(i)})
                    checkpoint.save_progress(i + 1)
                directory = checkpoint.directory
                self.assertEqual(len(list(directory.glob("batch_*.pt"))), 3)
            with ShardedTransformCheckpoint(
                base, "test", paths, base, "code", options=options
            ) as checkpoint:
                checkpoint.restore()
                self.assertEqual(checkpoint.state["completed_tables"], 2)
                self.assertEqual(len(list(directory.glob("batch_*.pt"))), 2)
                checkpoint.append_record({"id": "2"})
                checkpoint.save_progress(3, force=True)
                self.assertEqual(
                    [
                        r["id"]
                        for r in ShardedRecords(checkpoint.artifact_index(), base)
                    ],
                    ["0", "1", "2"],
                )


class ShardedPipelineTests(unittest.TestCase):
    setUp = resume_fixtures.TransformResumeTests.setUp
    sources = resume_fixtures.TransformResumeTests.sources

    def test_metadata_iteration_and_batch_planning_never_load_tensor_shards(self):
        paths = self.sources(5)
        self.config["transform"].update(batch_size_tables=2, batch_memory_limit_mb=1)
        with (
            patch("transform.discover_source_tables", return_value=paths),
            patch("transform.pd.read_parquet", return_value=self.frame),
        ):
            transform_next_patch(self.config)
        index = torch.load(self.base / "processed/windows.pt", weights_only=True)
        sys.path.insert(0, str(ROOT / "src/04_creating_embeddings"))
        sys.path.insert(0, str(ROOT / "src/05_model_training"))
        from types import SimpleNamespace

        from batching import make_batches
        from embed import LazyPatchRecord

        encoder = SimpleNamespace(
            config=SimpleNamespace(patch_size=16, context_length=2048)
        )
        with patch(
            "sharded_records.load_shard",
            side_effect=AssertionError("eager tensor loading"),
        ):
            records = ShardedRecords(index, self.base / "processed")
            train = [
                LazyPatchRecord(r, encoder) for r in records if r["split"] == "train"
            ]
            first = make_batches(train, 2, seed=7, bucket_by_length=True)
            second = make_batches(train, 2, seed=7, bucket_by_length=True)
            self.assertEqual(
                [[r["id"] for r in b] for b in first],
                [[r["id"] for r in b] for b in second],
            )
            self.assertEqual(len(list(records)), index["record_count"])
            self.assertIsNone(records.cached)

    def test_existing_shards_build_metadata_once_then_reopen_without_tensors(self):
        paths = self.sources(2)
        with (
            patch("transform.discover_source_tables", return_value=paths),
            patch("transform.pd.read_parquet", return_value=self.frame),
        ):
            transform_next_patch(self.config)
        base = self.base / "processed"
        for path in (base / ".record_metadata").glob("*.json"):
            path.unlink()
        index = torch.load(base / "windows.pt", weights_only=True)
        from sharded_records import load_shard

        with patch("sharded_records.load_shard", wraps=load_shard) as loader:
            first = list(ShardedRecords(index, base))
            self.assertEqual(loader.call_count, len(index["shards"]))
        with patch(
            "sharded_records.load_shard",
            side_effect=AssertionError("repeated tensor read"),
        ):
            second = list(ShardedRecords(index, base))
            self.assertEqual([r["id"] for r in first], [r["id"] for r in second])

    def test_table_batches_lazy_access_manifest_and_embed(self):
        paths = self.sources(5)
        self.config["transform"].update(batch_size_tables=2, batch_memory_limit_mb=1)
        with (
            patch("transform.discover_source_tables", return_value=paths),
            patch("transform.pd.read_parquet", return_value=self.frame),
        ):
            payload = transform_next_patch(self.config)
        raw = torch.load(self.base / "processed/windows.pt", weights_only=True)
        self.assertNotIn("records", raw)
        self.assertEqual(raw["format"], "sharded_windows_v1")
        self.assertEqual(len(raw["shards"]), 3)
        self.assertIsInstance(payload["records"], ShardedRecords)
        records = list(payload["records"])
        self.assertTrue(all(isinstance(r["batch"], ShardBatch) for r in records))
        first = records[0]["batch"]["context"]
        reference = weakref.ref(first)
        del first
        records[-1]["batch"]["context"]  # Evict first shard.
        gc.collect()
        self.assertIsNone(
            reference()
        )  # Retaining all metadata does not retain tensors.
        torch.testing.assert_close(
            records[0]["batch"]["context"][0, 0], torch.arange(64, dtype=torch.float32)
        )
        manifest = json.loads((self.base / "processed/manifest.json").read_text())
        self.assertEqual(manifest["table_count"], 5)
        self.assertEqual(len(manifest["tables"]), 5)
        self.assertEqual(manifest["excluded_tables"], [])

        sys.path.insert(0, str(ROOT / "src/04_creating_embeddings"))
        from embed import embed
        from sharded_records import EncodedRecords

        embed(self.config)
        embedded = read_artifact(self.base / "embedded/embeddings.pt", self.config)
        self.assertIsInstance(embedded["records"], EncodedRecords)
        prepared = embedded["records"][0]["prepared"]
        self.assertGreater(prepared["tokens"], 0)


if __name__ == "__main__":
    unittest.main()

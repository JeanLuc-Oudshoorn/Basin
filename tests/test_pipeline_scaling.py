"""Large-corpus costs: bounded record handles, linear splitting, shard locality."""

import hashlib
import json
import sys
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import torch

ROOT = Path(__file__).resolve().parents[1]
for stage in (
    "03_transforming_input_data",
    "04_creating_embeddings",
    "05_model_training",
):
    sys.path.insert(0, str(ROOT / "src" / stage))

from batching import BatchPlanner
from checkpoints import ordered_record_digest
from embed import LazyPatchRecord
from sharded_records import (
    EncodedRecords,
    RecordSelection,
    ShardedRecords,
    load_shard,
    write_record_metadata,
)
from training_protocol import split_datasets
from transform import file_digest, load_config, write_json


def corpus(directory, shards=3, tables=12):
    entries = []
    for shard_index in range(shards):
        records = []
        for row in range(tables):
            i = shard_index * tables + row
            length = 160 + row % 3 * 16
            metadata = dict(
                source=f"table_{i}",
                source_key=f"table_{i}",
                source_rows=length,
                source_sha256=f"hash_{i}",
                targets=["outcome"],
                roles={"targets": ["outcome"], "past": [], "known": []},
            )
            records.append(
                dict(
                    metadata,
                    id=f"{i}_train",
                    split="train",
                    cutoff=length,
                    context_rows=[0, length],
                    batch={
                        "context": torch.full((1, 1, length), float(i)),
                        "horizon": 16,
                    },
                )
            )
            records.append(
                dict(
                    metadata,
                    id=f"{i}_eval",
                    split="evaluation",
                    cutoff=length - 64,
                    context_rows=[0, length - 64],
                    label_rows=[length - 64, length],
                )
            )
        path = directory / f"shard_{shard_index}.pt"
        torch.save({"records": records}, path)
        entry = dict(
            path=path.name, sha256=file_digest(path), record_count=len(records)
        )
        entries.append(entry)
        write_record_metadata(directory, entry, records)
    index = dict(shards=entries, record_count=sum(s["record_count"] for s in entries))
    raw = ShardedRecords(index, directory)
    encoder = SimpleNamespace(
        config=SimpleNamespace(patch_size=16, context_length=2048)
    )

    class CountingRecords(EncodedRecords):
        passes = 0

        def __iter__(self):
            self.passes += 1
            for i in range(len(self)):
                yield self[i]

    return CountingRecords(raw, encoder, LazyPatchRecord)


class PipelineScalingTests(unittest.TestCase):
    def test_compact_sharded_training_resumes_exactly_mid_epoch(self):
        from test_timed_training import Clock, TimedTrainingTests, run_training, setup

        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            data = root / "data"
            data.mkdir()
            entries = []
            for i, length in enumerate((48, 64, 80)):
                record = dict(
                    id=str(i),
                    source=str(i),
                    source_rows=length,
                    source_sha256=f"source_{i}",
                    split="train",
                    cutoff=length,
                    context_rows=[0, length],
                    roles={"targets": ["outcome"], "past": [], "known": []},
                    batch={
                        "context": torch.linspace(0, 1, length)[None, None],
                        "horizon": 16,
                    },
                )
                path = data / f"shard_{i}.pt"
                torch.save({"records": [record]}, path)
                shard = dict(path=path.name, record_count=1, sha256=file_digest(path))
                entries.append(shard)
                write_record_metadata(data, shard, [record])

            def parts(directory):
                model, _, validation, config, optimizer = setup(directory)
                raw = ShardedRecords(dict(shards=entries, record_count=3), data)
                records = RecordSelection(
                    EncodedRecords(raw, model, LazyPatchRecord), range(3)
                )
                config["training"]["storage_locality"] = True
                return model, records, validation, config, optimizer

            run_training(parts(root / "full"), Clock())
            interrupted = parts(root / "partial")
            interrupted[3]["training"]["max_run_hours"] = 1 / 3600
            run_training(interrupted, Clock())
            run_training(parts(root / "partial"), Clock())
            full = torch.load(root / "full/latest_checkpoint.pt", weights_only=True)
            resumed = torch.load(
                root / "partial/latest_checkpoint.pt", weights_only=True
            )
            TimedTrainingTests.assert_checkpoint_equal(self, full, resumed)
            self.assertEqual(
                full["identity"]["dataset_identity_policy"], "ordered_stream_v1"
            )

    def test_split_scans_metadata_twice_and_returns_compact_disjoint_training(self):
        with tempfile.TemporaryDirectory() as temporary:
            records = corpus(Path(temporary))
            config = load_config()
            config["training"]["validation_sampling"]["tables"] = 2
            config["training"]["holdout_fraction"] = 0.2
            with patch(
                "sharded_records.load_shard",
                side_effect=AssertionError("tensor read during split"),
            ):
                train, validation, test, audit = split_datasets(records, config)
            self.assertEqual(records.passes, 2)
            self.assertIsInstance(train, RecordSelection)
            self.assertEqual(train.indices.itemsize, 8)
            self.assertEqual(len(train), 26)  # 36 sources - 8 test - 2 validation.
            self.assertFalse(
                {r["source"] for r in train} & {r["source"] for r in validation + test}
            )
            self.assertEqual(audit["validation_sampling"]["eligible_tables"], 28)
            reverse = split_datasets(list(reversed(records)), config)
            self.assertEqual(audit, reverse[3])

    def test_locality_reads_each_shard_once_and_preserves_coverage_and_seed(self):
        with tempfile.TemporaryDirectory() as temporary:
            records = corpus(Path(temporary))
            train = RecordSelection(records, range(0, len(records), 2))
            with patch(
                "sharded_records.load_shard",
                side_effect=AssertionError("tensor read while planning"),
            ):
                plan = BatchPlanner(
                    train,
                    4,
                    bucket_by_length=True,
                    shape_run_batches=2,
                    storage_locality=True,
                )
                first = plan.batches(7)
                same = plan.batches(7)
                next_epoch = plan.batches(8)

            def ids(batches):
                return [[r["id"] for r in b] for b in batches]

            self.assertEqual(ids(first), ids(same))
            self.assertNotEqual(ids(first), ids(next_epoch))
            self.assertEqual(
                sorted(r["id"] for b in first for r in b),
                sorted(r["id"] for r in train),
            )
            for batch in first:
                self.assertEqual(len({r.storage_key for r in batch}), 1)
            with patch("sharded_records.load_shard", wraps=load_shard) as loader:
                for batch in first:
                    for record in batch:
                        self.assertEqual(
                            record["batch"]["context"][0, 0, 0].item(),
                            int(record["id"].split("_")[0]),
                        )
                self.assertEqual(loader.call_count, 3)

    def test_streamed_id_digest_matches_join_and_binds_storage_layout(self):
        records = [
            {"id": "one", "source": "s"},
            {"id": "two\N{GREEK SMALL LETTER ALPHA}", "source": "s"},
        ]
        expected = hashlib.sha256(
            "\n".join(r["id"] for r in records).encode()
        ).hexdigest()
        self.assertEqual(ordered_record_digest(records), expected)
        self.assertEqual(ordered_record_digest([]), hashlib.sha256(b"").hexdigest())
        self.assertNotEqual(
            ordered_record_digest(records, include_storage=True), expected
        )

        class StoredRecord(dict):
            pass

        first = StoredRecord(id="same", source="same")
        second = StoredRecord(first)
        first.storage_key, second.storage_key = "shard_a", "shard_b"
        self.assertEqual(
            ordered_record_digest([first]), ordered_record_digest([second])
        )
        self.assertNotEqual(
            ordered_record_digest([first], include_storage=True),
            ordered_record_digest([second], include_storage=True),
        )

    def test_streaming_json_preserves_previous_file_on_encoding_failure(self):
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "audit.json"
            original = {"sources": ["a", "b"], "loss": 0.5}
            with patch(
                "transform.json.dumps",
                side_effect=AssertionError("full string allocation"),
            ):
                write_json(path, original)
            self.assertEqual(json.loads(path.read_text()), original)
            with self.assertRaises(ValueError):
                write_json(path, {"loss": float("nan")})
            self.assertEqual(json.loads(path.read_text()), original)
            self.assertEqual(list(Path(temporary).glob("*.tmp")), [])


if __name__ == "__main__":
    unittest.main()

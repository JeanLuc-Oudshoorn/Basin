"""Start-up index: the same split, batch order and identity, without metadata passes."""

import json
import sys
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
for stage in (
    "03_transforming_input_data",
    "04_patch_indexing",
    "05_model_training",
):
    sys.path.insert(0, str(ROOT / "src" / stage))

import startup_index
from batching import BatchPlanner
from checkpoints import checkpoint_identity, ordered_record_digest
from index_patches import ForecastConfig, LazyPatchRecord
from sharded_records import (
    EncodedRecords,
    RecordSelection,
    ShardedRecords,
    write_record_metadata,
)
from startup_index import LightRecord, fetch, light, scan
from test_pipeline_scaling import corpus
from training_protocol import split_datasets
from transform import load_config

NO_METADATA = AssertionError("record metadata read after the index was built")


def light_at(records, index):
    """The record as a start-up pass sees it through a corpus handle."""
    return light(RecordSelection(records, [index])[0])


def saved_index(records):
    return startup_index._corpus(records).startup_index()


def reopened(records):
    """The same corpus as a later start sees it: nothing held in memory."""
    raw = records.records
    index = dict(shards=raw.shards, record_count=len(raw))
    return EncodedRecords(
        ShardedRecords(index, raw.directory), records.encoder, records.factory
    )


def startup(records, config):
    """Everything training derives from the corpus before its first update."""
    train, validation, test, audit = split_datasets(records, config)
    planner = BatchPlanner(
        train, 4, bucket_by_length=True, shape_run_batches=2, storage_locality=True
    )
    model = SimpleNamespace(config=ForecastConfig())
    identity = checkpoint_identity(model, train, validation, config, object())
    return {
        "train": list(train.indices),
        "validation": json.dumps(validation, sort_keys=True),
        "test": json.dumps([dict(r) for r in test], sort_keys=True),
        "audit": json.dumps(audit),
        "plan": [
            [r.index for r in b] for seed in (7, 8) for b in planner.batches(seed)
        ],
        "identity": identity,
        "ordered": ordered_record_digest(train, include_storage=True),
    }


def split_config():
    config = load_config("configs/legacy_500m.json")
    config["training"]["validation_sampling"]["tables"] = 2
    config["training"]["holdout_fraction"] = 0.2
    return config


def odd_corpus(directory):
    """Records that differ in which fields they have, and in their text."""
    shared = dict(source_rows=96, targets=["outcome"], excluded_columns=[])
    shards = [
        [
            dict(
                shared,
                source="data/caf\N{LATIN SMALL LETTER E WITH ACUTE}/data.parquet",
                source_key="caf\N{LATIN SMALL LETTER E WITH ACUTE}/data.parquet",
                source_sha256="aa",
                id="one",
                split="train",
                cutoff=96,
                context_rows=[0, 96],
                roles={"targets": ["outcome"], "past": ["x"], "known": []},
                _shard_batch_keys=["context", "horizon", "past_covariates"],
            ),
            dict(
                shared,
                source="data/caf\N{LATIN SMALL LETTER E WITH ACUTE}/data.parquet",
                source_key="caf\N{LATIN SMALL LETTER E WITH ACUTE}/data.parquet",
                source_sha256="aa",
                id="",
                split="evaluation",
                cutoff=80,
                context_rows=[0, 80],
                label_rows=[80, 96],
                roles={"targets": ["outcome"], "past": ["x"], "known": []},
            ),
        ],
        [],  # the transform stage can publish a shard without records
        [
            # No source key and no id, as in artifacts of older versions.
            dict(
                shared, source="plain", source_sha256="bb", split="train", cutoff=None
            ),
            dict(
                shared,
                source="plain",
                source_sha256="bb",
                source_key="plain",
                id="1",
                split="train",
                cutoff=True,
                context_rows=[16, 96.0],
                roles=None,
                _shard_batch_keys=["context"],
            ),
            dict(shared, source="1", source_sha256="cc", id="last", cutoff=1),
        ],
    ]
    entries = []
    for number, records in enumerate(shards):
        entry = dict(
            path=f"shard_{number}.pt", sha256=f"sha_{number}", record_count=len(records)
        )
        (directory / entry["path"]).touch()
        write_record_metadata(directory, entry, records)
        entries.append(entry)
    return ShardedRecords(dict(shards=entries, record_count=5), directory)


class StartupIndexTests(unittest.TestCase):
    def assert_light_equals_whole(self, raw):
        fields = (*startup_index.TABLE_TEXT, *startup_index.RECORD_TEXT)
        fields += (*startup_index.CODED, "batch", "targets", "no_such_field")
        for index in range(len(raw)):
            whole, light = raw[index], light_at(raw, index)
            self.assertIsInstance(light, LightRecord)
            for field in fields:
                self.assertEqual(field in light, field in whole, (index, field))
                if field == "batch" and field in whole:
                    self.assertEqual(tuple(light[field]), tuple(whole[field]))
                    self.assertEqual(light[field].record, whole[field].record)
                    self.assertEqual(light[field].shard, whole[field].shard)
                    continue
                self.assertEqual(light.get(field, "none"), whole.get(field, "none"))
                # 1, 1.0 and True are equal: the type has to survive as well.
                self.assertIs(type(light.get(field)), type(whole.get(field)))
                if field not in whole:
                    with self.assertRaises(KeyError):
                        light[field]

    def test_light_record_answers_as_the_whole_record_does(self):
        with tempfile.TemporaryDirectory() as temporary:
            raw = odd_corpus(Path(temporary))
            self.assert_light_equals_whole(raw)
            self.assertEqual(
                [r.get("id") for r in scan(raw)], [r.get("id") for r in raw]
            )
            with self.assertRaises(IndexError):
                light_at(raw, 5)
            # A later start reads the saved index, not the record metadata.
            again = ShardedRecords(dict(shards=raw.shards, record_count=5), temporary)
            source = raw[0]["source"]
            with patch.object(
                ShardedRecords, "record_metadata", side_effect=NO_METADATA
            ):
                self.assertEqual(light_at(again, 3)["context_rows"], [16, 96.0])
                self.assertEqual(light_at(again, 0)["source"], source)

    def test_light_record_cannot_be_mistaken_for_the_whole_record(self):
        with tempfile.TemporaryDirectory() as temporary:
            record = light_at(odd_corpus(Path(temporary)), 0)
            for use in (dict, list, len, json.dumps, lambda r: r.items()):
                with self.assertRaises(TypeError):
                    use(record)

    def test_field_with_too_many_values_is_read_from_the_metadata(self):
        with (
            tempfile.TemporaryDirectory() as temporary,
            patch.object(startup_index, "VOCABULARY_LIMIT", 2),
        ):
            raw = odd_corpus(Path(temporary))
            self.assert_light_equals_whole(raw)
            self.assertNotIn("cutoff", saved_index(raw).fields)
            self.assertIn("split", saved_index(raw).fields)

    def test_field_that_is_not_text_is_read_from_the_metadata(self):
        with tempfile.TemporaryDirectory() as temporary:
            directory = Path(temporary)
            entry = dict(path="shard.pt", sha256="sha", record_count=2)
            write_record_metadata(
                directory,
                entry,
                [dict(source="a", id=7, split="train"), dict(source="b", id="x")],
            )
            raw = ShardedRecords(dict(shards=[entry], record_count=2), directory)
            self.assert_light_equals_whole(raw)
            self.assertNotIn("id", saved_index(raw).fields)

    def test_unusable_index_is_rebuilt(self):
        with tempfile.TemporaryDirectory() as temporary:
            directory = Path(temporary)
            raw = odd_corpus(directory)
            saved_index(raw)
            index = dict(shards=raw.shards, record_count=5)
            del raw
            saved = directory / startup_index.DIRECTORY

            def rebuilt():
                with patch.object(
                    startup_index.StartupIndex,
                    "build",
                    wraps=startup_index.StartupIndex.build,
                ) as build:
                    again = ShardedRecords(index, directory)
                    self.assert_light_equals_whole(again)
                    return build.call_count

            self.assertEqual(rebuilt(), 0)
            content = bytearray((saved / "source.bin").read_bytes())
            content[0] ^= 1
            (saved / "source.bin").write_bytes(bytes(content))
            self.assertEqual(rebuilt(), 1)
            self.assertEqual(rebuilt(), 0)
            # Another corpus in the same directory: other shards.
            index["shards"] = [dict(shard) for shard in index["shards"]]
            index["shards"][1]["sha256"] = "changed"
            metadata = saved.parent / ".record_metadata"
            (metadata / "changed.json").write_text(
                (metadata / "sha_1.json")
                .read_text("utf-8")
                .replace('"shard_sha256": "sha_1"', '"shard_sha256": "changed"'),
                encoding="utf-8",
            )
            self.assertEqual(rebuilt(), 1)
            self.assertFalse(list(directory.glob(f"{startup_index.DIRECTORY}.*")))

    def test_start_up_without_the_index_gives_the_same_results(self):
        with (
            tempfile.TemporaryDirectory() as first,
            tempfile.TemporaryDirectory() as second,
        ):
            config = split_config()
            indexed = corpus(Path(first))
            plain = corpus(Path(second))
            # As when the index cannot be saved.
            startup_index._corpus(plain).index = False
            with patch.object(
                startup_index.StartupIndex, "build", side_effect=NO_METADATA
            ):
                expected = startup(plain, config)
            self.assertFalse((Path(second) / startup_index.DIRECTORY).exists())
            self.assertEqual(startup(indexed, config), expected)
            # A later start: everything comes from the index and the saved
            # selection, and is again the same.
            with patch.object(
                ShardedRecords, "record_metadata", side_effect=NO_METADATA
            ):
                self.assertEqual(startup(reopened(indexed), config), expected)
            self.assertEqual(len(expected["train"]), 26)
            # A corpus that is not compact keeps its training records: whole ones.
            kept = split_datasets(indexed.records, config)[0]
            self.assertEqual({type(record) for record in kept}, {dict})
            self.assertEqual(
                [record["id"] for record in kept],
                [indexed[index]["id"] for index in expected["train"]],
            )
            self.assertGreater(len(expected["plan"]), 10)

    def test_whole_records_are_fetched_in_the_requested_order(self):
        with tempfile.TemporaryDirectory() as temporary:
            records = corpus(Path(temporary))
            wanted = [71, 3, 41, 3]  # evaluation records: metadata only
            expected = [dict(records[index]) for index in wanted]
            fetched = fetch(records, wanted)
            self.assertEqual([dict(r) for r in fetched], expected)
            self.assertIsInstance(fetched[0], LazyPatchRecord)
            (trained,) = fetch(records, [40])
            self.assertEqual(trained["id"], records[40]["id"])
            self.assertEqual(tuple(trained["batch"]), ("context", "horizon"))
            again = reopened(records)
            with patch.object(
                ShardedRecords, "record_metadata", side_effect=NO_METADATA
            ):
                self.assertEqual([dict(r) for r in fetch(again, wanted)], expected)
            self.assertEqual(fetch(records, []), [])
            selection = RecordSelection(records, [2, 4])
            self.assertEqual(light(selection[1])["id"], records[4]["id"])
            self.assertIs(light(expected[0]), expected[0])


if __name__ == "__main__":
    unittest.main()

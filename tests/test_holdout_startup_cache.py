"""Cache correctness, disjointness, and invalidation without GPU work."""

import json
import os
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src/06_holdout_test"))
from startup_cache import (
    artifact_digest,
    cached_family_exclusions,
    load_split,
    overlapping_sources,
    weights_digest,
)


class StartupCacheTests(unittest.TestCase):
    def test_weight_fingerprint_tracks_evaluated_tensors_and_invalidates(self):
        import torch

        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "checkpoint.pt"
            path.write_bytes(b"checkpoint version one")
            weights = {
                "weight": torch.ones(3, dtype=torch.bfloat16),
                "scalar": torch.tensor(1.0),
            }
            first = weights_digest(path, weights, field="best_state_dict")
            with patch(
                "startup_cache.hashlib.sha256",
                side_effect=AssertionError("must use cache"),
            ):
                self.assertEqual(
                    weights_digest(path, weights, field="best_state_dict"), first
                )
            path.write_bytes(b"checkpoint version two")
            weights["weight"][0] = 2
            self.assertNotEqual(
                weights_digest(path, weights, field="best_state_dict"), first
            )

    def test_streamed_split_preserves_overlap_checks_and_only_holdout_maps(self):
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "dataset_split.json"
            source = "data/date/table/data.parquet"
            split = {
                "source_sha256": {source: "digest", "train": "unused"},
                "train_sources": ["train"],
                "test_sources": [source],
                "validation_sources": ["validation"],
                "seed": 7,
                "requested_test_fraction": 0.1,
            }
            path.write_text(json.dumps(split))
            result = load_split(path)
            self.assertEqual(result["source_sha256"], {source: "digest"})
            self.assertNotIn("train_sources", result)
            self.assertEqual(
                overlapping_sources(result, "train", ["train", source]), {"train"}
            )
            self.assertEqual(
                overlapping_sources(result, "validation", ["validation"]),
                {"validation"},
            )
            with patch(
                "startup_cache.ijson.parse",
                side_effect=AssertionError("warm cache must not reparse"),
            ):
                warmed = load_split(path)
            self.assertEqual(overlapping_sources(warmed, "train", ["train"]), {"train"})
            split["train_sources"] = [source]
            path.write_text(json.dumps(split))
            self.assertEqual(
                overlapping_sources(load_split(path), "train", [source]), {source}
            )

    def test_checksum_invalidates_on_rewrite(self):
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "model.pt"
            path.write_bytes(b"first")
            first = artifact_digest(path)
            with patch(
                "startup_cache.hashlib.sha256",
                side_effect=AssertionError("must use cache"),
            ):
                self.assertEqual(artifact_digest(path), first)
            path.write_bytes(b"second")
            self.assertNotEqual(artifact_digest(path), first)

    def test_lineage_cache_notices_new_copy(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            folder = root / "date"
            folder.mkdir()
            source = str(folder / "original" / "data.parquet")
            calls = []

            def scan(sources, *, audit):
                calls.append(1)
                value = folder.stat()
                audit[str(folder)] = [
                    value.st_size,
                    value.st_mtime_ns,
                    value.st_ctime_ns,
                ]
                return {source} if (folder / "copy").exists() else set()

            self.assertEqual(cached_family_exclusions([source], root, scan), set())
            self.assertEqual(cached_family_exclusions([source], root, scan), set())
            self.assertEqual(len(calls), 1)
            (folder / "copy").mkdir()
            # Some filesystems (e.g. CI runners) do not move a directory's
            # timestamps for a create in the same instant; a real copy lands later.
            mtime = folder.stat().st_mtime_ns + 1_000_000_000
            os.utime(folder, ns=(mtime, mtime))
            self.assertEqual(cached_family_exclusions([source], root, scan), {source})
            self.assertEqual(len(calls), 2)


if __name__ == "__main__":
    unittest.main()

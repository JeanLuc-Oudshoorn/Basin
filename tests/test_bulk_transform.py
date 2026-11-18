"""Fast discovery, bulk reads and worker decoding leave the artifacts unchanged."""

import copy
import hashlib
import json
import os
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np
import pandas as pd
import torch
from loguru import logger

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src/03_transforming_input_data"))
from source_io import list_directory, read_files
from transform import (
    SourceTables,
    discover_source_tables,
    load_config,
    source_windows,
    throughput_options,
    transform_next_patch,
)
from transform_progress import TransformCheckpoint


class SourceIoTests(unittest.TestCase):
    def test_listing_agrees_with_scandir(self):
        with tempfile.TemporaryDirectory() as directory:
            base = Path(directory)
            (base / "folder").mkdir()
            (base / "Mixed Case").mkdir()
            (base / "file.parquet").write_bytes(b"x")
            listed = {(n, d) for n, d, _ in list_directory(base)}
            with os.scandir(base) as entries:
                expected = {(e.name, e.is_dir(follow_symlinks=False)) for e in entries}
            self.assertEqual(listed, expected)
            self.assertEqual(list(list_directory(base / "folder")), [])
            with self.assertRaises(OSError):
                list(list_directory(base / "missing"))

    def test_bulk_read_returns_every_file_exactly(self):
        random = np.random.default_rng(7)
        with tempfile.TemporaryDirectory() as directory:
            paths, contents = [], []
            for index, size in enumerate([0, 1, 4096, 5000, 70_001, 3_000_000]):
                path = Path(directory) / f"{index}" / "data.parquet"
                path.parent.mkdir()
                contents.append(random.bytes(size))
                path.write_bytes(contents[-1])
                paths.append(path)
            for threads, order in ((1, None), (4, None), (3, [5, 3, 4, 0, 2, 1])):
                with self.subTest(threads=threads, order=order):
                    read = read_files(paths, threads, order)
                    self.assertEqual([bytes(c) for c in read], contents)
            self.assertEqual(read_files([]), [])
            with self.assertRaises(FileNotFoundError):
                read_files([paths[0], Path(directory) / "absent" / "data.parquet"])

    def test_windows_respect_both_limits_and_cover_the_range(self):
        sizes = [10, 10, 10, 50, 10, 10]
        self.assertEqual(
            list(source_windows(sizes, 1, 6, max_tables=2, max_bytes=60)),
            [(1, 3), (3, 5), (5, 6)],
        )
        # A single table larger than the byte limit still forms its own window.
        self.assertEqual(
            list(source_windows(sizes, 0, 6, max_tables=10, max_bytes=25)),
            [(0, 2), (2, 3), (3, 4), (4, 6)],
        )
        self.assertEqual(list(source_windows(sizes, 6, 6, 2, 60)), [])

    def test_throughput_options_are_validated(self):
        config = {"transform": {}}
        self.assertIsNone(throughput_options(config)["workers"])
        for name, value in (("workers", -1), ("read_threads", 0), ("workers", True)):
            with self.subTest(name=name), self.assertRaises(ValueError):
                throughput_options({"transform": {name: value}})


class DiscoveryTests(unittest.TestCase):
    def setUp(self):
        environment = patch.dict(os.environ, {"BASIN_SOURCE_MANIFEST": ""})
        environment.start()
        self.addCleanup(environment.stop)
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name) / "raw"
        self.config = load_config()
        self.config["paths"]["raw"] = str(self.root)
        self.config["data"]["generated_dates"] = ["2026-10-31", "2026-10-30"]
        # Names chosen so that file-system order, plain string order and pathlib's
        # part-wise, case-insensitive-on-Windows order all disagree.
        for index, key in enumerate(
            (
                "2026-10-30/table_ab/data.parquet",
                "2026-10-30/table_a_x/data.parquet",
                "2026-10-30/table_a/data.parquet",
                "2026-10-30/Table_B/data.parquet",
                "2026-10-30/table_a-1/data.parquet",
                "2026-10-30/group/inner/deeper/data.parquet",
                "2026-10-30/group/other/data.parquet",
                "2026-10-31/data.parquet",
                "2026-10-31/table_000010/data.parquet",
                "2026-10-31/table_0000100/data.parquet",
                "2026-10-31/table_000011/data.parquet",
            )
        ):
            path = self.root / key
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_bytes(b"x" * (index + 1))
        (self.root / "2026-10-30/empty").mkdir()
        (self.root / "2026-10-30/table_a/other.parquet").touch()
        (self.root / "2026-10-29/unselected").mkdir(parents=True)
        (self.root / "2026-10-29/unselected/data.parquet").touch()

    def expected(self):
        """The selection as originally defined: a sorted recursive glob per date."""
        return [
            path
            for day in ("2026-10-30", "2026-10-31")
            for path in sorted((self.root / day).rglob("data.parquet"))
        ]

    def test_scan_matches_sorted_recursive_glob(self):
        paths = discover_source_tables(self.config)
        self.assertEqual(list(paths), self.expected())
        self.assertEqual(len(paths), 11)
        # The order in which folders are examined never changes the selection.
        with patch("transform.PROBE_BLOCK", 2):
            self.assertEqual(list(discover_source_tables(self.config)), self.expected())
        self.assertEqual(paths[2:4], self.expected()[2:4])
        self.assertIn(self.expected()[5], paths)

    def test_scan_keeps_what_the_file_system_reported(self):
        paths = discover_source_tables(self.config)
        stats = [path.stat() for path in self.expected()]
        self.assertEqual(paths.sizes.tolist(), [s.st_size for s in stats])
        self.assertEqual(paths.modified.tolist(), [s.st_mtime_ns for s in stats])
        self.assertEqual(
            paths.files(0, len(paths)), [str(path) for path in self.expected()]
        )
        if os.name == "nt":
            self.assertEqual(len(paths.records), len(paths))

    def test_checkpoint_identity_is_the_same_from_scan_or_plain_paths(self):
        scanned = discover_source_tables(self.config)
        processed = Path(self.temporary.name) / "processed"
        with patch(
            "transform_progress.examine",
            side_effect=AssertionError("the scan already examined every file"),
        ):
            fast = TransformCheckpoint(processed, "s", scanned, self.root, "code")
        plain = TransformCheckpoint(processed, "s", list(scanned), self.root, "code")
        self.assertEqual(fast.directory, plain.directory)
        self.assertEqual(list(fast.source_bytes), list(plain.source_bytes))
        # Limiting the tables keeps the recorded facts aligned with the paths.
        some = TransformCheckpoint(processed, "s", scanned[:4], self.root, "code")
        same = TransformCheckpoint(processed, "s", list(scanned)[:4], self.root, "code")
        self.assertEqual(some.directory, same.directory)

    def test_source_tables_behave_like_a_list_of_paths(self):
        keys = ["a/data.parquet", "b/data.parquet"]
        tables = SourceTables(self.root, keys)
        self.assertEqual(tables, [self.root / key for key in keys])
        self.assertNotEqual(tables, [self.root / keys[0]])
        self.assertEqual(tables[-1], self.root / keys[1])
        self.assertIsNone(tables.inventory(self.root))
        self.assertTrue(tables)
        self.assertFalse(SourceTables(self.root, []))


class WorkerEquivalenceTests(unittest.TestCase):
    """Decoding in worker processes writes byte-for-byte what decoding in place does."""

    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.base = Path(self.temporary.name)
        self.config = copy.deepcopy(load_config())
        self.config["data"].pop("generated_dates", None)
        self.config["data"].update(horizon=16, context_length=64, max_tables=None)
        self.config["model"].update(context_length=64, max_patches=16)
        self.config["validation"]["enabled"] = False
        self.config["transform"] = {"batch_size_tables": 7, "batch_memory_limit_mb": 1}
        self.config["paths"].update(raw=str(self.base / "raw"))
        logger.disable("transform")
        self.addCleanup(logger.enable, "transform")
        random = np.random.default_rng(11)
        for index, rows in enumerate([80, 20, 333, 64, 200, 97, 150, 48, 1000, 81]):
            frame = pd.DataFrame(
                {
                    "time_index": np.arange(rows),
                    "outcome": random.normal(size=rows),
                    "covariate_01": random.integers(0, 9, size=rows),
                    "outcome_lag_1": random.normal(size=rows),
                }
            )
            if index % 3 == 0:
                frame["covariate_02"] = random.normal(size=rows)
                frame.loc[frame.index[::7], "covariate_02"] = np.nan
            path = self.base / "raw" / f"table_{index:02d}" / "data.parquet"
            path.parent.mkdir(parents=True)
            frame.to_parquet(path, index=False, compression="zstd")

    def run_transform(self, name, policy, workers):
        config = copy.deepcopy(self.config)
        config["data"]["validation_policy"] = policy
        config["transform"]["workers"] = workers
        config["paths"]["processed"] = str(self.base / name)
        transform_next_patch(config)
        index = torch.load(self.base / name / "windows.pt", weights_only=True)
        shards = [
            hashlib.sha256((self.base / name / s["path"]).read_bytes()).hexdigest()
            for s in index["shards"]
        ]
        self.assertEqual(shards, [s["sha256"] for s in index["shards"]])
        manifest = json.loads((self.base / name / "manifest.json").read_text())
        del manifest["config"]
        return index["record_count"], shards, manifest

    def test_worker_processes_and_in_place_decoding_agree(self):
        for policy in ("sampled_training_origins", "reserved_tail"):
            with self.subTest(policy=policy):
                in_place = self.run_transform(f"in_place_{policy}", policy, workers=0)
                workers = self.run_transform(f"workers_{policy}", policy, workers=2)
                self.assertEqual(in_place, workers)
                self.assertGreater(len(in_place[1]), 1)
                excluded = in_place[2]["excluded_tables"]
                self.assertEqual(len(excluded), 1)  # the 20-row table
                self.assertEqual(in_place[2]["table_count"], 9)

    def test_decoding_errors_name_the_table_like_before(self):
        frame = pd.DataFrame({"time_index": [0, 2, 1] * 30, "outcome": range(90)})
        frame.to_parquet(self.base / "raw/table_03/data.parquet", index=False)
        for workers in (0, 2):
            config = copy.deepcopy(self.config)
            config["transform"]["workers"] = workers
            config["paths"]["processed"] = str(self.base / f"invalid_{workers}")
            with (
                self.subTest(workers=workers),
                self.assertRaisesRegex(ValueError, "Invalid time ordering: .*table_03"),
            ):
                transform_next_patch(config)


if __name__ == "__main__":
    unittest.main()

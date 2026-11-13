"""Native snapshots preserve recursive source selection without WSL metadata walks."""

import json
import os
import sys
import tempfile
import unittest
from contextlib import nullcontext
from pathlib import Path
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src/03_transforming_input_data"))
sys.path.insert(0, str(ROOT / "src/02_data_validation"))
sys.path.insert(0, str(ROOT / "src/05_model_training"))
from transform import discover_source_tables, load_config, write_source_manifest


class SourceDiscoveryTests(unittest.TestCase):
    @unittest.skipUnless(os.name == "nt", "Windows-to-WSL launcher")
    def test_validation_and_transform_forward_native_index_to_wsl(self):
        import transform
        import validate_generated_data

        with tempfile.TemporaryDirectory() as directory:
            config, _, _ = self.fixture(directory)
            config["validation"]["enabled"] = True
            config_path = Path(directory) / "config.json"
            config_path.write_text(json.dumps(config))
            manifest = ROOT / ".cache/source_indexes/launcher-test.json"
            for entry in (validate_generated_data.main, transform.main):
                with (
                    self.subTest(entry=entry.__module__),
                    patch.object(
                        sys, "argv", ["stage.py", "--config", str(config_path)]
                    ),
                    patch("transform.run_context", return_value=nullcontext()),
                    patch(
                        "transform.write_source_manifest", return_value=manifest
                    ) as scan,
                    patch("train.launch_wsl", return_value=True) as launch,
                ):
                    entry()
                    scan.assert_called_once()
                    self.assertEqual(
                        launch.call_args.kwargs["extra_args"],
                        (
                            "--source-manifest",
                            ".cache/source_indexes/launcher-test.json",
                        ),
                    )

    def setUp(self):
        environment = patch.dict(os.environ, {"BASIN_SOURCE_MANIFEST": ""})
        environment.start()
        self.addCleanup(environment.stop)

    def fixture(self, directory):
        config = load_config()
        root = Path(directory)
        config["paths"]["raw"] = str(root / "raw")
        config["data"]["generated_dates"] = ["2026-10-31", "2026-10-30"]
        expected = []
        for key in (
            "2026-10-30/table_a/data.parquet",
            "2026-10-31/table_b/nested/data.parquet",
        ):
            path = root / "raw" / key
            path.parent.mkdir(parents=True)
            path.write_bytes(b"fixture")
            expected.append(path)
        (root / "raw/2026-10-31/empty").mkdir()
        (root / "raw/2026-10-31/table_b/other.parquet").touch()
        with patch("transform.ROOT", root):
            manifest = write_source_manifest(config)
        return config, manifest, expected

    def test_recursive_order_matches_native_snapshot_and_environment_lookup(self):
        with tempfile.TemporaryDirectory() as directory:
            config, manifest, expected = self.fixture(directory)
            self.assertEqual(discover_source_tables(config), expected)
            self.assertEqual(
                discover_source_tables(config, manifest=manifest), expected
            )
            with patch.dict(os.environ, {"BASIN_SOURCE_MANIFEST": str(manifest)}):
                self.assertEqual(discover_source_tables(config), expected)

    def test_loading_does_not_rescan_or_stat_each_file(self):
        with tempfile.TemporaryDirectory() as directory:
            config, manifest, expected = self.fixture(directory)
            with (
                patch.object(
                    Path, "rglob", side_effect=AssertionError("must not rescan")
                ),
                patch.object(
                    Path,
                    "resolve",
                    side_effect=AssertionError("must not resolve every path"),
                ),
                patch.object(
                    Path, "is_file", side_effect=AssertionError("must not stat files")
                ),
            ):
                self.assertEqual(
                    discover_source_tables(config, manifest=manifest), expected
                )

    def test_membership_is_frozen_only_for_this_run(self):
        with tempfile.TemporaryDirectory() as directory:
            config, manifest, expected = self.fixture(directory)
            added = Path(directory) / "raw/2026-10-31/later/data.parquet"
            added.parent.mkdir()
            added.touch()
            self.assertEqual(
                discover_source_tables(config, manifest=manifest), expected
            )
            self.assertIn(added, discover_source_tables(config))

    def test_different_root_or_dates_rejects_snapshot(self):
        with tempfile.TemporaryDirectory() as directory:
            config, manifest, _ = self.fixture(directory)
            config["data"]["generated_dates"] = ["2026-10-31"]
            with self.assertRaisesRegex(ValueError, "does not match"):
                discover_source_tables(config, manifest=manifest)
            config["data"]["generated_dates"] = ["2026-10-31", "2026-10-30"]
            config["paths"]["raw"] = str(Path(directory) / "different")
            with self.assertRaisesRegex(ValueError, "does not match"):
                discover_source_tables(config, manifest=manifest)

    def test_invalid_paths_duplicates_and_unselected_dates_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            config, manifest, _ = self.fixture(directory)
            original = json.loads(manifest.read_text())
            for keys in (
                ["../outside/data.parquet"],
                # Rejection fixture only; no file is created or opened.
                ["/tmp/data.parquet"],  # nosec B108
                ["C:/outside/data.parquet"],
                ["2026-10-31\\table\\data.parquet"],
                ["2026-10-29/table/data.parquet"],
                ["2026-10-31/table/other.parquet"],
                ["2026-10-31/table/data.parquet"] * 2,
                [],
            ):
                manifest.write_text(json.dumps({**original, "paths": keys}))
                with self.subTest(keys=keys), self.assertRaises(ValueError):
                    discover_source_tables(config, manifest=manifest)


if __name__ == "__main__":
    unittest.main()

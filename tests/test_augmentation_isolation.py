"""Every member of a copied realization is training-only."""

import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
for stage in ("03_transforming_input_data", "05_model_training", "06_holdout_test"):
    sys.path.insert(0, str(ROOT / "src" / stage))

from augmentation_families import excluded_evaluation_sources
from holdout_test import evaluation_records
from training_protocol import split_datasets
from transform import load_config


class AugmentationIsolationTests(unittest.TestCase):
    def test_directory_alias_preserves_original_family_identity(self):
        with tempfile.TemporaryDirectory() as directory:
            base = Path(directory)
            (base / "original").mkdir()
            (base / "original_sign_inverted").mkdir()
            try:
                (base / "alias").symlink_to(base / "original", target_is_directory=True)
            except OSError as error:
                self.skipTest(f"Directory symlinks unavailable: {error}")
            source = str(base / "alias/data.parquet")
            self.assertEqual(excluded_evaluation_sources([source]), {source})

    def test_path_resolution_scales_with_date_folders_not_source_count(self):
        with tempfile.TemporaryDirectory() as directory:
            base = Path(directory)
            for date in ("date_a", "date_b"):
                (base / date).mkdir()
            (base / "date_a/original_sign_inverted").mkdir()
            sources = [
                str(base / date / f"table_{i}" / "data.parquet")
                for date in ("date_a", "date_b")
                for i in range(10000)
            ]
            sources.append(str(base / "date_a/original/data.parquet"))
            resolve = Path.resolve
            with patch.object(
                Path, "resolve", autospec=True, side_effect=resolve
            ) as resolver:
                excluded = excluded_evaluation_sources(sources)
                self.assertEqual(resolver.call_count, 2)
            self.assertEqual(excluded, {sources[-1]})

    def test_relative_and_absolute_date_prefixes_share_directory_scan(self):
        with tempfile.TemporaryDirectory() as directory:
            base = Path(directory)
            (base / "original").mkdir()
            (base / "original_coarse_2_0").mkdir()
            sources = [str(base / "original/data.parquet"), "original/data.parquet"]
            import augmentation_families

            scandir = augmentation_families.os.scandir
            with (
                patch.object(augmentation_families, "ROOT", base),
                patch.object(
                    augmentation_families.os, "scandir", wraps=scandir
                ) as scan,
            ):
                self.assertEqual(excluded_evaluation_sources(sources), set(sources))
                self.assertEqual(scan.call_count, 1)

    def fixtures(self, root):
        names = [
            "original",
            "original_coarse_3_1",
            "mirrored",
            "mirrored_sign_inverted",
        ]
        names += [f"independent_{i}" for i in range(12)]
        sources = []
        for name in names:
            directory = root / name
            directory.mkdir()
            sources.append(str(directory / "data.parquet"))
        # Original must be protected even though this copy was not admitted.
        (root / "independent_0_coarse_2_0").mkdir()
        return sources, set(sources[:4] + sources[4:5])

    def test_split_excludes_originals_and_copies_but_retains_training(self):
        with tempfile.TemporaryDirectory() as directory:
            sources, excluded = self.fixtures(Path(directory))
            self.assertEqual(excluded_evaluation_sources(sources), excluded)
            records = [
                dict(source=s, source_rows=192, source_sha256=s, split=split, id=s)
                for s in sources
                for split in ("train", "evaluation")
            ]
            config = load_config()
            config["training"]["holdout_fraction"] = 0.25
            config["training"]["validation_sampling"]["tables"] = 3
            for policy in ("reserved_tail", "sampled_training_origins"):
                config["data"]["validation_policy"] = policy
                train, validation, test, audit = split_datasets(records, config)
                self.assertTrue(excluded <= {r["source"] for r in train})
                self.assertFalse(excluded & {r["source"] for r in validation + test})
                self.assertEqual(
                    set(audit["evaluation_excluded_family_sources"]), excluded
                )
                self.assertEqual(audit["test_tables"], 3)
                if policy == "sampled_training_origins":
                    self.assertFalse(
                        {r["source"] for r in train} & {r["source"] for r in validation}
                    )
                    self.assertEqual(audit["validation_tables"], 3)
                reverse = split_datasets(list(reversed(records)), config)
                self.assertEqual(audit["test_sources"], reverse[3]["test_sources"])

    def test_old_checkpoint_holdout_is_filtered_before_sampling(self):
        with tempfile.TemporaryDirectory() as directory:
            sources, excluded = self.fixtures(Path(directory))
            metadata = {
                "dataset_split": {
                    "test_sources": sources,
                    "train_sources": [],
                    "source_sha256": {s: s for s in sources},
                }
            }
            actual = evaluation_records(metadata, metadata)
            self.assertEqual({r["source"] for r in actual}, set(sources) - excluded)

    def test_aggregates_exclude_the_whole_family(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            names = [
                "original",
                "original_aggregate_min_2",
                "original_aggregate_max_3",
                "original_aggregate_mean_5",
            ]
            for name in names:
                (root / name).mkdir()
            sources = [str(root / name / "data.parquet") for name in names]
            self.assertEqual(excluded_evaluation_sources(sources), set(sources))

    def test_new_copies_are_detected_without_stale_cache(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "original").mkdir()
            source = str(root / "original" / "data.parquet")
            self.assertEqual(excluded_evaluation_sources([source]), set())
            (root / "original_sign_inverted").mkdir()
            self.assertEqual(excluded_evaluation_sources([source]), {source})


if __name__ == "__main__":
    unittest.main()

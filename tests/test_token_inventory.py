"""Distinct content, source positions, real splits and rolling test coverage."""

import copy
import json
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np
import pandas as pd
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src/05_model_training"))
sys.path.insert(0, str(ROOT / "src/04_creating_embeddings"))
from embed import ForecastConfig, LazyPatchRecord, PatchEncoder
from token_inventory import (
    _Index,
    build_inventory,
    parameter_budget,
    prepare_token_inventory,
    save_inventory,
    update_parameter_ratios,
)
from training_protocol import split_datasets
from transform import file_digest, load_config


class TokenInventoryTests(unittest.TestCase):
    def fixture(self, root, length=64):
        config = copy.deepcopy(load_config())
        config["model"].update(
            width=128, layers=2, variate_heads=4, context_length=32, max_patches=16
        )
        config["data"]["context_length"] = 32
        config["training"]["holdout_fraction"] = 0.25
        settings = ForecastConfig(
            **dict(config["model"], quantiles=tuple(config["model"]["quantiles"]))
        )
        encoder = PatchEncoder(settings)
        records = []
        for i in range(4):
            path = Path(root) / str(i) / "data.parquet"
            path.parent.mkdir(parents=True)
            # Distinct patches by source/time, constant covariates deliberately repeat.
            x = np.arange(length, dtype=np.float32) + i * 1000
            pd.DataFrame({"outcome": x, "covariate": np.ones(length)}).to_parquet(path)
            cutoff, patch = length - 16, 16
            metadata = {
                "source": str(path),
                "source_key": f"{i}/data.parquet",
                "source_sha256": file_digest(path),
                "source_rows": length,
                "targets": ["outcome"],
                "variable_names": ["outcome", "covariate"],
                "roles": {"past": ["covariate"]},
            }
            # Cover all training transitions, including chunk boundaries.
            n = (cutoff + (-length) % patch) // patch
            for token_start in range(0, n - 1, 2):
                start = max(0, token_start * patch - (-length) % patch)
                stop = min(cutoff, (token_start + 3) * patch - (-length) % patch)
                batch = {
                    "context": torch.tensor(x[start:stop])[None, None],
                    "past_covariates": torch.ones(1, 1, stop - start),
                }
                prepared = encoder.tokenize_next(**batch, training=True)
                records.append(
                    {
                        **metadata,
                        "split": "train",
                        "batch": batch,
                        "prepared": prepared,
                        "context_rows": [start, stop],
                    }
                )
            start = max(0, cutoff - 32)
            batch = {
                "context": torch.tensor(x[start:cutoff])[None, None],
                "past_covariates": torch.ones(1, 1, cutoff - start),
                "future_target": torch.tensor(x[cutoff:])[None, None],
            }
            prepared = encoder.tokenize_next(
                batch["context"], past_covariates=batch["past_covariates"]
            )
            records.append(
                {
                    **metadata,
                    "split": "evaluation",
                    "batch": batch,
                    "prepared": prepared,
                    "cutoff": cutoff,
                    "context_rows": [start, cutoff],
                }
            )
        return config, {
            "signature": "fixture",
            "processed_sha256": "fixture",
            "records": records,
        }

    def test_exact_counts_duplicates_and_real_split(self):
        with tempfile.TemporaryDirectory() as root:
            config, data = self.fixture(root)
            report = build_inventory(data, config)
            train, val, test = [
                report["splits"][s] for s in ("training", "validation", "test")
            ]
            self.assertEqual(
                (train["tables"], val["tables"], test["tables"]), (3, 3, 1)
            )
            # Two input patches * two variables * three training tables.
            self.assertEqual(train["input_token_occurrences"], 12)
            self.assertEqual(train["unique_input_source_patches"], 12)
            self.assertEqual(
                train["unique_input_contents"], 7
            )  # 6 outcome + one constant covariate
            self.assertEqual(train["unique_target_contents"], 6)
            self.assertEqual(val["unique_input_contents"], 7)
            self.assertEqual(val["unique_target_contents"], 3)
            # Test origins 32 and 48, each with a 32-row context; source patch 1 repeats.
            self.assertEqual(test["windows"], 2)
            self.assertEqual(test["input_token_occurrences"], 8)
            self.assertEqual(test["unique_input_source_patches"], 6)
            self.assertEqual(test["unique_input_contents"], 4)
            self.assertEqual(test["unique_target_contents"], 2)
            self.assertEqual(test["observed_target_points"], 32)
            self.assertEqual(
                report["overlap_unique"]["training__test"]["input_source_patches"], 0
            )
            self.assertEqual(
                report["overlap_unique"]["training__test"]["input_contents"], 1
            )
            self.assertEqual(
                report["overlap_unique"]["training__validation"][
                    "input_source_patches"
                ],
                6,
            )
            self.assertEqual(val["input_contents_not_in_training"], 3)
            # Replay does not create unique data (nor extra tables).
            groups = split_datasets(data["records"], config)
            replay = build_inventory(data, config, groups=(groups[0] * 2, *groups[1:]))
            self.assertEqual(
                replay["splits"]["training"]["input_token_occurrences"], 24
            )
            self.assertEqual(replay["splits"]["training"]["unique_input_contents"], 7)

    def test_missing_padding_roles_and_canonicalization(self):
        with tempfile.TemporaryDirectory() as root:
            index = _Index(Path(root) / "index.sqlite", 16)
            try:
                for split, values, role, start in (
                    ("training", np.zeros(16), 0, 0),
                    ("training", -np.zeros(16), 0, 16),
                    ("training", np.zeros(16), 1, 32),
                    ("training", np.full(16, np.nan), 0, 48),
                    ("validation", np.r_[np.nan, np.zeros(15)], 0, 0),
                ):
                    index.add(split, values, role, "table", "var", start, start + 16)
                counts, _, _ = index.summarize()
                self.assertEqual(counts["training"]["unique_input_contents"], 2)
                self.assertEqual(counts["training"]["unique_input_source_patches"], 3)
                self.assertEqual(counts["training"]["empty_input_patches_excluded"], 1)
                self.assertEqual(counts["validation"]["partial_input_occurrences"], 1)
                self.assertEqual(
                    counts["validation"]["input_contents_not_in_training"], 1
                )
            finally:
                index.db.close()

    def test_partial_tail_and_chunk_padding(self):
        with tempfile.TemporaryDirectory() as root:
            config, data = self.fixture(root, length=83)
            report = build_inventory(data, config)
            test = report["splits"]["test"]
            # Origins 32,48,64,80; last target has 3 real observations.
            self.assertEqual(test["windows"], 4)
            self.assertEqual(test["observed_target_points"], 51)
            self.assertEqual(test["prediction_target_occurrences"], 4)
            self.assertEqual(test["input_token_occurrences"], 16)
            self.assertEqual(test["unique_input_source_patches"], 10)
            self.assertEqual(test["unique_input_contents"], 6)
            self.assertGreater(
                report["splits"]["training"]["partial_input_occurrences"], 0
            )

    def test_analytic_parameter_budget_matches_measured_architectures(self):
        config = load_config()
        for width, layers, expected in ((960, 19, 301774368), (1024, 25, 447482144)):
            config["model"].update(width=width, layers=layers)
            self.assertEqual(parameter_budget(config)["parameters"], expected)

    def test_reports_and_measured_parameter_ratios(self):
        with tempfile.TemporaryDirectory() as root:
            config, data = self.fixture(root)
            report = build_inventory(data, config)
            update_parameter_ratios(report, 100)
            self.assertEqual(report["model"]["method"], "measured")
            self.assertAlmostEqual(
                report["splits"]["training"]["unique_input_contents_per_parameter"],
                0.07,
            )
            save_inventory(report, Path(root) / "report")
            self.assertEqual(
                json.loads((Path(root) / "report/token_inventory.json").read_text()),
                report,
            )
            csv = pd.read_csv(Path(root) / "report/token_inventory.csv")
            self.assertEqual(csv.split.tolist(), ["training", "validation", "test"])
            self.assertIn(
                "embedding width: 128",
                (Path(root) / "report/token_inventory.txt").read_text(),
            )

    def test_training_writes_inventory_before_model_construction(self):
        import train

        with tempfile.TemporaryDirectory() as root:
            config, data = self.fixture(root)
            config["training"]["token_inventory"] = {"mode": "exact"}
            config["paths"].update(
                training=str(Path(root) / "training"), logs=str(Path(root) / "logs")
            )

            def stop_before_cuda(settings):
                report = json.loads(
                    (Path(root) / "training/token_inventory.json").read_text()
                )
                self.assertEqual(
                    report["splits"]["training"]["unique_input_contents"], 7
                )
                raise RuntimeError("preflight verified")

            with (
                patch.object(train, "read_artifact", return_value=data),
                patch.object(train, "file_digest", return_value="fixture"),
                patch.object(train, "RWKVForecaster", side_effect=stop_before_cuda),
                patch.object(train.torch.cuda, "synchronize"),
            ):
                with self.assertRaisesRegex(RuntimeError, "preflight verified"):
                    train.train(config)

    def test_stale_holdout_source_fails(self):
        with tempfile.TemporaryDirectory() as root:
            config, data = self.fixture(root)
            test = split_datasets(data["records"], config)[2][0]
            pd.DataFrame({"outcome": np.arange(64)}).to_parquet(test["source"])
            with self.assertRaisesRegex(ValueError, "Holdout source changed"):
                build_inventory(data, config)

    def test_default_startup_never_builds_missing_inventory(self):
        import token_inventory
        import train

        with tempfile.TemporaryDirectory() as root:
            config, data = self.fixture(root)
            config["paths"].update(
                training=str(Path(root) / "training"), logs=str(Path(root) / "logs")
            )
            groups = split_datasets(data["records"], config)
            with patch.object(
                token_inventory,
                "build_inventory",
                side_effect=AssertionError("must not scan"),
            ):
                self.assertIsNone(prepare_token_inventory(data, config, groups=groups))
                data["format"] = (
                    "lazy_patch_index_v1"  # Loader has already verified its checksum.
                )
                with (
                    patch.object(train, "read_artifact", return_value=data),
                    patch.object(
                        train,
                        "file_digest",
                        side_effect=AssertionError("redundant checksum scan"),
                    ),
                    patch.object(
                        train,
                        "RWKVForecaster",
                        side_effect=RuntimeError("reached model initialization"),
                    ),
                    patch.object(train.torch.cuda, "synchronize"),
                ):
                    with self.assertRaisesRegex(
                        RuntimeError, "reached model initialization"
                    ):
                        train.train(config)
            status = json.loads(
                (Path(root) / "training/token_inventory_status.json").read_text()
            )
            self.assertEqual(status["status"], "not_computed")
            self.assertFalse(status["corpus_scan_performed"])
            self.assertFalse((Path(root) / "training/token_inventory.json").exists())

    def test_cached_inventory_reused_only_for_same_artifact_and_partition(self):
        import token_inventory

        with tempfile.TemporaryDirectory() as root:
            config, data = self.fixture(root)
            config["paths"]["training"] = str(Path(root) / "training")
            groups = split_datasets(data["records"], config)
            exact = build_inventory(data, config, groups=groups)
            save_inventory(exact, config["paths"]["training"])
            with patch.object(
                token_inventory,
                "build_inventory",
                side_effect=AssertionError("must reuse cache"),
            ):
                cached = prepare_token_inventory(data, config, groups=groups)
                self.assertEqual(cached["splits"], exact["splits"])
                other_data = dict(data, processed_sha256="changed")
                self.assertIsNone(
                    prepare_token_inventory(other_data, config, groups=groups)
                )
                changed = (*groups[:3], dict(groups[3], seed=999))
                self.assertIsNone(prepare_token_inventory(data, config, groups=changed))

    def test_exact_inventory_does_not_tokenize_lazy_records(self):
        with tempfile.TemporaryDirectory() as root:
            config, data = self.fixture(root)
            encoder = PatchEncoder(
                ForecastConfig(
                    **dict(
                        config["model"], quantiles=tuple(config["model"]["quantiles"])
                    )
                )
            )
            records = [
                LazyPatchRecord(
                    {k: v for k, v in record.items() if k != "prepared"}, encoder
                )
                for record in data["records"]
            ]
            with patch.object(
                encoder,
                "tokenize_next",
                side_effect=AssertionError("audit must use raw patches"),
            ):
                report = build_inventory(dict(data, records=records), config)
            self.assertEqual(report["splits"]["training"]["unique_input_contents"], 7)


if __name__ == "__main__":
    unittest.main()

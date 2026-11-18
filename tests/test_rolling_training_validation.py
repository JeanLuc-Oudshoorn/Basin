"""Dataset-disjoint validation, full training labels, and streamed evaluation."""

import copy
import sys
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np
import pandas as pd
import torch

ROOT = Path(__file__).resolve().parents[1]
for stage in (
    "03_transforming_input_data",
    "04_patch_indexing",
    "05_model_training",
):
    sys.path.insert(0, str(ROOT / "src" / stage))
import gc

from epoch_training import fit_epochs
from index_patches import ForecastConfig, LazyPatchRecord, build_patch_index
from rolling_validation import (
    PANEL_INDEX,
    PANEL_VALUES,
    select_validation_origins,
    validate_origins,
    validation_windows,
)
from test_training_performance import CausalBlock
from token_inventory import build_inventory
from train import RWKVForecaster
from training_protocol import split_datasets
from transform import load_config, read_artifact, transform_next_patch


class RollingTrainingValidationTests(unittest.TestCase):
    def check_validation_panel(self, model, validation, config, expected, root):
        """A saved panel must give every later check exactly the same windows."""
        panel = root / "panel"
        direct = list(validation_windows(validation, config))
        first = list(validation_windows(validation, config, cache_directory=panel))
        self.assertTrue((panel / PANEL_VALUES).is_file())
        self.assertTrue((panel / PANEL_INDEX).is_file())
        saved = (panel / PANEL_VALUES).read_bytes()
        # The second pass must not open a source table at all.
        with patch("rolling_validation.pd.read_parquet", side_effect=AssertionError):
            with patch("rolling_validation.file_digest", side_effect=AssertionError):
                cached = list(
                    validation_windows(validation, config, cache_directory=panel)
                )
        self.assertEqual(len(cached), len(direct))
        for reference, built, loaded in zip(direct, first, cached):
            self.assertEqual(set(reference["batch"]), set(loaded["batch"]))
            self.assertEqual(
                {k: v for k, v in reference.items() if k != "batch"},
                {k: v for k, v in loaded.items() if k != "batch"},
            )
            for name, value in reference["batch"].items():
                for other in (built["batch"][name], loaded["batch"][name]):
                    if isinstance(value, torch.Tensor):
                        self.assertEqual(value.dtype, other.dtype)
                        self.assertEqual(value.shape, other.shape)
                        self.assertTrue(
                            torch.equal(
                                value.view(torch.int32), other.view(torch.int32)
                            )
                        )
                    else:
                        self.assertEqual(value, other)
        self.assertEqual((panel / PANEL_VALUES).read_bytes(), saved)
        # Validation through the panel reports exactly what it reported without.
        cached_config = dict(
            config,
            paths=dict(config["paths"], training=str(root / "panel_training")),
            training=dict(config["training"], validation_cache=True),
        )
        building = validate_origins(model, validation, cached_config)
        reading = validate_origins(model, validation, cached_config)
        for result in (building, reading):
            for key in ("loss", "rmse", "mae", "mase", "points", "loss_mass"):
                self.assertEqual(result[key], expected[key], key)
            self.assertEqual(
                [{k: v for k, v in row.items()} for row in result["tables"]],
                expected["tables"],
            )
        # A folder path keeps the panel elsewhere, one subfolder per training run.
        elsewhere = dict(
            cached_config,
            training=dict(config["training"], validation_cache=str(root / "caches")),
        )
        self.assertEqual(
            validate_origins(model, validation, elsewhere)["loss"], expected["loss"]
        )
        (subfolder,) = (root / "caches").iterdir()
        self.assertTrue((subfolder / PANEL_VALUES).is_file())
        self.assertFalse((root / "panel_training" / "caches").exists())
        # Another request list is another panel: it is rebuilt, not reused.
        # (Windows cannot replace a file that is still mapped; drop the views.)
        del direct, first, cached, reference, built, loaded, value, other
        gc.collect()
        fewer = validation[:-1]
        rebuilt = list(validation_windows(fewer, config, cache_directory=panel))
        self.assertEqual(len(rebuilt), len(fewer))
        self.assertNotEqual((panel / PANEL_VALUES).read_bytes(), saved)
        # An interrupted first pass leaves no panel behind.
        partial = root / "partial"
        stream = validation_windows(validation, config, cache_directory=partial)
        next(stream)
        stream.close()
        self.assertEqual(list(partial.iterdir()), [])

    def test_full_training_data_five_origins_and_streamed_validation(self):
        config = copy.deepcopy(load_config("configs/legacy_500m.json"))
        config["data"].pop("generated_dates", None)
        config["data"].update(
            validation_policy="sampled_training_origins", context_length=256
        )
        config["model"].update(
            width=64, layers=2, variate_heads=4, context_length=256, max_patches=16
        )
        config["validation"]["enabled"] = False
        config["training"]["validation_sampling"] = {
            "tables": 2,
            "seed": 13,
            "percentiles": [1, 25, 50, 75, 99],
        }
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            config["paths"].update(
                raw=str(root / "raw"),
                processed=str(root / "processed"),
                patch_index=str(root / "patch_index"),
            )
            for i, length in enumerate((48, 160, 192, 224)):
                path = root / "raw" / str(i)
                path.mkdir(parents=True)
                pd.DataFrame(
                    {
                        "time_index": range(length),
                        "outcome": np.arange(length),
                        "cov": np.arange(length) * 2,
                    }
                ).to_parquet(path / "data.parquet")
            payload = transform_next_patch(config)
            self.assertTrue(
                all(
                    "batch" not in r
                    for r in payload["records"]
                    if r["split"] == "evaluation"
                )
            )
            build_patch_index(config)
            indexed = read_artifact(root / "patch_index/patch_index.pt", config)
            self.assertEqual(len(indexed["records"]), len(payload["records"]))
            short_record = next(
                r
                for r in payload["records"]
                if r["source_rows"] == 48 and r["split"] == "train"
            )
            self.assertEqual(short_record["batch"]["context"].shape[-1], 48)
            train, validation, test, audit = split_datasets(payload["records"], config)
            self.assertEqual(len(validation), 10)
            inventory = build_inventory(
                dict(payload, processed_sha256="fixture"),
                config,
                groups=(train, validation, test, audit),
                directory=root,
            )
            self.assertEqual(
                inventory["splits"]["validation"]["observed_target_points"], 640
            )
            self.assertEqual(
                inventory["overlap_unique"]["training__validation"][
                    "target_source_patches"
                ],
                0,
            )
            self.assertTrue(
                {r["source"] for r in validation}.isdisjoint(
                    {r["source"] for r in train}
                )
            )
            self.assertFalse(
                {r["source"] for r in validation} & {r["source"] for r in test}
            )
            for record in train:
                self.assertEqual(record["cutoff"], record["source_rows"])
                self.assertEqual(
                    record["batch"]["context"].max(), record["source_rows"] - 1
                )
            metadata = [
                r
                for r in payload["records"]
                if r["split"] == "evaluation"
                and r["source"] in {x["source"] for x in validation}
            ]
            reversed_panel, _ = select_validation_origins(
                list(reversed(metadata)), config
            )
            self.assertEqual(
                [(r["source"], r["validation_origin"]) for r in validation],
                [(r["source"], r["validation_origin"]) for r in reversed_panel],
            )
            for record in validation_windows(validation, config):
                origin = record["validation_origin"]
                self.assertLess(record["batch"]["context"].max(), origin)
                self.assertEqual(
                    record["batch"]["future_target"].flatten().tolist(),
                    list(range(origin, origin + 64)),
                )
            settings = ForecastConfig(
                **dict(config["model"], quantiles=tuple(config["model"]["quantiles"]))
            )
            with patch(
                "train.load_rwkv", return_value=SimpleNamespace(Block=CausalBlock)
            ):
                model = RWKVForecaster(settings).train()
            serial = validate_origins(
                model,
                validation,
                dict(
                    config, training=dict(config["training"], validation_batch_size=1)
                ),
            )
            batched = validate_origins(model, validation, config)
            self.assertTrue(model.training)
            self.assertFalse(audit["validation_sampling"]["in_sample"])
            self.assertTrue(audit["validation_sampling"]["dataset_disjoint"])
            self.assertFalse(batched["in_sample"])
            self.assertEqual(
                set(audit["validation_sources"]), {r["source"] for r in validation}
            )
            self.assertEqual(audit["train_windows"], len(train))
            self.assertEqual(
                set(audit["train_sources"])
                | set(audit["validation_sources"])
                | set(audit["test_sources"]),
                {r["source"] for r in payload["records"]},
            )
            self.assertEqual(batched["forecast_count"], 10)
            self.assertEqual(batched["points"], 640)
            self.assertAlmostEqual(serial["loss"], batched["loss"], places=5)
            self.assertAlmostEqual(serial["rmse"], batched["rmse"], places=4)
            self.check_validation_panel(model, validation, config, batched, root)
            config["paths"]["training"] = str(root / "training")
            config["training"].update(
                steps=2,
                epochs=None,
                resume=False,
                validate_before_training=False,
                checkpoint_interval_seconds=1e-9,
                validation_interval_seconds=1e-9,
                batch_size=1,
                gradient_accumulation_steps=2,
            )
            optimizer = torch.optim.AdamW(
                model.parameters(), lr=config["training"]["learning_rate"]
            )
            _, summary, _ = fit_epochs(
                model,
                [LazyPatchRecord(r, model) for r in train],
                validation,
                config,
                optimizer,
            )
            self.assertEqual(summary["completed_updates"], 2)
            checkpoint = torch.load(
                root / "training/latest_checkpoint.pt", weights_only=True
            )
            self.assertEqual(checkpoint["format"], "rwkv_timed_checkpoint_v1")
            self.assertTrue(checkpoint["optimizer_state_dict"]["state"])
            self.assertEqual(checkpoint["loop_state"]["validation_checks"], 2)

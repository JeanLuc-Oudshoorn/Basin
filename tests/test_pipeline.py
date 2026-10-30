"""CPU tests for the persisted stage boundaries and leakage protection."""

import copy
import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np
import pandas as pd
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src/04_creating_embeddings"))
from embed import ForecastConfig, PatchEncoder, embed
from transform import load_config, read_artifact, signature, transform


class PipelineTests(unittest.TestCase):
    def test_label_isolation_and_gradient_through_saved_features(self):
        encoder = PatchEncoder(ForecastConfig(patch_size=4))
        context = torch.arange(17, dtype=torch.float32)[None, None]
        first = encoder.tokenize(context, 7, future_target=torch.ones(1, 1, 7))
        second = encoder.tokenize(context, 7, future_target=torch.ones(1, 1, 7) * 999)
        torch.testing.assert_close(
            first["features"], second["features"], rtol=0, atol=0
        )
        encoder.encode_tokens(first).square().mean().backward()
        self.assertGreater(float(encoder.embedding.skip.weight.grad.abs().sum()), 0)
        # Forecast target patches contain no observed target values.
        self.assertTrue((first["features"][:, :, first["cp"] :, 4:8] == 0).all())

    def test_saved_stages_and_temporal_holdout(self):
        with tempfile.TemporaryDirectory() as temporary:
            base = Path(temporary)
            config = load_config("configs/smoke_masked_horizon.json")
            config["paths"].update(
                {k: str(base / k) for k in ("raw", "processed", "embedded")}
            )
            config["data"].update(
                context_length=17,
                horizon=7,
                stride=5,
                max_tables=2,
                known_covariates=["calendar"],
            )
            config["model"]["patch_size"] = 4
            (base / "raw/table").mkdir(parents=True)
            frame = pd.DataFrame(
                {
                    "time_index": np.arange(80),
                    "outcome": np.sin(np.arange(80)),
                    "outcome_lag_1": np.cos(np.arange(80)),
                    "calendar": np.arange(80) % 7,
                }
            )
            frame.to_parquet(base / "raw/table/data.parquet", index=False)
            (base / "raw/table_b").mkdir()
            other = pd.DataFrame(
                {
                    "time_index": np.arange(95),
                    "outcome": np.arange(95),
                    "temperature": np.arange(95) / 10,
                    "pressure": np.ones(95),
                    "calendar": np.arange(95) % 7,
                    "temperature_lag_2": np.zeros(95),
                }
            )
            other.to_parquet(base / "raw/table_b/data.parquet", index=False)
            payload = transform(config)
            for record in payload["records"]:
                if record["split"] == "train":
                    self.assertLessEqual(
                        record["cutoff"] + config["data"]["horizon"],
                        record["evaluation_cutoff"],
                    )
                self.assertFalse(
                    any("lag" in name for name in record["variable_names"])
                )
                exported = pd.read_parquet(
                    base / "processed" / f"{record['id']}.parquet"
                )
                self.assertEqual(len(exported), 24)
                self.assertFalse(any("lag" in name for name in exported.columns))
                source = pd.read_parquet(record["source"])
                pd.testing.assert_frame_equal(
                    exported,
                    source[exported.columns]
                    .iloc[record["context_rows"][0] : record["label_rows"][1]]
                    .reset_index(drop=True),
                )
            self.assertFalse(list((base / "processed").glob("*.csv")))
            embed(config)
            saved = read_artifact(base / "embedded/embeddings.pt", config)
            encoder = PatchEncoder(
                ForecastConfig(
                    **dict(
                        config["model"], quantiles=tuple(config["model"]["quantiles"])
                    )
                )
            )
            encoder.load_state_dict(saved["encoder_state"])
            self.assertEqual(
                {r["initial_embeddings"].shape[1] for r in saved["records"]}, {2, 4}
            )
            self.assertEqual({r["source_rows"] for r in saved["records"]}, {80, 95})
            for record in saved["records"]:
                torch.testing.assert_close(
                    encoder.encode_tokens(record["prepared"]),
                    record["initial_embeddings"],
                    rtol=0,
                    atol=0,
                )
            changed = copy.deepcopy(config)
            changed["model"]["patch_size"] = 8
            self.assertNotEqual(signature(config), signature(changed))
            with self.assertRaisesRegex(ValueError, "Stale artifact"):
                read_artifact(base / "embedded/embeddings.pt", changed)
            explicit_lag = copy.deepcopy(config)
            explicit_lag["data"]["past_covariates"] = ["outcome_lag_1"]
            with self.assertRaisesRegex(ValueError, "excluded lag columns"):
                transform(explicit_lag)
            config["data"]["export_processed_tables"] = False
            config["paths"]["processed"] = str(base / "without_exports")
            transform(config)
            self.assertFalse(list((base / "without_exports").glob("*.parquet")))


if __name__ == "__main__":
    unittest.main()

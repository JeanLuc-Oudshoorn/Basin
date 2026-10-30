"""Causal next-token invariants and final-token holdout."""

import copy
import sys
import tempfile
import unittest
from dataclasses import replace
from pathlib import Path

import numpy as np
import pandas as pd
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src/05_model_training"))
from train import ForecastConfig, PatchEncoder, RWKVForecaster
from transform import load_config, transform_next_patch


def settings(**kwargs):
    return ForecastConfig(
        objective="next_patch",
        patch_size=16,
        width=128,
        layers=2,
        context_length=64,
        max_patches=16,
        **kwargs,
    )


class CausalTokenTests(unittest.TestCase):
    def test_prefix_statistics_and_label_isolation(self):
        encoder = PatchEncoder(settings())
        x = torch.randn(1, 1, 49)
        prepared = encoder.tokenize_next(x, training=True)
        prefix = encoder.tokenize_next(x[..., :-16])
        torch.testing.assert_close(
            prepared["features"], prefix["features"], rtol=0, atol=0
        )
        changed = x.clone()
        changed[..., -16:] += 100000
        other = encoder.tokenize_next(changed, training=True)
        torch.testing.assert_close(
            prepared["features"], other["features"], rtol=0, atol=0
        )
        self.assertEqual(prepared["labels"].shape, (1, 1, 3, 16))

    def test_short_padding_long_truncation(self):
        encoder = PatchEncoder(settings())
        short = encoder.tokenize_next(torch.ones(1, 1, 1))
        self.assertEqual(short["tokens"], 1)
        self.assertEqual(encoder.encode_tokens(short).shape, (1, 1, 16, 128))
        self.assertEqual(int(short["features"][..., 16:32].sum()), 1)
        x, cov = torch.randn(1, 1, 101), torch.randn(1, 3, 101)
        long = encoder.tokenize_next(x, past_covariates=cov)
        tail = encoder.tokenize_next(x[..., -64:], past_covariates=cov[..., -64:])
        torch.testing.assert_close(long["features"], tail["features"], rtol=0, atol=0)

    def test_2048_cap_discards_entire_old_prefix(self):
        encoder = PatchEncoder(
            replace(settings(), context_length=2048, max_patches=128)
        )
        x, cov = torch.randn(1, 1, 2500), torch.randn(1, 2, 2500)
        x[..., :452] = float("inf")
        cov[..., :452] = float("inf")
        long = encoder.tokenize_next(x, past_covariates=cov)
        tail = encoder.tokenize_next(x[..., -2048:], past_covariates=cov[..., -2048:])
        self.assertEqual(long["tokens"], 128)
        torch.testing.assert_close(long["features"], tail["features"], rtol=0, atol=0)

    def test_all_tables_short_and_chunked_exclude_final_patch(self):
        config = copy.deepcopy(load_config())
        config["data"].pop("generated_dates", None)  # Legacy flat-table fixture.
        config["validation"]["enabled"] = (
            False  # Synthetic split fixture, no screening stage.
        )
        with tempfile.TemporaryDirectory() as temp:
            base = Path(temp)
            config["paths"].update(
                raw=str(base / "raw"), processed=str(base / "processed")
            )
            config["data"].update(context_length=64, max_tables=None)
            config["model"].update(
                context_length=64, width=128, layers=2, max_patches=16, variate_heads=4
            )
            for i, n in enumerate((48, 140)):
                directory = base / "raw" / str(i)
                directory.mkdir(parents=True)
                pd.DataFrame(
                    {
                        "time_index": np.arange(n),
                        "outcome": np.arange(n),
                        "covariate": np.arange(n) * 2,
                        "outcome_lag_1": np.arange(n),
                    }
                ).to_parquet(directory / "data.parquet")
            payload = transform_next_patch(config)
            covered = {0: [], 1: []}
            encoder = PatchEncoder(settings())
            for record in payload["records"]:
                self.assertNotIn("outcome_lag_1", record["variable_names"])
                if record["split"] == "train":
                    prepared = encoder.tokenize_next(
                        record["batch"]["context"], training=True
                    )
                    covered[record["table_index"]] += (
                        prepared["labels"].flatten().tolist()
                    )
                    self.assertLess(
                        float(prepared["labels"].max()), record["evaluation_cutoff"]
                    )
                else:
                    self.assertEqual(
                        record["batch"]["future_target"].flatten().tolist(),
                        list(range(record["source_rows"] - 16, record["source_rows"])),
                    )
            self.assertEqual(covered[0], list(range(16, 32)))
            self.assertEqual(covered[1], list(range(12, 124)))


@unittest.skipUnless(torch.cuda.is_available(), "CUDA required")
class CausalModelTests(unittest.TestCase):
    def test_parallel_prefix_equivalence_checkpoint_and_crop(self):
        torch.manual_seed(3)
        model = RWKVForecaster(settings()).to("cuda", torch.bfloat16).eval()
        x = torch.randn(1, 1, 64, device="cuda")
        cov = torch.randn(1, 2, 64, device="cuda")
        prepared = model.tokenize_next(x, past_covariates=cov, training=True)
        with torch.no_grad():
            parallel = model.forward_encoded(model.encode_tokens(prepared), prepared)
            for i in range(3):
                single = model(
                    x[..., : (i + 1) * 16], 16, past_covariates=cov[..., : (i + 1) * 16]
                )
                torch.testing.assert_close(
                    parallel.quantiles[:, :, i], single.quantiles, atol=0.03, rtol=0.03
                )
            long = torch.cat([torch.randn(1, 1, 17, device="cuda"), x], dim=-1)
            torch.testing.assert_close(
                model.forecast(long, 16).point,
                model.forecast(x, 16).point,
                atol=0,
                rtol=0,
            )
        model.config = replace(model.config, gradient_checkpointing=True)
        model.train()
        result = model.forward_encoded(model.encode_tokens(prepared), prepared)
        result.loss.backward()
        self.assertTrue(torch.isfinite(result.loss))
        self.assertGreater(float(model.embedding.skip.weight.grad.abs().sum()), 0)
        self.assertGreater(float(model.blocks[0].att.output.weight.grad.abs().sum()), 0)


if __name__ == "__main__":
    unittest.main()

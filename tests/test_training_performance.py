"""Batching preserves causal predictions, labels, gradients and table coverage."""

import copy
import sys
import unittest
from dataclasses import replace
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
from batching import make_batches
from checkpoints import checkpoint_identity
from embed import ForecastConfig, LazyPatchRecord
from epoch_training import accumulated_update
from inference import forecast_contexts
from performance import prepare_update, prepared_updates
from train import RWKVForecaster
from training_protocol import validate
from transform import load_config


class CausalBlock(torch.nn.Module):
    """CPU stand-in: causal time mixing, real forecaster embedding/mixing/head."""

    def __init__(self, args, layer):
        super().__init__()
        self.projection = torch.nn.Linear(args.n_embd, args.n_embd)

    def forward(self, x, first):
        count = torch.arange(1, x.shape[1] + 1, device=x.device)[None, :, None]
        return x + self.projection(x.cumsum(1) / count), first


def model_and_records(split="evaluation"):
    torch.manual_seed(41)
    with patch("train.load_rwkv", return_value=SimpleNamespace(Block=CausalBlock)):
        model = RWKVForecaster(
            ForecastConfig(
                objective="next_patch",
                patch_size=16,
                width=64,
                layers=2,
                context_length=512,
                max_patches=32,
            )
        )
    records = []
    for i, length in enumerate((48, 64, 112, 256, 272, 400, 512)):
        context = torch.randn(1, 1, length)
        context[..., 4] = float("nan")
        batch = {
            "context": context,
            "past_covariates": torch.randn(1, 2, length),
            "horizon": 16,
            "future_target": torch.randn(1, 1, 16),
        }
        batch["future_target"][..., :i] = float("nan")
        records.append(
            LazyPatchRecord(
                {"id": str(i), "source": str(i), "split": split, "batch": batch}, model
            )
        )
    return model, records


class TrainingPerformanceTests(unittest.TestCase):
    def test_batched_forecasts_match_separate_contexts_including_missing_and_partial(
        self,
    ):
        model, records = model_and_records()
        contexts = [
            {k: r["batch"][k] for k in ("context", "past_covariates")} for r in records
        ]
        expected = [model.forecast(**c, horizon=16).quantiles[0] for c in contexts]
        actual = list(forecast_contexts(model, contexts, 3))
        self.assertTrue(model.training)
        for a, b in zip(expected, actual):
            torch.testing.assert_close(a, b, atol=2e-6, rtol=2e-5)
        # Changing a different batch member cannot influence the first forecast.
        contexts[1] = {k: torch.randn_like(v) for k, v in contexts[1].items()}
        torch.testing.assert_close(
            list(forecast_contexts(model, contexts, 3))[0], actual[0]
        )

    def test_execution_limits_do_not_invalidate_epoch_resume(self):
        model, records = model_and_records("train")
        optimizer = torch.optim.SGD(model.parameters(), lr=1e-3)
        config = load_config()
        before = checkpoint_identity(model, records, [], config, optimizer)
        config["training"].update(
            microbatch_size=1, microbatch_token_limit=64, cuda_memory_limit_gib=6.0
        )
        after = checkpoint_identity(model, records, [], config, optimizer)
        self.assertEqual(before, after)

    def test_physical_microbatch_split_preserves_update_and_padding(self):
        model, records = model_and_records("train")
        split = copy.deepcopy(model)
        group = [records[:4]]
        whole = prepare_update(group)
        pieces = prepare_update(group, microbatch_size=3, microbatch_token_limit=96)
        self.assertGreater(len(pieces), 1)
        self.assertEqual(sum(p for _, _, p in whole), sum(p for _, _, p in pieces))
        torch.testing.assert_close(
            torch.cat([p[0]["features"] for p in pieces]), whole[0][0]["features"]
        )
        a = accumulated_update(
            model, group, torch.optim.SGD(model.parameters(), lr=1e-3), 1e9, whole
        )
        b = accumulated_update(
            split, group, torch.optim.SGD(split.parameters(), lr=1e-3), 1e9, pieces
        )
        self.assertEqual(a["windows"], b["windows"])
        self.assertEqual(b["microbatches"], len(pieces))
        self.assertAlmostEqual(a["loss"], b["loss"], places=6)
        for left, right in zip(model.parameters(), split.parameters()):
            torch.testing.assert_close(left, right, atol=1e-7, rtol=1e-5)

    def test_selective_checkpointing_preserves_gradients(self):
        baseline, records = model_and_records("train")
        selective = copy.deepcopy(baseline)
        selective.config = replace(selective.config, gradient_checkpointing=True)
        selective.checkpoint_token_threshold = 96
        a = accumulated_update(
            baseline,
            [records[-2:]],
            torch.optim.SGD(baseline.parameters(), lr=1e-3),
            1e9,
        )
        b = accumulated_update(
            selective,
            [records[-2:]],
            torch.optim.SGD(selective.parameters(), lr=1e-3),
            1e9,
        )
        self.assertEqual(a["loss"], b["loss"])
        for left, right in zip(baseline.parameters(), selective.parameters()):
            torch.testing.assert_close(left, right, atol=1e-7, rtol=1e-5)

    def test_batched_validation_preserves_every_last_token_and_metrics(self):
        model, records = model_and_records()
        original = validate(model, records, batch_size=1)
        batched = validate(model, records, batch_size=4)
        self.assertTrue(model.training)
        self.assertEqual(original["points"], batched["points"])
        self.assertAlmostEqual(original["loss"], batched["loss"], places=6)
        self.assertAlmostEqual(original["rmse"], batched["rmse"], places=6)
        before = {t["source"]: t for t in original["tables"]}
        after = {t["source"]: t for t in batched["tables"]}
        self.assertEqual(before.keys(), after.keys())
        for key in before:
            self.assertAlmostEqual(before[key]["loss"], after[key]["loss"], places=6)
            self.assertAlmostEqual(before[key]["rmse"], after[key]["rmse"], places=6)

    def test_batched_training_preserves_causal_loss_and_gradients(self):
        model, records = model_and_records("train")
        combined = copy.deepcopy(model)
        a = accumulated_update(
            model,
            [[r] for r in records[:3]],
            torch.optim.SGD(model.parameters(), lr=1e-3),
            1e9,
        )
        b = accumulated_update(
            combined,
            [records[:3]],
            torch.optim.SGD(combined.parameters(), lr=1e-3),
            1e9,
        )
        self.assertAlmostEqual(a["loss"], b["loss"], places=6)
        for left, right in zip(model.parameters(), combined.parameters()):
            torch.testing.assert_close(left, right, atol=1e-7, rtol=1e-5)

    def test_bucket_shuffle_coverage_and_length_bound(self):
        _, records = model_and_records("train")
        batches = make_batches(records, 4, seed=19, bucket_by_length=True)
        self.assertEqual(
            sorted(r["id"] for b in batches for r in b), list(map(str, range(7)))
        )
        for batch in batches:
            self.assertEqual(len({r["prepared"]["padded_tokens"] for r in batch}), 1)
        again = make_batches(records, 4, seed=19, bucket_by_length=True)
        self.assertEqual(
            [[r["id"] for r in b] for b in batches],
            [[r["id"] for r in b] for b in again],
        )

    def test_prefetch_matches_sync_and_does_not_cache_features(self):
        _, records = model_and_records("train")
        batches = make_batches(records, 2)
        synchronous = list(prepared_updates(batches, 2, prefetch=False))
        asynchronous = list(prepared_updates(batches, 2, prefetch=True))
        self.assertEqual(len(synchronous), len(asynchronous))
        for a, b in zip(synchronous, asynchronous):
            self.assertEqual(a[0], b[0])
            for (pa, la, ca), (pb, lb, cb) in zip(a[2], b[2]):
                self.assertEqual(ca, cb)
                torch.testing.assert_close(pa["features"], pb["features"])
                torch.testing.assert_close(la, lb, equal_nan=True)
        self.assertTrue(all("prepared" not in r for r in records))

    def test_nonfinite_loss_never_updates_weights(self):
        model, records = model_and_records("train")
        optimizer = torch.optim.SGD(model.parameters(), lr=1e-3)
        with (
            patch.object(
                model,
                "forward_encoded",
                return_value=SimpleNamespace(
                    loss=model.embedding.skip.weight.sum() * float("nan")
                ),
            ),
            patch.object(optimizer, "step") as step,
        ):
            with self.assertRaises(FloatingPointError):
                accumulated_update(model, [[records[0]]], optimizer, 1.0)
            step.assert_not_called()


if __name__ == "__main__":
    unittest.main()

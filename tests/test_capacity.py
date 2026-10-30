"""Ragged batch loss must match complete independent-table evaluation."""

import csv
import io
import sys
import unittest
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src/05_model_training"))
from capacity_test import MasterAdamW, evaluate, make_batches, pack_records
from learning_rate import make_scheduler
from train import ForecastConfig, PatchEncoder, RWKVForecaster


def examples():
    encoder = PatchEncoder(
        ForecastConfig(
            objective="next_patch",
            patch_size=16,
            width=128,
            layers=2,
            context_length=512,
            max_patches=32,
        )
    )
    records = []
    for i, n in enumerate([33, 300]):
        context = torch.randn(1, 1, n)
        prepared = encoder.tokenize_next(context, training=True)
        records.append({"split": "train", "source": str(i), "prepared": prepared})
    return encoder, records


class PackingTests(unittest.TestCase):
    def test_schedule_updates_actual_fp32_master_optimizer(self):
        model = torch.nn.Linear(1, 1, bias=False).to(torch.bfloat16)
        with torch.no_grad():
            model.weight.fill_(1)
        settings = {
            "learning_rate": 0.01,
            "weight_decay": 0,
            "scheduler": {
                "name": "warmup_cosine",
                "warmup_steps": 2,
                "min_learning_rate": 0.001,
            },
        }
        optimizer = MasterAdamW(model, settings)
        scheduler, _ = make_scheduler(optimizer.optimizer, settings, 4)
        rates = []
        for _ in range(4):
            rates.append(optimizer.optimizer.param_groups[0]["lr"])
            optimizer.zero_grad()
            model.weight.grad = torch.ones_like(model.weight)
            optimizer.step(1)
            scheduler.step()
        for actual, expected in zip(rates, [0.005, 0.01, 0.0055, 0.001]):
            self.assertAlmostEqual(actual, expected)
        self.assertAlmostEqual(
            float(optimizer.masters[0].detach()), 1 - sum(rates), places=6
        )

    def test_padding_does_not_create_supervision(self):
        _, records = examples()
        packed = pack_records(records)
        self.assertEqual(packed["b"], 2)
        self.assertEqual(
            int(torch.isfinite(packed["labels"]).sum()),
            sum(int(torch.isfinite(r["prepared"]["labels"]).sum()) for r in records),
        )
        self.assertTrue(torch.isnan(packed["labels"][0, :, 2:]).all())
        self.assertEqual(len(make_batches(records, 8)), 1)
        records[0]["split"] = "evaluation"
        records[0]["batch"] = {"future_target": torch.ones(1, 1, 16)}
        packed = pack_records(records[:1])
        self.assertEqual(int(torch.isfinite(packed["labels"]).sum()), 16)
        self.assertTrue(torch.isnan(packed["labels"][:, :, :-1]).all())

    def test_master_weights_accumulate_small_bf16_updates(self):
        model = torch.nn.Linear(1, 1, bias=False).to(torch.bfloat16)
        with torch.no_grad():
            model.weight.fill_(1)
        optimizer = MasterAdamW(model, {"learning_rate": 0.0001, "weight_decay": 0})
        for _ in range(40):
            optimizer.zero_grad()
            model.weight.grad = torch.ones_like(model.weight)
            optimizer.step(1)
        self.assertEqual(optimizer.masters[0].dtype, torch.float32)
        self.assertLess(float(model.weight.detach()), 1)


@unittest.skipUnless(torch.cuda.is_available(), "CUDA required")
class BatchedEvaluationTests(unittest.TestCase):
    def test_batch_matches_table_at_a_time(self):
        encoder, records = examples()
        model = RWKVForecaster(encoder.config).cuda().to(torch.bfloat16).train()
        fields = [
            "step",
            "split",
            "source",
            "points",
            "pinball_loss",
            "rmse",
            "normalized_rmse",
        ]

        def scores(size):
            writer = csv.DictWriter(io.StringIO(), fieldnames=fields)
            return evaluate(model, make_batches(records, size), 0, "train", writer)

        single = scores(1)
        batched = scores(8)
        self.assertTrue(model.training)
        self.assertEqual(single["points"], batched["points"])
        self.assertEqual(single["tables"], 2)
        self.assertAlmostEqual(
            single["pinball_loss"], batched["pinball_loss"], delta=0.01
        )
        self.assertAlmostEqual(
            single["normalized_rmse"], batched["normalized_rmse"], delta=0.03
        )


if __name__ == "__main__":
    unittest.main()

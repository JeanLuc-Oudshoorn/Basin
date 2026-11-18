"""Four-lead arithmetic, causal tail isolation, batching and horizon sampling."""

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
    "06_holdout_test",
):
    sys.path.insert(0, str(ROOT / "src" / stage))
from epoch_training import accumulated_update
from horizon_policy import horizon_probabilities, sampled_horizon
from index_patches import LazyPatchRecord
from inference import forecast_contexts
from multi_horizon import (
    weighted_forecast_loss,
    weighted_label_mass,
    weighted_training_loss,
)
from test_training_performance import CausalBlock
from train import ForecastConfig, RWKVForecaster
from training_protocol import validate
from transform import load_config, transform_next_patch

WEIGHTS = (0.8, 0.16, 0.03, 0.01)


def model_fixture():
    with patch("train.load_rwkv", return_value=SimpleNamespace(Block=CausalBlock)):
        return RWKVForecaster(
            ForecastConfig(
                objective="next_patch",
                patch_size=16,
                width=64,
                layers=2,
                context_length=256,
                max_patches=16,
                output_patches=4,
                horizon_loss_weights=WEIGHTS,
            )
        )


class FourTokenTests(unittest.TestCase):
    def test_archived_one_patch_model_rolls_out_matching_requested_horizon(self):
        from dataclasses import replace

        model = model_fixture().eval()
        model.config = replace(
            model.config, output_patches=1, horizon_loss_weights=None
        )
        context = {"context": torch.randn(1, 1, 80), "past_covariates": None}
        with patch.object(
            model.blocks[0], "forward", wraps=model.blocks[0].forward
        ) as forward:
            actual = list(forecast_contexts(model, [context], 1, prediction_patches=4))[
                0
            ]
        self.assertEqual(forward.call_count, 4)
        torch.testing.assert_close(
            actual, model.forecast(**context, horizon=64).quantiles[0]
        )

    def test_multi_horizon_holdout_sends_identical_requests_to_all_methods(self):
        from unittest.mock import Mock

        from holdout import evaluate_holdout
        from transform import file_digest

        config = load_config("configs/legacy_500m.json")
        config["plot"]["max_plots"] = 0
        config["holdout"].update(
            benchmark_models=["naive", "chronos", "timestral"],
            latency={"warmup_requests": 0, "repeats": 1},
        )
        model = torch.nn.Linear(1, 1)
        model.config = SimpleNamespace(
            patch_size=16, context_length=256, quantiles=(0.1, 0.5, 0.9)
        )
        suite = Mock()
        suite.names = config["holdout"]["benchmark_models"]
        suite.metadata = {name: {"device": "cpu"} for name in suite.names}
        suite.predict.side_effect = lambda contexts, horizon, model_names: [
            {model_names[0]: np.zeros((1, horizon))}
        ]

        def neural(_model, contexts, _batch_size, prediction_patches=1):
            return [torch.zeros(1, prediction_patches * 16, 3)]

        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            source = root / "data.parquet"
            pd.DataFrame({"outcome": np.arange(160)}).to_parquet(source)
            config["paths"]["holdout_test"] = str(root / "holdout")
            record = {
                "source": str(source),
                "source_sha256": file_digest(source),
                "targets": ["outcome"],
                "roles": {"past": []},
            }
            short_source = root / "short.parquet"
            pd.DataFrame({"outcome": np.arange(80)}).to_parquet(short_source)
            short_record = dict(
                record,
                source=str(short_source),
                source_rows=80,
                source_sha256=file_digest(short_source),
            )
            with (
                patch("holdout.FoundationBenchmarks", return_value=suite),
                patch("holdout.forecast_contexts", side_effect=neural),
            ):
                result = evaluate_holdout(model, [record, short_record], config)
            self.assertEqual(result["skipped_short_tables"], 1)
            self.assertEqual(result["table_count"], 1)
            requests = pd.read_parquet(root / "holdout/request_latencies.parquet")
            self.assertTrue((requests.groupby("origin").horizon.nunique() == 1).all())
            self.assertEqual(set(requests[requests.origin == 32].horizon), {64})
            self.assertEqual(set(requests[requests.origin == 96].horizon), {64})
            self.assertEqual(requests.origin.max(), 96)
            self.assertNotIn("tables", result)
            self.assertNotIn("origins", result)
            self.assertTrue((root / "holdout/latency_distributions.html").is_file())
            predictions = pd.read_parquet(root / "holdout/test_predictions.parquet")
            self.assertTrue((predictions.groupby("origin").size() == 64).all())
            self.assertTrue(predictions.actual.notna().all())
            self.assertTrue(
                (predictions.groupby("origin").metric_weight.sum().round(8) == 1).all()
            )
            self.assertEqual(set(result["by_forecast_token"]), {"1", "2", "3", "4"})
            self.assertIn("4", result["by_forecast_token"])

    def test_known_weighted_forecast_loss_and_missing_tail(self):
        model = SimpleNamespace(
            config=SimpleNamespace(
                horizon_loss_weights=WEIGHTS, patch_size=16, quantiles=(0.1, 0.5, 0.9)
            )
        )
        labels = torch.arange(1, 5).repeat_interleave(16).float().reshape(1, 1, 64)
        prediction = torch.zeros(1, 1, 64, 3, requires_grad=True)
        loc, scale = torch.zeros(1, 1, 1), torch.ones(1, 1, 1)
        loss, per_sample, mass = weighted_forecast_loss(
            model, prediction, labels, loc, scale
        )
        self.assertAlmostEqual(loss.item(), 0.625, places=6)
        self.assertAlmostEqual(mass.item(), 16, places=5)
        loss.backward()
        self.assertGreater(prediction.grad.abs().sum().item(), 0)
        labels[..., 16:] = float("nan")
        loss, _, mass = weighted_forecast_loss(model, prediction, labels, loc, scale)
        self.assertAlmostEqual(loss.item(), 0.5, places=6)
        self.assertAlmostEqual(mass.item(), 12.8, places=5)

    def test_exact_training_leads_mask_boundaries_and_have_expected_mass(self):
        model = model_fixture()
        with torch.no_grad():
            for parameter in model.head.parameters():
                parameter.zero_()
        hidden = torch.randn(1, 1, 4, 64)
        labels = (
            torch.arange(1, 5).float()[None, None, :, None].expand(1, 1, 4, 16).clone()
        )
        loc, scale = torch.zeros(1, 1, 4, 1), torch.ones(1, 1, 4, 1)
        first = model.head(model.norm(hidden)).reshape(1, 1, 4, 16, 9)
        loss, _, mass = weighted_training_loss(model, hidden, labels, loc, scale, first)
        self.assertAlmostEqual(mass.item(), 60.0, places=5)
        self.assertAlmostEqual(
            weighted_label_mass(labels, WEIGHTS).item(), 60.0, places=5
        )
        self.assertAlmostEqual(loss.item(), 1.292, places=6)
        loss.backward()
        self.assertTrue(torch.isfinite(model.head.skip.weight.grad).all())

    def test_training_tail_reservation_and_full_validation_batch_equivalence(self):
        config = copy.deepcopy(load_config("configs/legacy_500m.json"))
        config["data"]["validation_policy"] = "reserved_tail"
        config["data"].pop("generated_dates", None)
        config["validation"]["enabled"] = False
        config["data"].update(context_length=256, horizon=64, max_tables=None)
        config["model"].update(
            context_length=256, max_patches=16, width=64, layers=2, variate_heads=4
        )
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            config["paths"].update(
                raw=str(root / "raw"), processed=str(root / "processed")
            )
            for length in (48, 96, 140):
                path = root / "raw" / str(length)
                path.mkdir(parents=True)
                pd.DataFrame(
                    {
                        "time_index": range(length),
                        "outcome": np.arange(length),
                        "cov": np.arange(length) * 2,
                    }
                ).to_parquet(path / "data.parquet")
            payload = transform_next_patch(config)
            model = model_fixture()
            records = [LazyPatchRecord(r, model) for r in payload["records"]]
            validation = []
            for record in records:
                cutoff = record["source_rows"] - 64
                if record["split"] == "train":
                    torch.testing.assert_close(
                        record["batch"]["past_covariates"],
                        record["batch"]["context"] * 2,
                    )
                    self.assertLess(record["batch"]["context"].max().item(), cutoff)
                    self.assertLess(
                        record["prepared"]["labels"].nan_to_num().max().item(), cutoff
                    )
                else:
                    validation.append(record)
                    self.assertEqual(
                        record["batch"]["future_target"].flatten().tolist(),
                        list(range(cutoff, cutoff + 64)),
                    )
            self.assertEqual(len(validation), 2)
            serial = validate(model, validation, batch_size=1)
            batched = validate(model, validation, batch_size=2)
            self.assertEqual(serial["points"], 128)
            self.assertAlmostEqual(serial["loss"], batched["loss"], places=5)
            self.assertAlmostEqual(serial["rmse"], batched["rmse"], places=4)

    def test_accumulated_weighted_gradients_match_unsplit_update(self):
        torch.manual_seed(4)
        first = model_fixture().train()
        second = copy.deepcopy(first)
        raw = [torch.randn(1, 1, 128), torch.randn(1, 1, 64)]
        records = []
        for index, context in enumerate(raw):
            records.append(
                {
                    "id": str(index),
                    "source": str(index),
                    "split": "train",
                    "prepared": first.tokenize_next(context, training=True),
                }
            )
        a = torch.optim.SGD(first.parameters(), lr=0.01)
        b = torch.optim.SGD(second.parameters(), lr=0.01)
        left = accumulated_update(first, [records], a, 1000)
        right = accumulated_update(second, [[r] for r in records], b, 1000)
        self.assertAlmostEqual(left["loss"], right["loss"], places=6)
        for x, y in zip(first.parameters(), second.parameters()):
            torch.testing.assert_close(x, y, atol=2e-7, rtol=2e-6)

    def test_horizons_are_seeded_order_independent_and_fit_future(self):
        config = {
            "holdout": {"horizon_probabilities": list(WEIGHTS), "horizon_seed": 7}
        }
        draws = [
            sampled_horizon({"source": str(i)}, 32, 1024, 16, config)
            for i in range(10000)
        ]
        for token, expected in enumerate(WEIGHTS, 1):
            self.assertLess(abs(draws.count(token) / len(draws) - expected), 0.02)
        self.assertEqual(
            draws[:10],
            [
                sampled_horizon({"source": str(i)}, 32, 1024, 16, config)
                for i in range(10)
            ],
        )
        for remaining in (1, 15, 16, 17, 32, 48, 64):
            tokens = sampled_horizon({"source": "test"}, 32, 32 + remaining, 16, config)
            self.assertLessEqual(tokens, max(1, remaining // 16))
        for value in ([0.8, 0.1], [-1, 2], [float("nan")], [True]):
            with self.assertRaises(ValueError):
                horizon_probabilities({"holdout": {"horizon_probabilities": value}})


if __name__ == "__main__":
    unittest.main()

"""Whole-table isolation, rolling origins, and validation semantics."""

import copy
import sys
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import pandas as pd
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src/05_model_training"))
sys.path.insert(0, str(ROOT / "src/06_holdout_test"))
sys.path.insert(0, str(ROOT / "src/03_transforming_input_data"))
from holdout import summarize
from train import load_config as pipeline_config
from training_protocol import rolling_origins, sampled_origins, split_datasets, validate
from transform import file_digest


def load_config():
    config = pipeline_config()
    # Protocol unit tests do not download foundation model weights.
    config["holdout"]["benchmark_models"] = ["prophet", "naive"]
    config["prophet"]["use_covariates"] = False
    return config


def records(count=20):
    result = []
    for i in range(count):
        for split in ("train", "train", "evaluation"):
            result.append(
                {
                    "source": f"table_{i}",
                    "source_rows": 64,
                    "source_sha256": f"digest_{i}",
                    "split": split,
                }
            )
    return result


class HoldoutProtocolTests(unittest.TestCase):
    def test_console_progress_every_100_tables_and_final_partial_interval(self):
        import numpy as np
        from holdout import evaluate_holdout

        config = load_config()
        config["plot"]["max_plots"] = 0
        model = torch.nn.Linear(1, 1)
        model.config = SimpleNamespace(
            patch_size=16, context_length=2048, quantiles=(0.1, 0.5, 0.9)
        )
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "data.parquet"
            pd.DataFrame({"outcome": np.arange(48)}).to_parquet(path)
            config["paths"]["holdout_test"] = str(Path(directory) / "output")
            record = {
                "source": str(path),
                "source_sha256": file_digest(path),
                "targets": ["outcome"],
                "roles": {"past": []},
            }
            with (
                patch("holdout.logger.info") as info,
                patch(
                    "holdout.forecast_contexts", return_value=[torch.zeros(1, 16, 3)]
                ),
                patch(
                    "holdout.baseline_forecasts",
                    return_value={"prophet": np.zeros(16), "naive": np.zeros(16)},
                ),
                patch("holdout.write_json"),
                patch.object(pd.DataFrame, "to_parquet"),
            ):
                evaluate_holdout(model, [record] * 201, config)
            updates = [
                call
                for call in info.call_args_list
                if call.args[0].startswith("HOLDOUT tables")
            ]
            self.assertEqual([call.args[1] for call in updates], [100, 200, 201])
            self.assertTrue(all("ETA" in call.args[0] for call in updates))

    def test_naive_carries_last_observed_value_over_whole_token(self):
        import numpy as np
        from holdout import baseline_forecasts

        with patch("prophet.Prophet") as prophet:
            prophet.return_value.predict.return_value = pd.DataFrame(
                {"yhat": [3.0] * 16}
            )
            result = baseline_forecasts(
                np.array([1.0, 2.0, np.nan, 6.0, np.nan]), 16, load_config()
            )
            np.testing.assert_array_equal(result["naive"], np.full(16, 6.0))
            self.assertEqual(set(result), {"naive", "prophet"})
            supplied = prophet.return_value.fit.call_args.args[0]
            self.assertEqual(len(supplied), 5)
            with self.assertRaisesRegex(ValueError, "entirely missing"):
                baseline_forecasts(np.full(4, np.nan), 16, load_config())

    def test_whole_tables_are_disjoint_and_order_independent(self):
        config = load_config()
        data = records()
        train, validation, test, audit = split_datasets(data, config)
        self.assertEqual((len(train), len(validation), len(test)), (36, 18, 2))
        self.assertFalse({r["source"] for r in train} & {r["source"] for r in test})
        self.assertEqual(
            {r["source"] for r in train}, {r["source"] for r in validation}
        )
        other = split_datasets(list(reversed(data)), config)[3]
        self.assertEqual(audit, other)

    def test_short_sources_excluded_and_small_pool_rounding(self):
        data = records(3)
        for record in data[:3]:
            record["source_rows"] = 47
        train, validation, test, audit = split_datasets(data, load_config())
        self.assertEqual(audit["admitted_eligible_tables"], 2)
        self.assertEqual(len(test), 1)
        self.assertEqual(audit["excluded_short_sources"], ["table_0"])
        self.assertNotIn("table_0", {r["source"] for r in train + validation + test})
        with self.assertRaisesRegex(ValueError, "at least two"):
            split_datasets(records(1), load_config())

    def test_missing_validation_fails_and_fraction_validated(self):
        config = copy.deepcopy(load_config())
        with self.assertRaisesRegex(ValueError, "Missing"):
            split_datasets([r for r in records() if r["split"] == "train"], config)
        config["training"]["holdout_fraction"] = 0
        with self.assertRaisesRegex(ValueError, "fraction"):
            split_datasets(records(), config)

    def test_one_token_origins_and_partial_tail(self):
        self.assertEqual(list(rolling_origins(48, 16)), [32])
        self.assertEqual(list(rolling_origins(64, 16)), [32, 48])
        self.assertEqual(list(rolling_origins(67, 16)), [32, 48, 64])
        with self.assertRaises(ValueError):
            rolling_origins(47, 16)

    def test_pooled_rmse_uses_observations(self):
        frame = pd.DataFrame(
            {
                "actual": [0.0, 0.0, 0.0, 0.0],
                "rwkv": [0.0, 0.0, 0.0, 4.0],
                "prophet": [1.0] * 4,
                "naive": [2.0] * 4,
                "context_scale": [2.0] * 4,
                "normalized_pinball": [0.1] * 4,
            }
        )
        result = summarize(frame)
        self.assertEqual(result["rwkv"]["rmse"], 2.0)
        self.assertEqual(result["rwkv"]["normalized_rmse"], 1.0)
        self.assertEqual(result["observations"], 4)

    def test_validation_is_full_no_grad_and_restores_train_mode(self):
        class FakeModel(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.weight = torch.nn.Parameter(torch.ones(()))
                self.calls = 0

            def forward(self, **batch):
                self.calls += 1
                assert not self.training and not torch.is_grad_enabled()
                return SimpleNamespace(
                    loss=self.weight * self.calls,
                    point=torch.zeros_like(batch["future_target"]),
                )

        model = FakeModel().train()
        data = [
            {"source": str(i), "batch": {"future_target": torch.ones(1, 1, i)}}
            for i in (1, 3)
        ]
        result = validate(model, data)
        self.assertEqual(result["loss"], 1.75)
        self.assertEqual(result["points"], 4)
        self.assertEqual(result["rmse"], 1.0)
        self.assertEqual(model.calls, 2)
        self.assertTrue(model.training)
        self.assertIsNone(model.weight.grad)

    def test_rolling_test_reveals_actuals_and_truncates_all_inputs(self):
        from holdout import evaluate_holdout

        class FakeModel(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.weight = torch.nn.Parameter(torch.ones(()))
                self.config = SimpleNamespace(
                    patch_size=16, context_length=2048, quantiles=(0.1, 0.5, 0.9)
                )
                self.contexts = []

            def forecast(self, context, horizon, past_covariates):
                assert not self.training and not torch.is_grad_enabled()
                torch.testing.assert_close(past_covariates, context * 2)
                self.contexts.append(context.clone())
                point = context[..., -1:].expand(1, 1, horizon)
                return SimpleNamespace(
                    point=point, quantiles=point[..., None].expand(1, 1, horizon, 3)
                )

        config = copy.deepcopy(load_config())
        config["plot"]["max_plots"] = 0
        with tempfile.TemporaryDirectory() as temporary:
            base = Path(temporary)
            path = base / "data.parquet"
            values = list(range(2099))
            pd.DataFrame(
                {"outcome": values, "covariate": [2 * v for v in values]}
            ).to_parquet(path)
            config["paths"]["holdout_test"] = str(base / "holdout_test")
            record = {
                "source": str(path),
                "source_sha256": file_digest(path),
                "targets": ["outcome"],
                "roles": {"past": ["covariate"]},
            }
            model = FakeModel().train()
            seen = []

            def baseline(history, horizon, _config, **kwargs):
                seen.append(history.copy())
                return {
                    "prophet": [history[-1]] * horizon,
                    "naive": [history[-1]] * horizon,
                }

            def predict(actual_model, contexts, batch_size):
                for context in contexts:
                    yield actual_model.forecast(**context, horizon=16).quantiles[0]

            with (
                patch("holdout.baseline_forecasts", side_effect=baseline),
                patch("holdout.forecast_contexts", side_effect=predict),
            ):
                result = evaluate_holdout(model, [record], config)
            self.assertEqual(result["pooled"]["observations"], 4 * 16 + 3)
            self.assertEqual(result["origin_target_count"], 5)
            self.assertEqual(model.contexts[0].shape[-1], 32)
            self.assertEqual(
                model.contexts[1][0, 0, -1], 527
            )  # actual, never previous prediction
            self.assertEqual(model.contexts[-1].shape[-1], 2048)
            self.assertEqual(model.contexts[-1][0, 0, 0], 48)
            self.assertEqual(len(seen), len(model.contexts))
            self.assertEqual(seen[-1][-1], 2095)
            self.assertTrue(model.training)
            self.assertIsNone(model.weight.grad)

    def test_sampled_origins_percentiles_short_series_and_ties(self):
        self.assertEqual(sampled_origins(2048, 16), [32, 512, 1024, 1536, 2032])
        self.assertEqual(sampled_origins(2056, 16), [32, 512, 1024, 1536, 2048])
        self.assertEqual(sampled_origins(160, 16), [32, 48, 80, 112, 144])
        for length in range(48, 2080):
            origins = sampled_origins(length, 16)
            eligible = list(rolling_origins(length, 16))
            self.assertEqual(len(origins), min(5, len(eligible)))
            self.assertEqual(origins, sorted(set(origins)))
            self.assertEqual((origins[0], origins[-1]), (eligible[0], eligible[-1]))
            self.assertTrue(set(origins).issubset(eligible))


if __name__ == "__main__":
    unittest.main()

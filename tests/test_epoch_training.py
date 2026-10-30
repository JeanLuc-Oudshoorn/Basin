"""Epoch boundaries, weighted accumulation, plateau stopping and best weights."""

import copy
import json
import sys
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src/05_model_training"))
from batching import make_batches
from epoch_training import (
    EarlyStopping,
    accumulated_update,
    fit_epochs,
    training_options,
)
from train import ForecastConfig, PatchEncoder
from transform import load_config


class ToyForecaster(PatchEncoder):
    def __init__(self):
        super().__init__(
            ForecastConfig(
                objective="next_patch",
                patch_size=16,
                width=64,
                layers=2,
                context_length=128,
                max_patches=16,
            )
        )
        self.head = torch.nn.Linear(64, 16)

    def forward_encoded(self, x, prepared, future_target=None):
        predictions = self.head(x[:, :1, : prepared["tokens"]])
        labels = prepared["labels"]
        valid = torch.isfinite(labels)
        return SimpleNamespace(
            loss=(predictions[valid] - labels[valid]).square().mean()
        )


def examples(model):
    result = []
    for i, length in enumerate((48, 64, 80)):
        context = torch.linspace(0, 1, length)[None, None]
        result.append(
            {
                "id": str(i),
                "source": str(i),
                "split": "train",
                "prepared": model.tokenize_next(context, training=True),
            }
        )
    return result


class EpochTrainingTests(unittest.TestCase):
    def test_epoch_zero_is_pre_update_persisted_and_not_repeated_on_resume(self):
        model = ToyForecaster()
        initial = copy.deepcopy(model.state_dict())
        records = examples(model)
        with tempfile.TemporaryDirectory() as temporary:
            config = copy.deepcopy(load_config())
            config["paths"]["training"] = temporary
            config["training"].update(
                epochs=1,
                steps=None,
                batch_size=1,
                gradient_accumulation_steps=1,
                validate_before_training=True,
            )
            optimizer = torch.optim.AdamW(model.parameters(), lr=0.001)
            calls = []

            def validation(actual_model, actual_records):
                calls.append(len(optimizer.state))
                if len(calls) == 1:
                    for name, weight in actual_model.state_dict().items():
                        torch.testing.assert_close(
                            weight, initial[name], rtol=0, atol=0
                        )
                return {
                    "loss": 1.0 / len(calls),
                    "rmse": 1.0,
                    "points": 16,
                    "tables": [],
                    "seconds": 0.0,
                }

            _, summary, _ = fit_epochs(
                model, records, [{"source": "v"}], config, optimizer, validation
            )
            rows = json.loads((Path(temporary) / "validation_history.json").read_text())
            self.assertEqual([r["epoch"] for r in rows], [0, 1])
            self.assertEqual(calls[0], 0)
            self.assertIsNone(rows[0]["training_loss"])
            self.assertEqual(summary["best_epoch"], 1)
            config["training"]["epochs"] = 2
            fit_epochs(model, records, [{"source": "v"}], config, optimizer, validation)
            self.assertEqual(len(calls), 3)
            rows = json.loads((Path(temporary) / "validation_history.json").read_text())
            self.assertEqual([r["epoch"] for r in rows], [0, 1, 2])

    def test_accumulation_matches_combined_batch_with_unequal_label_counts(self):
        torch.manual_seed(4)
        separate = ToyForecaster()
        together = copy.deepcopy(separate)
        records = examples(separate)
        first = accumulated_update(
            separate,
            [[records[0]], [records[2]]],
            torch.optim.SGD(separate.parameters(), lr=0.001),
            1e9,
        )
        second = accumulated_update(
            together,
            [[records[0], records[2]]],
            torch.optim.SGD(together.parameters(), lr=0.001),
            1e9,
        )
        self.assertAlmostEqual(first["loss"], second["loss"], places=6)
        for a, b in zip(separate.parameters(), together.parameters()):
            torch.testing.assert_close(a, b, atol=1e-7, rtol=1e-5)

    def test_validate_only_at_epoch_end_stop_after_three_and_restore_best(self):
        torch.manual_seed(5)
        model = ToyForecaster()
        records = examples(model)
        validation_records = [{"source": "v0"}, {"source": "v1"}]
        losses = iter([1.0, 0.9, 0.95, 0.96, 0.97])
        saved = []

        def validation(actual_model, actual_records):
            self.assertIs(actual_records, validation_records)
            saved.append(copy.deepcopy(actual_model.state_dict()))
            loss = next(losses)
            return {
                "loss": loss,
                "rmse": loss * 2,
                "points": 32,
                "tables": [
                    {"source": "v0", "loss": loss},
                    {"source": "v1", "loss": loss},
                ],
                "seconds": 0.0,
            }

        with tempfile.TemporaryDirectory() as temporary:
            config = copy.deepcopy(load_config())
            config["paths"]["training"] = temporary
            config["training"].update(
                validate_before_training=False,
                steps=None,
                epochs=None,
                batch_size=1,
                gradient_accumulation_steps=2,
                checkpoint_every=0,
                shuffle=True,
                early_stopping={
                    "patience_epochs": 3,
                    "min_delta": 0.0,
                    "monitor": "loss",
                },
            )
            optimizer = torch.optim.AdamW(model.parameters(), lr=0.001)
            history, summary, scheduler = fit_epochs(
                model, records, validation_records, config, optimizer, validation
            )
            self.assertEqual(len(history), 10)
            self.assertEqual(len(saved), 5)
            self.assertEqual(summary["reason"], "early_stopping")
            self.assertEqual(summary["completed_epochs"], 5)
            self.assertEqual(summary["best_epoch"], 2)
            self.assertEqual(summary["best_step"], 4)
            self.assertEqual(summary["epochs_without_improvement"], 3)
            self.assertEqual(scheduler["completed_validation_epochs"], 5)
            self.assertEqual([row["windows"] for row in history], [2, 1] * 5)
            self.assertEqual([row["epoch_update"] for row in history], [1, 2] * 5)
            self.assertTrue(all(row["epoch_total_updates"] == 2 for row in history))
            self.assertEqual(
                [row["epoch_progress_percent"] for row in history], [50.0, 100.0] * 5
            )
            self.assertTrue(
                all(row["epoch_eta_seconds"] == 0.0 for row in history[1::2])
            )
            self.assertTrue(all(row["validation_loss"] is None for row in history[::2]))
            for a, b in zip(model.state_dict().values(), saved[1].values()):
                torch.testing.assert_close(a, b, rtol=0, atol=0)
            epochs = json.loads(
                (Path(temporary) / "validation_history.json").read_text()
            )
            self.assertTrue(all(row["completed_epoch"] for row in epochs))
            self.assertEqual(
                len(list(Path(temporary).glob("validation_epoch_*.parquet"))), 5
            )
            for epoch in range(1, 6):
                ids = [
                    id
                    for row in history
                    if row["epoch"] == epoch
                    for id in row["window_ids"]
                ]
                self.assertEqual(sorted(ids), ["0", "1", "2"])

    def test_patience_resets_and_min_delta_is_absolute(self):
        early = EarlyStopping(3, 0.01)
        self.assertEqual(early.observe(1.0, 1), (True, False))
        self.assertEqual(early.observe(0.995, 2), (False, False))
        self.assertEqual(early.observe(0.98, 3), (True, False))
        self.assertEqual(early.bad_epochs, 0)
        for epoch in (4, 5):
            self.assertEqual(early.observe(0.98, epoch), (False, False))
        self.assertEqual(early.observe(0.98, 6), (False, True))

    def test_partial_final_accumulation_not_dropped_and_shuffle_reproducible(self):
        model = ToyForecaster()
        records = examples(model)
        first = make_batches(records, 2, seed=7)
        second = make_batches(records, 2, seed=7)
        self.assertEqual(
            [[r["id"] for r in group] for group in first],
            [[r["id"] for r in group] for group in second],
        )
        update = accumulated_update(
            model, first, torch.optim.SGD(model.parameters(), lr=0.001), 1.0
        )
        self.assertEqual(update["windows"], 3)
        self.assertEqual(update["target_points"], 144)

    def test_invalid_loop_settings_fail(self):
        for key, value in (
            ("batch_size", 0),
            ("gradient_accumulation_steps", True),
            ("epochs", -1),
            ("steps", 0),
            ("early_stopping", {"patience_epochs": 0}),
        ):
            with self.subTest(key=key):
                with self.assertRaises(ValueError):
                    training_options({key: value})

    def test_debug_step_cap_marks_partial_epoch_and_still_flushes_update(self):
        model = ToyForecaster()
        records = examples(model)
        with tempfile.TemporaryDirectory() as temporary:
            config = copy.deepcopy(load_config())
            config["paths"]["training"] = temporary
            config["training"].update(
                validate_before_training=False,
                steps=1,
                epochs=None,
                batch_size=1,
                gradient_accumulation_steps=2,
                checkpoint_every=0,
            )
            calls = []

            def validation(actual_model, actual_records):
                calls.append(1)
                return {
                    "loss": 1.0,
                    "rmse": 1.0,
                    "points": 16,
                    "tables": [],
                    "seconds": 0.0,
                }

            history, summary, _ = fit_epochs(
                model,
                records,
                [{"source": "validation"}],
                config,
                torch.optim.SGD(model.parameters(), lr=0.001),
                validation,
            )
            self.assertEqual(summary["reason"], "step_limit")
            self.assertFalse(summary["last_epoch_complete"])
            self.assertEqual(summary["completed_epochs"], 0)
            self.assertEqual(len(calls), 1)
            self.assertEqual(history[0]["windows"], 2)


if __name__ == "__main__":
    unittest.main()

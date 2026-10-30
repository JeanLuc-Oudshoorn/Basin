"""Crash recovery preserves optimizer, scheduler, RNG, history and stopping state."""

import copy
import json
import random
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
for stage in (
    "03_transforming_input_data",
    "04_creating_embeddings",
    "05_model_training",
):
    sys.path.insert(0, str(ROOT / "src" / stage))
from checkpoints import atomic_torch_save
from epoch_training import (
    EarlyStopping,
    accumulated_update,
    fit_epochs,
    training_options,
)
from test_epoch_training import ToyForecaster, examples
from transform import load_config


class StochasticForecaster(ToyForecaster):
    def encode_tokens(self, prepared):
        return torch.nn.functional.dropout(
            super().encode_tokens(prepared), p=0.2, training=self.training
        )


def setup(directory, *, epochs=None):
    torch.manual_seed(7)
    random.seed(7)
    np.random.seed(7)
    model = StochasticForecaster()
    config = copy.deepcopy(load_config())
    config["paths"]["training"] = str(directory)
    config["training"].update(
        validate_before_training=False,
        epochs=epochs,
        steps=None,
        checkpoint_every=0,
        batch_size=1,
        gradient_accumulation_steps=2,
        early_stopping={"patience_epochs": 1, "min_delta": 0.0, "monitor": "loss"},
    )
    optimizer = torch.optim.AdamW(model.parameters(), lr=0.001)
    return model, examples(model), config, optimizer


def validation_values(values):
    values = iter(values)

    def validate(model, records):
        value = next(values)
        # Consume all RNGs so restored RNG state is observable in subsequent dropout.
        random.random()  # nosec B311 - Exercise checkpointed RNG, not cryptography.
        np.random.random()
        torch.rand(1)
        return {
            "loss": value,
            "rmse": value * 2,
            "points": 16,
            "tables": [{"source": "v", "loss": value}],
            "seconds": 0.0,
        }

    return validate


class EpochCheckpointTests(unittest.TestCase):
    def test_interrupted_epoch_resumes_exactly_and_stops_after_first_bad_epoch(self):
        with tempfile.TemporaryDirectory() as temporary:
            reference_dir, crashed_dir = (
                Path(temporary) / "reference",
                Path(temporary) / "crashed",
            )
            model, records, config, optimizer = setup(reference_dir)
            expected_history, expected_summary, expected_scheduler = fit_epochs(
                model,
                records,
                [{"source": "v"}],
                config,
                optimizer,
                validation_values([1.0, 0.9, 0.95]),
            )
            expected_weights = copy.deepcopy(model.state_dict())
            # Reproducible RNG state is the subject of this recovery test.
            expected_rng = (torch.rand(5), random.random(), np.random.random())  # nosec B311
            model, records, config, optimizer = setup(crashed_dir)
            calls = 0

            def interrupted(*args, **kwargs):
                nonlocal calls
                calls += 1
                if calls == 4:
                    raise RuntimeError("simulated interruption during epoch 2")
                return accumulated_update(*args, **kwargs)

            with patch("epoch_training.accumulated_update", side_effect=interrupted):
                with self.assertRaisesRegex(RuntimeError, "simulated interruption"):
                    fit_epochs(
                        model,
                        records,
                        [{"source": "v"}],
                        config,
                        optimizer,
                        validation_values([1.0]),
                    )
            state = torch.load(crashed_dir / "latest_checkpoint.pt", weights_only=True)
            self.assertEqual(state["epoch"], 1)
            self.assertEqual(state["step"], 2)
            self.assertTrue(state["optimizer_state_dict"]["state"])
            del state
            # A failed next epoch can leave unrelated/newer external best weights.
            atomic_torch_save(
                {"state_dict": model.state_dict()}, crashed_dir / "best_weights.pt"
            )
            model, records, config, optimizer = setup(crashed_dir)
            torch.rand(30)
            history, summary, scheduler = fit_epochs(
                model,
                records,
                [{"source": "v"}],
                config,
                optimizer,
                validation_values([0.9, 0.95]),
            )
            self.assertEqual(summary["reason"], "early_stopping")
            self.assertEqual(summary["completed_epochs"], 3)
            self.assertEqual(summary["best_epoch"], 2)
            self.assertEqual(
                summary["completed_updates"], expected_summary["completed_updates"]
            )
            self.assertEqual(
                [r["loss"] for r in history], [r["loss"] for r in expected_history]
            )
            self.assertEqual(scheduler, expected_scheduler)
            for name, value in model.state_dict().items():
                torch.testing.assert_close(
                    value, expected_weights[name], rtol=0, atol=0
                )
            torch.testing.assert_close(torch.rand(5), expected_rng[0], rtol=0, atol=0)
            # Compare the restored PRNG stream, not a security-sensitive value.
            self.assertEqual(random.random(), expected_rng[1])  # nosec B311
            self.assertEqual(np.random.random(), expected_rng[2])
            terminal = torch.load(
                crashed_dir / "latest_checkpoint.pt", weights_only=True
            )
            self.assertEqual(terminal["early_stopping"]["bad_epochs"], 1)
            self.assertEqual(terminal["stop_reason"], "early_stopping")
            self.assertEqual(
                len(json.loads((crashed_dir / "validation_history.json").read_text())),
                3,
            )
            # Retrying export/test after an already-stopped run performs no training.
            model, records, config, optimizer = setup(crashed_dir)
            with patch("epoch_training.accumulated_update") as update:
                _, retry_summary, _ = fit_epochs(
                    model,
                    records,
                    [{"source": "v"}],
                    config,
                    optimizer,
                    validation_values([]),
                )
                update.assert_not_called()
            self.assertEqual(retry_summary["completed_epochs"], 3)
            self.assertEqual(retry_summary["reason"], "early_stopping")

    def test_atomic_failed_write_preserves_previous_checkpoint(self):
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "checkpoint.pt"
            atomic_torch_save({"epoch": 1}, path)

            def fail_mid_write(payload, stream):
                stream.write(b"partial checkpoint")
                raise OSError("disk failure")

            with patch("checkpoints.torch.save", side_effect=fail_mid_write):
                with self.assertRaises(OSError):
                    atomic_torch_save({"epoch": 2}, path)
            self.assertEqual(torch.load(path, weights_only=True)["epoch"], 1)
            self.assertEqual(list(Path(temporary).glob("*.tmp")), [])

    def test_changed_optimization_rejected_and_explicit_fresh_run_supported(self):
        with tempfile.TemporaryDirectory() as temporary:
            model, records, config, optimizer = setup(temporary, epochs=1)
            fit_epochs(
                model,
                records,
                [{"source": "v"}],
                config,
                optimizer,
                validation_values([1.0]),
            )
            config["training"]["batch_size"] = 2
            with self.assertRaisesRegex(ValueError, "optimization settings differ"):
                fit_epochs(
                    model,
                    records,
                    [{"source": "v"}],
                    config,
                    optimizer,
                    validation_values([]),
                )
            config["training"]["resume"] = False
            _, summary, _ = fit_epochs(
                model,
                records,
                [{"source": "v"}],
                config,
                optimizer,
                validation_values([0.5]),
            )
            self.assertIsNone(summary["resumed_from"])

    def test_default_patience_one_and_resume_settings(self):
        self.assertEqual(training_options({})["patience_epochs"], 1)
        early = EarlyStopping(1, 0.0)
        self.assertEqual(early.observe(0.5, 1), (True, False))
        self.assertEqual(early.observe(0.5, 2), (False, True))
        for value in (True, None, 3, ""):
            with self.subTest(value=value), self.assertRaises(ValueError):
                training_options({"resume": value})


if __name__ == "__main__":
    unittest.main()

"""Mid-epoch continuation, clocks, validation cadence and pending-check recovery."""

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
from epoch_training import accumulated_update
from test_epoch_checkpoints import StochasticForecaster
from test_epoch_training import examples
from timed_training import fit_timed
from transform import load_config


class Clock:
    def __init__(self):
        self.value = 0.0

    def __call__(self):
        return self.value


def setup(directory):
    torch.manual_seed(7)
    random.seed(7)
    np.random.seed(7)
    model = StochasticForecaster()
    records = examples(model)
    validation = [
        dict(records[0], source="reserved_validation_table", validation_origin=32)
    ]
    config = copy.deepcopy(load_config())
    config["paths"]["training"] = str(directory)
    config["training"].update(
        epochs=None,
        steps=6,
        batch_size=1,
        gradient_accumulation_steps=2,
        resume="auto",
        prefetch=True,
        validate_before_training=False,
        learning_rate=0.001,
        checkpoint_interval_seconds=1,
        validation_interval_seconds=2,
        max_run_hours=None,
        early_stopping={"patience_epochs": 100, "monitor": "loss", "min_delta": 0.0},
    )
    optimizer = torch.optim.AdamW(model.parameters(), lr=0.001)
    return model, records, validation, config, optimizer


def run_training(parts, clock, *, crash_validation=False, validation_cost=0):
    model, records, validation, config, optimizer = parts

    def update(*args, **kwargs):
        result = accumulated_update(*args, **kwargs)
        clock.value += 1.0
        return result

    def validate(model, records):
        if crash_validation:
            raise RuntimeError("validation interrupted")
        # Validation must not perturb subsequent stochastic training.
        torch.rand(3)
        random.random()  # nosec B311 - exercise restored nonsecurity RNG
        np.random.random()
        clock.value += validation_cost
        value = float(model.head.weight.detach().square().sum())
        return {
            "loss": value,
            "rmse": value,
            "points": 16,
            "tables": [],
            "seconds": validation_cost,
        }

    with patch("epoch_training.accumulated_update", side_effect=update):
        return fit_timed(
            model,
            records,
            validation,
            config,
            optimizer,
            validation_fn=validate,
            clock=clock,
        )


class TimedTrainingTests(unittest.TestCase):
    def test_resume_can_extend_validation_interval_without_repeating_old_due_check(
        self,
    ):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            with self.assertRaisesRegex(RuntimeError, "validation interrupted"):
                run_training(setup(root), Clock(), crash_validation=True)
            interrupted = torch.load(root / "latest_checkpoint.pt", weights_only=True)
            self.assertEqual(interrupted["step"], 2)
            self.assertTrue(interrupted["loop_state"]["pending_validation"])
            parts = setup(root)
            parts[3]["training"].update(
                validation_interval_seconds=4,
                checkpoint_interval_seconds=2,
            )
            run_training(parts, Clock())
            history = json.loads((root / "validation_history.json").read_text())
            self.assertEqual([row["step"] for row in history], [4])
            continued = torch.load(root / "latest_checkpoint.pt", weights_only=True)
            self.assertEqual(continued["step"], 6)
            self.assertFalse(continued["loop_state"]["pending_validation"])

    def test_resume_accepts_safer_physical_limits_and_restores_memory_policy(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            parts = setup(root)
            parts[3]["training"]["max_run_hours"] = 1 / 3600
            parts[0]._microbatch_limits = {((3, 32),): 1}
            run_training(parts, Clock())
            saved = torch.load(root / "latest_checkpoint.pt", weights_only=True)
            resumed = setup(root)
            resumed[3]["training"].update(microbatch_size=1, microbatch_token_limit=64)
            run_training(resumed, Clock())
            current = torch.load(root / "latest_checkpoint.pt", weights_only=True)
            self.assertGreater(current["step"], saved["step"])
            self.assertEqual(resumed[0]._microbatch_limits, {((3, 32),): 1})

    def test_training_rejects_validation_source_overlap_before_updates(self):
        with tempfile.TemporaryDirectory() as temporary:
            parts = setup(Path(temporary))
            parts[2][0]["source"] = parts[1][0]["source"]
            with self.assertRaisesRegex(ValueError, "dataset-disjoint"):
                run_training(parts, Clock())
            self.assertFalse((Path(temporary) / "latest_checkpoint.pt").exists())

    def test_old_in_sample_checkpoint_cannot_silently_resume(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            parts = setup(root)
            parts[3]["training"]["max_run_hours"] = 1 / 3600
            run_training(parts, Clock())
            path = root / "latest_checkpoint.pt"
            saved = torch.load(path, weights_only=True)
            del saved["identity"]["optimization"]["validation_split_policy"]
            torch.save(saved, path)
            with self.assertRaisesRegex(
                ValueError, "datasets or optimization settings differ"
            ):
                run_training(setup(root), Clock())

    def assert_checkpoint_equal(self, first, second):
        self.assertEqual(first["step"], second["step"])
        for name, value in first["state_dict"].items():
            torch.testing.assert_close(
                value, second["state_dict"][name], rtol=0, atol=0
            )
        for key, state in first["optimizer_state_dict"]["state"].items():
            for name, value in state.items():
                other = second["optimizer_state_dict"]["state"][key][name]
                if isinstance(value, torch.Tensor):
                    torch.testing.assert_close(value, other, rtol=0, atol=0)
                else:
                    self.assertEqual(value, other)
        self.assertEqual(first["scheduler_state_dict"], second["scheduler_state_dict"])
        self.assertEqual(
            first["optimizer_state_dict"]["param_groups"],
            second["optimizer_state_dict"]["param_groups"],
        )
        self.assertEqual(first["rng_state"]["python"], second["rng_state"]["python"])
        self.assertEqual(first["rng_state"]["numpy"], second["rng_state"]["numpy"])
        torch.testing.assert_close(
            first["rng_state"]["torch"], second["rng_state"]["torch"], rtol=0, atol=0
        )
        self.assertEqual(
            first["loop_state"]["next_batch_offset"],
            second["loop_state"]["next_batch_offset"],
        )
        self.assertEqual(
            [r["loss"] for r in first["history"]],
            [r["loss"] for r in second["history"]],
        )

    def test_time_limit_checkpoint_and_mid_epoch_resume_match_uninterrupted(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            full = setup(root / "full")
            _, full_summary, _ = run_training(full, Clock())
            interrupted = setup(root / "interrupted")
            interrupted[3]["training"]["max_run_hours"] = 1 / 3600
            _, paused, _ = run_training(interrupted, Clock())
            self.assertEqual(paused["reason"], "time_limit")
            saved = torch.load(
                root / "interrupted/latest_checkpoint.pt", weights_only=True
            )
            self.assertEqual(saved["step"], 1)
            self.assertEqual(saved["history"][0]["preparation_wait_seconds"], 0)
            self.assertEqual(saved["history"][0]["optimizer_update_seconds"], 1)
            self.assertEqual(saved["loop_state"]["next_batch_offset"], 2)
            self.assertEqual(saved["loop_state"]["completed_epochs"], 0)
            self.assertEqual(
                saved["loop_state"]["epoch_windows"], saved["windows_processed"]
            )
            self.assertEqual(
                saved["loop_state"]["epoch_target_points_processed"],
                saved["points_processed"],
            )
            resumed = setup(root / "interrupted")
            _, resumed_summary, _ = run_training(resumed, Clock())
            self.assertIsNotNone(resumed_summary["resumed_from"])
            self.assertEqual(
                full_summary["completed_updates"], resumed_summary["completed_updates"]
            )
            self.assert_checkpoint_equal(
                torch.load(root / "full/latest_checkpoint.pt", weights_only=True),
                torch.load(
                    root / "interrupted/latest_checkpoint.pt", weights_only=True
                ),
            )
            continued = torch.load(
                root / "interrupted/latest_checkpoint.pt", weights_only=True
            )
            self.assertEqual(
                continued["loop_state"]["epoch_target_points_processed"],
                sum(
                    r["target_points"]
                    for r in continued["history"]
                    if r["epoch"] == continued["epoch"]
                ),
            )

    def test_validation_history_and_individual_results_survive_resume(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            parts = setup(root)
            parts[3]["training"]["max_run_hours"] = 2 / 3600
            run_training(parts, Clock())
            first_history = json.loads((root / "validation_history.json").read_text())
            self.assertEqual([r["validation_check"] for r in first_history], [1])
            originals = {
                path: path.read_bytes() for path in root.glob("validation_check_*")
            }
            self.assertEqual(len(originals), 2)  # Aggregate JSON plus origin Parquet.
            run_training(setup(root), Clock())
            history = json.loads((root / "validation_history.json").read_text())
            self.assertEqual([r["validation_check"] for r in history], [1, 2, 3])
            self.assertEqual([r["step"] for r in history], [2, 4, 6])
            self.assertEqual([r["active_training_seconds"] for r in history], [2, 4, 6])
            self.assertEqual(history[0], first_history[0])
            for path, content in originals.items():
                self.assertEqual(path.read_bytes(), content)
            snapshots = sorted(root.glob("validation_check_*.json"))
            self.assertEqual(len(snapshots), 3)
            self.assertEqual(
                [json.loads(p.read_text())["loss"] for p in snapshots],
                [r["loss"] for r in history],
            )

    def test_interrupted_validation_is_repeated_before_next_update(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            run_training(setup(root / "full"), Clock())
            with self.assertRaisesRegex(RuntimeError, "validation interrupted"):
                run_training(setup(root / "failed"), Clock(), crash_validation=True)
            saved = torch.load(root / "failed/latest_checkpoint.pt", weights_only=True)
            self.assertEqual(saved["step"], 2)
            self.assertTrue(saved["loop_state"]["pending_validation"])
            run_training(setup(root / "failed"), Clock())
            self.assert_checkpoint_equal(
                torch.load(root / "full/latest_checkpoint.pt", weights_only=True),
                torch.load(root / "failed/latest_checkpoint.pt", weights_only=True),
            )

    def test_validation_time_does_not_count_as_active_training(self):
        with tempfile.TemporaryDirectory() as temporary:
            _, summary, _ = run_training(
                setup(Path(temporary)), Clock(), validation_cost=10
            )
            self.assertEqual(summary["active_training_seconds"], 6.0)
            self.assertEqual(summary["validation_checks"], 3)
            self.assertGreaterEqual(summary["run_elapsed_seconds"], 36.0)

    def test_wall_limit_includes_validation_and_saves_completed_check(self):
        with tempfile.TemporaryDirectory() as temporary:
            parts = setup(Path(temporary))
            parts[3]["training"]["max_run_hours"] = 5 / 3600
            _, summary, _ = run_training(parts, Clock(), validation_cost=10)
            self.assertEqual(summary["reason"], "time_limit")
            self.assertEqual(summary["completed_updates"], 2)
            self.assertEqual(summary["validation_checks"], 1)
            saved = torch.load(
                Path(temporary) / "latest_checkpoint.pt", weights_only=True
            )
            self.assertFalse(saved["loop_state"]["pending_validation"])

    def test_duration_expiry_on_hourly_boundary_completes_due_validation(self):
        with tempfile.TemporaryDirectory() as temporary:
            parts = setup(Path(temporary))
            parts[3]["training"]["max_run_hours"] = 2 / 3600
            _, summary, _ = run_training(parts, Clock(), validation_cost=10)
            self.assertEqual(summary["completed_updates"], 2)
            self.assertEqual(summary["validation_checks"], 1)
            saved = torch.load(
                Path(temporary) / "latest_checkpoint.pt", weights_only=True
            )
            self.assertEqual(saved["loop_state"]["last_validation_step"], 2)
            self.assertFalse(saved["loop_state"]["pending_validation"])

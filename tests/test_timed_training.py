"""Mid-epoch continuation, clocks, validation cadence and pending-check recovery."""

import copy
import io
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
    "04_patch_indexing",
    "05_model_training",
):
    sys.path.insert(0, str(ROOT / "src" / stage))
from epoch_training import accumulated_update
from test_checkpoint_writer import FakeStore, FakeUpload
from test_epoch_checkpoints import StochasticForecaster
from test_epoch_training import examples
from timed_training import fit_timed
from training_log import training_run
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
    config = copy.deepcopy(load_config("configs/legacy_500m.json"))
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
    def setUp(self):
        # Keep checkpoints on the background route whatever memory is free.
        import checkpoints

        original = checkpoints.available_host_memory
        checkpoints.available_host_memory = lambda: None
        self.addCleanup(setattr, checkpoints, "available_host_memory", original)

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

    def journal_metrics(self, root, **training):
        parts = setup(root / "run")
        parts[3]["paths"]["logs"] = str(root / "logs")
        parts[3]["training"].update(
            checkpoint_interval_seconds=100, validation_interval_seconds=100, **training
        )
        with training_run(parts[3], "train") as journal:
            run_training(parts, Clock())
            path = journal.path
        events = [json.loads(line) for line in path.read_text().splitlines()]
        return [e for e in events if e["event"] == "metrics"], parts

    def test_journal_has_one_record_per_update_unless_summaries_are_requested(self):
        with tempfile.TemporaryDirectory() as temporary:
            metrics, _ = self.journal_metrics(Path(temporary))
            self.assertEqual([e["global_step"] for e in metrics], [1, 2, 3, 4, 5, 6])
            # Three tables per epoch, two per update: updates of two and one.
            self.assertEqual([len(e["window_ids"]) for e in metrics], [2, 1] * 3)
            self.assertTrue(all("updates" not in e for e in metrics))

    def test_summaries_per_table_count_cover_every_update_exactly_once(self):
        with tempfile.TemporaryDirectory() as temporary:
            # Updates hold two tables, then one. A summary follows the first
            # update, then every three or more tables, then the last update.
            metrics, parts = self.journal_metrics(Path(temporary), log_every_tables=3)
            self.assertEqual([e["global_step"] for e in metrics], [1, 3, 5, 6])
            self.assertEqual([e["updates"] for e in metrics], [1, 2, 2, 1])
            self.assertEqual([e["windows"] for e in metrics], [2, 3, 3, 1])
            self.assertTrue(all("window_ids" not in e for e in metrics))
            history = json.loads(
                (Path(temporary) / "run" / "loss_history.json").read_text()
            )
            self.assertEqual(len(history), 6)  # the per-update history is unchanged
            self.assertEqual(
                sum(e["target_points"] for e in metrics),
                sum(h["target_points"] for h in history),
            )
            # Each summary's loss is the weighted mean of its updates' losses.
            start = 0
            for event in metrics:
                rows = history[start : start + event["updates"]]
                start += event["updates"]
                mass = sum(
                    r.get("target_loss_weight", r["target_points"]) for r in rows
                )
                expected = (
                    sum(
                        r["loss"] * r.get("target_loss_weight", r["target_points"])
                        for r in rows
                    )
                    / mass
                )
                self.assertAlmostEqual(event["train_loss"], expected, places=12)
                self.assertEqual(event["train_loss_last_update"], rows[-1]["loss"])
            self.assertEqual(
                torch.load(
                    Path(temporary) / "run" / "latest_checkpoint.pt", weights_only=True
                )["step"],
                6,
            )

    def test_summaries_are_flushed_before_every_checkpoint_and_validation(self):
        with tempfile.TemporaryDirectory() as temporary:
            parts = setup(Path(temporary) / "run")
            parts[3]["paths"]["logs"] = str(Path(temporary) / "logs")
            parts[3]["training"]["log_every_tables"] = 1000  # never reached
            with training_run(parts[3], "train") as journal:
                run_training(parts, Clock())  # a checkpoint is due after every update
                path = journal.path
            events = [json.loads(line) for line in path.read_text().splitlines()]
            metrics = [e for e in events if e["event"] == "metrics"]
            self.assertEqual([e["global_step"] for e in metrics], [1, 2, 3, 4, 5, 6])
            self.assertEqual(sum(e["windows"] for e in metrics), 9)

    def test_background_and_blocking_checkpoints_hold_the_same_training_state(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            _, first, _ = run_training(setup(root / "background"), Clock())
            blocking = setup(root / "blocking")  # seeds the generators again
            blocking[3]["training"]["background_checkpoints"] = False
            _, second, _ = run_training(blocking, Clock())
            self.assertEqual(first["completed_updates"], second["completed_updates"])
            self.assert_checkpoint_equal(
                torch.load(root / "background/latest_checkpoint.pt", weights_only=True),
                torch.load(root / "blocking/latest_checkpoint.pt", weights_only=True),
            )
            for name in ("validation_history.json", "loss_history.json"):
                left = json.loads((root / "background" / name).read_text())
                right = json.loads((root / "blocking" / name).read_text())
                self.assertEqual(
                    [row.get("loss") for row in left],
                    [row.get("loss") for row in right],
                )

    def test_unfinished_checkpoint_file_of_a_killed_run_is_removed_at_start(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            parts = setup(root)
            parts[3]["training"]["max_run_hours"] = 1 / 3600
            run_training(parts, Clock())
            complete = (root / "latest_checkpoint.pt").read_bytes()
            # What a run killed in the middle of a write leaves behind.
            stale = root / ".latest_checkpoint.pt.0123456789abcdef.tmp"
            stale.write_bytes(complete[: len(complete) // 2])
            other = root / ".unrelated.tmp"
            other.write_bytes(b"keep")
            run_training(setup(root), Clock())
            self.assertFalse(stale.exists())
            self.assertTrue(other.exists())
            self.assertEqual(
                torch.load(root / "latest_checkpoint.pt", weights_only=True)["step"], 6
            )

    def test_checkpoint_upload_needs_working_storage_before_training_starts(self):
        with tempfile.TemporaryDirectory() as temporary:
            parts = setup(Path(temporary))
            parts[3]["training"]["checkpoint_upload"] = {"container": "checkpoints"}

            class Unreachable:
                def describe(self):
                    return "fake://checkpoints"

                def preflight(self):
                    raise RuntimeError("Checkpoint upload check failed")

            with patch(
                "timed_training.make_checkpoint_store", return_value=Unreachable()
            ):
                with self.assertRaisesRegex(RuntimeError, "upload check failed"):
                    run_training(parts, Clock())
            self.assertFalse((Path(temporary) / "latest_checkpoint.pt").exists())

    def test_every_checkpoint_is_uploaded_and_the_first_must_succeed(self):
        class Store(FakeStore):
            def __init__(self, fail_first=False):
                super().__init__()
                self.fail_first, self.steps, self.checked = fail_first, [], 0

            def preflight(self):
                self.checked += 1

            def begin(self, path, *, cancel=None):
                store = self

                class Upload(FakeUpload):
                    def commit(self, digests, size, metadata=None):
                        if store.fail_first and not store.steps:
                            store.steps.append("failed")
                            raise ConnectionError("storage unreachable")
                        result = super().commit(digests, size, metadata)
                        # What arrived is a loadable checkpoint of that step.
                        saved = torch.load(
                            io.BytesIO(bytes(self.data)), weights_only=True
                        )
                        store.steps.append(saved["step"])
                        return result

                return Upload(self, path, cancel)

        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            store = Store()
            with patch("timed_training.make_checkpoint_store", return_value=store):
                _, summary, _ = run_training(setup(root / "ok"), Clock())
            self.assertEqual(store.checked, 1)
            self.assertEqual(summary["completed_updates"], 6)
            # The first and the final checkpoint are always uploaded; in
            # between, an upload gives way when the next checkpoint is due.
            self.assertEqual(store.steps[0], 1)
            self.assertEqual(store.steps[-1], 6)
            self.assertEqual(store.steps, sorted(store.steps))
            failing = Store(fail_first=True)
            with patch("timed_training.make_checkpoint_store", return_value=failing):
                with self.assertRaisesRegex(RuntimeError, "could not be uploaded"):
                    run_training(setup(root / "failing"), Clock())
            # Training stopped at the first checkpoint, which is on disk.
            saved = torch.load(root / "failing/latest_checkpoint.pt", weights_only=True)
            self.assertEqual(saved["step"], 1)

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

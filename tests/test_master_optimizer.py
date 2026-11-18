"""Small BF16 updates must survive optimization and checkpoint continuation."""

import copy
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import torch

ROOT = Path(__file__).resolve().parents[1]
for stage in (
    "03_transforming_input_data",
    "04_patch_indexing",
    "05_model_training",
):
    sys.path.insert(0, str(ROOT / "src" / stage))
from master_optimizer import FP32MasterAdamW
from performance import make_optimizer


class MasterOptimizerTests(unittest.TestCase):
    def test_bounded_staging_matches_fp32_adamw_with_skipped_gradients(self):
        torch.manual_seed(7)
        parameters = [
            torch.nn.Parameter(torch.randn(size, dtype=torch.bfloat16))
            for size in (16, 7, 25, 3, 37)
        ]
        reference = [torch.nn.Parameter(p.detach().float()) for p in parameters]
        options = dict(lr=1e-4, weight_decay=0.01, foreach=False)
        actual = FP32MasterAdamW(parameters, gradient_bucket_bytes=128, **options)
        expected = torch.optim.AdamW(reference, **options)
        groups = actual.param_groups
        parent_step = torch.optim.AdamW.step
        staged_bytes = []

        def observe_staging(optimizer, *args, **kwargs):
            live = [p for p in optimizer.master_parameters if p.grad is not None]
            size = sum(p.grad.numel() * p.grad.element_size() for p in live)
            staged_bytes.append(size)
            self.assertTrue(size <= 128 or len(live) == 1)
            return parent_step(optimizer, *args, **kwargs)

        for step in range(9):
            for i, (parameter, target) in enumerate(zip(parameters, reference)):
                gradient = (
                    None
                    if i == 1 or (i + step) % 3 == 0
                    else torch.randn_like(parameter)
                )
                parameter.grad = gradient
                target.grad = None if gradient is None else gradient.float()
            with patch.object(torch.optim.AdamW, "step", new=observe_staging):
                actual.step()
            expected.step()
            self.assertIs(actual.param_groups, groups)
            for master, target in zip(actual.master_parameters, reference):
                torch.testing.assert_close(master, target, rtol=0, atol=0)
                self.assertIsNone(master.grad)
                if target in expected.state:
                    for key in ("step", "exp_avg", "exp_avg_sq"):
                        torch.testing.assert_close(
                            actual.state[master][key],
                            expected.state[target][key],
                            rtol=0,
                            atol=0,
                        )
                else:
                    self.assertNotIn(master, actual.state)
        self.assertGreater(len(staged_bytes), 9)
        self.assertLessEqual(max(staged_bytes), 37 * 4)

    def test_bucket_failure_cleans_staging_and_does_not_retry_partial_update(self):
        parameters = [
            torch.nn.Parameter(torch.ones(8, dtype=torch.bfloat16)) for _ in range(3)
        ]
        optimizer = FP32MasterAdamW(parameters, lr=1e-4, gradient_bucket_bytes=32)
        for parameter in parameters:
            parameter.grad = torch.ones_like(parameter)
        parent_step = torch.optim.AdamW.step
        calls = 0

        def fail_second_bucket(instance):
            nonlocal calls
            calls += 1
            if calls == 2:
                raise torch.OutOfMemoryError("simulated optimizer failure")
            return parent_step(instance)

        with patch.object(torch.optim.AdamW, "step", new=fail_second_bucket):
            with self.assertRaisesRegex(torch.OutOfMemoryError, "simulated"):
                optimizer.step()
        self.assertEqual(calls, 2)
        self.assertTrue(torch.all(optimizer.master_parameters[0] < 1))
        self.assertTrue(torch.all(optimizer.master_parameters[1] == 1))
        self.assertTrue(all(p.grad is None for p in optimizer.master_parameters))

    def test_saved_master_state_resumes_with_a_different_staging_budget(self):
        parameters = [
            torch.nn.Parameter(torch.ones(8, dtype=torch.bfloat16)) for _ in range(3)
        ]
        original = FP32MasterAdamW(parameters, lr=1e-4, gradient_bucket_bytes=32)
        for _ in range(8):
            for parameter in parameters:
                parameter.grad = torch.ones_like(parameter)
            original.step()
        self.assertTrue(all(torch.all(p == 1) for p in parameters))
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "optimizer.pt"
            torch.save(
                {
                    "weights": [p.detach() for p in parameters],
                    "optimizer": original.state_dict(),
                },
                path,
            )
            saved = torch.load(path, weights_only=True)
            resumed_parameters = [
                torch.nn.Parameter(w.clone()) for w in saved["weights"]
            ]
            resumed = FP32MasterAdamW(
                resumed_parameters, lr=0.5, gradient_bucket_bytes=64
            )
            resumed.load_state_dict(saved["optimizer"])
        for _ in range(32):
            for group in (parameters, resumed_parameters):
                for parameter in group:
                    parameter.grad = torch.ones_like(parameter)
            original.step()
            resumed.step()
        for left, right in zip(original.master_parameters, resumed.master_parameters):
            torch.testing.assert_close(left, right, rtol=0, atol=0)
            for key in ("step", "exp_avg", "exp_avg_sq"):
                torch.testing.assert_close(
                    original.state[left][key], resumed.state[right][key], rtol=0, atol=0
                )

    def test_updates_below_bf16_resolution_accumulate_like_fp32_adamw(self):
        parameter = torch.nn.Parameter(torch.ones(16, dtype=torch.bfloat16))
        reference = torch.nn.Parameter(parameter.detach().float())
        stalled = torch.nn.Parameter(parameter.detach().clone())
        settings = dict(lr=1e-4, weight_decay=0.01, foreach=False)
        optimizer = FP32MasterAdamW([parameter], **settings)
        expected = torch.optim.AdamW([reference], **settings)
        legacy = torch.optim.AdamW([stalled], **settings)
        for i in range(64):
            gradient = torch.full_like(parameter, 0.125 + i / 256)
            parameter.grad = gradient
            reference.grad = gradient.float()
            stalled.grad = gradient.clone()
            optimizer.step()
            expected.step()
            legacy.step()
        self.assertTrue(torch.all(stalled == 1))
        self.assertTrue(torch.all(parameter < 1))
        torch.testing.assert_close(
            optimizer.master_parameters[0], reference, rtol=0, atol=0
        )
        torch.testing.assert_close(parameter, reference.bfloat16(), rtol=0, atol=0)
        self.assertIsNone(optimizer.master_parameters[0].grad)
        for key in ("exp_avg", "exp_avg_sq"):
            self.assertEqual(
                optimizer.state[optimizer.master_parameters[0]][key].dtype,
                torch.float32,
            )

    def test_checkpoint_preserves_invisible_updates_and_exact_next_step(self):
        p = torch.nn.Parameter(torch.ones(3, dtype=torch.bfloat16))
        optimizer = FP32MasterAdamW([p], lr=1e-4, weight_decay=0.01)
        for _ in range(8):
            p.grad = torch.ones_like(p)
            optimizer.step()
        self.assertTrue(torch.all(p == 1))
        saved = copy.deepcopy(optimizer.state_dict())
        resumed_p = torch.nn.Parameter(p.detach().clone())
        resumed = FP32MasterAdamW([resumed_p], lr=0.5)
        resumed.load_state_dict(saved)
        for _ in range(32):
            p.grad = torch.ones_like(p)
            resumed_p.grad = torch.ones_like(resumed_p)
            optimizer.step()
            resumed.step()
        torch.testing.assert_close(resumed_p, p, rtol=0, atol=0)
        torch.testing.assert_close(
            resumed.master_parameters[0], optimizer.master_parameters[0], rtol=0, atol=0
        )
        self.assertEqual(resumed.param_groups[0]["lr"], 1e-4)
        resumed.zero_grad()
        self.assertIsNone(resumed_p.grad)

    def test_legacy_migration_keeps_moments_and_uses_restored_model_weights(self):
        p = torch.nn.Parameter(torch.full((4,), 1.5, dtype=torch.bfloat16))
        old = torch.optim.AdamW([p], lr=1e-4)
        p.grad = torch.full_like(p, 0.25)
        old.step()
        saved = copy.deepcopy(old.state_dict())
        resumed_p = torch.nn.Parameter(torch.zeros_like(p))
        resumed = FP32MasterAdamW([resumed_p], lr=0.5)
        with torch.no_grad():
            resumed_p.copy_(p)  # restore_checkpoint restores the model first.
        resumed.load_state_dict(saved)
        master = resumed.master_parameters[0]
        torch.testing.assert_close(master, p.float(), rtol=0, atol=0)
        for key in ("step", "exp_avg", "exp_avg_sq"):
            torch.testing.assert_close(
                resumed.state[master][key],
                saved["state"][0][key].float(),
                rtol=0,
                atol=0,
            )

    def test_missing_gradients_do_not_apply_decay_and_scheduler_controls_master_lr(
        self,
    ):
        p = torch.nn.Parameter(torch.ones(4, dtype=torch.bfloat16))
        optimizer = FP32MasterAdamW([p], lr=1e-3, weight_decay=0.1)
        scheduler = torch.optim.lr_scheduler.StepLR(optimizer, 1, gamma=0.5)
        optimizer.step()
        self.assertTrue(torch.all(p == 1))
        self.assertFalse(optimizer.state)
        p.grad = torch.ones_like(p)
        optimizer.step()
        scheduler.step()
        self.assertEqual(optimizer.param_groups[0]["lr"], 5e-4)
        optimizer.zero_grad(set_to_none=False)
        self.assertTrue(torch.all(p.grad == 0))

    def test_factory_uses_masters_only_for_low_precision_models(self):
        model = torch.nn.Linear(2, 1)
        settings = {"learning_rate": 1e-4, "weight_decay": 0.01}
        self.assertIs(type(make_optimizer(model, settings)), torch.optim.AdamW)
        self.assertIsInstance(
            make_optimizer(model.bfloat16(), settings), FP32MasterAdamW
        )

    def test_full_checkpoint_restore_allows_only_precision_upgrade(self):
        from checkpoints import restore_checkpoint
        from epoch_training import fit_epochs
        from learning_rate import make_scheduler
        from test_epoch_checkpoints import setup, validation_values

        with tempfile.TemporaryDirectory() as temporary:
            base = Path(temporary)
            model, records, config, optimizer = setup(base, epochs=1)
            fit_epochs(
                model,
                records,
                [{"source": "v"}],
                config,
                optimizer,
                validation_values([1.0]),
            )
            checkpoint = base / "latest_checkpoint.pt"
            saved = torch.load(checkpoint, weights_only=True)
            fresh_model, _, _, _ = setup(base, epochs=1)
            upgraded = FP32MasterAdamW(fresh_model.parameters(), lr=0.001)
            scheduler, _ = make_scheduler(upgraded, config["training"], 2)
            identity = copy.deepcopy(saved["identity"])
            identity["optimizer_class"] = "FP32MasterAdamW"
            restored = restore_checkpoint(
                checkpoint,
                fresh_model,
                upgraded,
                scheduler,
                identity,
                saved["schedule_plan"],
                base / "best.pt",
            )
            self.assertEqual(restored["step"], saved["step"])
            self.assertEqual(
                restored["execution_changes"]["optimizer_precision"]["current"],
                "FP32 master weights and moments",
            )
            for parameter, master in zip(
                fresh_model.parameters(), upgraded.master_parameters
            ):
                torch.testing.assert_close(parameter.float(), master, rtol=0, atol=0)
            for actual, expected in zip(
                upgraded.state.values(), saved["optimizer_state_dict"]["state"].values()
            ):
                torch.testing.assert_close(actual["step"], expected["step"])
            identity["optimization"]["learning_rate"] = 0.99
            with self.assertRaisesRegex(ValueError, "optimization settings differ"):
                restore_checkpoint(
                    checkpoint,
                    fresh_model,
                    upgraded,
                    scheduler,
                    identity,
                    saved["schedule_plan"],
                    base / "best.pt",
                )


if __name__ == "__main__":
    unittest.main()

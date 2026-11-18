"""Activation OOM retries never skip labels or apply partial optimizer updates."""

import copy
import sys
import unittest
from dataclasses import replace
from pathlib import Path
from unittest.mock import patch

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "tests"))
for stage in (
    "03_transforming_input_data",
    "04_patch_indexing",
    "05_model_training",
):
    sys.path.insert(0, str(ROOT / "src" / stage))
from epoch_training import accumulated_update
from performance import prepare_update
from test_training_performance import model_and_records


class TrainingOOMTests(unittest.TestCase):
    def test_allocator_uses_eleven_gib_without_old_eighty_percent_cap(self):
        from types import SimpleNamespace

        from performance import configure_cuda_memory

        gib = 2**30
        with (
            patch("torch.cuda.is_available", return_value=True),
            patch("torch.cuda.current_device", return_value=0),
            patch(
                "torch.cuda.get_device_properties",
                return_value=SimpleNamespace(total_memory=12 * gib),
            ),
            patch("torch.cuda.mem_get_info", return_value=(11.75 * gib, 12 * gib)),
            patch("torch.cuda.memory_reserved", return_value=0),
            patch("torch.cuda.set_per_process_memory_fraction") as fraction,
        ):
            configure_cuda_memory({"cuda_memory_limit_gib": 11.0})
            self.assertAlmostEqual(fraction.call_args.args[0], 11 / 12)
        with (
            patch("torch.cuda.is_available", return_value=True),
            patch("torch.cuda.current_device", return_value=0),
            patch(
                "torch.cuda.get_device_properties",
                return_value=SimpleNamespace(total_memory=12 * gib),
            ),
            patch("torch.cuda.mem_get_info", return_value=(10 * gib, 12 * gib)),
            patch("torch.cuda.memory_reserved", return_value=0),
            patch("torch.cuda.set_per_process_memory_fraction") as fraction,
        ):
            configure_cuda_memory({"cuda_memory_limit_gib": 11.0})
            self.assertAlmostEqual(fraction.call_args.args[0], 9.5 / 12)

    def test_retry_after_partial_backward_equals_smaller_reference_update(self):
        model, records = model_and_records("train")
        reference = copy.deepcopy(model)
        group = [records[:4]]
        prepared = prepare_update(group, microbatch_size=2)
        ref_prepared = prepare_update(group, microbatch_size=1)
        optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
        ref_optimizer = torch.optim.AdamW(reference.parameters(), lr=1e-3)
        expected = accumulated_update(
            reference, group, ref_optimizer, 1.0, ref_prepared
        )
        original = model.forward_encoded
        calls = 0

        def fail_second(x, prepared, labels):
            nonlocal calls
            calls += 1
            if calls == 2:
                torch.rand(8)  # Failed work must not advance the RNG.
                raise torch.OutOfMemoryError("simulated activation OOM")
            return original(x, prepared, labels)

        before_rng = torch.get_rng_state().clone()
        with (
            patch.object(model, "forward_encoded", side_effect=fail_second),
            patch.object(optimizer, "step", wraps=optimizer.step) as step,
        ):
            actual = accumulated_update(model, group, optimizer, 1.0, prepared)
        self.assertEqual(actual["oom_retries"], 1)
        self.assertEqual(actual["microbatches"], 4)
        self.assertEqual(actual["target_points"], expected["target_points"])
        self.assertAlmostEqual(actual["loss"], expected["loss"], places=6)
        self.assertEqual(step.call_count, 1)
        torch.testing.assert_close(torch.get_rng_state(), before_rng, rtol=0, atol=0)
        for a, b in zip(model.parameters(), reference.parameters()):
            torch.testing.assert_close(a, b, atol=1e-7, rtol=1e-5)
            torch.testing.assert_close(
                optimizer.state[a]["exp_avg"],
                ref_optimizer.state[b]["exp_avg"],
                atol=1e-7,
                rtol=1e-5,
            )
        # Learned shape limit prevents the same oversized execution next time.
        next_result = accumulated_update(model, group, optimizer, 1.0, prepared)
        self.assertEqual(next_result["microbatches"], 4)
        self.assertEqual(next_result["oom_retries"], 0)

    def test_optimizer_oom_is_never_retried(self):
        model, records = model_and_records("train")
        optimizer = torch.optim.SGD(model.parameters(), lr=1e-3)
        with patch.object(
            optimizer, "step", side_effect=torch.OutOfMemoryError("optimizer OOM")
        ) as step:
            with self.assertRaisesRegex(torch.OutOfMemoryError, "optimizer OOM"):
                accumulated_update(model, [records[:2]], optimizer, 1.0)
        self.assertEqual(step.call_count, 1)

    def test_retry_preserves_four_lead_weighting_with_unequal_label_counts(self):
        from test_four_token_policy import model_fixture

        model = model_fixture()
        reference = copy.deepcopy(model)
        _, records = model_and_records("train")
        group = [records[:4]]
        expected = accumulated_update(
            reference,
            group,
            torch.optim.SGD(reference.parameters(), lr=1e-3),
            1.0,
            prepare_update(group, microbatch_size=1),
        )
        original = model.forward_encoded

        def require_single(x, prepared, labels):
            if x.shape[0] > 1:
                raise torch.OutOfMemoryError("simulated weighted loss OOM")
            return original(x, prepared, labels)

        with patch.object(model, "forward_encoded", side_effect=require_single):
            actual = accumulated_update(
                model, group, torch.optim.SGD(model.parameters(), lr=1e-3), 1.0
            )
        self.assertEqual(actual["oom_retries"], 2)
        self.assertEqual(actual["target_loss_weight"], expected["target_loss_weight"])
        self.assertAlmostEqual(actual["loss"], expected["loss"], places=6)
        for a, b in zip(model.parameters(), reference.parameters()):
            torch.testing.assert_close(a, b, atol=1e-7, rtol=1e-5)

    def test_single_table_can_retry_with_all_layers_checkpointed(self):
        model, records = model_and_records("train")
        model.config = replace(model.config, gradient_checkpointing=True)
        model.checkpoint_token_threshold = 1024
        optimizer = torch.optim.SGD(model.parameters(), lr=1e-3)
        original = model.forward_encoded
        calls = 0

        def fail_once(x, prepared, labels):
            nonlocal calls
            calls += 1
            if calls == 1:
                raise torch.OutOfMemoryError("single table needs recomputation")
            return original(x, prepared, labels)

        with patch.object(model, "forward_encoded", side_effect=fail_once):
            result = accumulated_update(model, [[records[0]]], optimizer, 1.0)
        self.assertEqual(result["oom_retries"], 1)
        self.assertEqual(result["windows"], 1)
        self.assertEqual(model.checkpoint_token_threshold, 0)

    def test_one_table_oom_fails_without_stepping_or_skipping(self):
        model, records = model_and_records("train")
        optimizer = torch.optim.SGD(model.parameters(), lr=1e-3)
        with (
            patch.object(
                model, "forward_encoded", side_effect=torch.OutOfMemoryError("single")
            ),
            patch.object(optimizer, "step") as step,
        ):
            with self.assertRaisesRegex(torch.OutOfMemoryError, "single"):
                accumulated_update(model, [[records[0]]], optimizer, 1.0)
        step.assert_not_called()
        self.assertTrue(all(p.grad is None for p in model.parameters()))


if __name__ == "__main__":
    unittest.main()

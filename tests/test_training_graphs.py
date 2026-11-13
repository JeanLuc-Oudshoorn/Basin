"""Training graph correctness through real RWKV, optimizer and checkpointing."""

import copy
import sys
import unittest
from pathlib import Path
from unittest.mock import patch

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src/05_model_training"))
sys.path.insert(0, str(ROOT / "src/03_transforming_input_data"))
sys.path.insert(0, str(ROOT))
from epoch_training import accumulated_update, training_options
from performance import make_optimizer
from train import ForecastConfig, RWKVForecaster
from training_graphs import TrainingGraphs, configure_training_graphs
from variate_attention import configure_variate_execution


class GraphConfigTests(unittest.TestCase):
    def test_capture_admission_does_not_thrash_on_mixed_shapes(self):
        manager = TrainingGraphs(torch.nn.Linear(2, 2), {})
        short = (4, 4, 16, 1024)
        other = (4, 5, 32, 1024)
        for shape in [short, other] * 30:
            manager.begin_update()
            self.assertFalse(manager.admit(shape))
        for _ in range(7):
            manager.begin_update()
            self.assertFalse(manager.admit(short))
        # Accumulation/microbatch size must not satisfy admission in one update.
        for _ in range(32):
            self.assertFalse(manager.admit(short))
        manager.begin_update()
        self.assertTrue(manager.admit(short))
        manager.shapes[short] = None
        self.assertFalse(manager.admit(other))
        self.assertTrue(manager.admit(short))  # Cached profiles replay immediately.
        for _ in range(20):
            self.assertFalse(manager.admit((4, 4, 128, 1024)))
        self.assertEqual(manager.counts["length_skips"], 20)

    def test_graph_pool_retirement_waits_for_all_device_work(self):
        events = []
        model = torch.nn.Linear(2, 2)
        manager = TrainingGraphs(model, {})
        shape = (1, 1, 16, 2)

        class Pool:
            def __del__(self):
                events.append("released")

        manager.entries[(0, shape)] = Pool()
        manager.costs[(0, shape)] = manager.bytes = 16
        manager.shapes[shape] = None
        with patch(
            "torch.cuda.synchronize",
            side_effect=lambda device: events.append("synchronized"),
        ):
            manager.retire(shape)
        self.assertEqual(events[:2], ["synchronized", "released"])
        self.assertEqual(manager.bytes, 0)
        self.assertFalse(manager.entries)

    def test_only_capture_and_allocation_errors_can_fall_back(self):
        for message in (
            "CUDA driver error: operation failed due to a previous error during capture",
            "CUDA error: operation not permitted when stream is capturing",
            "cudaErrorStreamCaptureInvalidated",
            "CUDA error: operation failed due to a previous error during capture\n"
            "Compile with TORCH_USE_CUDA_DSA to enable device-side assertions.",
        ):
            self.assertTrue(
                TrainingGraphs._recoverable_capture_error(RuntimeError(message))
            )
        self.assertTrue(
            TrainingGraphs._recoverable_capture_error(torch.OutOfMemoryError())
        )
        self.assertTrue(
            TrainingGraphs._recoverable_capture_error(
                torch.AcceleratorError(
                    "CUDA error: operation failed due to a previous error during capture"
                )
            )
        )
        for message in (
            "unexpected tensor shape",
            "CUDA error: an illegal memory access during capture",
            "CUDA error: device-side assert triggered during capture",
        ):
            self.assertFalse(
                TrainingGraphs._recoverable_capture_error(RuntimeError(message))
            )

    def test_invalid_graph_options(self):
        for options in (
            {"training_cuda_graphs": "yes"},
            {"training_graph_cache_shapes": 0},
            {"training_graph_memory_gib": -1},
            {"training_graph_memory_gib": True},
            {"shape_run_batches": 0},
            {"shape_run_batches": 128, "bucket_by_length": False},
            {"training_graph_min_reuse": 0},
            {"training_graph_max_tokens": False},
        ):
            with self.assertRaises(ValueError):
                training_options(options)


@unittest.skipUnless(torch.cuda.is_available(), "CUDA required")
class TrainingGraphTests(unittest.TestCase):
    def test_real_capture_failure_recovers_without_losing_an_update_or_rng_state(self):
        torch.set_num_threads(4)
        torch.manual_seed(51)
        config = ForecastConfig(
            objective="next_patch",
            patch_size=16,
            width=128,
            layers=2,
            max_patches=16,
            context_length=256,
            output_patches=4,
        )
        reference = RWKVForecaster(config).cuda().bfloat16().train()
        optimized = copy.deepcopy(reference)
        options = dict(
            compile_variate=True,
            compile_variate_inference=True,
            training_cuda_graphs=False,
            learning_rate=0.001,
            weight_decay=0.01,
            training_graph_memory_gib=0.25,
            training_graph_min_reuse=1,
            training_graph_max_tokens=None,
        )
        configure_variate_execution(reference, options, ROOT)
        configure_variate_execution(
            optimized, dict(options, training_cuda_graphs=True), ROOT
        )
        optimizers = [
            make_optimizer(model, options) for model in (reference, optimized)
        ]
        prepared = reference.tokenize_next(
            torch.randn(4, 1, 256),
            past_covariates=torch.randn(4, 3, 256),
            training=True,
        )
        points = int(torch.isfinite(prepared["labels"]).sum())
        prepared["observed_points"] = points
        batches = [(prepared, prepared["labels"], points)] * 2
        factory = torch.cuda.make_graphed_callables
        calls = 0

        def fail_second_capture(proxy, args):
            nonlocal calls
            calls += 1
            if calls == 2:
                execute = proxy.execute

                def invalid_capture(mix, x):
                    if torch.cuda.is_current_stream_capturing():
                        # Real CUDA capture invalidation, not a mocked Python error.
                        torch.cuda.synchronize()
                    return execute(mix, x)

                proxy.execute = invalid_capture
            return factory(proxy, args)

        stream = torch.cuda.current_stream()
        for update in range(3):
            results, rng_states = [], []
            for model, optimizer in zip((reference, optimized), optimizers):
                torch.manual_seed(700 + update)
                with patch(
                    "torch.cuda.make_graphed_callables", side_effect=fail_second_capture
                ):
                    results.append(
                        accumulated_update(
                            model, [[{}]] * 2, optimizer, 1.0, prepared_batches=batches
                        )
                    )
                rng_states.append(torch.cuda.get_rng_state())
            self.assertEqual(torch.cuda.current_stream(), stream)
            self.assertEqual(results[0]["loss"], results[1]["loss"])
            torch.testing.assert_close(rng_states[0], rng_states[1], rtol=0, atol=0)
            for p, q in zip(reference.parameters(), optimized.parameters()):
                torch.testing.assert_close(p.grad, q.grad, rtol=0, atol=0)
                torch.testing.assert_close(p, q, rtol=0, atol=0)
        self.assertEqual(calls, 2)  # Never retry a broken capture each microbatch.
        self.assertEqual(optimized.training_graphs.counts["capture_failures"], 1)
        self.assertEqual(optimized.training_graphs.summary()["entries"], 0)
        self.assertEqual(optimized.training_graphs.hooks, {})
        self.assertIsNotNone(optimized.training_graphs.disabled_reason)
        optimized.training_graphs.close()

    def test_real_updates_gradients_checkpoint_recompute_and_resume(self):
        torch.set_num_threads(4)
        torch.manual_seed(23)
        config = ForecastConfig(
            objective="next_patch",
            patch_size=16,
            width=128,
            layers=2,
            max_patches=16,
            context_length=256,
            gradient_checkpointing=True,
            output_patches=4,
        )
        reference = RWKVForecaster(config).cuda().bfloat16().train()
        with torch.no_grad():
            for block in reference.blocks:
                block.att.output.weight.normal_(std=0.01)
                block.ffn.value.weight.normal_(std=0.01)
        optimized = copy.deepcopy(reference)
        options = dict(
            compile_variate=True,
            compile_variate_inference=False,
            training_cuda_graphs=False,
            learning_rate=0.001,
            weight_decay=0.01,
            training_graph_cache_shapes=2,
            training_graph_memory_gib=0.25,
            training_graph_min_reuse=1,
            training_graph_max_tokens=None,
        )
        configure_variate_execution(reference, options, ROOT)
        configure_variate_execution(
            optimized, dict(options, training_cuda_graphs=True), ROOT
        )
        parameters = list(optimized.state_dict())
        optimizers = [
            make_optimizer(model, options) for model in (reference, optimized)
        ]
        # One graphed layer and one activation-checkpointed layer at V=3,T=16.
        reference.checkpoint_token_threshold = optimized.checkpoint_token_threshold = 24
        for update in range(3):
            batches = []
            for length in (80, 128, 80):
                context = torch.randn(1, 1, length)
                context[..., 4] = float("nan")
                prepared = reference.tokenize_next(
                    context, past_covariates=torch.randn(1, 2, length), training=True
                )
                points = int(torch.isfinite(prepared["labels"]).sum())
                prepared["observed_points"] = points
                batches.append((prepared, prepared["labels"], points))
            results = []
            for model, optimizer in zip((reference, optimized), optimizers):
                torch.manual_seed(100 + update)
                results.append(
                    accumulated_update(
                        model, [[{}]] * 3, optimizer, 1.0, prepared_batches=batches
                    )
                )
            self.assertEqual(results[0]["loss"], results[1]["loss"])
            for p, q in zip(reference.parameters(), optimized.parameters()):
                torch.testing.assert_close(p.grad, q.grad, rtol=0, atol=0)
                torch.testing.assert_close(p, q, rtol=0, atol=0)
            self.assertEqual(parameters, list(optimized.state_dict()))
        self.assertGreater(optimized.training_graphs.counts["replays"], 0)
        self.assertEqual(len(optimized.training_graphs.entries), 1)
        # Evaluation bypasses all training graphs; weights retain ordinary names.
        optimized.eval()
        reference.eval()
        x = torch.randn(1, 1, 80, device="cuda")
        torch.testing.assert_close(
            reference.forecast(x, 16).quantiles,
            optimized.forecast(x, 16).quantiles,
            rtol=0,
            atol=0,
        )
        saved_model = copy.deepcopy(optimized.state_dict())
        saved_optimizer = copy.deepcopy(optimizers[1].state_dict())
        optimized.training_graphs.close()
        optimized.load_state_dict(saved_model)
        optimizers[1].load_state_dict(saved_optimizer)
        optimized.train()
        configure_training_graphs(optimized, dict(options, training_cuda_graphs=True))
        self.assertEqual(optimized.training_graphs.summary()["entries"], 0)
        for model, optimizer in zip((reference, optimized), optimizers):
            model.train()
            torch.manual_seed(200)
            accumulated_update(
                model, [[{}]] * 3, optimizer, 1.0, prepared_batches=batches
            )
        for p, q in zip(reference.parameters(), optimized.parameters()):
            torch.testing.assert_close(p, q, rtol=0, atol=0)
        configure_training_graphs(
            optimized,
            dict(options, training_cuda_graphs=True, training_graph_cache_shapes=1),
        )
        for variables in (2, 3, 2):
            prepared = reference.tokenize_next(
                torch.randn(1, 1, 80),
                past_covariates=torch.randn(1, variables - 1, 80),
                training=True,
            )
            points = int(torch.isfinite(prepared["labels"]).sum())
            prepared["observed_points"] = points
            for model, optimizer in zip((reference, optimized), optimizers):
                torch.manual_seed(301)
                accumulated_update(
                    model,
                    [[{}]],
                    optimizer,
                    1.0,
                    prepared_batches=[(prepared, prepared["labels"], points)],
                )
            self.assertLessEqual(len(optimized.training_graphs.shapes), 1)
            self.assertLessEqual(
                optimized.training_graphs.bytes, optimized.training_graphs.budget
            )
            for p, q in zip(reference.parameters(), optimized.parameters()):
                torch.testing.assert_close(p, q, rtol=0, atol=0)
        self.assertGreaterEqual(
            optimized.training_graphs.counts["profile_evictions"], 2
        )


if __name__ == "__main__":
    unittest.main()

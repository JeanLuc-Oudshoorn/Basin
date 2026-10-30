"""Execution-only attention changes preserve weights, predictions and gradients."""

import copy
import sys
import unittest
from pathlib import Path
from unittest.mock import Mock

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src/05_model_training"))
from train import VariateMix
from variate_attention import compiled_attention, projected_attention


class VariateAttentionTests(unittest.TestCase):
    def test_math_path_preserves_predictions_and_all_gradients(self):
        torch.manual_seed(13)
        for variables in (1, 3, 8, 11):
            with self.subTest(variables=variables):
                original = VariateMix(64, 4).double()
                optimized = copy.deepcopy(original)
                x = torch.randn(
                    2, variables, 7, 64, dtype=torch.float64, requires_grad=True
                )
                y = x.detach().clone().requires_grad_(True)
                expected = original(x)
                actual = projected_attention(optimized, y)
                torch.testing.assert_close(actual, expected, atol=1e-10, rtol=1e-10)
                gradient = torch.randn_like(actual)
                expected.backward(gradient)
                actual.backward(gradient)
                torch.testing.assert_close(x.grad, y.grad, atol=1e-10, rtol=1e-10)
                for a, b in zip(original.parameters(), optimized.parameters()):
                    torch.testing.assert_close(a.grad, b.grad, atol=1e-10, rtol=1e-10)

    def test_evaluation_and_no_grad_keep_original_path_and_checkpoint_names(self):
        module = VariateMix(64, 4)
        original = copy.deepcopy(module)
        module.compiled_forward = Mock(side_effect=AssertionError("should use eager"))
        self.assertEqual(original.state_dict().keys(), module.state_dict().keys())
        original.load_state_dict(module.state_dict(), strict=True)
        x = torch.randn(2, 3, 7, 64)
        with torch.no_grad():
            torch.testing.assert_close(module(x), original(x))
        module.eval()
        original.eval()
        torch.testing.assert_close(module(x), original(x))
        module.train()
        original.train()
        wider = torch.randn(2, 11, 7, 64)
        torch.testing.assert_close(module(wider), original(wider))
        module.compiled_forward.assert_not_called()

    @unittest.skipUnless(torch.cuda.is_available(), "CUDA required for Inductor")
    def test_cuda_compiled_forward_backward_and_checkpoint_recomputation(self):
        from torch.utils.checkpoint import checkpoint

        torch.manual_seed(29)
        import os

        os.environ.setdefault(
            "TORCHINDUCTOR_CACHE_DIR", str(ROOT / ".cache/torchinductor")
        )
        os.environ.setdefault(
            "TRITON_CACHE_DIR", str(ROOT / ".cache/torchinductor/triton")
        )
        original = VariateMix(128, 4).cuda().bfloat16()
        optimized = copy.deepcopy(original)
        optimized.compiled_forward = compiled_attention()
        for variables in (1, 3, 8, 11):
            for recompute in (False, True):
                with self.subTest(variables=variables, recompute=recompute):
                    original.zero_grad(set_to_none=True)
                    optimized.zero_grad(set_to_none=True)
                    x = torch.randn(
                        2,
                        variables,
                        16,
                        128,
                        device="cuda",
                        dtype=torch.bfloat16,
                        requires_grad=True,
                    )
                    y = x.detach().clone().requires_grad_(True)
                    a = original(x)
                    b = (
                        checkpoint(optimized, y, use_reentrant=False)
                        if recompute
                        else optimized(y)
                    )
                    gradient = torch.randn_like(a)
                    a.backward(gradient)
                    b.backward(gradient)
                    pairs = [(a, b), (x.grad, y.grad)] + [
                        (p.grad, q.grad)
                        for p, q in zip(original.parameters(), optimized.parameters())
                    ]
                    for expected, actual in pairs:
                        self.assertTrue(torch.isfinite(actual).all())
                        # Relative vector error, avoiding loose elementwise tolerances
                        # on gradients close to zero. BF16 kernels round differently.
                        error = (expected.float() - actual.float()).norm()
                        bound = expected.float().norm().clamp_min(1e-6)
                        self.assertLess(float((error / bound).detach()), 0.02)


if __name__ == "__main__":
    unittest.main()

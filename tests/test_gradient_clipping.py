"""The trainer's clipping must give exactly what clip_grad_norm_ gives."""

import sys
import unittest
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src" / "05_model_training"))
from gradient_clipping import clip_gradients, measure_gradients, scale_gradients


def bits(tensor):
    return (
        tensor.detach()
        .contiguous()
        .view(torch.int16 if tensor.dtype == torch.bfloat16 else torch.int32)
    )


class GradientClippingTests(unittest.TestCase):
    SHAPES = ((), (3,), (17, 5), (64,), (9, 4, 2), (1,), (33,), (128, 16))

    def parameters(self, device, scale, *, mixed=True, seed=3):
        generator = torch.Generator().manual_seed(seed)
        result = []
        for index, shape in enumerate(self.SHAPES):
            dtype = torch.float32 if mixed and index % 3 == 1 else torch.bfloat16
            parameter = torch.nn.Parameter(torch.zeros(shape, dtype=dtype).to(device))
            if index != 5:  # one parameter never receives a gradient
                gradient = torch.randn(shape, generator=generator) * scale
                parameter.grad = gradient.to(device, dtype)
            result.append(parameter)
        return result

    def compare(self, device, scale, max_norm, *, mixed=True):
        expected = self.parameters(device, scale, mixed=mixed)
        actual = self.parameters(device, scale, mixed=mixed)
        reference = torch.nn.utils.clip_grad_norm_(
            expected, max_norm, error_if_nonfinite=False, foreach=True
        )
        before = [None if p.grad is None else bits(p.grad).clone() for p in actual]
        norm, coefficient, groups = measure_gradients(actual, max_norm)
        self.assertEqual(norm.dtype, reference.dtype)
        self.assertTrue(torch.equal(bits(norm), bits(reference)))
        for parameter, saved in zip(actual, before):  # measuring changes nothing
            if saved is not None:
                self.assertTrue(torch.equal(bits(parameter.grad), saved))
        clipped = float(coefficient) != 1.0
        if clipped:
            scale_gradients(groups, coefficient)
        for left, right in zip(actual, expected):
            self.assertEqual(left.grad is None, right.grad is None)
            if left.grad is not None:
                self.assertTrue(torch.equal(bits(left.grad), bits(right.grad)))
        return clipped

    def devices(self):
        return ["cpu"] + (["cuda"] if torch.cuda.is_available() else [])

    def test_matches_clip_grad_norm_when_clipping_and_when_not(self):
        for device in self.devices():
            for mixed in (True, False):
                with self.subTest(device=device, mixed=mixed):
                    # Small gradients: below the limit, nothing may change.
                    self.assertFalse(self.compare(device, 1e-3, 1.0, mixed=mixed))
                    # Large gradients: clipped exactly as PyTorch clips them.
                    self.assertTrue(self.compare(device, 3.0, 1.0, mixed=mixed))
                    self.assertTrue(self.compare(device, 1e-3, 1e-4, mixed=mixed))

    def test_norms_near_the_limit_follow_pytorch_on_either_side(self):
        for device in self.devices():
            outcomes = set()
            for step in range(40):
                scale = 0.019 + step * 1e-4  # total norm runs from 0.9 to 1.1
                outcomes.add(self.compare(device, scale, 1.0))
            self.assertEqual(outcomes, {True, False}, device)

    def test_drop_in_returns_the_norm_and_handles_no_gradients(self):
        parameters = self.parameters("cpu", 3.0)
        reference = self.parameters("cpu", 3.0)
        norm = clip_gradients(parameters, 1.0)
        expected = torch.nn.utils.clip_grad_norm_(reference, 1.0, foreach=True)
        self.assertTrue(torch.equal(bits(norm), bits(expected)))
        for left, right in zip(parameters, reference):
            if left.grad is not None:
                self.assertTrue(torch.equal(bits(left.grad), bits(right.grad)))
        empty = [torch.nn.Parameter(torch.zeros(3))]
        norm, coefficient, groups = measure_gradients(empty, 1.0)
        self.assertEqual((float(norm), float(coefficient), groups), (0.0, 1.0, []))


if __name__ == "__main__":
    unittest.main()

"""Run in the configured CUDA environment: python -m unittest discover -s tests -p test_forecaster.py."""

import sys
import unittest
from pathlib import Path

import pandas as pd
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src/05_model_training"))
from train import ForecastConfig, RWKVForecaster
from transform import frame_window


class ConfigTests(unittest.TestCase):
    def test_window_roles_exclude_unavailable_future(self):
        frame = pd.DataFrame(
            {"outcome": range(12), "outcome_lag_1": range(12), "calendar": range(12)}
        )
        batch = frame_window(
            frame,
            cutoff=8,
            context_length=5,
            horizon=3,
            past_covariates=["outcome_lag_1"],
            known_covariates=["calendar"],
            device="cpu",
        )
        self.assertEqual(batch["past_covariates"].flatten().tolist(), [3, 4, 5, 6, 7])
        self.assertEqual(batch["future_covariates"].flatten().tolist(), [8, 9, 10])
        self.assertEqual(batch["future_target"].flatten().tolist(), [8, 9, 10])
        with self.assertRaises(ValueError):
            frame_window(
                frame,
                cutoff=8,
                context_length=5,
                horizon=3,
                past_covariates=["outcome"],
                device="cpu",
            )

    def test_invalid_config(self):
        for kwargs in (
            {"width": 65},
            {"layers": 1},
            {"patch_size": 0},
            {"quantiles": (0.5, 0.1)},
            {"max_patches": 17},
        ):
            with self.assertRaises(ValueError):
                ForecastConfig(**kwargs)


@unittest.skipUnless(
    torch.cuda.is_available() and torch.cuda.is_bf16_supported(), "CUDA bf16 required"
)
class ForecastTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.manual_seed(11)
        cls.model = RWKVForecaster(ForecastConfig(patch_size=4, max_patches=32)).to(
            device="cuda", dtype=torch.bfloat16
        )

    def test_missing_constant_and_partial_patches(self):
        x = torch.ones(2, 3, 17, device="cuda")
        x[0, 0] = float("nan")
        output = self.model.forecast(x, 7)
        self.assertEqual(output.point.shape, (2, 3, 7))
        self.assertTrue(torch.isfinite(output.quantiles).all())

    def test_permutation_equivariance_and_batch_isolation(self):
        x = torch.randn(2, 3, 19, device="cuda")
        original = self.model.forecast(x, 9).quantiles
        permuted = self.model.forecast(x[:, [2, 0, 1]], 9).quantiles
        torch.testing.assert_close(
            original[:, [2, 0, 1]], permuted, rtol=0.03, atol=0.03
        )
        x[1] += 100
        changed = self.model.forecast(x, 9).quantiles
        torch.testing.assert_close(original[0], changed[0], rtol=0, atol=0)

    def test_validation(self):
        x = torch.ones(1, 1, 17, device="cuda")
        for kwargs in (
            {"horizon": 0},
            {"horizon": 200},
            {"horizon": 7, "future_covariates": x},
            {"horizon": 7, "future_target": x[..., :3]},
            {"horizon": 7, "future_target": torch.full_like(x[..., :7], float("nan"))},
        ):
            with self.assertRaises(ValueError):
                self.model(x, **kwargs)

    def test_covariate_permutation_and_variable_count(self):
        x = torch.randn(1, 1, 19, device="cuda")
        covariates = torch.randn(1, 3, 19, device="cuda")
        original = self.model.forecast(x, 9, past_covariates=covariates).point
        reordered = self.model.forecast(
            x, 9, past_covariates=covariates[:, [2, 0, 1]]
        ).point
        torch.testing.assert_close(original, reordered, rtol=0.03, atol=0.03)
        fewer = self.model.forecast(
            x[..., :17], 7, past_covariates=covariates[:, :1, :17]
        )
        self.assertEqual(fewer.point.shape, (1, 1, 7))
        self.assertTrue(torch.isfinite(fewer.point).all())


if __name__ == "__main__":
    unittest.main()

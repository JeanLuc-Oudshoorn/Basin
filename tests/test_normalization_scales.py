"""Tiny positive scales survive normalization; zero/missing inputs stay finite."""

import sys
import unittest
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src/04_creating_embeddings"))
from embed import ForecastConfig, PatchEncoder


def encoder(objective):
    return PatchEncoder(
        ForecastConfig(
            objective=objective,
            patch_size=16,
            width=64,
            context_length=64,
            max_patches=16,
        )
    )


class NormalizationScaleTests(unittest.TestCase):
    def test_positive_scales_are_preserved_for_targets_and_covariates(self):
        base = torch.linspace(-1, 1, 64).reshape(1, 1, 64)
        for objective in ("next_patch", "masked_horizon"):
            model = encoder(objective)
            for amplitude in (1e-3, 1e-6, 1e-8, 1e-12):
                with self.subTest(objective=objective, amplitude=amplitude):
                    context = base * amplitude
                    result = model.tokenize(
                        context, horizon=16, past_covariates=context * 2
                    )
                    expected = torch.cat([context, context * 2], dim=1).std(
                        -1, correction=0
                    )
                    actual = (
                        result["scale"][:, :, -1, 0]
                        if objective == "next_patch"
                        else result["scale"][:, :, 0]
                    )
                    torch.testing.assert_close(actual, expected, atol=0, rtol=1e-5)
                    self.assertTrue(torch.isfinite(result["features"]).all())

    def test_unit_changes_do_not_change_normalized_features_below_old_cutoff(self):
        context = torch.linspace(-1e-8, 1e-8, 64).reshape(1, 1, 64)
        for objective in ("next_patch", "masked_horizon"):
            model = encoder(objective)
            first = model.tokenize(context, horizon=16)
            second = model.tokenize(context * 100, horizon=16)
            torch.testing.assert_close(
                first["features"], second["features"], atol=2e-6, rtol=2e-6
            )
            torch.testing.assert_close(
                first["scale"] * 100, second["scale"], atol=0, rtol=2e-6
            )

    def test_zero_and_all_missing_histories_keep_safe_finite_scales(self):
        for objective in ("next_patch", "masked_horizon"):
            model = encoder(objective)
            for value in (0.0, 7.0, float("nan")):
                result = model.tokenize(torch.full((1, 1, 64), value), horizon=16)
                self.assertTrue(torch.isfinite(result["features"]).all())
                torch.testing.assert_close(
                    result["scale"], torch.ones_like(result["scale"])
                )

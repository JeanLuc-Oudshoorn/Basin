"""Graph admission limits are explicit and never require GPU work on fallback."""

import sys
import unittest
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src/05_model_training"))
from fast_inference import GraphCache


class GraphAdmissionTests(unittest.TestCase):
    def test_disabled_cache_reports_fallbacks(self):
        cache = GraphCache(enabled=False)
        value = cache.run("shape", lambda inputs: inputs["x"] * 2, {"x": torch.ones(2)})
        torch.testing.assert_close(value, torch.full((2,), 2.0))
        summary = cache.summary()
        self.assertEqual(summary["requests"], 1)
        self.assertEqual(summary["fallback_reasons"], {"disabled": 1})
        self.assertEqual(summary["replay_coverage"], 0)

    def test_profile_limit_and_memory_limit_do_not_capture_or_evict(self):
        for reason, cache in (
            ("profile_limit", GraphCache(1)),
            ("memory_limit", GraphCache(256, memory_gib=1e-6)),
        ):
            if reason == "profile_limit":
                cache.entries["keep"] = object()
            value = cache.run(
                "other", lambda inputs: inputs["x"] + 3, {"x": torch.ones(2)}
            )
            torch.testing.assert_close(value, torch.full((2,), 4.0))
            self.assertEqual(cache.summary()["fallback_reasons"], {reason: 1})
            self.assertEqual(cache.summary()["captures"], 0)
            if reason == "profile_limit":
                self.assertIn("keep", cache.entries)
            cache.clear()
            self.assertEqual(cache.summary()["retained_cache_bytes"], 0)
            self.assertEqual(len(cache.entries), 0)

    def test_invalid_memory_limits(self):
        for value in (True, 0, -1, float("inf"), float("nan"), "1.5"):
            with self.subTest(value=value), self.assertRaises(ValueError):
                GraphCache(memory_gib=value)

    def test_rare_profiles_do_not_capture(self):
        cache = GraphCache(256)
        cache.allowed_keys = {"frequent"}
        cache.run("rare", lambda inputs: inputs["x"], {"x": torch.ones(1)})
        self.assertEqual(cache.summary()["fallback_reasons"], {"low_reuse": 1})
        self.assertEqual(cache.summary()["captures"], 0)
        cache.clear()
        self.assertIsNone(cache.allowed_keys)


if __name__ == "__main__":
    unittest.main()

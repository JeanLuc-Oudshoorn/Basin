"""Shape admission matches observed request layouts and handles legacy rollout."""

import sys
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
for stage in ("03_transforming_input_data", "05_model_training", "06_holdout_test"):
    sys.path.insert(0, str(ROOT / "src" / stage))
from graph_profiles import profile_counts


class GraphProfileTests(unittest.TestCase):
    def test_direct_head_and_legacy_rollout_profiles(self):
        records = [
            {
                "source": "table",
                "source_rows": 320,
                "targets": ["outcome"],
                "roles": {"past": ["cov"]},
            }
        ]
        config = {
            "data": {"context_length": 2048},
            "holdout": {"horizon_probabilities": [0.8, 0.16, 0.03, 0.01]},
        }
        model = SimpleNamespace(
            context_length=2048, patch_size=16, width=1024, output_patches=4
        )
        with patch("graph_profiles.sampled_origins", return_value=[240]):
            direct = profile_counts(records, model, config, "cuda:0")
            self.assertEqual(direct, {((1, 2, 16, 1024), 1, 4, "cuda:0"): 1})
            model.output_patches = 1
            legacy = profile_counts(records, model, config, "cuda:0")
            self.assertEqual(
                legacy,
                {
                    ((1, 2, 16, 1024), 1, 1, "cuda:0"): 2,
                    ((1, 2, 32, 1024), 1, 1, "cuda:0"): 2,
                },
            )

    def test_contexts_near_a_standard_length_share_its_graph(self):
        config = {
            "data": {"context_length": 2048},
            "holdout": {"horizon_probabilities": [0.8, 0.16, 0.03, 0.01]},
        }
        model = SimpleNamespace(
            context_length=2048, patch_size=16, width=768, output_patches=4
        )

        def shape(points, covariates=1):
            record = {
                "source": "table",
                "source_rows": 4096,
                "targets": ["outcome"],
                "roles": {"past": ["cov"] * covariates},
            }
            with patch("graph_profiles.sampled_origins", return_value=[points]):
                (key,) = profile_counts([record], model, config, "cuda:0")
            return key[0]

        for standard in (2048, 1536, 1024, 512):
            expected = (1, 2, standard // 16, 768)
            # At most 10% shorter: padded up to the standard length.
            for points in (standard, standard - 1, -(-9 * standard // 10)):
                self.assertEqual(shape(points), expected, points)
            # Clearly shorter contexts use a smaller graph, not more padding.
            self.assertLess(shape(standard - 256)[2], standard // 16)
            # A different covariate count is a different shape at any length.
            self.assertEqual(shape(standard, covariates=3)[1], 4)
        # Every padded length is a whole number of 16-patch kernel blocks.
        self.assertEqual({shape(n)[2] % 16 for n in range(32, 2049, 37)}, {0})


if __name__ == "__main__":
    unittest.main()

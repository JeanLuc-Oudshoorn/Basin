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


if __name__ == "__main__":
    unittest.main()

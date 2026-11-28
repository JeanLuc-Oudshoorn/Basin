"""Per-table statistics retain extreme cases without changing pooled metrics."""

import copy
import json
import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src/06_holdout_test"))
from error_distribution import (
    reflected_density,
    summarize_tables,
    table_mase_tails,
    write_error_distribution,
    write_latency_distribution,
)


class ErrorDistributionTests(unittest.TestCase):
    def test_latency_plot_keeps_tails_excludes_warmups_and_is_offline(self):
        result = {"models": ["rwkv", "chronos"]}
        requests = pd.DataFrame(
            {
                "model": ["rwkv"] * 4 + ["chronos"],
                "phase": ["scored"] * 3 + ["warmup", "scored"],
                "seconds": [0.001, 0.002, 10.0, 999.0, 0.003],
            }
        )
        with tempfile.TemporaryDirectory() as directory:
            write_latency_distribution(result, requests, directory, {})
            html = (Path(directory) / "latency_distributions.html").read_text(
                encoding="utf-8"
            )
            self.assertIn("Plotly.newPlot", html)
            self.assertIn("chronos (constant)", html)
            self.assertIn("warmups excluded", html)
            self.assertEqual(result["latency_distribution_plot"]["axis"], "log1p")

    def test_equal_table_statistics_and_undefined_coverage(self):
        tables = [
            {"observations": 80, "rwkv": {"mase": 1.0, "rmse": 2.0}},
            {"observations": 16, "rwkv": {"mase": 3.0, "rmse": 4.0}},
            {"observations": 16, "rwkv": {"mase": 101.0, "rmse": 6.0}},
            {"observations": 16, "rwkv": {"mase": None, "rmse": 8.0}},
        ]
        result = summarize_tables(tables, ["rwkv"])
        mase = result["models"]["rwkv"]["mase"]
        self.assertEqual(mase["mean"], 35.0)
        self.assertEqual(mase["median"], 3.0)
        self.assertAlmostEqual(mase["p5"], 1.2)
        self.assertAlmostEqual(mase["p10"], 1.4)
        self.assertAlmostEqual(mase["p90"], 81.4)
        self.assertAlmostEqual(mase["p95"], 91.2)
        # The leaderboard rows get both ends and keep what they had.
        metrics = {
            "table_summary": result,
            "leaderboard": [{"model": "rwkv", "table_mase_median": 3.0}],
        }
        table_mase_tails(metrics)
        self.assertEqual(
            metrics["leaderboard"],
            [
                {
                    "model": "rwkv",
                    "table_mase_median": 3.0,
                    "table_mase_p5": mase["p5"],
                    "table_mase_p95": mase["p95"],
                }
            ],
        )
        self.assertEqual((mase["valid_tables"], mase["undefined_tables"]), (3, 1))
        self.assertEqual(result["models"]["rwkv"]["rmse"]["mean"], 5.0)
        json.dumps(result, allow_nan=False)

    def test_boundary_corrected_density_has_unit_mass(self):
        grid = np.linspace(0, 12, 20001)
        density = reflected_density(np.array([0, 0.1, 1, 2]), grid, 0.3)
        self.assertAlmostEqual(float(np.trapezoid(density, grid)), 1.0, places=6)
        self.assertTrue(np.all(density >= 0))

    def test_plot_preserves_extreme_scores_and_pooled_metrics(self):
        result = {
            "models": ["rwkv", "naive", "chronos", "timestral"],
            "pooled": {"rwkv": {"mase": 99.0}},
            "tables": [
                {
                    method: {"mase": value, "rmse": 2.0}
                    for method in ("rwkv", "naive", "chronos", "timestral")
                }
                for value in (0.0, 1.0, 1e9)
            ],
        }
        original = copy.deepcopy(result)
        with tempfile.TemporaryDirectory() as directory:
            write_error_distribution(result, directory, {})
            html = (Path(directory) / "error_distributions.html").read_text(
                encoding="utf-8"
            )
            self.assertIn("Plotly.newPlot", html)
            for method in result["models"]:
                self.assertIn(method, html)
            self.assertEqual(result["error_distribution_plot"]["axis"], "log1p")
        self.assertEqual(result["tables"], original["tables"])
        self.assertEqual(result["pooled"], original["pooled"])
        self.assertGreater(
            result["table_summary"]["models"]["rwkv"]["mase"]["p90"], 1e8
        )

    def test_disabled_plot_retains_statistics_and_null_metrics(self):
        result = {"models": ["rwkv"], "tables": [{"rwkv": {"mase": None}}]}
        write_error_distribution(
            result, ".", {"plot": {"error_distribution": {"enabled": False}}}
        )
        self.assertIsNone(result["table_summary"]["models"]["rwkv"]["mase"]["median"])
        self.assertNotIn("error_distribution_plot", result)


if __name__ == "__main__":
    unittest.main()

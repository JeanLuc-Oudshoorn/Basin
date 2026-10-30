"""Interactive plots preserve forecast segments and work without network access."""

import json
import sys
import tempfile
import unittest
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
for stage in ("03_transforming_input_data", "05_model_training", "06_holdout_test"):
    sys.path.insert(0, str(ROOT / "src" / stage))
from plotting import regenerate_saved_plots, write_holdout_plot
from transform import load_config


class HoldoutPlottingTests(unittest.TestCase):
    def test_interactive_offline_html_preserves_all_models_and_forecast_gaps(self):
        config = load_config()
        frame = pd.DataFrame({"outcome": range(64)})
        predictions = pd.DataFrame(
            {
                "source": ["table"] * 4,
                "target": ["outcome"] * 4,
                "origin": [32, 32, 48, 48],
                "time_index": [32, 33, 48, 49],
                **{
                    name: [1.0, 2.0, 3.0, 4.0]
                    for name in ("rwkv", "prophet", "naive", "chronos", "timestral")
                },
            }
        )
        methods = ["rwkv", "prophet", "naive", "chronos", "timestral"]
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory) / "comparison.html"
            figure = write_holdout_plot(
                frame, predictions, "outcome", methods, output, config, 3
            )
            self.assertEqual(
                [trace.name for trace in figure.data], ["Actual"] + methods
            )
            self.assertEqual(list(figure.data[1].x), [32, 33, None, 48, 49, None])
            self.assertEqual(list(figure.data[1].y), [1.0, 2.0, None, 3.0, 4.0, None])
            self.assertFalse(figure.data[1].connectgaps)
            self.assertTrue(figure.layout.xaxis.rangeslider.visible)
            html = output.read_text(encoding="utf-8")
            self.assertIn("Plotly.newPlot", html)
            self.assertIn('"scrollZoom": true', html)
            self.assertGreater(
                output.stat().st_size, 1_000_000
            )  # Embedded JS, not a remote script.
            source = Path(directory) / "data.parquet"
            frame.to_parquet(source)
            predictions["source"] = str(source)
            predictions.to_parquet(Path(directory) / "test_table_003.parquet")
            selection = {
                "tables": [
                    {"table_index": 3, "source": str(source), "plots": ["old.png"]}
                ]
            }
            (Path(directory) / "plot_selection.json").write_text(json.dumps(selection))
            metrics = {
                "models": methods,
                "tables": [{"source": str(source)}] * 4,
                "pooled": {"rwkv": {"rmse": 0.5}},
            }
            (Path(directory) / "test_metrics.json").write_text(json.dumps(metrics))
            config["paths"]["holdout_test"] = directory
            self.assertEqual(regenerate_saved_plots(config), 1)
            updated = json.loads((Path(directory) / "test_metrics.json").read_text())
            self.assertEqual(updated["pooled"], metrics["pooled"])
            self.assertEqual(
                updated["plot_selection"]["tables"][0]["plots"],
                ["test_table_003_target_0.html"],
            )


if __name__ == "__main__":
    unittest.main()

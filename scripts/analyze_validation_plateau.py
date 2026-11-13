"""CPU-only validation summaries and a small optimizer precision experiment.

Does not import the training entry point, initialize CUDA, or modify run files.
"""

import argparse
import json
from pathlib import Path

import pandas as pd


def validation_summary(directory):
    history = json.loads((directory / "validation_history.json").read_text())
    summaries, previous = [], None
    for row in history:
        name = row["table_metrics_path"].replace("\\", "/").split("/")[-1]
        frame = pd.read_parquet(directory / name)
        table = frame.groupby("source", sort=False)[["loss", "mase"]].mean()
        summary = {
            "check": row["validation_check"],
            "step": row["step"],
            "hours": row["active_training_seconds"] / 3600,
            "forecast_median": frame[["loss", "mase"]].median().to_dict(),
            "forecast_p90": frame[["loss", "mase"]].quantile(0.9).to_dict(),
            "table_median": table.median().to_dict(),
            "table_p90": table.quantile(0.9).to_dict(),
            "aggregate_loss": row["loss"],
            "aggregate_mase": row["mase"],
        }
        current = frame.set_index(["source", "origin"])[["loss", "mase"]]
        if previous is not None:
            if not current.index.equals(previous.index):
                if set(current.index) != set(previous.index):
                    raise ValueError("Validation requests changed between checks")
                previous = previous.reindex(current.index)
            summary["fraction_requests_improved"] = {
                key: float((current[key] < previous[key]).mean())
                for key in ("loss", "mase")
            }
        previous = current
        summaries.append(summary)
    return summaries


def precision_probe():
    import torch

    torch.set_num_threads(1)
    results = []
    for dtype in (torch.bfloat16, torch.float32):
        for value in (1.0, 0.5, 0.02):
            parameter = torch.nn.Parameter(torch.full((32,), value, dtype=dtype))
            optimizer = torch.optim.AdamW(
                [parameter], lr=0.0001, weight_decay=0.01, fused=True
            )
            initial = parameter.detach().float().clone()
            for _ in range(1000):
                parameter.grad = torch.full_like(parameter, 0.1)
                optimizer.step()
            results.append(
                {
                    "dtype": str(dtype),
                    "initial": float(initial[0]),
                    "after_1000_steps": float(parameter[0].detach()),
                    "changed": bool((parameter.detach().float() != initial).any()),
                    "moment_dtypes": {
                        key: str(tensor.dtype)
                        for key, tensor in optimizer.state[parameter].items()
                        if isinstance(tensor, torch.Tensor)
                    },
                }
            )
    return results


def write_plot(summaries, output):
    import plotly.graph_objects as go
    from plotly.subplots import make_subplots

    figure = make_subplots(
        rows=1, cols=2, subplot_titles=("Median normalized pinball", "Median MASE")
    )
    for column, metric in enumerate(("loss", "mase"), start=1):
        for scope, label in (
            ("forecast_median", "Forecast requests"),
            ("table_median", "Tables (average of five origins)"),
        ):
            figure.add_trace(
                go.Scatter(
                    x=[item["hours"] for item in summaries],
                    y=[item[scope][metric] for item in summaries],
                    name=label,
                    legendgroup=scope,
                    showlegend=column == 1,
                    mode="lines+markers",
                ),
                row=1,
                col=column,
            )
    figure.update_xaxes(title_text="Cumulative active training hours")
    figure.update_layout(title="Basin validation checks: observed medians")
    figure.write_html(output, include_plotlyjs=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--training", type=Path, default=Path("results/model/training"))
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--precision-probe", action="store_true")
    args = parser.parse_args()
    # Explicit output prevents an accidental write into the production results.
    output = args.output.resolve()
    if output.is_relative_to(args.training.resolve()):
        raise ValueError("Diagnostic output must be outside the training directory")
    output.mkdir(parents=True, exist_ok=True)
    summaries = validation_summary(args.training)
    (output / "validation_medians.json").write_text(
        json.dumps(summaries, indent=2) + "\n", encoding="utf-8"
    )
    if args.precision_probe:
        (output / "precision_probe.json").write_text(
            json.dumps(precision_probe(), indent=2) + "\n", encoding="utf-8"
        )
    write_plot(summaries, output / "validation_medians.html")
    print(f"Saved {len(summaries)} validation checks to {output}")


if __name__ == "__main__":
    main()

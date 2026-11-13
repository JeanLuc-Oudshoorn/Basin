"""Audit saved holdout scaled-error contributions without rerunning models."""

import json
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.colors import qualitative
from plotly.subplots import make_subplots

ROOT = Path(__file__).resolve().parents[1]


def main():
    directory = ROOT / "results/model/holdout_test"
    output = ROOT / "docs/audits/2026-11-04/mase_tail"
    output.mkdir(parents=True, exist_ok=True)
    metrics = json.loads((directory / "test_metrics.json").read_text())
    tables = json.loads((directory / "table_metrics.json").read_text())
    predictions = pd.read_parquet(directory / "test_predictions.parquet")
    frame = predictions
    frame = frame[np.isfinite(frame.actual)].copy()
    valid = np.isfinite(frame.mase_scale) & (frame.mase_scale > 0)
    frame = frame[valid].copy()
    methods = metrics["models"]
    total_mass = frame.metric_weight.sum()
    evidence = {
        "tables": len(tables),
        "valid_points": len(frame),
        "total_metric_mass": total_mass,
        "models": {},
    }
    contributions = pd.DataFrame(index=pd.Index(frame.source.unique(), name="source"))
    for name in methods:
        frame[f"{name}_mass"] = (
            np.abs(frame[name] - frame.actual) / frame.mase_scale * frame.metric_weight
        )
        per_table = (
            frame.groupby("source")[f"{name}_mass"].sum().sort_values(ascending=False)
        )
        contributions[name] = per_table / total_mass
        total = per_table.sum()
        statistics = {
            "recomputed_mase": total / total_mass,
            "saved_mase": metrics["pooled"][name]["mase"],
            "top_tables": [],
        }
        for source, mass in per_table.head(10).items():
            subset = frame[frame.source == source]
            statistics["top_tables"].append(
                {
                    "source": source,
                    "mase": mass / subset.metric_weight.sum(),
                    "contribution_to_pooled_mase": mass / total_mass,
                    "fraction_of_scaled_error": mass / total,
                }
            )
        for count in (1, 2, 3, 5, 10):
            excluded = set(per_table.head(count).index)
            remaining = frame[~frame.source.isin(excluded)]
            statistics[f"top_{count}_fraction"] = per_table.head(count).sum() / total
            statistics[f"mase_without_top_{count}"] = (
                remaining[f"{name}_mass"].sum() / remaining.metric_weight.sum()
            )
        evidence["models"][name] = statistics
    contributions.sort_values("rwkv", ascending=False).to_csv(
        output / "table_contributions.csv"
    )
    worst_sources = contributions.sort_values("rwkv", ascending=False).head(3).index
    origins = []
    raw_summaries = []
    for source in worst_sources:
        subset = frame[frame.source == source]
        raw_path = ROOT / source
        raw = pd.read_parquet(raw_path)
        for (origin, target), group in subset.groupby(["origin", "target"]):
            history = (
                raw[target].iloc[max(0, origin - 2048) : origin].to_numpy(dtype=float)
            )
            row = {
                "source": source,
                "origin": int(origin),
                "target": target,
                "scale": float(group.mase_scale.iloc[0]),
                "history_min": float(np.nanmin(history)),
                "history_max": float(np.nanmax(history)),
                "history_last": float(history[-1]),
                "history_std": float(np.nanstd(history)),
                "basin_unit_scale_fallback": bool(np.nanstd(history) <= 1e-5),
                "history_step_changes": int(np.count_nonzero(np.diff(history))),
                "actual_min": float(group.actual.min()),
                "actual_max": float(group.actual.max()),
            }
            for name in methods:
                row[f"{name}_mase"] = float(
                    group[f"{name}_mass"].sum() / group.metric_weight.sum()
                )
                row[f"{name}_mae"] = float(
                    np.average(
                        np.abs(group[name] - group.actual), weights=group.metric_weight
                    )
                )
                row[f"{name}_pooled_contribution"] = float(
                    group[f"{name}_mass"].sum() / total_mass
                )
            origins.append(row)
        raw_summaries.append(
            {
                "source": source,
                "rows": len(raw),
                "columns": list(raw.columns),
                "sibling_files": [
                    p.name for p in raw_path.parent.iterdir() if p.is_file()
                ],
            }
        )
    evidence["worst_origins"] = origins
    evidence["raw_tables"] = raw_summaries
    # Same exclusions for all models, selected solely from Basin's scaled-error tail.
    shared = set(contributions.sort_values("rwkv", ascending=False).head(1).index)
    remaining = frame[~frame.source.isin(shared)]
    evidence["all_models_excluding_same_worst_table"] = {
        name: float(remaining[f"{name}_mass"].sum() / remaining.metric_weight.sum())
        for name in methods
    }
    evidence["shared_exclusion_sensitivity"] = {}
    for count in (3, 5, 10):
        excluded = contributions.sort_values("rwkv", ascending=False).head(count).index
        remaining = frame[~frame.source.isin(excluded)]
        evidence["shared_exclusion_sensitivity"][str(count)] = {
            name: float(remaining[f"{name}_mass"].sum() / remaining.metric_weight.sum())
            for name in methods
        }
    figure = make_subplots(
        rows=3,
        cols=1,
        subplot_titles=[Path(source).parent.name for source in worst_sources],
    )
    for panel, source in enumerate(worst_sources, 1):
        raw = pd.read_parquet(ROOT / source)
        figure.add_trace(
            go.Scatter(
                x=raw.time_index,
                y=raw.outcome,
                name="actual",
                legendgroup="actual",
                showlegend=panel == 1,
                line={"color": "black"},
            ),
            row=panel,
            col=1,
        )
        saved = predictions[
            (predictions.source == source) & (predictions.target == "outcome")
        ]
        for color_index, name in enumerate(methods):
            x, y = [], []
            for _, group in saved.groupby("origin"):
                x.extend(group.time_index.tolist() + [None])
                y.extend(group[name].tolist() + [None])
            figure.add_trace(
                go.Scatter(
                    x=x,
                    y=y,
                    name=name,
                    legendgroup=name,
                    showlegend=panel == 1,
                    line={"color": qualitative.Plotly[color_index]},
                    connectgaps=False,
                ),
                row=panel,
                col=1,
            )
    figure.update_layout(
        height=1100,
        template="plotly_white",
        title="Three largest MASE contributors: saved forecasts and actuals",
    )
    figure.write_html(
        output / "worst_tables.html",
        include_plotlyjs=True,
        config={"scrollZoom": True, "displaylogo": False},
    )
    (output / "evidence.json").write_text(
        json.dumps(evidence, indent=2, allow_nan=False) + "\n", encoding="utf-8"
    )
    print(json.dumps(evidence, indent=2, allow_nan=False))


if __name__ == "__main__":
    main()

"""Recover the overwritten 20% summary from its immutable run journal."""

import json
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "docs/audits/2026-11-02/forecast_regression"


def main():
    journal = (
        ROOT / "logs/rwkv7_holdout_test_20261102_054242_743315Z_pid3696_a6cf0231.jsonl"
    )
    metrics = None
    with journal.open() as stream:
        for line in stream:
            row = json.loads(line)
            if row.get("event") == "out_of_sample_evaluation":
                metrics = row["out_of_sample_metrics"]
    if metrics is None:
        raise ValueError("Completed 20% metrics missing")
    (OUT / "recovered_20_percent_metrics.json").write_text(
        json.dumps(metrics, indent=2) + "\n", newline="\n"
    )
    origins = metrics["origins"]
    details = []
    methods = [
        name
        for name in ("rwkv", "naive", "chronos", "timestral")
        if name in metrics["pooled"]
    ]
    for origin in origins:
        entry = dict(
            source=origin["source"],
            target=origin["target"],
            origin=origin["origin"],
            points=origin["observations"],
        )
        for name in methods:
            entry[name] = origin[name]["mase"]
            entry[name + "_mae"] = origin[name]["mae"]
        details.append(entry)
    frame = pd.DataFrame(details)
    valid = frame.rwkv.notna()
    frame["rwkv_contribution"] = frame.rwkv.fillna(0) * frame.points
    total = frame.rwkv_contribution.sum()
    top = frame.nlargest(20, "rwkv_contribution").copy()
    for index, row in top.iterrows():
        raw = pd.read_parquet(ROOT / row.source, columns=[row.target])
        history = (
            raw[row.target].iloc[max(0, row.origin - 2048) : row.origin].to_numpy()
        )
        change = np.abs(np.diff(history))
        top.loc[index, "historical_naive_mae"] = float(np.nanmean(change))
        top.loc[index, "context_std"] = float(np.nanstd(history))
    sensitivity = {}
    for discard in (0, 1, 5, 10, 20):
        keep = frame.drop(frame.nlargest(discard, "rwkv_contribution").index)
        sensitivity[str(discard)] = {
            name: float(
                np.average(
                    keep.loc[keep[name].notna(), name],
                    weights=keep.loc[keep[name].notna(), "points"],
                )
            )
            for name in methods
        }
    summary = dict(
        table_count=metrics["table_count"],
        pooled=metrics["pooled"],
        top1_share=float(top.rwkv_contribution.iloc[0] / total),
        top5_share=float(top.rwkv_contribution.iloc[:5].sum() / total),
        sensitivity_drop_origins=sensitivity,
        source_journal=str(journal),
        undefined_origins=int((~valid).sum()),
    )
    top.to_parquet(OUT / "twenty_percent_worst_origins.parquet", index=False)
    (OUT / "twenty_percent_diagnostics.json").write_text(
        json.dumps(summary, indent=2) + "\n", newline="\n"
    )
    print(json.dumps({k: v for k, v in summary.items() if k != "pooled"}, indent=2))
    print(
        top[
            ["source", "origin", "rwkv", "naive", "historical_naive_mae", "context_std"]
        ]
        .head()
        .to_string(index=False)
    )


if __name__ == "__main__":
    main()

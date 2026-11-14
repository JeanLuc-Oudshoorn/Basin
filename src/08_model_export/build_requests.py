"""Draw a reproducible sample of real holdout forecast requests for parity/latency checks.

Requests are (table, origin) pairs that the holdout stage already evaluated, so
each carries the median forecast the original model saved at that time.
CPU only; needs pandas + pyarrow + torch.
"""

import argparse
import json
import re
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch

ROOT = Path(__file__).resolve().parents[2]
BUCKETS = ((1, 128), (129, 512), (513, 1024), (1025, 2048))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--per-bucket", type=int, default=60)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--output", default=str(ROOT / ".cache/port_work/requests.pt"))
    args = parser.parse_args()
    config = json.loads((ROOT / "configs/config.json").read_text(encoding="utf-8"))
    data = config["data"]
    cap = data["context_length"]
    predictions = pd.read_parquet(
        ROOT / config["paths"]["holdout_test"] / "test_predictions.parquet",
        columns=[
            "source",
            "target",
            "origin",
            "time_index",
            "context_points",
            "rwkv",
            "actual",
            "metric_weight",
            "mase_scale",
        ],
    )
    keys = predictions[["source", "origin", "context_points"]].drop_duplicates()
    rng = np.random.default_rng(args.seed)
    chosen = []
    for lower, upper in BUCKETS:
        bucket = keys[(keys.context_points >= lower) & (keys.context_points <= upper)]
        rows = rng.choice(
            len(bucket), size=min(args.per_bucket, len(bucket)), replace=False
        )
        chosen.append(bucket.iloc[np.sort(rows)])
    chosen = pd.concat(chosen, ignore_index=True)
    saved = predictions.merge(chosen[["source", "origin"]], on=["source", "origin"])
    requests = []
    for row in chosen.itertuples(index=False):
        frame = pd.read_parquet(ROOT / row.source)
        excluded = [
            c
            for c in frame.columns
            if any(re.search(p, c) for p in data["excluded_column_patterns"])
        ]
        targets = list(data["targets"])
        covariates = [
            c
            for c in frame.columns
            if c not in targets + [data["time_column"]] + excluded
        ]
        history = frame.iloc[max(0, row.origin - cap) : row.origin]
        if len(history) != row.context_points:
            raise ValueError(f"Context length mismatch for {row.source}@{row.origin}")
        holdout = saved[(saved.source == row.source) & (saved.origin == row.origin)]
        holdout = holdout.sort_values(["target", "time_index"])
        requests.append(
            {
                "source": row.source,
                "origin": int(row.origin),
                "context_points": int(row.context_points),
                "variables": len(targets) + len(covariates),
                "context": torch.tensor(
                    history[targets].to_numpy(dtype=np.float32).T.copy()
                )[None],
                "past_covariates": None
                if not covariates
                else torch.tensor(
                    history[covariates].to_numpy(dtype=np.float32).T.copy()
                )[None],
                # Median forecast stored by the holdout run (may be shorter than 64 at a table's end).
                "holdout_point": torch.tensor(
                    holdout["rwkv"].to_numpy(dtype=np.float64)
                ).reshape(len(targets), -1),
                # Ground truth and the holdout's own scoring weights / MASE scale.
                "actual": torch.tensor(
                    holdout["actual"].to_numpy(dtype=np.float64)
                ).reshape(len(targets), -1),
                "metric_weight": torch.tensor(
                    holdout["metric_weight"].to_numpy(dtype=np.float64)
                ).reshape(len(targets), -1),
                "mase_scale": torch.tensor(
                    holdout.groupby("target", sort=True)["mase_scale"]
                    .first()
                    .to_numpy(dtype=np.float64)
                ),
            }
        )
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    torch.save(requests, output)
    summary = pd.DataFrame(
        {k: [r[k] for r in requests] for k in ("context_points", "variables")}
    )
    print(f"{len(requests)} requests -> {output}")
    print(summary.describe().loc[["min", "50%", "max"]].T)
    print("variables:", summary.variables.value_counts().sort_index().to_dict())


if __name__ == "__main__":
    sys.exit(main())

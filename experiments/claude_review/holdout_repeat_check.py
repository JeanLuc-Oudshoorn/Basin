"""Do Basin's later forecast patches repeat patch 1? CPU-only, reads saved holdout predictions.

Usage: python experiments/claude_review/holdout_repeat_check.py [predictions.parquet]
"""

import sys

import numpy as np
import pandas as pd

path = (
    sys.argv[1]
    if len(sys.argv) > 1
    else "results/model/holdout_test/test_predictions.parquet"
)
columns = [
    "source",
    "origin",
    "time_index",
    "context_points",
    "forecast_token",
    "actual",
    "rwkv",
    "naive",
    "chronos",
    "timestral",
    "mase_scale",
]
frame = pd.read_parquet(path, columns=columns)
frame = frame.sort_values(["source", "origin", "time_index"], kind="stable")
requests = len(frame) // 64
series = ["rwkv", "chronos", "timestral", "actual"]
values = {
    name: frame[name].to_numpy().reshape(requests, 4, 16) for name in series + ["naive"]
}
scale = frame["mase_scale"].to_numpy().reshape(requests, 64)[:, 0]
context = frame["context_points"].to_numpy().reshape(requests, 64)[:, 0]
defined = np.isfinite(scale) & (scale > 0)
print(f"requests={requests:,} defined MASE scale={defined.sum():,}")

print(
    "\n1. Distance between a model's own patch k and its patch 1 (MASE units, median over requests)"
)
for name in series:
    v = values[name][defined]
    s = scale[defined, None]
    row = [
        np.nanmedian(np.nanmean(np.abs(v[:, k] - v[:, 0]), axis=1) / s[:, 0])
        for k in (1, 2, 3)
    ]
    a, b = (
        v[:, 0] - v[:, 0].mean(1, keepdims=True),
        v[:, 1] - v[:, 1].mean(1, keepdims=True),
    )
    den = np.sqrt((a * a).sum(1) * (b * b).sum(1))
    corr = np.where(den > 0, (a * b).sum(1) / np.where(den > 0, den, 1), np.nan)
    print(
        f"  {name:10s} patch2-1={row[0]:.3f} patch3-1={row[1]:.3f} patch4-1={row[2]:.3f} "
        f"median corr(p1,p2)={np.nanmedian(corr):.3f} share corr>0.95={np.nanmean(corr > 0.95):.3f}"
    )

print(
    "\n2. Is patch k closer to its own actuals than a copy of patch 1 would be? (share of requests)"
)
actual = values["actual"][defined]
for name in ("rwkv", "chronos", "timestral"):
    v = values[name][defined]
    shares = []
    for k in (1, 2, 3):
        own = np.nanmean(np.abs(v[:, k] - actual[:, k]), axis=1)
        copy = np.nanmean(np.abs(v[:, 0] - actual[:, k]), axis=1)
        shares.append(np.mean(own < copy))
    print(
        f"  {name:10s} "
        + " ".join(f"patch{k + 1}: {s:.3f}" for k, s in zip((1, 2, 3), shares))
    )

print("\n3. Median per-request MASE by forecast patch")
for name in ("rwkv", "chronos", "timestral", "naive"):
    v = values[name][defined]
    row = [
        np.nanmedian(
            np.nanmean(np.abs(v[:, k] - actual[:, k]), axis=1) / scale[defined]
        )
        for k in range(4)
    ]
    print(f"  {name:10s} " + " ".join(f"p{k + 1}={r:.3f}" for k, r in enumerate(row)))

print(
    "\n4. Median per-request MASE of patch 1 and of patches 1-4 (equal weight) by context points"
)
edges = [0, 128, 256, 512, 1024, 2048]
for low, high in zip(edges, edges[1:]):
    keep = defined & (context > low) & (context <= high)
    line = f"  {low + 1:>4}-{high:<4} n={keep.sum():>6,}"
    for name in ("rwkv", "chronos", "timestral"):
        v, a, s = values[name][keep], values["actual"][keep], scale[keep]
        first = np.nanmedian(np.nanmean(np.abs(v[:, 0] - a[:, 0]), axis=1) / s)
        every = np.nanmedian(np.nanmean(np.abs(v - a), axis=(1, 2)) / s)
        line += f" | {name} p1={first:.3f} all={every:.3f}"
    print(line)

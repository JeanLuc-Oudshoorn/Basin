"""Is Basin's long-context gap concentrated at anchors that never saw later-lead labels?

Training chunks hold at most 128 input patches plus ONE label patch, so the anchor
with the full 2,048-point context is supervised on lead 1 only, the one before it
on leads 1-2, and so on. CPU-only; reads saved holdout predictions.
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
    "actual",
    "rwkv",
    "chronos",
    "timestral",
    "mase_scale",
]
frame = pd.read_parquet(path, columns=columns).sort_values(
    ["source", "origin", "time_index"], kind="stable"
)
n = len(frame) // 64
value = {
    c: frame[c].to_numpy().reshape(n, 4, 16)
    for c in ("actual", "rwkv", "chronos", "timestral")
}
scale = frame["mase_scale"].to_numpy().reshape(n, 64)[:, 0]
context = frame["context_points"].to_numpy().reshape(n, 64)[:, 0]
ok = np.isfinite(scale) & (scale > 0)
print(
    "context tokens -> anchor index (0-based) = tokens-1; leads supervised in training at that anchor = 128 - tokens + 1 (max 4)"
)
groups = [
    ("<=1024", context <= 1024),
    ("1025-1984", (context > 1024) & (context <= 1984)),
    ("2000 (3 leads)", context == 2000),
    ("2016 (2 leads)", context == 2016),
    ("2032 (1 lead... 2 leads)", context == 2032),
    ("2048 (lead 1 only)", context == 2048),
]
print(sorted(pd.Series(context[context > 1984]).value_counts().items()))
for name, keep in groups:
    keep = keep & ok
    if not keep.sum():
        continue
    line = f"{name:26s} n={keep.sum():>6,}"
    for k in range(4):
        scores = {
            m: np.nanmean(
                np.abs(value[m][keep][:, k] - value["actual"][keep][:, k]), axis=1
            )
            / scale[keep]
            for m in ("rwkv", "chronos", "timestral")
        }
        med = {m: np.nanmedian(s) for m, s in scores.items()}
        win = np.mean(scores["rwkv"] < scores["timestral"])
        line += f" | p{k + 1} basin/timestral={med['rwkv'] / med['timestral']:.3f} basin/chronos={med['rwkv'] / med['chronos']:.3f} win%={100 * win:.0f}"
    print(line)

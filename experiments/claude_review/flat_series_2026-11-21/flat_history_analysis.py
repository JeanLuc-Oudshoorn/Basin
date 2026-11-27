"""How do the histories look where the holdout's actual value equals the last observed one?

CPU only. Reads the epoch-1 holdout predictions and the source tables.
"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[3]
RUN = ROOT / "results/model/holdout_test/20261120T172837Z_epoch1_step1457590"
OUT = Path(__file__).resolve().parent
MODELS = ["rwkv", "timestral", "chronos"]

p = pd.read_parquet(
    RUN / "test_predictions.parquet",
    columns=[
        "source",
        "target",
        "origin",
        "time_index",
        "context_points",
        "actual",
        "naive",
        "context_scale",
        *MODELS,
    ],
)
p = p[np.isfinite(p.actual)]
p["flat"] = p.actual == p.naive
print("points", len(p), "flat share", p.flat.mean())
g = p.groupby(["source", "target", "origin"], sort=False)
f = g.agg(
    flat_share=("flat", "mean"),
    n=("flat", "size"),
    context_points=("context_points", "first"),
    naive=("naive", "first"),
    scale=("context_scale", "first"),
).reset_index()
print(
    "forecasts",
    len(f),
    "with any flat point",
    (f.flat_share > 0).mean(),
    "entirely flat",
    (f.flat_share == 1).mean(),
)
print(
    "flat points lying in entirely flat forecasts:",
    f[f.flat_share == 1].n.sum() / p.flat.sum(),
)
flat = p[p.flat]
for m in MODELS:
    flat["e_" + m] = (flat[m] - flat.actual).abs() / flat.context_scale
err = (
    flat.groupby(["source", "target", "origin"])[["e_" + m for m in MODELS]]
    .mean()
    .reset_index()
)
f = f[f.flat_share > 0].merge(err, on=["source", "target", "origin"])
rng = np.random.default_rng(7)
sources = f.source.drop_duplicates()
sources = sources.iloc[
    rng.permutation(len(sources))[: int(sys.argv[1]) if len(sys.argv) > 1 else 1200]
]
f = f[f.source.isin(sources)]
rows = []
for source, part in f.groupby("source"):
    frame = pd.read_parquet(ROOT / source)
    for r in part.itertuples(index=False):
        h = frame[r.target].to_numpy(dtype=float)[max(0, r.origin - 2048) : r.origin]
        h = h[np.isfinite(h)]
        same = h == r.naive
        differ = np.flatnonzero(~same)
        run = len(h) if len(differ) == 0 else len(h) - 1 - differ[-1]
        rows.append(
            (
                source,
                r.target,
                r.origin,
                len(h),
                run,
                same.mean(),
                np.ptp(h) if len(h) else np.nan,
            )
        )
h = pd.DataFrame(
    rows,
    columns=[
        "source",
        "target",
        "origin",
        "observed",
        "trailing_run",
        "share_equal_last",
        "range",
    ],
)
f = f.merge(h, on=["source", "target", "origin"])
f.to_parquet(OUT / "flat_forecasts.parquet")
f["whole_history_constant"] = f.trailing_run == f.observed
print(
    "forecasts analysed",
    len(f),
    "whole history constant",
    f.whole_history_constant.mean(),
)
bins = [0, 1, 4, 16, 64, 256, 5000]
f["run_bin"] = pd.cut(f.trailing_run, bins, right=True).astype(str)
f.loc[f.whole_history_constant, "run_bin"] = "whole history"
t = f.groupby("run_bin").agg(
    forecasts=("n", "size"),
    flat_share=("flat_share", "mean"),
    **{m: ("e_" + m, "median") for m in MODELS},
)
print(t.to_string())
f["hist_bin"] = pd.cut(f.context_points, [0, 64, 256, 1024, 2048]).astype(str)
print(
    f.groupby("hist_bin")
    .agg(forecasts=("n", "size"), **{m: ("e_" + m, "median") for m in MODELS})
    .to_string()
)
print(
    "share_equal_last quantiles",
    f.share_equal_last.quantile([0.1, 0.25, 0.5, 0.75, 0.9]).round(3).to_dict(),
)

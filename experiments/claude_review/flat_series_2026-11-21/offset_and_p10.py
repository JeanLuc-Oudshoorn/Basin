"""Is the level offset on flat series stable between two checkpoints, and what would fixing constant histories do to p10?"""

from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
RUN = ROOT / "results/model/holdout_test/20261120T172837Z_epoch1_step1457590"

a = pd.read_parquet(HERE.parent / "precision_2026-11-20/predictions_by_arm.parquet")
a = a[(a.group == "flat") & (a.actual == a.naive)]
cols = ["A", "C", "E1", "timestral", "chronos"]
for c in cols:
    a[c] = (a[c] - a.actual) / a.context_scale
off = a.groupby(["source", "origin"])[cols].mean()
print("flat forecasts", len(off))
print("Spearman correlation of per-forecast level offset:")
print(off.corr(method="spearman").round(2).to_string())
print(
    "sign agreement A vs E1",
    (np.sign(off.A) == np.sign(off.E1)).mean().round(3),
    "A vs timestral",
    (np.sign(off.A) == np.sign(off.timestral)).mean().round(3),
)
print("median |offset|", off.abs().median().round(5).to_dict())

p = pd.read_parquet(
    RUN / "test_predictions.parquet",
    columns=[
        "source",
        "target",
        "origin",
        "actual",
        "naive",
        "mase_scale",
        "rwkv",
        "timestral",
        "chronos",
    ],
)
p = p[
    np.isfinite(p.actual)
    & np.isfinite(p.mase_scale)
    & (p.mase_scale > 0)
    & np.isfinite(p.rwkv)
]
f = pd.read_parquet(HERE / "flat_forecasts.parquet")
f["constant"] = f.trailing_run == f.observed
p = p.merge(
    f[["source", "target", "origin", "constant", "trailing_run"]],
    on=["source", "target", "origin"],
    how="left",
)
p["constant"] = p.constant.fillna(False).astype(bool)
e = {
    m: (p[m] - p.actual).abs() / p.mase_scale
    for m in ["rwkv", "timestral", "chronos", "naive"]
}


def q(x):
    return float(np.quantile(x, 0.10))


print("p10 pooled:", {m: round(q(v), 4) for m, v in e.items()})
print(
    "share of points in forecasts with constant visible history",
    p.constant.mean().round(4),
)
print(
    "rwkv p10, naive on constant histories:",
    round(q(np.where(p.constant, e["naive"], e["rwkv"])), 4),
)
long_run = p.trailing_run.fillna(0) >= 16
print("share of points after a flat run of 16+:", long_run.mean().round(4))
print(
    "rwkv p10, naive after flat runs of 16+:",
    round(q(np.where(long_run, e["naive"], e["rwkv"])), 4),
)
flat = p.actual == p.naive
print(
    "rwkv p10, exact on every flat point (bound):",
    round(q(np.where(flat, 0, e["rwkv"])), 4),
)
c = f[f.constant]
print(
    "constant histories: holdout scale quantiles",
    c["scale"].quantile([0.1, 0.5, 0.9]).to_dict(),
    "| level |naive| quantiles",
    c.naive.abs().quantile([0.1, 0.5, 0.9]).round(3).to_dict(),
)
print(
    "constant histories: rwkv abs error / max(1,|level|) median",
    (c.e_rwkv * c["scale"] / np.maximum(1, c.naive.abs())).median(),
)

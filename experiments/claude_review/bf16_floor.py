"""CPU-only: does bf16 I/O rounding bound Basin's holdout accuracy? Reads saved predictions."""

from pathlib import Path

import numpy as np
import pandas as pd
import torch

ROOT = Path(__file__).resolve().parents[2]
p = pd.read_parquet(ROOT / "results/model/holdout_test/test_predictions.parquet")


def bf16(x):
    return (
        torch.tensor(np.asarray(x, dtype=np.float32))
        .to(torch.bfloat16)
        .float()
        .numpy()
        .astype(np.float64)
    )


rows = []
cache = {}
for (source, origin), g in p.groupby(["source", "origin"], sort=False):
    if source not in cache:
        cache.clear()
        cache[source] = pd.read_parquet(ROOT / source)
    frame = cache[source]
    y = frame["outcome"].to_numpy(dtype=np.float64)
    ctx = y[max(0, origin - 2048) : origin]
    loc, scale = ctx.mean(), ctx.std()
    if not np.isfinite(scale) or scale <= 0:
        scale = 1.0
    g = g.sort_values("time_index")
    z_pred = (g["rwkv"].to_numpy() - loc) / scale
    z_true = (g["actual"].to_numpy() - loc) / scale
    z_last = (ctx[-16:] - loc) / scale
    ms = g["mase_scale"].iloc[0]
    w = g["metric_weight"].to_numpy()
    w = w / w.sum()

    def mase(col):
        return (
            float((np.abs(g[col].to_numpy() - g["actual"].to_numpy()) * w).sum() / ms)
            if ms > 0
            else np.nan
        )

    rows.append(
        dict(
            source=source,
            origin=origin,
            context=len(ctx),
            ncov=frame.shape[1] - 2,
            ms=ms,
            scale=scale,
            smooth=ms / scale if scale > 0 else np.nan,
            # how far Basin's normalized median sits from the bf16 grid, relative to grid spacing
            pred_offgrid=float(
                np.median(
                    np.abs(z_pred - bf16(z_pred))
                    / np.maximum(
                        np.abs(
                            np.spacing(bf16(z_pred).astype(np.float32)).astype(
                                np.float64
                            )
                        )
                        * 2**16,
                        1e-30,
                    )
                )
            ),
            # error floors in MASE units
            out_floor=float((np.abs(z_true - bf16(z_true)) * w).sum() * scale / ms)
            if ms > 0
            else np.nan,
            in_floor=float(np.abs(z_last - bf16(z_last)).mean() * scale / ms)
            if ms > 0
            else np.nan,
            zabs=float(np.abs(z_true).mean()),
            rwkv=mase("rwkv"),
            chronos=mase("chronos"),
            timestral=mase("timestral"),
            naive=mase("naive"),
            **{
                f"{m}_t{t}": float(
                    np.abs(
                        g.loc[g.forecast_token == t, m]
                        - g.loc[g.forecast_token == t, "actual"]
                    ).mean()
                    / ms
                )
                if ms > 0
                else np.nan
                for m in ("rwkv", "chronos", "timestral", "naive")
                for t in (1, 2, 3, 4)
            },
        )
    )
d = pd.DataFrame(rows)
d.to_parquet(ROOT / "experiments/claude_review/bf16_floor.parquet")
print(len(d), "requests")

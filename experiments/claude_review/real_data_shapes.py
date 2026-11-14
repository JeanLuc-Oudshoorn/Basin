"""Describe the operational datasets added to data/00_public on 2026-11-13.

CPU-only and read-only: prints per-column shape statistics and writes one PNG
of 2048-row windows per dataset next to this script.

Usage: python real_data_shapes.py
"""

import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from generator_coverage import ROOT, signal_statistics

PUBLIC = ROOT / "data" / "00_public"
WINDOW = 2048
SOURCES = {
    "skab": ("anomaly-free.csv", {"sep": ";"}, "datetime"),
    "metro": (
        "Metro_PT3_(AirCompressor).csv",
        {"nrows": 400000, "index_col": 0},
        "timestamp",
    ),
    "kelmarsh": ("Kelmarsh_era5.csv", {"nrows": 60000}, "datetime"),
    "batadal": ("BATADAL_dataset03.csv", {}, None),
    "etth1": ("ETTh1.csv", {}, "date"),
}


def timing(frame, column):
    stamps = pd.to_datetime(frame[column], errors="coerce")
    delta = stamps.diff().dt.total_seconds().dropna()
    median = delta.median()
    return {
        "median_dt_s": float(median),
        "share_dt_not_median": float((delta != median).mean()),
        "share_dt_gt_2x": float((delta > 2 * median).mean()),
        "longest_gap_in_samples": float(delta.max() / median),
    }


def main():
    for label, (name, options, time_column) in SOURCES.items():
        frame = pd.read_csv(PUBLIC / name, **options)
        frame.columns = [str(c).strip() for c in frame.columns]
        print(f"\n===== {label}: rows={len(frame)} columns={len(frame.columns)}")
        if time_column:
            print("timing", timing(frame, time_column))
        numeric = frame.select_dtypes("number")
        print("missing share", float(numeric.isna().mean().mean()))
        rows = []
        for column in numeric:
            values = numeric[column].to_numpy(dtype=float)
            values = values[~np.isnan(values)]
            stats = signal_statistics(values[: 10 * WINDOW])
            if stats["constant"]:
                rows.append({"column": column, "constant": True})
                continue
            head = values[:WINDOW]
            rows.append(
                {
                    "column": column,
                    "unique": stats["unique"],
                    "flat_share": round(stats["flat_share"], 3),
                    "flat_run": stats["flat_run"],
                    "at_ext": round(
                        max(stats["at_max_share"], stats["at_min_share"]), 3
                    ),
                    "zero": round(stats["zero_share"], 3),
                    "hf_noise": round(stats["hf_noise"], 4),
                    "rough": round(stats["roughness"], 3),
                    "peak_share": round(stats["peak_share"], 3),
                    "peak_period": round(stats["peak_period"], 1),
                    "skew": round(float(pd.Series(head).skew()), 2),
                    "kurt": round(float(pd.Series(head).kurt()), 1),
                }
            )
        print(pd.DataFrame(rows).to_string(index=False))
        columns = list(numeric.columns)[:16]
        figure, axes = plt.subplots(
            len(columns), 2, figsize=(20, 1.7 * len(columns)), squeeze=False
        )
        middle = max(0, len(numeric) // 2 - WINDOW // 2)
        for row, column in enumerate(columns):
            values = numeric[column].to_numpy(dtype=float)
            axes[row, 0].plot(values[middle : middle + WINDOW], linewidth=0.6)
            axes[row, 0].set_ylabel(column[:14], fontsize=7)
            axes[row, 1].plot(values[middle : middle + 300], linewidth=0.8)
            for axis in axes[row]:
                axis.tick_params(labelsize=6)
        axes[0, 0].set_title(f"{label}: 2048 rows")
        axes[0, 1].set_title("first 300 rows of the same window")
        figure.tight_layout()
        figure.savefig(Path(__file__).with_name(f"real_shapes_{label}.png"), dpi=70)
        plt.close(figure)


if __name__ == "__main__":
    main()

"""Forecast the tail of one CSV column with a portable Basin model, plot it and score it.

Shared by run_pytorch.py and run_fla.py. The first ``--context-fraction`` of the
series is the model's context; the rest is held out and forecast in one call.
"""

import argparse
import os
import sys
from pathlib import Path
from time import perf_counter

import numpy as np
import pandas as pd
import torch

ROOT = Path(__file__).resolve().parents[2]
MODELS = ROOT / "results/model/portable"
VARIANTS = {
    "pytorch": ("basin-rwkv7-pytorch", "modeling_basin_pytorch", "BasinPyTorch"),
    "fla": ("basin-rwkv7-fla", "modeling_basin_fla", "BasinFLA"),
}


def parse(variant, argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--data", default=str(ROOT / "data/00_public/airline-passengers.csv")
    )
    parser.add_argument(
        "--column", default=None, help="value column; default: last numeric column"
    )
    parser.add_argument("--context-fraction", type=float, default=0.8)
    parser.add_argument(
        "--device", default=None, help="default: cuda if available, else cpu"
    )
    parser.add_argument("--dtype", default="bfloat16", choices=("bfloat16", "float32"))
    parser.add_argument("--models", default=str(MODELS))
    parser.add_argument("--no-show", action="store_true", help="save the plot only")
    args = parser.parse_args(argv)
    args.variant = variant
    return args


def metrics(actual, point, lower, upper, context):
    error = point - actual
    naive = np.abs(np.diff(context)).mean()
    return {
        "MAE": float(np.abs(error).mean()),
        "RMSE": float(np.sqrt((error**2).mean())),
        "MAPE %": float((np.abs(error) / np.abs(actual)).mean() * 100),
        "MASE (vs in-context one-step naive)": float(np.abs(error).mean() / naive),
        "MAE of repeating the last value": float(np.abs(context[-1] - actual).mean()),
        "Share of actuals inside the 10-90% band": float(
            ((actual >= lower) & (actual <= upper)).mean()
        ),
    }


def run(args):
    folder_name, module, class_name = VARIANTS[args.variant]
    folder = Path(args.models) / folder_name
    if not (folder / "model.safetensors").is_file():
        raise FileNotFoundError(
            f"{folder} not found: run `python src/08_model_export/export_models.py` first"
        )
    frame = pd.read_csv(args.data)
    column = args.column or frame.select_dtypes("number").columns[-1]
    series = frame[column].to_numpy(dtype=np.float32)
    split = int(len(series) * args.context_fraction)
    horizon = len(series) - split
    if split < 1 or horizon < 1:
        raise ValueError("context-fraction must leave both a context and a horizon")
    context, actual = series[:split], series[split:]

    sys.path.insert(0, str(folder))
    device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    started = perf_counter()
    model = getattr(__import__(module), class_name).from_pretrained(
        folder, device=device, dtype=getattr(torch, args.dtype)
    )
    load_seconds = perf_counter() - started
    started = perf_counter()
    forecast = model.forecast(torch.tensor(context)[None], horizon)
    forecast_seconds = perf_counter() - started
    levels = model.config.quantiles
    quantiles = forecast.quantiles[0].numpy()
    point = forecast.point[0].numpy()
    lower, upper = quantiles[:, 0], quantiles[:, -1]

    print(f"Model: {folder_name} on {device} ({args.dtype})")
    print(
        f"Data: {args.data} column '{column}': {split} context points, {horizon} forecast"
    )
    print(
        f"Load {load_seconds:.1f} s, forecast {forecast_seconds * 1000:.0f} ms (first call)"
    )
    for name, value in metrics(actual, point, lower, upper, context).items():
        print(f"  {name:42s} {value:10.3f}")

    import matplotlib

    if args.no_show or not (os.name == "nt" or os.environ.get("DISPLAY")):
        matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    steps = np.arange(len(series))
    figure, axis = plt.subplots(figsize=(11, 4.5))
    axis.plot(steps[:split], context, color="gray", label="Context")
    axis.plot(steps[split:], actual, color="black", label="Actual (held out)")
    axis.plot(steps[split:], point, color="tab:blue", label="Basin median forecast")
    axis.fill_between(
        steps[split:],
        lower,
        upper,
        color="tab:blue",
        alpha=0.2,
        label=f"{levels[0]:.0%}-{levels[-1]:.0%} quantile band",
    )
    axis.axvline(split, color="gray", linestyle="--")
    axis.set(
        title=f"{Path(args.data).name} / {column}: {folder_name}",
        xlabel="Row",
        ylabel=column,
    )
    axis.legend()
    figure.tight_layout()
    output = MODELS / "demo" / f"{args.variant}_{Path(args.data).stem}.png"
    output.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(output, dpi=140)
    print(f"Plot saved: {output}")
    if matplotlib.get_backend().lower() != "agg":
        plt.show()
    return output

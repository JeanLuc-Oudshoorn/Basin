"""Compare an annealing experiment's holdout run with the run of the weights it started from.

CPU only; reads saved results, forecasts nothing.

    python src/10_annealing/compare_holdout.py --experiment flat25_lr1e-5

"Before" is the main run's holdout test of exactly the experiment's start
weights (matched by their checksum), "after" the experiment's most recent
holdout test. Both must have forecast the same points. Besides the headline
numbers, the points are split into the kinds the annealing mix is aimed at.
"""

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
for directory in (
    HERE,
    ROOT / "src/03_transforming_input_data",
    ROOT / "src/06_holdout_test",
):
    sys.path.insert(0, str(directory))
from annealing_experiment import experiment_config_path  # noqa: E402
from holdout_runs import holdout_runs, latest_holdout_run  # noqa: E402

KEYS = ["source", "target", "origin", "time_index"]
HEADLINE = (
    ("Table MASE, median", "table_mase_median"),
    ("Table MASE, 10th percentile", "table_mase_p10"),
    ("Table MASE, 90th percentile", "table_mase_p90"),
)
POOLED = (
    ("Pointwise scaled error, median", "median_absolute_scaled_error"),
    ("Pointwise scaled error, 10th percentile", "absolute_scaled_error_p10"),
    ("Pointwise scaled error, 90th percentile", "absolute_scaled_error_p90"),
)


def baseline_run(config, start):
    """The main run's holdout test of the experiment's start weights."""
    base = json.loads(
        (ROOT / config["annealing"]["base_config"]).read_text(encoding="utf-8")
    )
    matching = []
    for run in holdout_runs(base):
        used = run / "model_used.json"
        if used.is_file() and (
            json.loads(used.read_text(encoding="utf-8")).get("evaluated_weights_sha256")
            == start["weights_sha256"]
        ):
            matching.append(run)
    if not matching:
        raise FileNotFoundError(
            "No holdout run of the main model evaluated the weights this experiment "
            f"started from (sha256 {start['weights_sha256']}). Run "
            f"src/06_holdout_test/holdout_test.py --config {config['annealing']['base_config']} "
            "first, or name a run folder with --baseline."
        )
    return matching[-1]


def scaled_errors(run, column="rwkv", extra=()):
    frame = pd.read_parquet(
        run / "test_predictions.parquet",
        columns=[
            *KEYS,
            "forecast_token",
            "actual",
            "naive",
            "context_scale",
            "mase_scale",
            column,
            *extra,
        ],
    )
    frame = frame[
        np.isfinite(frame.actual)
        & np.isfinite(frame.mase_scale)
        & (frame.mase_scale > 0)
    ]
    for name in (column, *extra):
        frame[name] = (frame[name] - frame.actual).abs() / frame.mase_scale
    return frame


def point_groups(frame, smooth_roughness=0.03):
    """Kind of each scored point: flat, on a smooth series, or other."""
    flat = (frame.actual == frame.naive).to_numpy()
    smooth = (frame.mase_scale / frame.context_scale < smooth_roughness).to_numpy()
    return np.where(flat, "flat", np.where(smooth, "smooth", "other"))


def compare(before_run, after_run):
    before_metrics = json.loads((before_run / "test_metrics.json").read_text("utf-8"))
    after_metrics = json.loads((after_run / "test_metrics.json").read_text("utf-8"))
    references = [
        name
        for name in before_metrics["models"]
        if name not in ("rwkv", "naive", "prophet")
    ]
    before = scaled_errors(before_run, extra=references)
    after = scaled_errors(after_run).rename(columns={"rwkv": "after"})
    merged = before.merge(after[[*KEYS, "after"]], on=KEYS, how="inner")
    if not len(before) == len(after) == len(merged):
        raise ValueError(
            f"The runs did not forecast the same points (before {len(before):,}, after "
            f"{len(after):,}, shared {len(merged):,}). Compare runs made with the same "
            "holdout sample settings."
        )
    merged = merged.rename(columns={"rwkv": "before"}).dropna(
        subset=["before", "after"]
    )
    merged["group"] = point_groups(merged)
    rows = []
    leaders = {
        name: {row["model"]: row for row in metrics["leaderboard"]}["rwkv"]
        for name, metrics in (("before", before_metrics), ("after", after_metrics))
    }
    for label, key in HEADLINE:
        rows.append((label, leaders["before"][key], leaders["after"][key]))
    for label, key in POOLED:
        rows.append(
            (
                label,
                before_metrics["pooled"]["rwkv"][key],
                after_metrics["pooled"]["rwkv"][key],
            )
        )
    headline = pd.DataFrame(rows, columns=["metric", "before", "after"])
    headline["change_percent"] = 100 * (headline.after / headline.before - 1)
    models = ["before", "after", *references]
    groups = []
    selections = [(name, merged.group == name) for name in ("flat", "smooth", "other")]
    selections.append(("first forecast patch", merged.forecast_token == 1))
    selections.append(("all points", np.ones(len(merged), dtype=bool)))
    for name, mask in selections:
        part = merged[mask]
        row = {"points": name, "share_percent": 100 * len(part) / len(merged)}
        for statistic, level in (("p10", 0.10), ("median", 0.50)):
            for model in models:
                row[f"{statistic}_{model}"] = float(part[model].quantile(level))
        groups.append(row)
    return headline, pd.DataFrame(groups)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--experiment", required=True)
    parser.add_argument(
        "--run", help="experiment holdout run folder name; default: most recent"
    )
    parser.add_argument(
        "--baseline", help="holdout run folder of the model before annealing"
    )
    parser.add_argument("--json", help="also save the tables to this file")
    args = parser.parse_args()
    config = json.loads(experiment_config_path(args.experiment).read_text("utf-8"))
    after_run = latest_holdout_run(config, args.run)
    if args.baseline:
        before_run = Path(args.baseline)
        before_run = before_run if before_run.is_absolute() else ROOT / before_run
    else:
        start = ROOT / config["paths"]["training"] / "annealing_start.json"
        before_run = baseline_run(config, json.loads(start.read_text("utf-8")))
    headline, groups = compare(before_run, after_run)
    print(f"before: {before_run}\nafter:  {after_run}\n")
    with pd.option_context("display.width", 200, "display.max_columns", 30):
        print(headline.to_string(index=False, float_format=lambda v: f"{v:.4g}"))
        print("\nPointwise scaled error (MASE units) by kind of point; lower is better")
        print(groups.to_string(index=False, float_format=lambda v: f"{v:.4g}"))
    if args.json:
        Path(args.json).write_text(
            json.dumps(
                {
                    "before": str(before_run),
                    "after": str(after_run),
                    "headline": headline.to_dict("records"),
                    "groups": groups.to_dict("records"),
                },
                indent=2,
            ),
            encoding="utf-8",
        )


if __name__ == "__main__":
    main()

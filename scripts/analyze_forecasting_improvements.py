"""Read saved evaluations for the November 2026 forecasting design review.

No model loading, GPU work, corpus scan or changes to training artifacts.
"""

import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
OUTPUT = ROOT / "docs/model_training/forecasting_review_20261110"


def digest(path):
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def trim(values):
    values = np.sort(np.asarray(values)[np.isfinite(values)])
    k = int(len(values) * 0.05)
    return float(values[k : len(values) - k].mean())


def summary(values):
    values = np.asarray(values)
    values = values[np.isfinite(values)]
    return {
        "count": len(values),
        "mean": float(values.mean()),
        "median": float(np.median(values)),
        "trimmed_mean_5_percent_each_tail": trim(values),
        "p90": float(np.quantile(values, 0.9)),
        "p99": float(np.quantile(values, 0.99)),
    }


def paired(left, right):
    pair = pd.concat([left.rename("before"), right.rename("after")], axis=1).dropna()
    change = (pair.after - pair.before).to_numpy()
    rng = np.random.default_rng(7)
    medians = []
    for _ in range(1000):
        medians.append(
            float(np.median(change[rng.integers(0, len(change), len(change))]))
        )
    return {
        "paired_count": len(pair),
        "fraction_improved": float((change < 0).mean()),
        "mean_delta_after_minus_before": float(change.mean()),
        "median_delta_after_minus_before": float(np.median(change)),
        "median_delta_bootstrap_95_percent_interval": np.quantile(
            medians, [0.025, 0.975]
        ).tolist(),
        "before": summary(pair.before),
        "after": summary(pair.after),
    }


def main():
    training = ROOT / "results/model/training"
    history_path = training / "validation_history.json"
    history_bytes = history_path.read_bytes()
    history = json.loads(history_bytes)
    # Freeze the set of completed checks before reading a live training directory.
    history = [row for row in history if row["validation_check"] <= 8]
    evidence = {
        "scope": "completed checks 0 through 8; seed-7 table bootstrap, 1000 replicates",
        "history_sha256": hashlib.sha256(history_bytes).hexdigest(),
        "validation": [],
    }
    tables = {}
    for row in history:
        path = (
            training / row["table_metrics_path"].replace("\\", "/").rsplit("/", 1)[-1]
        )
        frame = pd.read_parquet(path)
        if frame.duplicated(["source", "origin"]).any():
            raise ValueError(f"Duplicate validation requests: {path}")
        table = frame.groupby("source")[["loss", "mase"]].mean()
        weighted_loss = frame.loss * frame.loss_mass
        actual_loss = float(weighted_loss.sum() / frame.loss_mass.sum())
        if not np.isclose(actual_loss, row["loss"], rtol=1e-10):
            raise ValueError(f"Validation loss does not reconstruct: {path}")
        entry = {
            "check": row["validation_check"],
            "epoch": row["epoch"],
            "step": row["step"],
            "active_hours": row["active_training_seconds"] / 3600,
            "training_loss": row["training_loss"],
            "epoch_windows_processed": row["epoch_windows_processed"],
            "loss": row["loss"],
            "mase": row["mase"],
            "next_learning_rate": row["next_learning_rate"],
            "table_loss": summary(table.loss),
            "table_mase": summary(table.mase),
            "top_five_forecasts_loss_share": float(
                weighted_loss.nlargest(5).sum() / weighted_loss.sum()
            ),
            "forecast_count": len(frame),
            "source_file": str(path.relative_to(ROOT)),
            "source_sha256": digest(path),
            "request_panel_sha256": hashlib.sha256(
                frame.sort_values(["source", "origin"])[["source", "origin"]]
                .to_csv(index=False)
                .encode()
            ).hexdigest(),
        }
        evidence["validation"].append(entry)
        tables[row["validation_check"]] = table
    recent_path = training / "loss_history.json"
    recent_bytes = recent_path.read_bytes()
    recent = json.loads(recent_bytes)
    losses = np.array([row["loss"] for row in recent])
    masses = np.array(
        [row.get("target_loss_weight", row["target_points"]) for row in recent]
    )
    gradients = np.array([row["gradient_norm"] for row in recent])
    evidence["recent_training_sample"] = {
        "source_sha256": hashlib.sha256(recent_bytes).hexdigest(),
        "step_start": recent[0]["step"],
        "step_stop": recent[-1]["step"],
        "updates": len(recent),
        "loss": summary(losses),
        "label_mass_weighted_mean_loss": float(np.average(losses, weights=masses)),
        "top_five_updates_loss_share": float(
            np.sort(losses * masses)[-5:].sum() / (losses * masses).sum()
        ),
        "gradient_norm": summary(gradients),
        "maximum_gradient_norm": float(gradients.max()),
        "maximum_loss": float(losses.max()),
        "limitation": "latest saved update sample only; not a census of training tables or historical updates",
    }
    for before, after in ((5, 7), (6, 7), (7, 8)):
        if not tables[before].index.equals(tables[after].index):
            raise ValueError("Validation panels differ")
        signatures = {
            row["check"]: row["request_panel_sha256"] for row in evidence["validation"]
        }
        if signatures[before] != signatures[after]:
            raise ValueError("Paired validation forecast origins differ")
        evidence[f"paired_tables_{before}_to_{after}"] = {
            metric: paired(tables[before][metric], tables[after][metric])
            for metric in ("loss", "mase")
        }
    directory = ROOT / "results/model/holdout_test"
    metrics = json.loads((directory / "test_metrics.json").read_text())
    origins = json.loads((directory / "origin_metrics.json").read_text())
    frame = pd.DataFrame(
        {
            "source": row["source"],
            **{name: row[name]["mase"] for name in ("rwkv", "chronos", "timestral")},
        }
        for row in origins
    )
    table = frame.groupby("source")[["rwkv", "chronos", "timestral"]].mean()
    evidence["holdout"] = {
        "model_identity": json.loads((directory / "model_used.json").read_text()),
        "benchmark_metadata": metrics["benchmark_metadata"],
        "leaderboard": metrics["leaderboard"],
        "table_count": metrics["table_count"],
        "origin_target_count": metrics["origin_target_count"],
        "per_forecast_token_mase": {
            key: {
                name: value[name]["mase"] for name in ("rwkv", "chronos", "timestral")
            }
            for key, value in metrics["by_forecast_token"].items()
        },
        "paired_tables_rwkv_to_baseline": {
            name: paired(table.rwkv, table[name]) for name in ("chronos", "timestral")
        },
    }
    OUTPUT.mkdir(parents=True, exist_ok=True)
    path = OUTPUT / "evidence.json"
    path.write_text(
        json.dumps(evidence, indent=2, allow_nan=False) + "\n", encoding="utf-8"
    )
    print(path)
    for row in evidence["validation"]:
        print(
            row["check"],
            "train",
            row["training_loss"],
            "loss",
            row["loss"],
            "table median loss",
            row["table_loss"]["median"],
            "table median MASE",
            row["table_mase"]["median"],
            "trim loss",
            row["table_loss"]["trimmed_mean_5_percent_each_tail"],
        )
    print(json.dumps(evidence["paired_tables_7_to_8"], indent=2))


if __name__ == "__main__":
    main()

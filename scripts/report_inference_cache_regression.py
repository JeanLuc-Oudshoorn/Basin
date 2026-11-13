"""Audit saved cache coverage and matched accuracy; never rerun forecasts."""

import ast
import json
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
OUTPUT = ROOT / "docs/audits/2026-11-03/inference_cache"


def errors(frame, name):
    difference = frame[name] - frame.actual
    valid = np.isfinite(frame.mase_scale) & (frame.mase_scale > 0)
    return {
        "mase": float(np.mean(np.abs(difference[valid]) / frame.mase_scale[valid])),
        "rmse": float(np.sqrt(np.mean(difference**2))),
        "observations": len(frame),
        "mase_observations": int(valid.sum()),
    }


def main():
    base = ROOT / "results/model/holdout_test"
    metrics = json.loads((base / "test_metrics.json").read_text())
    architecture = json.loads(
        (ROOT / "results/model/training/architecture.json").read_text()
    )
    saved_identity = json.loads((base / "model_used.json").read_text())
    events = ROOT / "logs/20261103T170354_564971Z_holdout_test_2f206829/events.log"
    keys = {
        ast.literal_eval(line.split("shape=", 1)[1].split(" seconds=", 1)[0])
        for line in events.read_text().splitlines()
        if "Inference CUDA graph captured shape=" in line
    }
    requests = pd.read_parquet(base / "request_latencies.parquet")
    requests = requests[
        (requests.model == "rwkv")
        & (requests.phase == "scored")
        & (requests.repeat == 0)
    ].copy()
    width = architecture["config"]["width"]
    patch = architecture["config"]["patch_size"]
    requests["bucket"] = (
        ((requests.context_points + patch - 1) // patch + 15) // 16 * 16
    )
    requests["cached"] = [
        (
            (1, row.targets + row.covariates, row.bucket, width),
            row.targets,
            row.horizon // patch,
            "cuda:0",
        )
        in keys
        for row in requests.itertuples()
    ]
    evidence = {
        "model_identity": saved_identity,
        "original_requests": len(requests),
        "unique_profiles": len(
            requests.groupby(["targets", "covariates", "bucket", "horizon"])
        ),
        "captured_profiles": len(keys),
        "replay_coverage": float(requests.cached.mean()),
        "latency_by_cache_coverage": {
            str(bool(cached)): {
                "requests": len(group),
                "median_ms": float(np.median(group.seconds) * 1000),
            }
            for cached, group in requests.groupby("cached")
        },
        "latency_by_horizon": {
            str(int(horizon)): {
                "requests": len(group),
                "median_ms": float(np.median(group.seconds) * 1000),
            }
            for horizon, group in requests.groupby("horizon")
        },
        "accuracy_by_tokens": metrics["by_forecast_tokens"],
    }
    new = pd.read_parquet(base / "test_predictions.parquet")
    previous = (
        ROOT
        / "results/447m_model/holdout_test/current_holdout/test_predictions.parquet"
    )
    if previous.exists():
        old = pd.read_parquet(previous)
        join_keys = ["source", "target", "origin", "time_index"]
        matched = new[new.forecast_tokens == 1].merge(
            old[join_keys + ["rwkv", "actual", "mase_scale"]],
            on=join_keys,
            suffixes=("", "_447m"),
            validate="one_to_one",
        )
        if not np.array_equal(
            matched.actual, matched.actual_447m
        ) or not np.array_equal(
            matched.mase_scale, matched.mase_scale_447m, equal_nan=True
        ):
            raise RuntimeError("Matched historical targets or MASE scales differ")
        evidence["matched_one_token_accuracy"] = {
            "tables": int(matched.source.nunique()),
            "origins": len(matched[["source", "origin"]].drop_duplicates()),
            "scores": {
                name: errors(matched, name)
                for name in ("rwkv", "rwkv_447m", "chronos", "timestral", "naive")
            },
            "scope": "intersection of saved runs, one-token requests only; not the full current holdout",
        }
    (OUTPUT / "saved_run_evidence.json").write_text(
        json.dumps(evidence, indent=2) + "\n", encoding="utf-8"
    )


if __name__ == "__main__":
    main()

"""Exercise the default-size trainer and real baselines on ten tiny synthetic tables."""
# ruff: noqa: E402

import argparse
import copy
import json
import sys
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src/05_model_training"))
sys.path.insert(0, str(ROOT / "src/04_creating_embeddings"))
sys.path.insert(0, str(ROOT / "src/03_transforming_input_data"))
from embed import embed
from train import load_config, run_context, train
from transform import transform_next_patch, write_json


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--audit-dir", default=str(ROOT / "docs/audits/2026-10-25/default_holdout")
    )
    parser.add_argument("--exercise-early-stopping", action="store_true")
    args = parser.parse_args()
    config = copy.deepcopy(load_config())
    audit = Path(args.audit_dir).resolve()
    audit.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix="basin_holdout_") as temporary:
        base = Path(temporary)
        config["data"].pop("generated_dates", None)
        config["validation"]["enabled"] = (
            False  # Synthetic protocol fixture, not an admission-quality test.
        )
        config["training"].update(steps=2, checkpoint_every=0)
        if args.exercise_early_stopping:
            config["training"].update(
                steps=None,
                epochs=None,
                early_stopping={
                    "patience_epochs": 3,
                    "min_delta": 1e9,
                    "monitor": "loss",
                },
            )
        config["logging"]["level"] = "INFO"
        for name in ("raw", "processed", "embedded", "training"):
            config["paths"][name] = str(base / name)
        config["paths"]["forecasts"] = str(audit / "smoke_forecasts")
        config["paths"]["holdout_test"] = str(audit / "holdout_test")
        for index in range(10):
            path = base / "raw" / f"table_{index:02d}"
            path.mkdir(parents=True)
            time = np.arange(64)
            pd.DataFrame(
                {
                    "time_index": time,
                    "outcome": 10
                    + 0.03 * time
                    + np.sin(2 * np.pi * time / 16 + index / 10),
                    "covariate": np.cos(2 * np.pi * time / 16),
                }
            ).to_parquet(path / "data.parquet", index=False)
        with run_context(config, "transform"):
            transform_next_patch(config)
        with run_context(config, "embed"):
            embed(config)
        with run_context(config, "train"):
            train(config)
        # Exercise the independent holdout entry point against its saved split.
        sys.path.insert(0, str(ROOT / "src/06_holdout_test"))
        from holdout_test import run

        replay = run(config)
        report = {}
        for filename in (
            "architecture.json",
            "dataset_split.json",
            "loss_history.json",
            "validation_history.json",
            "embedding_optimization.json",
            "early_stopping.json",
            "learning_rate_scheduler.json",
        ):
            content = json.loads((base / "training" / filename).read_text())
            write_json(audit / filename, content)
            report[filename] = content
        report["test_metrics.json"] = replay
        write_json(audit / "test_metrics.json", replay)
        assert (
            440_000_000
            <= report["architecture.json"]["total_parameters"]
            <= 460_000_000
        )
        assert report["dataset_split.json"]["test_tables"] == 1
        assert report["dataset_split.json"]["train_tables"] == 9
        assert len(report["loss_history.json"]) == (
            8 if args.exercise_early_stopping else 2
        )
        assert len(report["validation_history.json"]) == (
            5 if args.exercise_early_stopping else 2
        )
        assert all(
            np.isfinite(row["loss"]) for row in report["validation_history.json"]
        )
        if args.exercise_early_stopping:
            assert report["early_stopping.json"]["reason"] == "early_stopping"
            assert report["early_stopping.json"]["best_epoch"] == 1
            assert report["early_stopping.json"]["epochs_without_improvement"] == 3
        assert report["test_metrics.json"]["origin_target_count"] == 2
        assert report["test_metrics.json"]["pooled"]["observations"] == 32
        assert replay["pooled"] == report["test_metrics.json"]["pooled"]
        print(
            "PASS: full-size default train, full validation after each epoch, disjoint rolling holdout and both real baselines"
        )


if __name__ == "__main__":
    main()

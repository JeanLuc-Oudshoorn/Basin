"""Bounded, real-model smoke of all benchmark adapters on a saved holdout subset."""

# ruff: noqa: E402
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
for stage in ("03_transforming_input_data", "05_model_training", "06_holdout_test"):
    sys.path.insert(0, str(ROOT / "src" / stage))
from holdout_test import run
from transform import load_config, run_context, write_json


def main():
    config = load_config()
    output = ROOT / "docs/audits/2026-10-29/foundation_benchmarks"
    config["paths"]["holdout_test"] = str(output / "smoke")
    config["holdout"].update(
        sample_percent=0.02,
        benchmark_models=["prophet", "naive", "chronos", "timestral"],
    )
    config["plot"]["max_plots"] = 2
    with run_context(config, "holdout_test"):
        result = run(config)
    report = {
        "status": "passed",
        "models": result["models"],
        "metadata": result["benchmark_metadata"],
        "table_count": result["table_count"],
        "origin_count": result["origin_target_count"],
        "seconds": result["seconds"],
        "selection": result["holdout_selection"],
        "pooled": result["pooled"],
    }
    write_json(output / "gpu_verification.json", report)
    print(
        "PASS: real RWKV, Prophet, naive, Chronos-2 and Timestral on sampled holdout tables"
    )


if __name__ == "__main__":
    main()

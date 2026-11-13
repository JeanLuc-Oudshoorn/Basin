"""One-table GPU holdout smoke without opening the processed training corpus."""
# ruff: noqa: E402

import json
import sys
from pathlib import Path
from time import perf_counter

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
for stage in ("03_transforming_input_data", "05_model_training", "06_holdout_test"):
    sys.path.insert(0, str(ROOT / "src" / stage))
from holdout_test import run
from transform import load_config, run_context, write_json


def main():
    config = load_config()
    output = ROOT / "docs/audits/2026-10-30/bolt_screening_startup/holdout_smoke"
    config["paths"]["holdout_test"] = str(output)
    config["holdout"].update(
        sample_percent=0.001, benchmark_models=["naive", "chronos", "timestral"]
    )
    config["plot"]["max_plots"] = 0
    reference_path = output.parent / "gpu_holdout_verification.json"
    reference = (
        json.loads(reference_path.read_text()) if reference_path.is_file() else None
    )
    # These do not exist: the smoke must succeed without upstream training files.
    config["paths"].update(
        embedded=str(output / "absent_embeddings"),
        processed=str(output / "absent_windows"),
    )
    started = perf_counter()
    with run_context(config, "holdout_test"):
        result = run(config)
    difference = None
    if reference is not None:
        names = ["mase", "rmse", "mae", "normalized_rmse", "normalized_pinball"]
        current = np.array([result["pooled"]["rwkv"][name] for name in names])
        previous = np.array([reference["pooled"]["rwkv"][name] for name in names])
        difference = float(np.abs(current - previous).max())
        if not np.allclose(current, previous, rtol=1e-6, atol=1e-6):
            raise RuntimeError(
                f"RWKV reference metrics changed: maximum difference {difference}"
            )
    write_json(
        output.parent / "gpu_holdout_verification.json",
        {
            "status": "passed",
            "seconds_including_startup": perf_counter() - started,
            "table_count": result["table_count"],
            "models": result["models"],
            "primary_metric": result["primary_metric"],
            "pooled": result["pooled"],
            "latency": result["latency"],
            "model_load_seconds": result["rwkv_model_load_seconds"],
            "rwkv_max_absolute_reference_metric_difference": difference,
        },
    )


if __name__ == "__main__":
    main()

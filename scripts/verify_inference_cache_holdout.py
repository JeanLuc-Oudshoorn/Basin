"""Repeat the paired holdout in an audit directory; preserve production results."""
# ruff: noqa: E402

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch

ROOT = Path(__file__).resolve().parents[1]
for stage in ("03_transforming_input_data", "05_model_training", "06_holdout_test"):
    sys.path.insert(0, str(ROOT / "src" / stage))
from holdout_test import run
from transform import load_config, run_context, write_json


def main():
    config = load_config()
    original = ROOT / "docs/audits/2026-11-03/inference_cache/reference"
    selection = json.loads((original / "sample_selection.json").read_text())
    original_metrics = json.loads((original / "test_metrics.json").read_text())
    # Pin the original cohort even if the user's live config changes meanwhile.
    config["holdout"]["sample_percent"] = selection["requested_sample_percent"]
    config["holdout"]["sample_seed"] = selection["sample_seed"]
    config["holdout"]["horizon_probabilities"] = original_metrics["horizon_policy"][
        "probabilities"
    ]
    config["holdout"]["horizon_seed"] = original_metrics["horizon_policy"]["seed"]
    config["holdout"]["percentiles"] = original_metrics["percentiles"]
    output = ROOT / "docs/audits/2026-11-03/inference_cache/paired_holdout_final"
    config["paths"]["holdout_test"] = str(output)
    with run_context(config, "inference_cache_paired_holdout"):
        metrics = run(config)
        old_identity = json.loads((original / "model_used.json").read_text())
        new_identity = json.loads((output / "model_used.json").read_text())
        if old_identity["checkpoint_sha256"] != new_identity["checkpoint_sha256"]:
            raise RuntimeError("Trained checkpoint differs from the original holdout")
        before = pd.read_parquet(original / "test_predictions.parquet")
        after = pd.read_parquet(output / "test_predictions.parquet")
        keys = ["source", "target", "origin", "time_index", "forecast_horizon"]
        joined = before.merge(
            after, on=keys, suffixes=("_before", "_after"), validate="one_to_one"
        )
        if len(joined) != len(before) or len(joined) != len(after):
            raise RuntimeError("Paired holdout sources, origins or horizons differ")
        differences = {
            name: float(
                np.max(np.abs(joined[name + "_before"] - joined[name + "_after"]))
            )
            for name in metrics["models"]
        }
        if differences["rwkv"] != 0:
            raise RuntimeError("RWKV forecasts changed")
        write_json(
            output / "verification.json",
            {
                "checkpoint_sha256": new_identity["checkpoint_sha256"],
                "observations": len(joined),
                "maximum_absolute_prediction_differences": differences,
                "rwkv_inference": metrics["rwkv_inference"],
                "cuda_allocator": torch.cuda.memory.get_allocator_backend(),
                "device_memory_free_total": torch.cuda.mem_get_info(),
            },
        )


if __name__ == "__main__":
    main()

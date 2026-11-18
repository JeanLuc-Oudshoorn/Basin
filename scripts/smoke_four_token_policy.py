"""Isolated CUDA train/validate/forecast smoke; production artifacts stay intact."""
# ruff: noqa: E402

import argparse
import json
import sys
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from loguru import logger

ROOT = Path(__file__).resolve().parents[1]
for stage in (
    "03_transforming_input_data",
    "04_patch_indexing",
    "05_model_training",
    "06_holdout_test",
):
    sys.path.insert(0, str(ROOT / "src" / stage))
from fast_inference import configure_fast_inference
from holdout_test import run as holdout_run
from index_patches import build_patch_index
from train import RWKVForecaster, train
from transform import load_config, run_context, transform_next_patch, write_json


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--full-size", action="store_true")
    args = parser.parse_args()
    if not torch.cuda.is_available():
        raise RuntimeError("Run inside the configured WSL CUDA runtime")
    config = load_config()
    full_model_settings = dict(config["model"])
    config["data"].pop("generated_dates", None)
    config["validation"]["enabled"] = False
    config["model"].update(
        width=64,
        layers=2,
        variate_heads=4,
        context_length=256,
        max_patches=16,
        gradient_checkpointing=False,
    )
    config["data"].update(context_length=256, horizon=64, max_tables=None)
    config["training"].update(
        epochs=2,
        steps=None,
        batch_size=2,
        microbatch_size=1,
        validation_batch_size=4,
        compile_variate=False,
        compile_variate_inference=False,
        prefetch=True,
        resume="auto",
        token_inventory={"mode": "off"},
        early_stopping={"patience_epochs": 3, "min_delta": 0.0, "monitor": "loss"},
    )
    config["holdout"].update(
        benchmark_models=["naive"],
        sample_percent=100,
        latency={"warmup_requests": 0, "repeats": 1},
    )
    config["plot"].update(max_plots=1)
    if args.full_size:
        config["model"].update(full_model_settings)
        config["data"]["context_length"] = full_model_settings["context_length"]
        config["training"]["epochs"] = 1
    output = ROOT / (
        f"docs/audits/2026-11-02/four_token_policy_{config['model']['layers']}layer_smoke"
        if args.full_size
        else "docs/audits/2026-11-02/four_token_policy_smoke"
    )
    output.mkdir(parents=True, exist_ok=True)
    temporary_root = ROOT / ".cache/tmp/wsl"
    temporary_root.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(
        prefix="four_token_", dir=temporary_root
    ) as temporary:
        base = Path(temporary)
        config["paths"].update(
            raw=str(base / "raw"),
            processed=str(base / "processed"),
            patch_index=str(base / "patch_index"),
            training=str(base / "training"),
            holdout_test=str(output / "holdout"),
        )
        for index in range(12):
            length = ((96, 140, 512, 2048) if args.full_size else (96, 140, 256, 320))[
                index % 4
            ]
            path = base / "raw" / str(index)
            path.mkdir(parents=True)
            time = np.arange(length, dtype=float)
            pd.DataFrame(
                {
                    "time_index": time,
                    "outcome": np.sin(time / 8) + 0.002 * time + index,
                    "cov": np.cos(time / 8),
                }
            ).to_parquet(path / "data.parquet")
        with run_context(config, "four_token_policy_smoke"):
            transform_next_patch(config)
            build_patch_index(config)
            train(config)
            history = json.loads(
                (base / "training/validation_history.json").read_text()
            )
            model = RWKVForecaster.load(base / "training/model.pt").eval()
            context = torch.linspace(0, 1, 160).reshape(1, 1, -1)
            covariates = torch.cos(context)
            expected = {
                h: model.forecast(
                    context.to("cuda"), h, past_covariates=covariates.to("cuda")
                ).quantiles.cpu()
                for h in (16, 32, 48, 64, 80, 128)
            }
            configure_fast_inference(model, config["training"])
            differences = {}
            for horizon, eager in expected.items():
                actual = model.forecast(
                    context, horizon, past_covariates=covariates
                ).quantiles.cpu()
                torch.testing.assert_close(actual, eager, atol=0, rtol=0)
                differences[str(horizon)] = float((actual - eager).abs().max())
            model.fast_inference.clear()
            del model
            torch.cuda.empty_cache()
            metrics = holdout_run(config)
            result = {
                "device": torch.cuda.get_device_name(),
                "training_epochs": config["training"]["epochs"],
                "width": config["model"]["width"],
                "layers": config["model"]["layers"],
                "output_patches": 4,
                "horizon_loss_weights": config["model"]["horizon_loss_weights"],
                "validation_history": history,
                "graph_eager_max_abs_differences": differences,
                "holdout_tables": metrics["table_count"],
                "holdout_horizon_policy": metrics["horizon_policy"],
                "production_artifacts_modified": False,
            }
            write_json(output / "summary.json", result)
            logger.success("Four-token CUDA smoke passed: {}", output)


if __name__ == "__main__":
    main()

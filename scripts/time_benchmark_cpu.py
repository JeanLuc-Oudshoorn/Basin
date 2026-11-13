"""Supplementary FP32 CPU baselines; RWKV's vendored CUDA kernels are unsupported."""
# ruff: noqa: E402

import itertools
import json
import sys
from pathlib import Path
from time import perf_counter

import numpy as np
import pandas as pd
import torch
from loguru import logger

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
from time_benchmark import (
    FoundationBenchmarks,
    foundation_prediction,
    history,
    load_config,
    run_context,
    write_json,
)


def main():
    options = json.loads((ROOT / "configs/time_benchmark.json").read_text())
    settings = options["cpu"]
    config = load_config()
    torch.set_num_threads(options["cpu_threads"])
    config["holdout"].update(
        benchmark_models=["chronos", "timestral"], foundation_device="cpu"
    )
    output = ROOT / options["output"]
    rows = []
    with run_context(config, "time_benchmark_cpu"):
        suite = FoundationBenchmarks(config, 2048)
        models = ["chronos", "timestral", "timestral_default"]
        for case_index, (length, horizon, covariates) in enumerate(
            itertools.product(
                settings["contexts"], settings["horizons"], settings["covariates"]
            )
        ):
            context = history(length, covariates, options["seed"] + case_index)
            for repeat in range(settings["warmups"] + settings["repeats"]):
                shift = repeat % len(models)
                for name in models[shift:] + models[:shift]:
                    started = perf_counter()
                    result = foundation_prediction(
                        suite, name, context, horizon, options["quantiles"], "fp32"
                    )
                    elapsed = (perf_counter() - started) * 1000
                    if result.shape != (1, horizon, 9) or not np.isfinite(result).all():
                        raise ValueError(f"Invalid CPU forecast: {name}")
                    rows.append(
                        dict(
                            model=name,
                            precision="fp32",
                            context=length,
                            horizon=horizon,
                            covariates=covariates,
                            phase="warmup"
                            if repeat < settings["warmups"]
                            else "scored",
                            repeat=repeat,
                            milliseconds=elapsed,
                        )
                    )
            pd.DataFrame(rows).to_csv(output / "cpu_requests.csv", index=False)
            logger.info(
                "CPU case {} complete: context={} horizon={} covariates={}",
                case_index + 1,
                length,
                horizon,
                covariates,
            )
        frame = pd.DataFrame(rows)
        summary = (
            frame[frame.phase == "scored"]
            .groupby(["model", "context", "horizon", "covariates"])
            .milliseconds.agg(
                count="size",
                median_ms="median",
                mean_ms="mean",
                min_ms="min",
                max_ms="max",
            )
            .reset_index()
        )
        summary.to_csv(output / "cpu_summary.csv", index=False)
        cpu = next(
            (
                line.split(":", 1)[1].strip()
                for line in Path("/proc/cpuinfo").read_text().splitlines()
                if line.startswith("model name")
            ),
            "unknown",
        )
        write_json(
            output / "cpu_environment.json",
            {
                "cpu": cpu,
                "torch": torch.__version__,
                "cpu_threads": torch.get_num_threads(),
                "models": suite.metadata,
                "configuration": settings,
                "rwkv": "not run: requires CUDA/bf16; no alternative kernel substituted",
            },
        )


if __name__ == "__main__":
    main()

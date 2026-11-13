"""Memory-bounded isolated-provider GPU anchors matching the depth confirmation."""
# ruff: noqa: E402

import gc
import json
import os
import resource
import sys
from pathlib import Path

os.environ.setdefault("TORCHINDUCTOR_COMPILE_THREADS", "1")
os.environ.setdefault("MAX_JOBS", "1")
import pandas as pd
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
from time_benchmark import (
    FoundationBenchmarks,
    Meter,
    configure_cuda_memory,
    foundation_prediction,
    history,
    load_config,
    logger,
    run_context,
    write_json,
)


def main():
    settings = json.loads((ROOT / "configs/time_benchmark.json").read_text())
    confirmation = settings["confirmation"]
    config = load_config()
    config["training"]["cuda_memory_limit_gib"] = confirmation["cuda_memory_limit_gib"]
    torch.set_num_threads(settings["cpu_threads"])
    # Match the runtime flags set by the RWKV backend in the depth confirmation.
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.enabled = True
    torch.backends.cudnn.benchmark = True
    configure_cuda_memory(config["training"])
    cases = confirmation["cases"]
    meter = Meter()
    output = ROOT / settings["output"]
    metadata = {}
    with run_context(config, "time_benchmark_baseline_confirmation"):
        for provider in ("chronos", "timestral"):
            config["holdout"]["benchmark_models"] = [provider]
            suite = FoundationBenchmarks(config, 2048)
            suite.models[provider].model.to(dtype=torch.bfloat16)
            metadata[provider] = dict(
                suite.metadata[provider],
                dtype=str(next(suite.models[provider].model.parameters()).dtype),
            )
            names = (
                [provider]
                if provider == "chronos"
                else ["timestral", "timestral_default"]
            )
            for index, (length, horizon, covariates) in enumerate(cases):
                context = history(length, covariates, settings["seed"] + index)
                for name in names:
                    labels = dict(
                        model=name,
                        precision="bf16",
                        context=length,
                        horizon=horizon,
                        covariates=covariates,
                    )
                    for repeat in range(
                        confirmation["warmups"] + confirmation["repeats"]
                    ):
                        meter.call(
                            lambda provider_suite=suite: foundation_prediction(
                                provider_suite,
                                name,
                                context,
                                horizon,
                                settings["quantiles"],
                                "bf16",
                            ),
                            phase="warmup"
                            if repeat < confirmation["warmups"]
                            else "scored",
                            repeat=repeat,
                            **labels,
                        )
                logger.info(
                    "Isolated GPU {} case {}/{} complete",
                    provider,
                    index + 1,
                    len(cases),
                )
            del suite
            gc.collect()
            torch.cuda.empty_cache()
        frame = pd.DataFrame(meter.rows)
        frame.to_csv(output / "baseline_confirmation_requests.csv", index=False)
        result = (
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
        result.to_csv(output / "baseline_confirmation_summary.csv", index=False)
        write_json(
            output / "baseline_confirmation_environment.json",
            {
                "models": metadata,
                "torch": torch.__version__,
                "gpu": torch.cuda.get_device_name(),
                "cpu_threads": torch.get_num_threads(),
                "memory_cap_gib": 4,
                "configuration": confirmation,
                "tf32_allowed": torch.backends.cuda.matmul.allow_tf32,
                "peak_process_rss_gib": resource.getrusage(
                    resource.RUSAGE_SELF
                ).ru_maxrss
                / 2**20,
                "scope": "one provider resident at a time; same five payloads and nine-quantile contract as depth confirmation",
            },
        )


if __name__ == "__main__":
    main()

"""Low-memory, paired depth confirmation; reuse the completed baseline matrix."""
# ruff: noqa: E402

import json
import os
import resource
import sys
from pathlib import Path

import pandas as pd
import torch

os.environ.setdefault("TORCHINDUCTOR_COMPILE_THREADS", "1")
os.environ.setdefault("MAX_JOBS", "1")
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
from time_benchmark import (
    Meter,
    RWKVForecaster,
    configure_cuda_memory,
    configure_variate_execution,
    depth_variant,
    history,
    load_config,
    logger,
    run_context,
    summarize,
    write_json,
)


def main():
    settings = json.loads((ROOT / "configs/time_benchmark.json").read_text())
    confirmation = settings["confirmation"]
    config = load_config()
    output = ROOT / ".cache/time_benchmark_confirmation"
    output.mkdir(parents=True, exist_ok=True)
    torch.set_num_threads(settings["cpu_threads"])
    config["training"]["cuda_memory_limit_gib"] = confirmation["cuda_memory_limit_gib"]
    configure_cuda_memory(config["training"])
    # A few matched anchors suffice to confirm depth scaling; full matrices are saved.
    cases = confirmation["cases"]
    with run_context(config, "time_benchmark_confirmation"):
        full = RWKVForecaster.load(
            ROOT / config["paths"]["training"] / "model.pt"
        ).eval()
        configure_variate_execution(full, config["training"], ROOT)
        reduced = depth_variant(full, settings["reduced_layers"])
        from time_benchmark import rwkv_prediction

        meter = Meter()
        for index, (length, horizon, covariates) in enumerate(cases):
            context = history(length, covariates, settings["seed"] + index)
            models = {"rwkv_447m": full, "rwkv_307m": reduced}
            labels = dict(
                precision="bf16", context=length, horizon=horizon, covariates=covariates
            )
            for name, model in models.items():
                for repeat in range(confirmation["warmups"]):
                    meter.call(
                        lambda: rwkv_prediction(model, context, horizon),
                        model=name,
                        phase="warmup",
                        repeat=repeat,
                        **labels,
                    )
            for repeat in range(confirmation["repeats"]):
                order = list(models) if repeat % 2 == 0 else list(reversed(models))
                for name in order:
                    meter.call(
                        lambda: rwkv_prediction(models[name], context, horizon),
                        model=name,
                        phase="scored",
                        repeat=repeat,
                        **labels,
                    )
            summarize(meter.rows, output)
            logger.info(
                "PAIRED confirmation {}/{} context={} horizon={} covariates={} allocated_GiB={:.2f} peak_RSS_GiB={:.2f}",
                index + 1,
                len(cases),
                length,
                horizon,
                covariates,
                torch.cuda.memory_allocated() / 2**30,
                resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 2**20,
            )
        final = ROOT / settings["output"]
        pd.DataFrame(meter.rows).to_csv(
            final / "depth_confirmation_requests.csv", index=False
        )
        pd.read_csv(output / "summary.csv").to_csv(
            final / "depth_confirmation_summary.csv", index=False
        )
        write_json(
            final / "depth_confirmation_environment.json",
            {
                "gpu": torch.cuda.get_device_name(),
                "torch": torch.__version__,
                "cuda": torch.version.cuda,
                "cpu_threads": torch.get_num_threads(),
                "model_parameters": {
                    k: sum(p.numel() for p in v.parameters()) for k, v in models.items()
                },
                "precision": "bf16",
                "gpu_memory_cap_gib": confirmation["cuda_memory_limit_gib"],
                "configuration": confirmation,
                "compile_threads": 1,
                "peak_process_rss_gib": resource.getrusage(
                    resource.RUSAGE_SELF
                ).ru_maxrss
                / 2**20,
                "peak_allocated_gib": torch.cuda.max_memory_allocated() / 2**30,
                "note": "Both depth views share checkpoint weights. No external models resident. Full external comparisons remain in the completed primary matrix.",
            },
        )


if __name__ == "__main__":
    main()

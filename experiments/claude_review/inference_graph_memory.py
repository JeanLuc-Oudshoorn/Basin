"""How much graph memory does full inference replay need, and what does it buy?

Captures one CUDA graph per planned holdout shape (one request, `variables`
series, `patches` context patches, the full forecast block) on an UNTRAINED
model built from configs/config.json, most frequent shape first, with no
memory limit. Records the retained bytes and the replay and forward-only
latency of every shape. Weights do not affect graph size or latency.

The shape plan comes from the pipeline's own holdout planner applied to table
metadata sampled from the generator (graph_shape_plan.json, made on CPU).

Run inside the WSL CUDA environment with no training job on the GPU:

    source scripts/activate_rwkv7.sh
    python experiments/claude_review/inference_graph_memory.py
"""
# ruff: noqa: E402

import json
import statistics
import sys
from pathlib import Path
from time import perf_counter

import torch

ROOT = Path(__file__).resolve().parents[2]
for directory in (
    ROOT,
    ROOT / "src/03_transforming_input_data",
    ROOT / "src/04_creating_embeddings",
    ROOT / "src/05_model_training",
):
    sys.path.insert(0, str(directory))
from embed import ForecastConfig
from fast_inference import configure_fast_inference
from train import RWKVForecaster
from variate_attention import configure_variate_execution

OPTIONS = {
    "compile_variate": True,
    "compile_variate_inference": True,
    "inference_cuda_graphs": True,
    "inference_graph_cache_size": 256,
    "inference_graph_memory_gib": 64.0,  # Effectively unlimited: measure demand.
}


def timed(engine, request, horizon, repeats):
    seconds = []
    for _ in range(repeats):
        torch.cuda.synchronize()
        start = perf_counter()
        engine.forecast(request["context"], horizon, past_covariates=request["past"])
        torch.cuda.synchronize()
        seconds.append(perf_counter() - start)
    return statistics.median(seconds) * 1000


def main():
    here = Path(__file__).resolve().parent
    config = json.loads((ROOT / "configs/config.json").read_text())
    plan = json.loads((here / "graph_shape_plan.json").read_text())
    settings = dict(config["model"], quantiles=tuple(config["model"]["quantiles"]))
    torch.manual_seed(config["seed"])
    model = (
        RWKVForecaster(ForecastConfig(**settings))
        .to(device="cuda", dtype=torch.bfloat16)
        .eval()
    )
    configure_variate_execution(model, OPTIONS, ROOT)
    configure_fast_inference(model, OPTIONS)
    engine, graphs = model.fast_inference, model.fast_inference.graphs
    patch = model.config.patch_size
    horizon = model.config.output_patches * patch
    generator = torch.Generator().manual_seed(3)
    rows, total = [], sum(p["requests"] for p in plan["profiles"])
    covered = 0
    for profile in plan["profiles"]:
        length = profile["patches"] * patch
        covariates = profile["variables"] - 1
        request = {
            "context": torch.randn(1, 1, length, generator=generator),
            "past": torch.randn(1, covariates, length, generator=generator)
            if covariates
            else None,
        }
        before, captures = graphs.retained_bytes, graphs.captures
        engine.forecast(request["context"], horizon, past_covariates=request["past"])
        if graphs.captures != captures + 1:
            raise SystemExit(f"shape not captured: {profile} {graphs.summary()}")
        replay = timed(engine, request, horizon, 7)
        graphs.enabled = False
        forward = timed(engine, request, horizon, 5)
        graphs.enabled = True
        covered += profile["requests"]
        rows.append(
            {
                **profile,
                "graph_mib": (graphs.retained_bytes - before) / 2**20,
                "cumulative_gib": graphs.retained_bytes / 2**30,
                "cumulative_request_share": covered / total,
                "replay_ms": replay,
                "forward_only_ms": forward,
            }
        )
        print(json.dumps(rows[-1]), flush=True)
    result = {
        "parameters": sum(p.numel() for p in model.parameters()),
        "forecast_steps": horizon,
        "gpu": torch.cuda.get_device_name(0),
        "peak_allocated_gib": torch.cuda.max_memory_allocated() / 2**30,
        "reserved_gib": torch.cuda.memory_reserved() / 2**30,
        "total_graph_gib": graphs.retained_bytes / 2**30,
        "profiles": rows,
    }
    (here / "inference_graph_memory_output.json").write_text(
        json.dumps(result, indent=1)
    )
    weights = [r["requests"] for r in rows]
    for name in ("replay_ms", "forward_only_ms"):
        mean = sum(r[name] * w for r, w in zip(rows, weights)) / sum(weights)
        print(f"request-weighted mean {name}: {mean:.2f}")
    print(
        f"all {len(rows)} shapes: {result['total_graph_gib']:.2f} GiB of graphs, "
        f"peak allocated {result['peak_allocated_gib']:.2f} GiB"
    )


if __name__ == "__main__":
    main()

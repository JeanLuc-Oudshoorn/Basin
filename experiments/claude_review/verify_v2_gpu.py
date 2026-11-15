"""GPU checks that the unit tests cannot make, because they need the real RWKV kernels.

1. A hybrid checkpoint loaded with RWKVForecaster.load forecasts exactly like a
   normally constructed model with the same weights (derived constants survive
   the empty-model load path).
2. The CUDA-graph fast inference path agrees with the plain forward of the
   hybrid model (two independent implementations of the block loop).
3. The archived 500M checkpoint still loads and forecasts with the current code,
   and agrees with its pure-PyTorch port, which was verified against the model
   as it was evaluated.

Run inside the WSL CUDA environment with no training job on the GPU:

    source scripts/activate_rwkv7.sh
    python experiments/claude_review/verify_v2_gpu.py --hybrid results/smoke_v2/training/model.pt
"""
# ruff: noqa: E402

import argparse
import json
import sys
from pathlib import Path

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
from inference import forecast_contexts
from train import RWKVForecaster
from variate_attention import configure_variate_execution

OPTIONS = {
    "compile_variate": True,
    "compile_variate_inference": True,
    "inference_cuda_graphs": True,
}


def contexts(seed=11):
    generator = torch.Generator().manual_seed(seed)
    result = []
    for length, covariates in ((2048, 2), (700, 0), (333, 1), (96, 3), (1500, 0)):
        time = torch.arange(length).float()
        series = (
            torch.sin(time / 9.0) * 3
            + 0.004 * time
            + torch.randn(length, generator=generator) * 0.4
            + 50
        )
        result.append(
            {
                "context": series[None, None],
                "past_covariates": torch.randn(
                    1, covariates, length, generator=generator
                )
                if covariates
                else None,
            }
        )
    return result


def forecasts(model, requests, patches):
    return [
        q.float()
        for q in forecast_contexts(model, requests, 1, prediction_patches=patches)
    ]


def spread(a, b, requests):
    """Largest forecast difference as a share of each context's standard deviation."""
    return max(
        float((x - y).abs().max() / r["context"].std())
        for x, y, r in zip(a, b, requests)
    )


def typical(a, b, requests):
    """Mean forecast difference as a share of each context's standard deviation."""
    shares = [
        float((x - y).abs().mean() / r["context"].std())
        for x, y, r in zip(a, b, requests)
    ]
    return sum(shares) / len(shares)


def check_hybrid(path):
    requests = contexts()
    loaded = RWKVForecaster.load(path).eval()
    patches = loaded.config.output_patches
    if not loaded.hybrid:
        raise SystemExit("--hybrid must point at a model with temporal attention")
    plain = forecasts(loaded, requests, patches)

    saved = torch.load(path, map_location="cpu", weights_only=True, mmap=True)
    built = RWKVForecaster(ForecastConfig(**saved["config"])).to(
        device="cuda", dtype=torch.bfloat16
    )
    built.load_state_dict(saved["state_dict"])
    constructed = forecasts(built.eval(), requests, patches)
    del built

    configure_variate_execution(loaded, OPTIONS, ROOT)
    configure_fast_inference(loaded, OPTIONS)
    forecasts(loaded, requests, patches)  # capture
    replayed = forecasts(loaded, requests, patches)
    coverage = loaded.fast_inference.graphs.summary()["replay_coverage"]
    result = {
        "checkpoint": str(path),
        "parameters": sum(p.numel() for p in loaded.parameters()),
        "forecast_steps": patches * loaded.config.patch_size,
        "loaded_vs_constructed_max_share_of_std": spread(plain, constructed, requests),
        "fast_path_vs_plain_max_share_of_std": spread(replayed, plain, requests),
        "fast_path_vs_plain_mean_share_of_std": typical(replayed, plain, requests),
        "fast_path_replay_coverage": coverage,
        "finite": all(bool(torch.isfinite(q).all()) for q in plain + replayed),
    }
    del loaded
    torch.cuda.empty_cache()
    return result


def check_archive():
    training = ROOT / "results/500m_model/training"
    port = ROOT / "results/500m_model/portable/basin-rwkv7-pytorch"
    payload = torch.load(
        training / "latest_checkpoint.pt",
        map_location="cpu",
        weights_only=True,
        mmap=True,
    )
    model = RWKVForecaster.load(
        None,
        checkpoint={
            "config": payload["identity"]["model"],
            "state_dict": payload["best_state_dict"],
        },
    ).eval()
    del payload
    requests = contexts()
    original = forecasts(model, requests, model.config.output_patches)
    parameters = sum(p.numel() for p in model.parameters())
    # The same two-path comparison as for the hybrid model, as a yardstick for
    # how far BF16 rounding alone separates the fast path from the plain one.
    configure_variate_execution(model, OPTIONS, ROOT)
    configure_fast_inference(model, OPTIONS)
    forecasts(model, requests, model.config.output_patches)
    replayed = forecasts(model, requests, model.config.output_patches)
    del model
    torch.cuda.empty_cache()

    sys.path.insert(0, str(port))
    from modeling_basin_pytorch import BasinPyTorch

    ported = BasinPyTorch.from_pretrained(str(port), device="cuda")
    outputs = []
    for request in requests:
        covariates = request["past_covariates"]
        forecast = ported.forecast(
            request["context"][0],
            horizon=64,
            past_covariates=None if covariates is None else covariates[0],
        )
        outputs.append(forecast.quantiles.float().cpu())
    return {
        "parameters": parameters,
        "current_code_vs_pure_pytorch_port_max_share_of_std": spread(
            original, outputs, requests
        ),
        "current_code_vs_pure_pytorch_port_mean_share_of_std": typical(
            original, outputs, requests
        ),
        "fast_path_vs_plain_max_share_of_std": spread(replayed, original, requests),
        "fast_path_vs_plain_mean_share_of_std": typical(replayed, original, requests),
        "finite": all(bool(torch.isfinite(q).all()) for q in original),
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--hybrid", default=str(ROOT / "results/smoke_v2/training/model.pt")
    )
    parser.add_argument("--skip-archive", action="store_true")
    parser.add_argument("--output", default=None)
    args = parser.parse_args()
    report = {"hybrid": check_hybrid(Path(args.hybrid))}
    if not args.skip_archive:
        report["archived_500m"] = check_archive()
    print(json.dumps(report, indent=2))
    if args.output:
        Path(args.output).write_text(json.dumps(report, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()

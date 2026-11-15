"""Training throughput and memory of candidate model shapes on synthetic batches.

Measures one optimizer update (forward, backward, clipping, FP32-master AdamW)
for full-length tables, which is the expensive case: 128 input patches each.
Run inside the WSL CUDA environment, with no training job on the GPU:

    source scripts/activate_rwkv7.sh
    python experiments/claude_review/architecture_throughput.py

Data is random, so the losses mean nothing; only time and memory are reported.
"""
# ruff: noqa: E402

import argparse
import gc
import json
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
from performance import make_optimizer
from token_inventory import parameter_budget
from train import RWKVForecaster
from variate_attention import configure_variate_execution

QUANTILES = (0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9)
V2 = dict(
    output_patches=7,
    horizon_loss_weights="equal",
    head="direct",
    gated_mixing=True,
    value_transform="asinh",
    fp32_io=True,
    rwkv_init="constructor",  # initial values do not change speed; skips slow QR
)
SHAPES = {
    "500M archived (1024x28, mix every block)": dict(
        width=1024,
        layers=28,
        variate_heads=16,
        output_patches=4,
        horizon_loss_weights=(0.8, 0.16, 0.03, 0.01),
    ),
    "768x24, mix every 4": dict(
        V2, width=768, layers=24, variate_heads=12, variate_every=4, temporal_every=4
    ),
    "768x24, mix every 4, no temporal attention": dict(
        V2, width=768, layers=24, variate_heads=12, variate_every=4, temporal_every=0
    ),
    "832x20, mix every 4": dict(
        V2, width=832, layers=20, variate_heads=13, variate_every=4, temporal_every=4
    ),
    "896x17, mix every 4": dict(
        V2, width=896, layers=17, variate_heads=14, variate_every=4, temporal_every=4
    ),
    "1024x13, mix every 4": dict(
        V2, width=1024, layers=13, variate_heads=16, variate_every=4, temporal_every=4
    ),
}


def measure(name, settings, tables, variables, *, updates, checkpointing, threshold):
    torch.manual_seed(7)
    config = ForecastConfig(
        patch_size=16,
        max_patches=128,
        quantiles=QUANTILES,
        objective="next_patch",
        context_length=2048,
        gradient_checkpointing=checkpointing,
        **settings,
    )
    model = RWKVForecaster(config).to(device="cuda", dtype=torch.bfloat16).train()
    model.checkpoint_token_threshold = threshold
    training = {
        "learning_rate": 1e-4,
        "weight_decay": 0.01,
        "compile_variate": True,
        "compile_variate_inference": True,
        "optimizer": {"parameter_groups": "rwkv7"}
        if settings.get("head") == "direct"
        else {},
    }
    configure_variate_execution(model, training, ROOT)
    optimizer = make_optimizer(model, training)
    parameters = sum(p.numel() for p in model.parameters())
    context = torch.randn(tables, 1, 2064).cumsum(-1)
    covariates = torch.randn(tables, variables - 1, 2064) if variables > 1 else None
    prepared = model.tokenize_next(context, past_covariates=covariates, training=True)
    prepared = {
        k: v.cuda() if isinstance(v, torch.Tensor) else v for k, v in prepared.items()
    }
    prepared["observed_points"] = int(torch.isfinite(prepared["labels"]).sum())

    def update():
        optimizer.zero_grad(set_to_none=True)
        output = model.forward_encoded(model.encode_tokens(prepared), prepared)
        output.loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        return output.loss

    result = {
        "shape": name,
        "parameters_m": round(parameters / 1e6, 1),
        "tables": tables,
        "variables": variables,
        "gradient_checkpointing": checkpointing,
    }
    try:
        for _ in range(4):  # compilation, allocator and cuDNN warm-up
            update()
        torch.cuda.synchronize()
        torch.cuda.reset_peak_memory_stats()
        started = perf_counter()
        for _ in range(updates):
            loss = update()
        float(loss)
        torch.cuda.synchronize()
        seconds = (perf_counter() - started) / updates
        result.update(
            ms_per_update=round(1000 * seconds, 1),
            tables_per_second=round(tables / seconds, 1),
            peak_gib=round(torch.cuda.max_memory_allocated() / 2**30, 2),
        )
    except torch.OutOfMemoryError:
        result["error"] = "out of memory"
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--updates", type=int, default=12)
    parser.add_argument("--output", default=None)
    parser.add_argument("--only", default=None, help="substring of the shape name")
    parser.add_argument("--tables", type=int, nargs="+", default=[8])
    parser.add_argument("--variables", type=int, nargs="+", default=[1, 3])
    parser.add_argument("--no-checkpointing", action="store_true")
    parser.add_argument("--threshold", type=int, default=10**9)
    args = parser.parse_args()
    torch.cuda.set_per_process_memory_fraction(
        10.5 * 2**30 / torch.cuda.get_device_properties(0).total_memory
    )
    rows = []
    for name, settings in SHAPES.items():
        if args.only and args.only not in name:
            continue
        budget = parameter_budget(
            {"model": dict(settings, patch_size=16, quantiles=QUANTILES)}
        )
        print(f"\n{name}: {budget['parameters'] / 1e6:.1f}M parameters (analytic)")
        for tables in args.tables:
            for variables in args.variables:
                row = measure(
                    name,
                    settings,
                    tables,
                    variables,
                    updates=args.updates,
                    checkpointing=not args.no_checkpointing,
                    threshold=args.threshold,
                )
                rows.append(row)
                print(json.dumps(row), flush=True)
                # The measured model is out of scope: return its memory.
                gc.collect()
                torch.cuda.empty_cache()
    if args.output:
        Path(args.output).write_text(json.dumps(rows, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()

"""Bounded CUDA throughput benchmark on real training windows, isolated from runs."""

# ruff: noqa: E402
import argparse
import gc
import json
import os
import random
import sys
from dataclasses import replace
from datetime import datetime, timezone
from pathlib import Path
from time import perf_counter

import torch
from loguru import logger

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src/05_model_training"))
sys.path.insert(0, str(ROOT / "src/04_creating_embeddings"))
sys.path.insert(0, str(ROOT / "src/03_transforming_input_data"))
from batching import make_batches
from embed import LazyPatchRecord, PatchEncoder
from epoch_training import accumulated_update
from performance import configure_cuda_memory, prepare_update
from train import ForecastConfig, RWKVForecaster
from training_log import emit_training_event, training_run
from training_protocol import split_datasets
from transform import load_config, location, read_artifact, write_json


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--case", default="baseline")
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--accumulation", type=int, default=8)
    parser.add_argument("--updates", type=int, default=6)
    parser.add_argument("--no-checkpoint", action="store_true")
    parser.add_argument("--fused", action="store_true")
    parser.add_argument("--profile", action="store_true")
    parser.add_argument("--worst-case", action="store_true")
    parser.add_argument("--bucket", action="store_true")
    parser.add_argument("--validation", action="store_true")
    parser.add_argument("--checkpoint-threshold", type=int, default=0)
    parser.add_argument("--sweep", action="store_true")
    parser.add_argument("--sample", help="Reuse an existing bounded sample cache")
    parser.add_argument("--direct-attention", action="store_true")
    parser.add_argument("--reentrant-checkpoint", action="store_true")
    parser.add_argument(
        "--attention-backend", choices=("default", "math", "cudnn"), default="default"
    )
    parser.add_argument("--compile-attention", action="store_true")
    parser.add_argument("--skip-mix-checkpoint", action="store_true")
    parser.add_argument("--warmup-sweep", action="store_true")
    parser.add_argument(
        "--blas", choices=("default", "cublas", "cublaslt"), default="default"
    )
    parser.add_argument("--allocator-config")
    parser.add_argument("--production-compile", action="store_true")
    parser.add_argument("--microbatch-size", type=int)
    parser.add_argument("--microbatch-token-limit", type=int)
    parser.add_argument("--memory-limit-gib", type=float)
    parser.add_argument("--output-dir", default="docs/audits/2026-10-28/training_speed")
    args = parser.parse_args()
    for name in ("microbatch_size", "microbatch_token_limit", "memory_limit_gib"):
        value = getattr(args, name)
        if value is not None and not (0 < value < float("inf")):
            parser.error(f"{name} must be positive and finite")
    if args.allocator_config:
        os.environ["PYTORCH_ALLOC_CONF"] = args.allocator_config
    if (
        min(args.batch_size, args.accumulation, args.updates) < 1
        or args.checkpoint_threshold < 0
    ):
        parser.error(
            "batch-size, accumulation and updates must be positive; threshold must be nonnegative"
        )
    config = load_config()
    with training_run(config, "throughput_benchmark"):
        run_benchmark(args, config)


def run_benchmark(args, config):
    torch.set_num_threads(config["training"].get("cpu_threads", 4))
    if args.memory_limit_gib is not None:
        configure_cuda_memory({"cuda_memory_limit_gib": args.memory_limit_gib})
    if args.blas != "default":
        torch.backends.cuda.preferred_blas_library(args.blas)
    output = ROOT / args.output_dir
    output.mkdir(parents=True, exist_ok=True)
    logger.remove()
    logger.add(sys.stderr, level="INFO")
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S_%fZ")
    logger.add(ROOT / "logs" / f"throughput_{args.case}_{stamp}.log", level="DEBUG")
    sample_path = ROOT / args.sample if args.sample else output / "sample.pt"
    if not sample_path.exists():
        data = read_artifact(location(config, "embedded") / "embeddings.pt", config)
        training, validation, _, _ = split_datasets(data["records"], config)
        rng = random.Random(config["seed"])  # nosec B311
        sample = rng.sample(training, min(512, len(training)))
        worst = max(
            training,
            key=lambda r: (
                r["batch"]["context"].shape[-1]
                * (
                    r["batch"]["context"].shape[1]
                    + (
                        r["batch"]["past_covariates"].shape[1]
                        if r["batch"].get("past_covariates") is not None
                        else 0
                    )
                )
            ),
        )

        # Clone sampled tensors so saving does not retain the corpus storage.
        def materialize(r):
            return {
                **dict(r),
                "batch": {
                    k: v.clone() if isinstance(v, torch.Tensor) else v
                    for k, v in r["batch"].items()
                },
            }

        torch.save(
            {
                "train": [materialize(r) for r in sample],
                "validation": [materialize(r) for r in rng.sample(validation, 128)],
                "worst": materialize(worst),
            },
            sample_path,
        )
        del data, training, validation, sample, worst
        gc.collect()
    raw = torch.load(sample_path, weights_only=False, map_location="cpu")  # nosec B614
    settings = ForecastConfig(
        **dict(config["model"], quantiles=tuple(config["model"]["quantiles"]))
    )
    settings = replace(settings, gradient_checkpointing=not args.no_checkpoint)
    encoder = PatchEncoder(settings)
    sample = [LazyPatchRecord(r, encoder) for r in raw["train"]]
    if args.worst_case:
        sample = [LazyPatchRecord(raw["worst"], encoder) for _ in range(64)]
    batches = make_batches(
        sample, args.batch_size, seed=config["seed"], bucket_by_length=args.bucket
    )
    started = perf_counter()
    torch.manual_seed(config["seed"])
    model = RWKVForecaster(settings).to(device="cuda", dtype=torch.bfloat16).train()
    if args.direct_attention:
        from types import MethodType

        def direct_attention(self, x):
            b, v, t, d = x.shape
            flat = x.permute(0, 2, 1, 3).reshape(b * t, v, d)
            attention = self.attention
            qkv = torch.nn.functional.linear(
                self.norm(flat), attention.in_proj_weight, attention.in_proj_bias
            ).view(b * t, v, 3, attention.num_heads, d // attention.num_heads)
            q, k, value = qkv.permute(2, 0, 3, 1, 4).unbind(0)
            if args.attention_backend == "default":
                mixed = torch.nn.functional.scaled_dot_product_attention(q, k, value)
            else:
                from torch.nn.attention import SDPBackend, sdpa_kernel

                backend = (
                    SDPBackend.MATH
                    if args.attention_backend == "math"
                    else SDPBackend.CUDNN_ATTENTION
                )
                with sdpa_kernel(backend):
                    mixed = torch.nn.functional.scaled_dot_product_attention(
                        q, k, value
                    )
            mixed = mixed.transpose(1, 2).reshape(b * t, v, d)
            mixed = attention.out_proj(mixed)
            return (flat + mixed).reshape(b, t, v, d).permute(0, 2, 1, 3)

        for mix in model.variate:
            mix.forward = MethodType(direct_attention, mix)
    if args.compile_attention:
        os.environ.setdefault(
            "TORCHINDUCTOR_CACHE_DIR", str(ROOT / ".cache/torchinductor")
        )
        os.environ.setdefault("TRITON_CACHE_DIR", str(ROOT / ".cache/triton"))
        # Share the compiled function across identically shaped layer modules.
        compiled = torch.compile(type(model.variate[0]).forward, dynamic=True)
        from types import MethodType

        if args.direct_attention:
            compiled = torch.compile(direct_attention, dynamic=True)
        for mix in model.variate:
            mix.forward = MethodType(compiled, mix)
    if args.reentrant_checkpoint:
        import train
        from torch.utils.checkpoint import checkpoint

        def reentrant(function, *inputs, **kwargs):
            return checkpoint(function, *inputs, use_reentrant=True)

        train.checkpoint = reentrant
    if args.skip_mix_checkpoint:
        import train
        from torch.utils.checkpoint import checkpoint

        def block_only(function, *inputs, **kwargs):
            if isinstance(function, train.VariateMix):
                return function(*inputs)
            return checkpoint(function, *inputs, **kwargs)

        train.checkpoint = block_only
    model.checkpoint_token_threshold = args.checkpoint_threshold
    if args.production_compile:
        from variate_attention import configure_variate_execution

        configure_variate_execution(model, {"compile_variate": True}, ROOT)
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=1e-4, foreach=False, fused=args.fused
    )
    initialization = perf_counter() - started
    rows = []
    try:
        groups = [
            batches[i : i + args.accumulation]
            for i in range(0, len(batches), args.accumulation)
        ]
        warmups = groups if args.warmup_sweep else [groups[0], groups[0]]
        schedule = warmups + (
            groups
            if args.sweep
            else [groups[i % len(groups)] for i in range(args.updates)]
        )
        for step, group in enumerate(schedule):
            torch.cuda.synchronize()
            torch.cuda.reset_peak_memory_stats()
            started = perf_counter()
            prepared = prepare_update(
                group,
                microbatch_size=args.microbatch_size,
                microbatch_token_limit=args.microbatch_token_limit,
            )
            result = accumulated_update(model, group, optimizer, 1.0, prepared)
            torch.cuda.synchronize()
            elapsed = perf_counter() - started
            row = {
                **result,
                "seconds": elapsed,
                "peak_gib": torch.cuda.max_memory_allocated() / 2**30,
                "reserved_gib": torch.cuda.max_memory_reserved() / 2**30,
                "windows_per_second": result["windows"] / elapsed,
                "warmup": step < len(warmups),
            }
            rows.append(row)
            emit_training_event(
                "metrics",
                global_step=step + 1,
                train_loss=result["loss"],
                benchmark_case=args.case,
                benchmark_warmup=step < len(warmups),
                windows_per_second=row["windows_per_second"],
            )
            logger.info(
                "{} step={} windows/s={:.2f} time={:.3f}s peak={:.2f}GiB",
                args.case,
                step,
                row["windows_per_second"],
                elapsed,
                row["peak_gib"],
            )
        if args.profile:
            with torch.profiler.profile(
                activities=[
                    torch.profiler.ProfilerActivity.CPU,
                    torch.profiler.ProfilerActivity.CUDA,
                ]
            ) as prof:
                prepared = prepare_update(
                    group,
                    microbatch_size=args.microbatch_size,
                    microbatch_token_limit=args.microbatch_token_limit,
                )
                accumulated_update(model, group, optimizer, 1.0, prepared)
                torch.cuda.synchronize()
            (output / f"{args.case}_profile.txt").write_text(
                prof.key_averages().table(sort_by="self_cuda_time_total", row_limit=40)
                + "\n"
                + prof.key_averages().table(sort_by="self_cpu_time_total", row_limit=40)
            )
        if args.validation:
            from training_protocol import validate

            validation_records = [
                LazyPatchRecord(r, encoder) for r in raw["validation"]
            ]
            validation_results = {}
            for size in (1, 8, 16):
                torch.cuda.synchronize()
                validation_results[str(size)] = validate(
                    model, validation_records, batch_size=size
                )
                torch.cuda.synchronize()
            write_json(output / f"{args.case}_validation.json", validation_results)
        measured = rows[len(warmups) :]
        report = {
            "status": "ok",
            "checkpoint_policy": "proportional_layer_budget",
            "arguments": vars(args),
            "gpu": torch.cuda.get_device_name(),
            "parameters": sum(p.numel() for p in model.parameters()),
            "initialization_seconds": initialization,
            "updates": rows,
            "windows_per_second": sum(r["windows"] for r in measured)
            / sum(r["seconds"] for r in measured),
            "peak_gib": max(r["peak_gib"] for r in measured),
        }
    except torch.cuda.OutOfMemoryError as exc:
        report = {
            "status": "out_of_memory",
            "arguments": vars(args),
            "error": str(exc),
            "updates": rows,
        }
    write_json(output / f"{args.case}.json", report)
    print(json.dumps({k: v for k, v in report.items() if k != "updates"}, indent=2))


if __name__ == "__main__":
    main()

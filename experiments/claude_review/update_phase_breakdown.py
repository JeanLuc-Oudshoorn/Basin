"""Where one optimizer update spends its time, on synthetic batches.

Builds the model, optimizer and weight average exactly as `train.py` does for
the given config, then measures two things per batch shape:

* `end_to_end`: the trainer's own `accumulated_update` plus the weight-average
  update, timed as training runs it. This is the number to compare before and
  after a change.
* `phases`: the same update with a GPU synchronisation after every phase
  (forward, backward, clipping, optimizer, weight average). Synchronising adds
  a little time, so the phases sum to slightly more than `end_to_end`.

Run inside the WSL CUDA environment, with no training job on the GPU:

    source scripts/activate_rwkv7.sh
    python experiments/claude_review/update_phase_breakdown.py --label baseline

Data is random, so the losses mean nothing; only time and memory are reported.
The allocator is capped as in training (`training.cuda_memory_limit_gib`), so a
shape that does not fit fails with an out-of-memory error instead of spilling
into shared GPU memory.
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
    ROOT / "src/04_patch_indexing",
    ROOT / "src/05_model_training",
):
    sys.path.insert(0, str(directory))
import epoch_training
from gradient_clipping import measure_gradients, scale_gradients
from index_patches import ForecastConfig
from loguru import logger
from performance import configure_cuda_memory, make_optimizer, prepare_update
from train import RWKVForecaster
from transform import load_config
from variate_attention import configure_variate_execution
from weight_average import WeightAverage, weight_average_options

# tables x variables x input patches. 8x4x128 and 8x8x128 exceed the default
# 3,072-token pass limit and are therefore split into several passes.
DEFAULT_SHAPES = (
    "8x1x16",
    "8x1x64",
    "8x1x128",
    "8x3x64",
    "8x3x128",
    "8x4x128",
    "8x8x128",
)


def build(config, *, init):
    training = config["training"]
    configure_cuda_memory(training)
    torch.set_num_threads(training.get("cpu_threads", 4))
    torch.manual_seed(config["seed"])
    settings = ForecastConfig(
        **dict(
            config["model"],
            quantiles=tuple(config["model"]["quantiles"]),
            rwkv_init=init,
        )
    )
    model = RWKVForecaster(settings).to(device="cuda", dtype=torch.bfloat16).train()
    model.checkpoint_token_threshold = training.get("checkpoint_token_threshold", 0)
    model.auto_microbatch = False  # a shape that does not fit must fail, not shrink
    configure_variate_execution(model, training, ROOT)
    optimizer = make_optimizer(model, training)
    options = weight_average_options(training)
    average = WeightAverage(model, optimizer, **options) if options else None
    return model, optimizer, average


def synthetic_update(model, tables, variables, patches, training):
    """One logical batch, prepared and split exactly as the trainer does."""
    length = (patches + 1) * model.config.patch_size  # inputs plus one label patch
    records = []
    for index in range(tables):
        context = torch.randn(1, 1, length).cumsum(-1)
        covariates = torch.randn(1, variables - 1, length) if variables > 1 else None
        with torch.no_grad():
            prepared = model.tokenize_next(
                context, past_covariates=covariates, training=True
            )
        records.append(
            {"prepared": prepared, "split": "train", "id": index, "source": index}
        )
    group = [records]
    batches = prepare_update(
        group,
        next_patch=True,
        pin_memory=True,
        microbatch_size=training.get("microbatch_size", 4),
        microbatch_token_limit=training.get("microbatch_token_limit", 1536),
    )
    return group, batches


def clip_and_metrics(model, gradient_clip, losses):
    """Clipping plus the metrics transfer, as `_accumulated_gradients` does it."""
    norm, coefficient, groups = measure_gradients(model.parameters(), gradient_clip)
    values = torch.stack([norm.float(), coefficient.float(), *losses]).cpu().tolist()
    if values[1] != 1.0:
        scale_gradients(groups, coefficient)


def phase_update(model, optimizer, average, batches, gradient_clip, seconds):
    """The trainer's update with a synchronisation after each phase."""

    def lap(name, started):
        torch.cuda.synchronize()
        now = perf_counter()
        seconds[name] = seconds.get(name, 0.0) + now - started
        return now

    mark = perf_counter()
    optimizer.zero_grad(set_to_none=True)
    total = sum(points for _, _, points in batches)
    losses = []
    for prepared, _, points in batches:
        prepared = {
            k: v.to("cuda", non_blocking=True) if isinstance(v, torch.Tensor) else v
            for k, v in prepared.items()
        }
        output = model.forward_encoded(
            model.encode_tokens(prepared), prepared, prepared["labels"]
        )
        mark = lap("forward", mark)
        (output.loss * (points / total)).backward()
        losses.append(output.loss.detach().float())
        del output
        mark = lap("backward", mark)
    clip_and_metrics(model, gradient_clip, losses)
    mark = lap("clip_and_metrics", mark)
    optimizer.step()
    mark = lap("optimizer", mark)
    if average is not None:
        average.update()
        lap("weight_average", mark)


def measure(model, optimizer, average, shape, training, *, updates, warmup):
    tables, variables, patches = (int(part) for part in shape.split("x"))
    group, batches = synthetic_update(model, tables, variables, patches, training)
    gradient_clip = training["gradient_clip"]
    row = {
        "shape": shape,
        "tokens": tables * variables * batches[0][0]["padded_tokens"],
        "passes": len(batches),
    }

    def update():
        result = epoch_training.accumulated_update(
            model, group, optimizer, gradient_clip, batches
        )
        if average is not None:
            average.update()
        return result

    try:
        for _ in range(warmup):  # compilation, allocator and cuDNN warm-up
            update()
        torch.cuda.synchronize()
        torch.cuda.reset_peak_memory_stats()
        started = perf_counter()
        for _ in range(updates):
            update()
        torch.cuda.synchronize()
        row["end_to_end_ms"] = round(1000 * (perf_counter() - started) / updates, 2)
        row["peak_allocated_gib"] = round(torch.cuda.max_memory_allocated() / 2**30, 2)
        row["reserved_gib"] = round(torch.cuda.memory_reserved() / 2**30, 2)
        seconds = {}
        for _ in range(updates):
            phase_update(model, optimizer, average, batches, gradient_clip, seconds)
        row["phases_ms"] = {
            name: round(1000 * value / updates, 2) for name, value in seconds.items()
        }
    except torch.OutOfMemoryError:
        row["error"] = "out of memory under the configured allocator limit"
    optimizer.zero_grad(set_to_none=True)
    gc.collect()
    torch.cuda.empty_cache()
    return row


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/config.json")
    parser.add_argument("--shapes", nargs="+", default=list(DEFAULT_SHAPES))
    parser.add_argument("--updates", type=int, default=32)
    parser.add_argument("--warmup", type=int, default=8)
    parser.add_argument("--repeats", type=int, default=1)
    parser.add_argument("--label", default="run")
    parser.add_argument("--output", default=None)
    parser.add_argument(
        "--init",
        default="constructor",
        help="rwkv_init; 'constructor' skips the slow upstream initialisation, "
        "which does not change speed",
    )
    parser.add_argument(
        "--set",
        nargs="*",
        default=[],
        metavar="KEY=JSON",
        help="override training options, e.g. microbatch_token_limit=4096",
    )
    args = parser.parse_args()
    config = load_config(args.config)
    # The trainer's sinks: INFO to the console, DEBUG to a file on the project
    # drive. The file sink's cost is part of what a real update pays.
    logger.remove()
    logger.add(sys.stderr, level="INFO")
    log_directory = ROOT / ".cache" / "benchmark_logs"
    log_directory.mkdir(parents=True, exist_ok=True)
    logger.add(
        str(log_directory / "events.log"),
        level="DEBUG",
        rotation="10 MB",
        retention=2,
        diagnose=False,
    )
    for item in args.set:
        key, _, value = item.partition("=")
        config["training"][key] = json.loads(value)
    training = config["training"]
    model, optimizer, average = build(config, init=args.init)
    print(
        f"{args.label}: {sum(p.numel() for p in model.parameters()) / 1e6:.1f}M "
        f"parameters, optimizer={type(optimizer).__name__}, "
        f"pass limit={training.get('microbatch_token_limit')} tokens, "
        f"checkpoint threshold={model.checkpoint_token_threshold}",
        flush=True,
    )
    rows = []
    for repeat in range(args.repeats):
        for shape in args.shapes:
            row = dict(
                measure(
                    model,
                    optimizer,
                    average,
                    shape,
                    training,
                    updates=args.updates,
                    warmup=args.warmup,
                ),
                label=args.label,
                repeat=repeat,
            )
            rows.append(row)
            print(json.dumps(row), flush=True)
    if args.output:
        Path(args.output).write_text(json.dumps(rows, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()

"""Training speed on the corpus's own mix of batch shapes.

`update_phase_breakdown.py` times one shape at a time. A training run sees all
of them, in proportions that decide what an optimisation is worth. This script
weights every shape (variables x padded patches, eight tables per update) by
how often it occurs in a sample of the patch index, and visits the shapes in a
shuffled order, as training does, so that the memory allocator sees the same
kind of traffic.

    source scripts/activate_rwkv7.sh
    python experiments/claude_review/shape_mix_benchmark.py --label current
    python experiments/claude_review/shape_mix_benchmark.py --label earlier --earlier-update
    python experiments/claude_review/shape_mix_benchmark.py --label limit_4096 \
        --set microbatch_token_limit=4096

`--earlier-update` restores the three-pass optimizer update,
`torch.nn.utils.clip_grad_norm_` and the training thread's per-update DEBUG log
lines, for a before/after comparison.

Reported: the mean time per update weighted by shape frequency, tables per
second, the largest allocated and reserved VRAM, and how many shapes ran out
of memory under the training allocator limit (none should).
"""
# ruff: noqa: E402

import argparse
import gc
import json
import random
import sys
from pathlib import Path
from time import perf_counter

import torch

ROOT = Path(__file__).resolve().parents[2]
for directory in (
    Path(__file__).resolve().parent,
    ROOT,
    ROOT / "src/03_transforming_input_data",
    ROOT / "src/04_patch_indexing",
    ROOT / "src/05_model_training",
):
    sys.path.insert(0, str(directory))
import epoch_training
import update_phase_breakdown as benchmark
from loguru import logger

DEFAULT_SAMPLE = ROOT / "experiments/claude_review/speedup_2026-11-17/shape_sample.json"


def shape_weights(path, max_variables, tables):
    counts = json.loads(Path(path).read_text(encoding="utf-8"))
    total = sum(counts.values())
    weights = {}
    for name, count in counts.items():
        variables, patches = (int(part) for part in name.split("x"))
        if variables <= max_variables:
            weights[f"{tables}x{variables}x{patches}"] = count / total
    return weights, 1 - sum(weights.values())


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/config.json")
    parser.add_argument("--sample", default=str(DEFAULT_SAMPLE))
    parser.add_argument("--label", default="run")
    parser.add_argument("--output", default=None)
    parser.add_argument("--rounds", type=int, default=3)
    parser.add_argument("--updates-per-visit", type=int, default=4)
    parser.add_argument("--max-variables", type=int, default=8)
    parser.add_argument("--earlier-update", action="store_true")
    parser.add_argument("--set", nargs="*", default=[], metavar="KEY=JSON")
    args = parser.parse_args()
    logger.remove()
    logger.add(sys.stderr, level="WARNING")
    log_directory = ROOT / ".cache" / "benchmark_logs"
    log_directory.mkdir(parents=True, exist_ok=True)
    logger.add(
        str(log_directory / "events.log"),
        level="DEBUG",
        rotation="10 MB",
        retention=2,
        diagnose=False,
    )
    config = benchmark.load_config(args.config)
    training = config["training"]
    for item in args.set:
        key, _, value = item.partition("=")
        training[key] = json.loads(value)
    if args.earlier_update:
        training["fused_master_optimizer"] = False

        def torch_clipping(parameters, max_norm):
            norm = torch.nn.utils.clip_grad_norm_(
                parameters, max_norm, error_if_nonfinite=False, foreach=True
            )
            return norm, torch.ones((), device=norm.device), []

        epoch_training.measure_gradients = torch_clipping
    weights, left_out = shape_weights(
        args.sample, args.max_variables, training.get("batch_size", 8)
    )
    model, optimizer, average = benchmark.build(config, init="constructor")
    torch.manual_seed(99)
    updates, failed = {}, {}
    for shape in weights:
        tables, variables, patches = (int(part) for part in shape.split("x"))
        updates[shape] = benchmark.synthetic_update(
            model, tables, variables, patches, training
        )

    peak = 0.0

    def run(shape):
        nonlocal peak
        group, batches = updates[shape]
        result = epoch_training.accumulated_update(
            model, group, optimizer, training["gradient_clip"], batches
        )
        if average is not None:
            average.update()
        peak = max(peak, result["cuda_peak_allocated_gib"])
        if args.earlier_update:
            # The earlier trainer also wrote one DEBUG line per pass and one per
            # update from the training thread to the log file on the project
            # drive. (Its per-table lines came from the preparation thread and
            # are left out here.)
            for index, (prepared, _, points) in enumerate(batches, 1):
                logger.debug(
                    "Accumulation microbatch {}/{} embedding_shape[B,V,T,D]={} max_real_patches={} kernel_patches={} observed_labels={}",
                    index,
                    len(batches),
                    result["embedding_shapes"][index - 1],
                    prepared["tokens"],
                    prepared["padded_tokens"],
                    points,
                )
            logger.debug("Accumulation microbatch losses={}", [result["loss"]])

    for shape in weights:  # compilation and allocator warm-up for every shape
        try:
            run(shape)
            run(shape)
        except torch.OutOfMemoryError:
            failed[shape] = "out of memory during warm-up"
            optimizer.zero_grad(set_to_none=True)
            gc.collect()
            torch.cuda.empty_cache()
    torch.cuda.synchronize()
    peak = 0.0
    seconds = {shape: 0.0 for shape in weights}
    counts = {shape: 0 for shape in weights}
    reserved = 0
    order = random.Random(7)  # nosec B311 - reproducible visiting order
    for _ in range(args.rounds):
        visit = [shape for shape in weights if shape not in failed]
        order.shuffle(visit)
        for shape in visit:
            try:
                torch.cuda.synchronize()
                started = perf_counter()
                for _ in range(args.updates_per_visit):
                    run(shape)
                torch.cuda.synchronize()
                seconds[shape] += perf_counter() - started
                counts[shape] += args.updates_per_visit
            except torch.OutOfMemoryError:
                failed[shape] = "out of memory"
                optimizer.zero_grad(set_to_none=True)
                gc.collect()
                torch.cuda.empty_cache()
            reserved = max(reserved, torch.cuda.memory_reserved())
    measured = {s: seconds[s] / counts[s] for s in weights if counts[s]}
    covered = sum(weights[s] for s in measured)
    mean = sum(weights[s] * measured[s] for s in measured) / covered
    tables = training.get("batch_size", 8)
    summary = {
        "label": args.label,
        "earlier_update": args.earlier_update,
        "overrides": args.set,
        "weighted_ms_per_update": round(1000 * mean, 2),
        "tables_per_second": round(tables / mean, 1),
        "shapes_measured": len(measured),
        "share_of_windows_covered": round(covered, 5),
        "share_left_out_more_variables": round(left_out, 5),
        "out_of_memory": failed,
        "peak_allocated_gib": round(peak, 2),
        "peak_reserved_gib": round(reserved / 2**30, 2),
        "passes": {s: len(updates[s][1]) for s in weights},
        "ms_per_update": {s: round(1000 * v, 2) for s, v in sorted(measured.items())},
        "weights": {s: round(w, 5) for s, w in sorted(weights.items())},
    }
    print(
        json.dumps(
            {
                k: v
                for k, v in summary.items()
                if k not in ("ms_per_update", "weights", "passes")
            },
            indent=1,
        ),
        flush=True,
    )
    if args.output:
        Path(args.output).write_text(json.dumps(summary, indent=1), encoding="utf-8")


if __name__ == "__main__":
    main()

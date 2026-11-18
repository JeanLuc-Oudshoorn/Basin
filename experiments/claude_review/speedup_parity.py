"""Do the faster optimizer step and clipping change a single bit of training?

Training on this GPU is not reproducible bit for bit from one run to the next
(two runs of the unchanged trainer drift apart after a few updates), so two
separate runs cannot be compared. This script instead trains the full-size
model once and keeps a second, "shadow" copy of all weights, FP32 masters and
moments. On every update the shadow receives the very same gradients and is
updated the earlier way:

* `torch.nn.utils.clip_grad_norm_` for clipping, and
* the three-pass optimizer update (cast, PyTorch's fused AdamW, copy back).

After each update, the gradient norm, the clipped gradients, the model weights,
the masters, the moments and the step counters of both copies are compared bit
for bit. The clipping limit is lowered to the median gradient norm of the first
updates so that about half of the updates are actually clipped.

    source scripts/activate_rwkv7.sh
    python experiments/claude_review/speedup_parity.py --updates 96

Needs about 8 GiB of VRAM (the model's training memory for small batches plus
3.5 GiB for the shadow copy); the allocator is capped as in training.
"""
# ruff: noqa: E402

import argparse
import json
import statistics
import sys
from pathlib import Path
from types import SimpleNamespace

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
import gradient_clipping
import update_phase_breakdown as benchmark
from loguru import logger
from master_optimizer import FP32MasterAdamW

# Small batches leave room for the shadow copy. 8x3x64 includes variate
# attention, so every parameter receives a gradient.
SHAPES = ("8x1x16", "8x3x64", "8x1x64", "8x2x32", "8x1x128", "8x3x32")


def bits(tensor):
    flat = tensor.detach().contiguous().reshape(-1)
    return flat.view(torch.int16 if tensor.dtype == torch.bfloat16 else torch.int32)


def same(left, right):
    return left.dtype == right.dtype and torch.equal(bits(left), bits(right))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/config.json")
    parser.add_argument("--updates", type=int, default=96)
    parser.add_argument("--calibration-updates", type=int, default=12)
    parser.add_argument("--init", default="upstream")
    parser.add_argument("--output", default=None)
    args = parser.parse_args()
    logger.remove()
    logger.add(sys.stderr, level="WARNING")
    config = benchmark.load_config(args.config)
    training = config["training"]
    model, optimizer, average = benchmark.build(config, init=args.init)
    if getattr(optimizer, "_kernel", None) is None:
        raise RuntimeError("The one-pass optimizer kernel is not active")

    # Shadow weights in the optimizer's own grouping, updated the earlier way.
    weights = optimizer.model_parameters
    shadow_weights = [torch.nn.Parameter(w.detach().clone()) for w in weights]
    groups, offset = [], 0
    for group in optimizer.param_groups:
        count = len(group["params"])
        groups.append(
            {
                "params": shadow_weights[offset : offset + count],
                "lr": group["lr"],
                "weight_decay": group["weight_decay"],
            }
        )
        offset += count
    first = optimizer.param_groups[0]
    shadow = FP32MasterAdamW(
        groups,
        lr=first["lr"],
        weight_decay=first["weight_decay"],
        betas=first["betas"],
        eps=first["eps"],
        fused=True,
        foreach=False,
        fused_kernel=False,
    )
    by_weight = {id(w): s for w, s in zip(weights, shadow_weights)}
    report = {
        "updates": 0,
        "updates_clipped": 0,
        "tensors_compared_per_update": 0,
        "mismatches": [],
        "gradient_norms": [],
    }
    pending = {}
    real_measure = gradient_clipping.measure_gradients

    def checked_measure(parameters, max_norm):
        # Earlier clipping, on copies of the same gradients.
        copies = [
            SimpleNamespace(grad=None if p.grad is None else p.grad.clone())
            for p in parameters
        ]
        expected = torch.nn.utils.clip_grad_norm_(
            copies, max_norm, error_if_nonfinite=False, foreach=True
        )
        norm, coefficient, grouped = real_measure(parameters, max_norm)
        if not same(norm, expected):
            report["mismatches"].append(
                {"update": report["updates"], "what": "gradient norm"}
            )
        pending["clipped"] = {
            id(p): c.grad for p, c in zip(parameters, copies) if c.grad is not None
        }
        return norm, coefficient, grouped

    epoch_training.measure_gradients = checked_measure
    real_step = optimizer.step

    def checked_step():
        update = report["updates"]
        expected = pending.pop("clipped")
        compared = 0
        for weight in weights:
            reference = expected.get(id(weight))
            target = by_weight[id(weight)]
            if (weight.grad is None) != (reference is None):
                report["mismatches"].append({"update": update, "what": "gradient set"})
            elif reference is not None and not same(weight.grad, reference):
                report["mismatches"].append(
                    {"update": update, "what": "clipped gradient"}
                )
            target.grad = reference
            compared += reference is not None
        for scheduled, group in zip(optimizer.param_groups, shadow.param_groups):
            group["lr"] = scheduled["lr"]
        shadow.step()
        real_step()
        for weight, master, target, target_master in zip(
            weights,
            optimizer.master_parameters,
            shadow_weights,
            shadow.master_parameters,
        ):
            pairs = [("weight", weight, target), ("master", master, target_master)]
            if (master in optimizer.state) != (target_master in shadow.state):
                report["mismatches"].append({"update": update, "what": "state set"})
            elif master in optimizer.state:
                pairs += [
                    (
                        key,
                        optimizer.state[master][key],
                        shadow.state[target_master][key],
                    )
                    for key in ("step", "exp_avg", "exp_avg_sq")
                ]
            for name, left, right in pairs:
                compared += 1
                if not same(left, right):
                    report["mismatches"].append({"update": update, "what": name})
        for target in shadow_weights:
            target.grad = None
        report["tensors_compared_per_update"] = compared
        report["updates"] += 1

    optimizer.step = checked_step
    torch.manual_seed(1234)
    updates = [
        benchmark.synthetic_update(
            model, *(int(part) for part in shape.split("x")), training
        )
        for shape in SHAPES
    ]
    gradient_clip = training["gradient_clip"]
    base_rates = [group["lr"] for group in optimizer.param_groups]
    for step in range(args.updates):
        if step == args.calibration_updates:
            # From here on, clip about every second update.
            gradient_clip = statistics.median(report["gradient_norms"])
        for group, rate in zip(optimizer.param_groups, base_rates):
            # The scheduler changes the rate every update; so does this.
            group["lr"] = rate * min(1.0, (step + 1) / 20) * 0.995**step
        group, batches = updates[step % len(updates)]
        result = epoch_training.accumulated_update(
            model, group, optimizer, gradient_clip, batches
        )
        if average is not None:
            average.update()
        report["gradient_norms"].append(result["gradient_norm"])
        report["updates_clipped"] += result["gradient_norm"] > gradient_clip
        if report["mismatches"]:
            break
    torch.cuda.synchronize()
    summary = {
        "identical": not report["mismatches"],
        "updates_compared": report["updates"],
        "updates_clipped": int(report["updates_clipped"]),
        "clip_limit_after_calibration": gradient_clip,
        "tensors_compared_per_update": report["tensors_compared_per_update"],
        "parameters": sum(w.numel() for w in weights),
        "mismatches": report["mismatches"][:10],
        "peak_allocated_gib": round(torch.cuda.max_memory_allocated() / 2**30, 2),
        "reserved_gib": round(torch.cuda.memory_reserved() / 2**30, 2),
    }
    print(json.dumps(summary, indent=1), flush=True)
    if args.output:
        Path(args.output).write_text(json.dumps(summary, indent=1), encoding="utf-8")
    raise SystemExit(0 if summary["identical"] else 1)


if __name__ == "__main__":
    main()

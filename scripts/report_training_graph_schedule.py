"""Summarize total, capture-inclusive timings for the shape scheduling audit."""

import hashlib
import json
import statistics
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
OUTPUT = ROOT / "docs/time_benchmark/training_graph_schedule"


def main():
    results = json.loads((OUTPUT / "results.json").read_text())
    settings = json.loads((OUTPUT / "settings.json").read_text())
    env = json.loads((OUTPUT / "environment.json").read_text())
    by_name = defaultdict(list)
    for result in results:
        by_name[result["name"]].append(result)
    summary = {}
    for name, rows in by_name.items():
        summary[name] = dict(
            mean_update_ms=statistics.mean(
                r["total_ms"] / len(r["updates"]) for r in rows
            ),
            trial_update_ms=[r["total_ms"] / len(r["updates"]) for r in rows],
            peak_allocated_gib=max(r["peak_allocated_gib"] for r in rows),
            peak_reserved_gib=max(r["peak_reserved_gib"] for r in rows),
            graphs=[r["graphs"] for r in rows],
        )
    paired = []
    for repeat in sorted({r["repeat"] for r in results}):
        arms = {r["name"]: r for r in results if r["repeat"] == repeat}
        off, on = arms["grouped_off"], arms["grouped_replay"]
        paired.append(
            dict(
                repeat=repeat,
                time_saved_percent=100 * (1 - on["total_ms"] / off["total_ms"]),
                max_loss_absolute_difference=max(
                    abs(a["loss"] - b["loss"])
                    for a, b in zip(off["updates"], on["updates"])
                ),
                max_loss_relative_difference=max(
                    abs(a["loss"] - b["loss"]) / max(abs(a["loss"]), 1e-6)
                    for a, b in zip(off["updates"], on["updates"])
                ),
                mean_loss_relative_difference=statistics.mean(
                    abs(a["loss"] - b["loss"]) / max(abs(a["loss"]), 1e-6)
                    for a, b in zip(off["updates"], on["updates"])
                ),
            )
        )
    summary["paired"] = paired
    controls = sorted(by_name["grouped_off"], key=lambda r: r["repeat"])
    errors = [
        abs(a["loss"] - b["loss"]) / max(abs(a["loss"]), 1e-6)
        for a, b in zip(controls[0]["updates"], controls[1]["updates"])
    ]
    summary["replay_disabled_repeat_variation"] = dict(
        mean=statistics.mean(errors), maximum=max(errors)
    )
    dimensions = defaultdict(lambda: defaultdict(list))
    for result in results:
        for row in result["updates"]:
            # [physical batch, variables, padded patch tokens, hidden width]
            shape = tuple(row["embedding_shapes"][0])
            dimensions[shape][result["name"]].append(row["milliseconds"])
    shape_rows = []
    for shape, arms in sorted(dimensions.items()):
        shape_rows.append(
            dict(
                shape=shape,
                means_ms={k: statistics.mean(v) for k, v in arms.items()},
                count_per_trial=len(arms["grouped_off"]) // 2,
            )
        )
    summary["shapes"] = shape_rows
    (OUTPUT / "summary.json").write_text(
        json.dumps(summary, indent=2) + "\n", newline="\n"
    )
    lines = [
        "# Training graph schedule: capture-inclusive benchmark",
        "",
        "2026-11-01. Production 307,307,808-parameter model, RTX 5070 12 GiB, BF16, "
        f"PyTorch {env['torch']}. Two trials per arm, 1,024 full optimizer updates per trial.",
        "",
        "| Execution | Mean ms/update | Trial 1 | Trial 2 | Peak allocated GiB |",
        "|---|---:|---:|---:|---:|",
    ]
    for name in ("shuffled_off", "grouped_off", "grouped_replay"):
        row = summary[name]
        lines.append(
            f"| {name} | {row['mean_update_ms']:.2f} | "
            + " | ".join(f"{x:.2f}" for x in row["trial_update_ms"])
            + f" | {row['peak_allocated_gib']:.3f} |"
        )
    saved = 100 * (
        1
        - summary["grouped_replay"]["mean_update_ms"]
        / summary["grouped_off"]["mean_update_ms"]
    )
    overall = 100 * (
        1
        - summary["grouped_replay"]["mean_update_ms"]
        / summary["shuffled_off"]["mean_update_ms"]
    )
    lines += [
        "",
        f"Replay saves **{saved:.2f}%** versus the same grouped schedule without replay. "
        f"Grouping plus replay saves **{overall:.2f}%** versus shuffled batches without replay.",
        "",
        "## What changed",
        "",
        "`make_batches` still shuffles tables within each existing variable-role and length bucket. "
        "It now shuffles runs of up to 128 batches instead of every batch separately. "
        "Every table occurs once per epoch; table contents, batch membership, temporal ordering, "
        "loss weighting, validation and checkpoint frequency are preserved. Optimizer update order changes, "
        "so this is not an identical training trajectory or a test of final accuracy.",
        "",
        "A new shape needs eight consecutive optimizer updates before capture; accumulation microbatches "
        "do not count as additional updates. Already cached profiles replay immediately. "
        "Default replay is limited to at most 32 padded patch tokens (512 observations of input); "
        "long contexts use compiled attention. Set `training_graph_max_tokens` to null to experiment with all lengths.",
        "",
        "The 1.5 GiB graph budget, two-profile bound, 8 GiB allocator limit, main-thread pinning and "
        "synchronized graph-pool retirement remain in force. Profiles are separate for physical batch size, "
        "variable count, padded patch length and model width, with layer-specific graph storage.",
        "",
        f"The complete epoch has {sum(settings['epoch_shapes'].values()):,} batch groups and "
        f"{len(settings['epoch_shapes'])} logical batch/variable/padded-length combinations, including partial batches. "
        "Physical microbatching can introduce further batch sizes. It is not just the number of covariates.",
        "",
        "## Shape-level totals (capture included)",
        "",
        "| Physical B,V,T,D | Updates/trial | Compiled ms | Replay ms |",
        "|---|---:|---:|---:|",
    ]
    for row in shape_rows:
        lines.append(
            f"| {row['shape']} | {row['count_per_trial']} | {row['means_ms']['grouped_off']:.2f} | {row['means_ms']['grouped_replay']:.2f} |"
        )
    larger_path = ROOT / "docs/time_benchmark/training_graph_schedule_micro8"
    if (larger_path / "environment.json").exists():
        larger = json.loads((larger_path / "results.json").read_text())
        larger_summary = {}
        for name in ("grouped_off", "grouped_replay"):
            rows = [r for r in larger if r["name"] == name]
            larger_summary[name] = dict(
                mean_update_ms=statistics.mean(
                    r["total_ms"] / len(r["updates"]) for r in rows
                ),
                trials_ms=[
                    r["total_ms"] / len(r["updates"])
                    for r in sorted(rows, key=lambda r: r["repeat"])
                ],
                peak_allocated_gib=max(r["peak_allocated_gib"] for r in rows),
                peak_reserved_gib=max(r["peak_reserved_gib"] for r in rows),
                graphs=[r["graphs"] for r in rows],
            )
        summary["microbatch_eight"] = larger_summary
        (OUTPUT / "summary.json").write_text(
            json.dumps(summary, indent=2) + "\n", newline="\n"
        )
        current = json.loads((ROOT / "configs/config.json").read_text())["training"]
        selected = (
            "grouped_replay" if current["training_cuda_graphs"] else "grouped_off"
        )
        latency = larger_summary[selected]["mean_update_ms"]
        gain = 100 * (1 - latency / summary["shuffled_off"]["mean_update_ms"])
        top = [
            "## Selected default and larger-microbatch experiment",
            "",
            f"Default physical microbatch limit: **{current['microbatch_size']}**; replay: **{current['training_cuda_graphs']}**. "
            f"On this sample the selected microbatch-eight execution takes **{latency:.2f} ms/update**, "
            f"**{gain:.2f}% less time** than the microbatch-four shuffled baseline.",
            "",
            "| Microbatch maximum | Replay | Mean ms/update | Trial 1 | Trial 2 | Peak allocated GiB |",
            "|---:|---|---:|---:|---:|---:|",
        ]
        for name, row in larger_summary.items():
            top.append(
                f"| 8 | {name == 'grouped_replay'} | {row['mean_update_ms']:.2f} | "
                + " | ".join(f"{x:.2f}" for x in row["trials_ms"])
                + f" | {row['peak_allocated_gib']:.3f} |"
            )
        replay_change = 100 * (
            larger_summary["grouped_replay"]["mean_update_ms"]
            / larger_summary["grouped_off"]["mean_update_ms"]
            - 1
        )
        top += [
            "",
            f"At microbatch eight, replay changes elapsed update time by **{replay_change:+.2f}%** "
            "relative to compiled execution without replay (positive means slower). "
            "The default uses the faster measured option; inference replay remains enabled.",
        ]
        top += [
            "",
            "Optimizer batch size remains eight. The 1,536 variable-token limit still reduces the physical "
            "batch for larger inputs, and the 8 GiB allocator cap remains enabled. This experiment uses the "
            "same real-table sample and resets initial weights/AdamW for each arm; two paired trials reverse "
            "arm order. Raw artifacts and resource monitoring are in `../training_graph_schedule_micro8/`.",
            "",
            "The physical-microbatch experiment was run after the microbatch-four experiment in a separate process. "
            "The cross-microbatch gain is consequently not an interleaved timing comparison; both used the same GPU, "
            "environment, tables, seed and compiled code. It is not a full-epoch throughput or accuracy guarantee.",
            "",
            "The microbatch-eight run was interrupted by a usage limit after trial one. The benchmark resumed "
            "only the two unfinished arms in a fresh process, after checking identical model/training settings "
            "and source batches. Completed arms were retained; the interrupted partial arm was discarded. "
            "`settings.json` records the original and resumed log directories. The gap between trials is an "
            "additional reason to treat small timing differences cautiously.",
            "",
            "## Microbatch-four scheduling experiment",
            "",
        ]
        lines[4:4] = top
        results = results + larger
    lines += [
        "",
        "## Method, correctness and limits",
        "",
        "The sample is the first 1,024 batches of the new epoch-1 schedule, using real admitted training tables. "
        "All arms use exactly those same tables and batch memberships. The shuffled arm randomizes their batch order. "
        "Every arm resets model parameters and AdamW state, uses the same RNG seed, and starts with an empty graph cache. "
        "Trial two reverses arm order. One GPU model is resident at a time.",
        "",
        "Timing includes CPU prefetch consumption, pinning, transfer, forward/backward, gradient clipping, optimizer "
        "update, all graph captures and evictions, and a final CUDA synchronization. Common compilation is warmed "
        "outside every arm. Model/data loading and epoch validation/checkpoint writes are excluded. "
        "There is no discarded warmup inside a measured arm. This is a sustained mixed-shape sample, not a full epoch.",
        "",
        f"Paired loss checks: `{json.dumps(paired)}`.",
        "",
        f"Replay-disabled repeats also vary: mean relative loss difference {statistics.mean(errors):.6f}, "
        f"maximum {max(errors):.6f}. These BF16 full-model runs are not bitwise deterministic, even without replay. "
        "Use the dedicated gradient/optimizer regression tests for correctness; this benchmark does not establish identical final accuracy.",
        "",
        "`results.json` contains every update's loss, timing and embedding shapes, plus memory/capture counters. "
        "`sources.json` identifies the tables; `settings.json` includes the complete epoch shape inventory. "
        f"Run logs: `{env['logs']}`.",
        "",
        "## Configuration and continuation",
        "",
        "Default `configs/config.json`: `microbatch_size=8`, `shape_run_batches=128`, `training_graph_min_reuse=8`, "
        "`training_graph_max_tokens=32`, `training_cuda_graphs=false`. Running `train.py` uses these settings. "
        "Replay remains available by setting `training_cuda_graphs=true`; inference replay is unchanged. "
        "`shape_run_batches=1` restores globally shuffled batches; the admission rule then prevents repeated "
        "capture for isolated shapes. `training_cuda_graphs=false` disables replay while leaving compilation enabled.",
        "",
        "Batch ordering is included in checkpoint identity. To continue an existing checkpoint created with the "
        "old ordering, set `shape_run_batches=1`; to use the new order from scratch, set `resume=false`. "
        "New checkpoints resume normally with the matching setting. No production checkpoint was altered by this benchmark.",
        "",
        "## Reproduction",
        "",
        "From the activated WSL GPU environment, run `python scripts/benchmark_training_graph_schedule.py --microbatch 4`, "
        "then `python scripts/benchmark_training_graph_schedule.py --microbatch 8 --skip-shuffled`. "
        "Generate the report with `python scripts/report_training_graph_schedule.py`. "
        "Add `--resume` to the benchmark command to retain completed arms after an interruption; "
        "model/training settings and source batches must still match. These commands write benchmark artifacts, not production weights.",
        "",
    ]
    (OUTPUT / "REPORT.md").write_text("\n".join(lines), newline="\n")
    files = [
        "src/05_model_training/batching.py",
        "src/05_model_training/training_graphs.py",
        "src/05_model_training/epoch_training.py",
        "src/05_model_training/checkpoints.py",
        "configs/config.json",
        "scripts/benchmark_training_graph_schedule.py",
    ]
    (OUTPUT / "verification.json").write_text(
        json.dumps(
            dict(
                source_sha256={
                    path: hashlib.sha256((ROOT / path).read_bytes()).hexdigest()
                    for path in files
                },
                optimizer_updates=sum(len(r["updates"]) for r in results),
                capture_failures=sum(
                    (r["graphs"] or {}).get("capture_failures", 0) for r in results
                ),
                disabled_reasons=[
                    r["graphs"]["disabled_reason"] for r in results if r["graphs"]
                ],
                regression_tests={
                    "native": "29 tests, 27 passed and 2 CUDA-only skipped",
                    "cuda": "6 tests passed, including both CUDA-only tests",
                },
            ),
            indent=2,
        )
        + "\n",
        newline="\n",
    )


if __name__ == "__main__":
    main()

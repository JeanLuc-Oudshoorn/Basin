"""Summarize paired replay timings without hiding capture or cache churn."""

import csv
import hashlib
import json
import statistics
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DIRECTORY = ROOT / "docs/time_benchmark/training_graphs_post_fixes"


def main():
    rows = json.loads((DIRECTORY / "results.json").read_text())
    settings = json.loads((DIRECTORY / "settings.json").read_text())
    env = json.loads((DIRECTORY / "environment.json").read_text())
    names = [case["name"] for case in settings["benchmark"]["cases"]] + [
        "mixed_real_tables"
    ]
    summary = []
    for name in names:
        selected = [row for row in rows if row["name"] == name]
        off = sorted(
            (row for row in selected if not row["enabled"]), key=lambda r: r["repeat"]
        )
        on = sorted(
            (row for row in selected if row["enabled"]), key=lambda r: r["repeat"]
        )
        expected = settings["benchmark"][
            "mixed_repeats" if name == "mixed_real_tables" else "repeats"
        ]
        if len(off) != expected or len(on) != expected:
            raise ValueError(f"Incomplete benchmark: {name}")
        mixed = name == "mixed_real_tables"

        def metric(row):
            return (
                row["all_updates_total_ms"] / len(row["updates"])
                if mixed
                else row["median_ms"]
            )

        reference = statistics.median(metric(row) for row in off)
        replay = statistics.median(metric(row) for row in on)
        savings = [100 * (1 - metric(b) / metric(a)) for a, b in zip(off, on)]
        first_extra = statistics.median(
            b["updates"][0]["milliseconds"] - a["updates"][0]["milliseconds"]
            for a, b in zip(off, on)
        )
        summary.append(
            dict(
                case=name,
                reference_ms=reference,
                replay_ms=replay,
                time_reduction_percent=100 * (1 - replay / reference),
                paired_time_reduction_min_percent=min(savings),
                paired_time_reduction_max_percent=max(savings),
                first_update_extra_ms=first_extra,
                approximate_first_capture_break_even_updates=(
                    max(0, first_extra) / (reference - replay)
                    if replay < reference and not mixed
                    else None
                ),
                peak_allocated_gib=max(row["peak_allocated_gib"] for row in on),
                peak_reserved_gib=max(row["peak_reserved_gib"] for row in on),
                max_loss_relative_difference=max(
                    row["max_loss_relative_difference"] for row in on
                ),
                all_losses_within_one_percent=all(
                    row["loss_agreement_within_one_percent"] for row in on
                ),
                capture_failures=sum(
                    row["graphs"].get("capture_failures", 0) for row in on
                ),
                disabled_replay=any(
                    row["graphs"]["disabled_reason"] is not None for row in on
                ),
            )
        )
    (DIRECTORY / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    with (DIRECTORY / "summary.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(summary[0]))
        writer.writeheader()
        writer.writerows(summary)
    lines = [
        "# Training graph replay after the memory and prefetch fixes",
        "",
        "2026-11-01. Measured current execution; no production settings or weights changed.",
        "",
        "## Repeated shapes: warm full-update latency",
        "",
        "| Input observations | Variables | Maximum microbatch | Compiled reference ms | Replay ms | Time saved | Paired range |",
        "|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for case, row in zip(settings["benchmark"]["cases"], summary):
        lines.append(
            f"| {case['patches'] * settings['model']['patch_size']} | {case['variables']} | {case['microbatch']} | "
            f"{row['reference_ms']:.2f} | {row['replay_ms']:.2f} | {row['time_reduction_percent']:.1f}% | "
            f"{row['paired_time_reduction_min_percent']:.1f}% to {row['paired_time_reduction_max_percent']:.1f}% |"
        )
    mixed = summary[-1]
    lines += [
        "",
        "Positive time saved means replay reduced latency; negative means slower.",
        "",
        "## Actual changing shapes: capture and eviction included",
        "",
        f"Across the first {settings['benchmark']['mixed_batches']} real epoch-1 batch groups, "
        f"mean latency per update was {mixed['reference_ms']:.2f} ms without replay and "
        f"{mixed['replay_ms']:.2f} ms with replay. Replay took "
        f"{mixed['replay_ms'] / mixed['reference_ms']:.2f} times as long per update.",
        "",
        "This comparison includes every update, including new graph captures, synchronized "
        "profile retirement and cache churn. Common attention compilation was warmed for "
        "both arms. Each trial starts with an empty graph cache. It is not a full-epoch measurement.",
        "",
        "## Method and limits",
        "",
        f"- {env['parameters']:,} parameters; {env['gpu']}; PyTorch {env['torch']}, CUDA {env['cuda']}, BF16.",
        f"- One GPU model resident; {env['cpu_threads']} CPU threads; eight tables per update.",
        "- Identical initial model state and batch sequence, fresh AdamW state and the same RNG seed for each arm.",
        "- Compiled attention, prefetch, token-budget microbatch splitting, gradient accumulation, clipping and optimizer are enabled in both arms.",
        "- Host timing includes waiting for preparation, main-thread pinning, transfer, forward, both losses, backward, clipping, metric synchronization and completed AdamW update.",
        "- Per-update GPU synchronization bounds the measurement. Validation, checkpoint I/O and ordinary epoch/progress logging are excluded.",
        "- Synthetic repeated-shape tables have precomputed CPU token features. Real mixed-shape tables use the actual lazy tokenizer and normal prefetch path.",
        f"- Repeated-shape trials discard {settings['benchmark']['warmup_updates']} warmup updates and time {settings['benchmark']['measured_updates']} more; {settings['benchmark']['repeats']} paired repetitions alternate arm order.",
        "- Repeated-shape table reports median trial medians. Mixed-shape result reports median trial total divided by update count, retaining costly captures.",
        f"- Graph cache budget {env['graph_budget_gib']} GiB; allocator cap {env['allocator_limit_gib']} GiB. Configured microbatch four becomes two in the six-variable 128-patch case because of the 1536-token limit.",
        "- First-update and all-update timings, physical shapes, losses, graph counters, memory peaks and paired variation are retained in results.json and summary.json.",
        "- Estimated first-capture break-even counts use first-update excess latency divided by median warm savings; they exclude later shape changes and are approximate.",
        "- BF16 capture can change rounding; loss agreement is measured rather than assumed. This benchmark does not establish bitwise gradient equality.",
        "",
        f"Detailed logs: `{env['logs']}`.",
        "",
    ]
    if any(not row["all_losses_within_one_percent"] for row in summary):
        lines += [
            "Some paired loss sequences differed by more than 1%; see summary.json before interpreting those timings as equivalent training.",
            "",
        ]
    lines += [
        "## Interpretation",
        "",
        "Replay still reduces warm latency for short, repeated profiles. With the default "
        "four-table microbatch, the 256- and 512-observation cases saved approximately "
        "16% and 18% of update time. The 1024-observation effect was small and variable; "
        "the longest control saved only about 3%.",
        "",
        "The current shuffled mixed-profile pipeline did not receive an overall speedup: "
        f"replay was {mixed['replay_ms'] / mixed['reference_ms']:.2f}x slower in this bounded test. "
        "Each replay-enabled mixed trial recorded 388 captures, 825 replays and 22 "
        "profile evictions in just 24 updates. Most profiles are replaced before their "
        "capture cost is amortized. Thus the warm gains must not be presented as an "
        "epoch-level improvement for the current default cache policy.",
        "",
        "Warm replay timings apply only while the same batch profile stays cached. "
        "Changing shapes can require capture again, and safety synchronization adds cost when "
        "a profile is evicted. A warm speedup alone is insufficient to claim faster shuffled-corpus training.",
        "",
        "No production configuration was changed. These results identify profile reuse "
        "and capture amortization as the remaining performance limitation, rather than "
        "establishing which individual safety change contributed what fraction of the overhead.",
        "",
    ]
    (DIRECTORY / "REPORT.md").write_text("\n".join(lines), encoding="utf-8")
    sources = {
        str(path.relative_to(ROOT)): hashlib.sha256(path.read_bytes()).hexdigest()
        for path in (
            ROOT / "src/05_model_training/training_graphs.py",
            ROOT / "src/05_model_training/performance.py",
            ROOT / "src/05_model_training/epoch_training.py",
            ROOT / "scripts/benchmark_training_graphs_post_fixes.py",
            ROOT / "configs/training_graph_benchmark.json",
        )
    }
    (DIRECTORY / "verification.json").write_text(
        json.dumps(
            dict(
                complete_pairs=len(rows) // 2,
                source_sha256=sources,
                capture_failures=sum(row["capture_failures"] for row in summary),
                replay_disabled=any(row["disabled_replay"] for row in summary),
                loss_checks_all_passed=all(
                    row["all_losses_within_one_percent"] for row in summary
                ),
            ),
            indent=2,
        )
        + "\n"
    )
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()

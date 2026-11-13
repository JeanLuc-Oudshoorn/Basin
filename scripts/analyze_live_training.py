"""Read a bounded journal tail to summarize live throughput, without using CUDA."""

import argparse
import json
import statistics
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def percentile(values, q):
    values = sorted(values)
    return values[round((len(values) - 1) * q)] if values else None


def analyze(path, tail_bytes=16 * 1024 * 1024):
    with path.open("rb") as stream:
        stream.seek(0, 2)
        end = stream.tell()
        stream.seek(max(0, end - tail_bytes))
        lines = stream.read().splitlines()
    records = []
    for line in lines:
        try:
            event = json.loads(line)
        except (ValueError, UnicodeDecodeError):
            continue
        if isinstance(event, dict) and event.get("event") == "metrics":
            records.append(event)
    if len(records) < 2:
        raise ValueError("At least two training metric events are needed")
    pairs = [
        (a, b)
        for a, b in zip(records, records[1:])
        if b["global_step"] == a["global_step"] + 1 and b.get("epoch") == a.get("epoch")
    ]
    wall = [b["elapsed_seconds"] - a["elapsed_seconds"] for a, b in pairs]
    active = [
        b["active_training_seconds"] - a["active_training_seconds"] for a, b in pairs
    ]
    # Long wall gaps may include validation/checkpoints. Report the unfiltered
    # gap distribution rather than silently treating it as kernel execution.
    peaks = [
        e["cuda_peak_allocated_gib"]
        for e in records
        if e.get("cuda_peak_allocated_gib") is not None
    ]
    by_microbatches = {}
    for a, b in pairs:
        by_microbatches.setdefault(b["microbatches"], []).append(
            b["active_training_seconds"] - a["active_training_seconds"]
        )
    return {
        "sampled_at_utc": datetime.now(timezone.utc).isoformat(),
        "journal": str(path),
        "metric_events": len(records),
        "first_step": records[0]["global_step"],
        "last_step": records[-1]["global_step"],
        "epoch_updates": records[-1].get("epoch_updates"),
        "wall_ms_median": percentile(wall, 0.5) * 1000,
        "wall_ms_p90": percentile(wall, 0.9) * 1000,
        "active_ms_median": percentile(active, 0.5) * 1000,
        "active_ms_mean": statistics.mean(active) * 1000,
        "outside_active_seconds": sum(wall) - sum(active),
        "outside_active_scope": "between-update bookkeeping only; debug logging in preparation/backward is inside active time",
        "sample_wall_seconds": sum(wall),
        "physical_microbatches_mean": statistics.mean(
            e["microbatches"] for e in records
        ),
        "split_update_fraction": sum(e["microbatches"] > 1 for e in records)
        / len(records),
        "active_by_microbatches": {
            str(count): {
                "updates": len(values),
                "mean_ms": statistics.mean(values) * 1000,
                "share_of_active_time": sum(values) / sum(active),
            }
            for count, values in sorted(by_microbatches.items())
        },
        "oom_retries": sum(e.get("oom_retries", 0) for e in records),
        "peak_live_gib_median": percentile(peaks, 0.5),
        "peak_live_gib_p90": percentile(peaks, 0.9),
        "peak_live_gib_max": max(peaks) if peaks else None,
        "mean_windows_per_update": statistics.mean(e["windows"] for e in records),
        "observed_target_points_per_second_active": sum(
            b["target_points"] for _, b in pairs
        )
        / sum(active),
        "epoch_training_eta_hours": records[-1].get("epoch_eta_seconds", 0) / 3600,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--journal", type=Path)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    journals = (ROOT / "logs").glob("rwkv7_train_*.jsonl")
    path = args.journal or max(journals, key=lambda p: p.stat().st_mtime)
    result = analyze(path)
    content = json.dumps(result, indent=2, allow_nan=False) + "\n"
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(content, encoding="utf-8", newline="\n")
    print(content, end="")


if __name__ == "__main__":
    main()

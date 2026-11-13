"""Read bounded journal ranges without loading weights or using CUDA."""

import json
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
OLD = ROOT / "logs/rwkv7_train_20261108_190532_203118Z_pid23856_0fdb0d94.jsonl"
CURRENT = ROOT / "logs/rwkv7_train_20261110_043446_997744Z_pid18100_9da1bdc9.jsonl"
OUTPUT = ROOT / "docs/time_benchmark/epoch2_throughput_20261111"


def read_range(path, first, last):
    """Seek by monotonically increasing metrics steps; freeze the journal end."""
    rows = []
    with path.open("rb") as stream:
        stream.seek(0, 2)
        end = high = stream.tell()
        low = 0
        while high - low > 65536:
            middle = (low + high) // 2
            stream.seek(middle)
            stream.readline()
            row = None
            while stream.tell() < end:
                line = stream.readline()
                try:
                    item = json.loads(line)
                except ValueError:
                    continue
                if item.get("event") == "metrics":
                    row = item
                    break
            if row is None or row["global_step"] >= first:
                high = middle
            else:
                low = stream.tell()
        stream.seek(low)
        if low:
            stream.readline()
        while stream.tell() < end:
            line = stream.readline()
            try:
                row = json.loads(line)
            except ValueError:
                continue
            if row.get("event") != "metrics":
                continue
            step = row["global_step"]
            if step > last:
                break
            if step >= first:
                rows.append(row)
    return rows


def stats(values):
    values = np.asarray(values, dtype=float)
    return {
        "mean": float(values.mean()),
        "median": float(np.median(values)),
        "p90": float(np.quantile(values, 0.9)),
        "maximum": float(values.max()),
        "sum": float(values.sum()),
    }


def profile(row):
    shapes = row["embedding_shapes"]
    return str((sum(s[0] for s in shapes), shapes[0][1], shapes[0][2]))


def summarize(rows):
    groups = defaultdict(list)
    for row in rows:
        groups[profile(row)].append(row)
    fields = (
        "preparation_wait_seconds",
        "optimizer_update_seconds",
        "microbatches",
        "cuda_allocated_gib",
        "cuda_peak_allocated_gib",
        "epoch_eta_seconds",
    )
    return {
        "count": len(rows),
        "steps": [rows[0]["global_step"], rows[-1]["global_step"]],
        "first": {
            k: rows[0].get(k)
            for k in (
                "epoch",
                "epoch_update",
                "epoch_elapsed_seconds",
                "active_training_seconds",
                "epoch_windows",
                "epoch_eta_seconds",
            )
        },
        "last": {
            k: rows[-1].get(k)
            for k in (
                "epoch",
                "epoch_update",
                "epoch_elapsed_seconds",
                "active_training_seconds",
                "epoch_windows",
                "epoch_eta_seconds",
            )
        },
        "stats": {
            key: stats([r[key] for r in rows])
            for key in fields
            if all(r.get(key) is not None for r in rows)
        },
        "microbatch_distribution": dict(Counter(r["microbatches"] for r in rows)),
        "oom_retries": sum(r.get("oom_retries", 0) for r in rows),
        "profiles": {
            key: {
                "count": len(group),
                "seconds": stats([r["optimizer_update_seconds"] for r in group]),
                "microbatches": stats([r["microbatches"] for r in group]),
            }
            for key, group in groups.items()
        },
        "slowest": [
            {
                k: r.get(k)
                for k in (
                    "global_step",
                    "preparation_wait_seconds",
                    "optimizer_update_seconds",
                    "embedding_shapes",
                    "oom_retries",
                )
            }
            for r in sorted(
                rows,
                key=lambda r: (
                    r["preparation_wait_seconds"] + r["optimizer_update_seconds"]
                ),
                reverse=True,
            )[:5]
        ],
    }


def main():
    ranges = {
        "bf16_late_epoch1": (OLD, 420000, 424999),
        "fp32_start": (CURRENT, 465163, 480000),
        "fp32_epoch1_middle": (CURRENT, 580000, 584999),
        "fp32_epoch1_late": (CURRENT, 665000, 674999),
        "fp32_epoch1_end": (CURRENT, 688697, 693696),
        "fp32_epoch2_first9730": (CURRENT, 693697, 703426),
        "fp32_epoch2_recent": (CURRENT, 703427, 704500),
    }
    results = {}
    for name, (path, first, last) in ranges.items():
        rows = read_range(path, first, last)
        if [r["global_step"] for r in rows] != list(range(first, last + 1)):
            raise ValueError(f"Missing or reordered journal updates in {name}")
        results[name] = summarize(rows)
    OUTPUT.mkdir(parents=True, exist_ok=True)
    (OUTPUT / "evidence.json").write_text(
        json.dumps(results, indent=2) + "\n", encoding="utf-8"
    )
    for name, value in results.items():
        print(
            name,
            "steps",
            value["steps"],
            "count",
            value["count"],
            "mean_ms",
            {
                k: round(v["mean"] * 1000, 2)
                for k, v in value["stats"].items()
                if k.endswith("seconds") and k != "epoch_eta_seconds"
            },
            "microbatches",
            value["stats"]["microbatches"]["mean"],
            "oom",
            value["oom_retries"],
        )
    for name in ("fp32_epoch1_late", "fp32_epoch1_end", "fp32_epoch2_first9730"):
        print(name, "first", results[name]["first"], "last", results[name]["last"])


if __name__ == "__main__":
    main()

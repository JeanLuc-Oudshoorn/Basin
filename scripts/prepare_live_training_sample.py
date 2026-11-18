"""Cache a bounded CPU sample of batches recorded in the actual training journal."""

# ruff: noqa: E402
import argparse
import json
import os
import sys
from pathlib import Path

os.environ["CUDA_VISIBLE_DEVICES"] = ""
ROOT = Path(__file__).resolve().parents[1]
for stage in (
    "03_transforming_input_data",
    "04_patch_indexing",
    "05_model_training",
):
    sys.path.insert(0, str(ROOT / "src" / stage))
import torch
from sharded_records import ShardedRecords
from transform import load_config, location


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--batches", type=int, default=32)
    parser.add_argument("--output", default=".cache/training_tuning/live_sample.pt")
    args = parser.parse_args()
    if args.batches < 2:
        parser.error("batches must be at least two")
    torch.set_num_threads(1)
    config = load_config()
    journal = max(
        (ROOT / "logs").glob("rwkv7_train_*.jsonl"), key=lambda p: p.stat().st_mtime
    )
    with journal.open("rb") as stream:
        stream.seek(0, 2)
        stream.seek(max(0, stream.tell() - 16 * 1024 * 1024))
        lines = stream.read().splitlines()
    events = []
    for line in lines:
        try:
            value = json.loads(line)
        except (ValueError, UnicodeDecodeError):
            continue
        if value.get("event") == "metrics" and value.get("window_ids"):
            events.append(value)
    if len(events) < args.batches:
        raise ValueError("Not enough recorded batches")
    selected = [
        events[i * (len(events) - 1) // (args.batches - 1)] for i in range(args.batches)
    ]
    directory = location(config, "processed")
    index = torch.load(directory / "windows.pt", weights_only=True, mmap=True)
    raw = ShardedRecords(index, directory)
    requests = {}
    for event in selected:
        for identity, source in zip(event["window_ids"], event["sources"]):
            requests[identity] = source
    records = {}
    # This corpus has one train + one evaluation record per table. Verify the
    # locator's ID/source instead of assuming this layout for arbitrary artifacts.
    for identity in sorted(requests, key=lambda value: int(value.split("_")[1])):
        table = int(identity.split("_")[1])
        found = None
        for position in (2 * table, 2 * table + 1):
            if position >= len(raw):
                continue
            record = raw[position]
            if record["id"] == identity and record["source"] == requests[identity]:
                found = record
                break
        if found is None:
            raise ValueError(
                f"Corpus layout does not match verified journal locator: {identity}"
            )
        records[identity] = {
            "id": identity,
            "source": found["source"],
            "split": "train",
            "batch": {
                key: value.detach().clone()
                if isinstance(value, torch.Tensor)
                else value
                for key, value in found["batch"].items()
            },
        }
    sample = {
        "train": list(records.values()),
        "batch_ids": [event["window_ids"] for event in selected],
        "journal": str(journal),
        "recorded_steps": [event["global_step"] for event in selected],
    }
    sample["worst"] = max(
        sample["train"],
        key=lambda r: (
            r["batch"]["context"].numel()
            + (
                r["batch"]["past_covariates"].numel()
                if r["batch"].get("past_covariates") is not None
                else 0
            )
        ),
    )
    output = ROOT / args.output
    output.parent.mkdir(parents=True, exist_ok=True)
    torch.save(sample, output)
    print(
        json.dumps(
            {
                "sample": str(output),
                "tables": len(records),
                "batches": len(selected),
                "bytes": output.stat().st_size,
                "gpu_used": False,
            }
        )
    )


if __name__ == "__main__":
    main()

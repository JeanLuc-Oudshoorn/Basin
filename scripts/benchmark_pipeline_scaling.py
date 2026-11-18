"""Bounded CPU benchmark of real cached metadata and shuffled shard reads."""

# ruff: noqa: E402
import argparse
import json
import sys
from pathlib import Path
from time import perf_counter
from types import SimpleNamespace
from unittest.mock import patch

import torch

ROOT = Path(__file__).resolve().parents[1]
for stage in (
    "03_transforming_input_data",
    "04_patch_indexing",
    "05_model_training",
):
    sys.path.insert(0, str(ROOT / "src" / stage))
from batching import BatchPlanner
from index_patches import LazyPatchRecord
from sharded_records import EncodedRecords, RecordSelection, ShardedRecords, load_shard
from transform import load_config, location, write_json


def run(config, shards=2, per_shard=8):
    directory = location(config, "processed")
    index = torch.load(directory / "windows.pt", weights_only=True, mmap=True)
    entries = [s for s in index["shards"] if s["record_count"]][:shards]
    raw = ShardedRecords(
        dict(shards=entries, record_count=sum(s["record_count"] for s in entries)),
        directory,
    )
    encoder = SimpleNamespace(config=SimpleNamespace(**config["model"]))
    records = EncodedRecords(raw, encoder, LazyPatchRecord)
    indices, offset = [], 0
    for entry in entries:
        count = 0
        for i in range(entry["record_count"]):
            if records[offset + i]["split"] == "train":
                indices.append(offset + i)
                count += 1
            if count >= per_shard:
                break
        offset += entry["record_count"]
    selection = RecordSelection(records, indices)
    report = {
        "platform": sys.platform,
        "gpu_used": False,
        "records": len(selection),
        "shards": len(entries),
        "index_array_bytes": len(selection.indices) * selection.indices.itemsize,
        "scope": "bounded real-data sample, batch_size=1; no model/optimizer execution or whole-corpus timing",
    }
    coverage = []
    for enabled in (False, True):
        started = perf_counter()
        plan = BatchPlanner(
            selection,
            1,
            bucket_by_length=True,
            shape_run_batches=1,
            storage_locality=enabled,
        )
        batches = plan.batches(config["seed"])
        planning_seconds = perf_counter() - started
        raw.cached, raw.cached_index = None, None
        raw.verified.clear()
        seen = []
        started = perf_counter()
        with patch("sharded_records.load_shard", wraps=load_shard) as loader:
            for batch in batches:
                for record in batch:
                    record["batch"]["context"][0, 0, 0].item()
                    seen.append(record["id"])
        report["storage_locality" if enabled else "global_shuffle"] = {
            "planning_seconds": planning_seconds,
            "tensor_read_seconds": perf_counter() - started,
            "tensor_shard_loads": loader.call_count,
        }
        coverage.append(sorted(seen))
    if coverage[0] != coverage[1]:
        raise RuntimeError("Storage-local planning changed record coverage")
    report["same_record_coverage"] = True
    return report


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default=str(ROOT / "configs/config.json"))
    parser.add_argument("--shards", type=int, default=2)
    parser.add_argument("--records-per-shard", type=int, default=8)
    parser.add_argument(
        "--output",
        default=str(ROOT / "docs/audits/2026-11-08/pipeline_scaling_sample.json"),
    )
    args = parser.parse_args()
    if args.shards < 1 or args.records_per_shard < 1:
        parser.error("sample sizes must be positive")
    torch.set_num_threads(1)
    report = run(load_config(args.config), args.shards, args.records_per_shard)
    write_json(Path(args.output), report)
    print(json.dumps(report))

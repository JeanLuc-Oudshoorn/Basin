"""Bounded CPU smoke: profile real holdout metadata and screen three real tables."""
# ruff: noqa: E402

import argparse
import shutil
import sys
from pathlib import Path
from time import perf_counter

import torch

ROOT = Path(__file__).resolve().parents[1]
for stage in (
    "02_data_validation",
    "03_transforming_input_data",
    "05_model_training",
    "06_holdout_test",
):
    sys.path.insert(0, str(ROOT / "src" / stage))
from benchmarks import sample_holdout
from holdout_test import checkpoint_test_records, populate_test_schema
from transform import load_config, location, run_context, write_json
from validate_generated_data import validate


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--startup-only", action="store_true")
    args = parser.parse_args()
    config = load_config()
    output = ROOT / "docs/audits/2026-10-30/bolt_screening_startup"
    started = perf_counter()
    metadata = torch.load(
        location(config, "training") / "model.pt",
        map_location="cpu",
        weights_only=True,
        mmap=True,
    )
    loaded = perf_counter()
    records, selection = sample_holdout(checkpoint_test_records(metadata), config)
    populate_test_schema(records, metadata)
    ready = perf_counter()
    write_json(
        output / f"startup_probe_{sys.platform}.json",
        {
            "status": "passed",
            "runtime": f"{sys.platform} CPU; metadata/schema only, not GPU model initialization",
            "checkpoint_metadata_seconds": loaded - started,
            "selection_and_schema_seconds": ready - loaded,
            "total_seconds": ready - started,
            "available_tables": selection["available_holdout_tables"],
            "selected_tables": len(records),
            "training_corpus_opened": False,
            "model_weights_initialized": False,
        },
    )
    if args.startup_only:
        return
    raw = ROOT / ".cache/bolt_screening_smoke/raw"
    for index, record in enumerate(records[:3]):
        destination = raw / f"table_{index:03d}/data.parquet"
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(ROOT / record["source"], destination)
    del metadata
    config["paths"]["raw"] = str(raw)
    config["data"].pop("generated_dates", None)
    config["validation"].update(
        metrics_path=str(output / "screening/metrics.parquet"), run_folders=False
    )
    # This historical verification script deliberately exercises the legacy engine.
    config["validation"]["engine"] = "chronos_bolt"
    config["validation"]["chronos_bolt"] = {
        "model_id": "amazon/chronos-bolt-small",
        "revision": "772f3d25d38aec6d914c8949dab4462e2d46f5d8",
        "device": "cpu",
        "batch_size": 64,
        "context_length": 2048,
        "cpu_threads": 4,
        "cache_dir": ".cache/huggingface",
    }
    with run_context(config, "validate"):
        results = validate(config)
    if not results.status.eq("ok").all():
        raise RuntimeError("Chronos-Bolt smoke failed; inspect screening metrics")
    print(
        f"PASS: startup metadata/schema probe and real Chronos-Bolt screening on {len(results)} tables"
    )


if __name__ == "__main__":
    main()

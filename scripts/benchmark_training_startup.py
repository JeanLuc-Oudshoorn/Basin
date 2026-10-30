"""Bounded real-corpus startup/feature benchmark; never initialize or train a model."""
# ruff: noqa: E402

import argparse
import copy
import random
import sys
import tempfile
from pathlib import Path
from time import perf_counter
from unittest.mock import patch

import torch
from loguru import logger

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src/05_model_training"))
sys.path.insert(0, str(ROOT / "src/04_creating_embeddings"))
sys.path.insert(0, str(ROOT / "src/03_transforming_input_data"))
from embed import PatchEncoder
from token_inventory import prepare_token_inventory
from training_protocol import split_datasets
from transform import load_config, location, read_artifact, write_json


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default=str(ROOT / "configs/config.json"))
    parser.add_argument("--samples", type=int, default=256)
    parser.add_argument(
        "--output",
        default=str(ROOT / "docs/audits/2026-10-27/training_startup/benchmark.json"),
    )
    args = parser.parse_args()
    if args.samples < 1:
        parser.error("samples must be positive")
    config = copy.deepcopy(load_config(args.config))
    torch.set_num_threads(config["training"].get("cpu_threads", 4))
    logger.remove()
    logger.add(sys.stderr, level="INFO")
    output = Path(args.output).resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    started = perf_counter()
    data = read_artifact(location(config, "embedded") / "embeddings.pt", config)
    load_seconds = perf_counter() - started
    started = perf_counter()
    groups = split_datasets(data["records"], config)
    split_seconds = perf_counter() - started
    # An absent cache is the important regression case. Keep all benchmark writes
    # isolated from the user's training outputs and use no GPU or optimizer.
    with tempfile.TemporaryDirectory(
        prefix="startup_benchmark_", dir=output.parent
    ) as temporary:
        config["paths"]["training"] = temporary
        config["training"]["token_inventory"] = {"mode": "cached"}
        started = perf_counter()
        with patch.object(
            PatchEncoder,
            "tokenize_next",
            side_effect=AssertionError("startup must not tokenize corpus"),
        ):
            inventory = prepare_token_inventory(data, config, groups=groups)
        inventory_seconds = perf_counter() - started
        if inventory is not None:
            raise RuntimeError("Expected a cold-cache benchmark")
    # Reproducible benchmarking, not security-sensitive randomness.
    sample = random.Random(config["seed"]).sample(  # nosec B311
        groups[0], min(args.samples, len(groups[0]))
    )
    started = perf_counter()
    feature_elements = 0
    for record in sample:
        prepared = record["prepared"]
        feature_elements += prepared["features"].numel()
    prepare_seconds = perf_counter() - started
    write_json(
        output,
        {
            "status": "passed",
            "gpu_initialized": False,
            "training_started": False,
            "windows": len(data["records"]),
            "training_windows": len(groups[0]),
            "validation_tables": len(groups[1]),
            "test_tables": len(groups[2]),
            "load_and_checksum_seconds": load_seconds,
            "split_seconds": split_seconds,
            "cold_cache_inventory_seconds": inventory_seconds,
            "startup_corpus_tokenizer_calls": 0,
            "sampled_training_windows": len(sample),
            "feature_elements_prepared": feature_elements,
            "feature_preparation_seconds": prepare_seconds,
            "feature_windows_per_second": len(sample) / prepare_seconds,
            "cpu_threads": torch.get_num_threads(),
            "note": "Feature throughput excludes RWKV forward/backward and exact uniqueness counting. Artifact loading and checksums are reported separately.",
        },
    )


if __name__ == "__main__":
    main()

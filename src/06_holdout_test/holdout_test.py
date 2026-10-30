"""Stage 6: evaluate the latest exported trained model on its saved test split."""
# ruff: noqa: E402

import argparse
import sys
from pathlib import Path

import torch
from loguru import logger

ROOT = Path(__file__).resolve().parents[2]
for stage in (
    "03_transforming_input_data",
    "04_creating_embeddings",
    "05_model_training",
):
    sys.path.insert(0, str(ROOT / "src" / stage))
from benchmarks import benchmark_names, sample_holdout
from holdout import evaluate_holdout, holdout_directory
from performance import configure_cuda_memory
from train import RWKVForecaster, launch_wsl
from training_log import emit_training_event, logged_training
from transform import (
    file_digest,
    load_config,
    location,
    read_artifact,
    run_context,
    timed,
    write_json,
)
from variate_attention import configure_variate_execution


def select_test_records(metadata, data):
    """Replay the exported split; never draw a fresh split during testing."""
    if "dataset_split" not in metadata:
        raise ValueError(
            "Trained model has no saved holdout split; use a model exported by train.py"
        )
    split = metadata["dataset_split"]
    sources = set(split["test_sources"])
    records = [
        r
        for r in data["records"]
        if r["split"] == "evaluation" and r["source"] in sources
    ]
    if (
        not sources
        or len(records) != len(sources)
        or {r["source"] for r in records} != sources
    ):
        raise ValueError(
            "Checkpoint holdout sources missing or duplicated in embedding artifact"
        )
    expected_hashes = split["source_sha256"]
    if any(r["source_sha256"] != expected_hashes[r["source"]] for r in records):
        raise ValueError(
            "Holdout record checksum differs from the trained model's saved split"
        )
    return records


@logged_training("holdout_test")
def run(config):
    benchmark_names(config)
    checkpoint = location(config, "training") / "model.pt"
    logger.info("HOLDOUT model={} output={}", checkpoint, holdout_directory(config))
    if not checkpoint.is_file():
        raise FileNotFoundError(
            f"No exported trained model at {checkpoint}; finish train.py first"
        )
    torch.set_num_threads(config["training"].get("cpu_threads", 4))
    configure_cuda_memory(config["training"])
    data = read_artifact(location(config, "embedded") / "embeddings.pt", config)
    metadata = read_artifact(checkpoint, config)
    if metadata["embedding_sha256"] != file_digest(
        location(config, "embedded") / "embeddings.pt"
    ):
        raise ValueError(
            "Embedding artifact changed since training; select the original model and data"
        )
    records, selection = sample_holdout(select_test_records(metadata, data), config)
    logger.info(
        "Holdout selection: {} / {} tables (requested {}%, seed={})",
        len(records),
        selection["available_holdout_tables"],
        selection["requested_sample_percent"],
        selection["sample_seed"],
    )
    del data
    summary = metadata.get("training_summary", {})
    model_identity = {
        "checkpoint_path": str(checkpoint),
        "checkpoint_sha256": file_digest(checkpoint),
        "selected_epoch": summary.get("best_epoch"),
        "selected_step": summary.get("best_step"),
        "test_tables": len(records),
    }
    del metadata
    with timed("load exported trained model", cuda=torch.cuda.is_available()):
        model = RWKVForecaster.load(checkpoint).eval()
        configure_variate_execution(model, config["training"], ROOT)
    directory = holdout_directory(config)
    directory.mkdir(parents=True, exist_ok=True)
    write_json(directory / "model_used.json", model_identity)
    write_json(directory / "sample_selection.json", selection)
    emit_training_event("holdout_start", **model_identity)
    metrics = evaluate_holdout(model, records, config)
    metrics["holdout_selection"] = {
        k: v for k, v in selection.items() if k != "selected_sources"
    }
    write_json(directory / "test_metrics.json", metrics)
    emit_training_event("out_of_sample_evaluation", out_of_sample_metrics=metrics)
    logger.success("Holdout test complete: {}", directory)
    return metrics


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default=str(ROOT / "configs/config.json"))
    args = parser.parse_args()
    config = load_config(args.config)
    with run_context(config, "holdout_test"):
        if launch_wsl(__file__, args.config):
            return
        if not torch.cuda.is_available() or not torch.cuda.is_bf16_supported():
            raise RuntimeError("Use the configured WSL CUDA bf16 runtime")
        with timed("HOLDOUT TEST stage", cuda=True):
            run(config)


if __name__ == "__main__":
    main()

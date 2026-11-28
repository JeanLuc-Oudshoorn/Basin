"""Stage 6: evaluate exported weights on a saved, reproducible test split."""
# ruff: noqa: E402

import argparse
import hashlib
import json
import re
import sys
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from time import perf_counter

import pyarrow.parquet as pq
import torch
from loguru import logger

ROOT = Path(__file__).resolve().parents[2]
for stage in (
    "03_transforming_input_data",
    "04_patch_indexing",
    "05_model_training",
):
    sys.path.insert(0, str(ROOT / "src" / stage))
from benchmarks import benchmark_names, sample_holdout
from fast_inference import configure_fast_inference, inference_execution_summary
from holdout import evaluate_holdout
from holdout_runs import holdout_directory, new_holdout_run
from performance import configure_cuda_memory
from result_upload import make_result_store, upload_run
from startup_cache import (
    artifact_digest,
    atomic_json,
    cached_family_exclusions,
    load_split,
    overlapping_sources,
    weights_digest,
)
from train import RWKVForecaster, launch_wsl
from training_log import emit_training_event, logged_training
from transform import (
    load_config,
    location,
    run_context,
    signature,
    timed,
    write_json,
)
from variate_attention import configure_variate_execution


def holdout_checkpoint(config):
    """Prefer the current recovery snapshot over an older final export."""
    option = config.get("holdout", {}).get("checkpoint")
    if option:
        return ROOT / option
    directory = location(config, "training")
    exported, recovery = directory / "model.pt", directory / "latest_checkpoint.pt"
    if recovery.is_file() and (
        not exported.is_file()
        or recovery.stat().st_mtime_ns > exported.stat().st_mtime_ns
    ):
        return recovery
    return exported


def load_holdout_metadata(checkpoint, config):
    """Use recovery's best weights and saved split without resuming training."""
    metadata = torch.load(checkpoint, map_location="cpu", weights_only=True, mmap=True)
    if "best_state_dict" not in metadata:
        return metadata
    identity = metadata["identity"]
    if identity.get("input_signature") != signature(config):
        raise ValueError(
            f"Training checkpoint {checkpoint} does not match the configured model/data "
            "policy. Use its original config or set holdout.checkpoint to an exported model.pt."
        )
    split_path = checkpoint.parent / "dataset_split.json"
    if not split_path.is_file():
        raise FileNotFoundError(
            f"Training checkpoint requires its saved split: {split_path}"
        )
    if split_path.stat().st_mtime_ns > checkpoint.stat().st_mtime_ns:
        raise ValueError(
            f"Saved split is newer than the training checkpoint: {split_path}"
        )
    result = {
        "config": identity["model"],
        "state_dict": metadata["best_state_dict"],
        "pipeline_config": config,
        "training_summary": {
            "exported_epoch": metadata["early_stopping"]["best_epoch"],
            "exported_step": metadata["best_step"],
        },
        "checkpoint_weights": "best_state_dict",
        "training_split_path": str(split_path),
    }
    # Release optimizer/current weights before loading the potentially large split.
    del metadata
    logger.info("HOLDOUT reading saved training split={}", split_path)
    split = load_split(split_path)
    from training_protocol import holdout_request

    tables, fraction = holdout_request(config)
    if (
        split.get("seed") != config["seed"]
        or split.get("requested_test_tables") != tables
        or (tables is None and split.get("requested_test_fraction") != fraction)
    ):
        raise ValueError(
            "Saved dataset split does not match the training checkpoint policy"
        )
    if (
        split.get("input_signature", identity["input_signature"])
        != identity["input_signature"]
    ):
        raise ValueError("Saved dataset split belongs to a different training run")
    result["dataset_split"] = split
    checkpoint_test_records(result)
    dates = config["data"].get("generated_dates")
    if dates and any(
        Path(source).parent.parent.name not in dates for source in split["test_sources"]
    ):
        raise ValueError(
            "Saved holdout dates do not match the training checkpoint policy"
        )
    # Keep membership lists for leakage checks; retain only the holdout maps.
    sources = split["test_sources"]
    for key in ("source_sha256", "source_keys"):
        values = split.get(key, {})
        split[key] = {source: values[source] for source in sources if source in values}
    return result


def checkpoint_test_records(metadata):
    """Replay the saved split without opening or hashing the training corpus."""
    split = metadata.get("dataset_split", {})
    sources = split.get("test_sources", [])
    if not sources or len(sources) != len(set(sources)):
        raise ValueError("Checkpoint must contain unique saved holdout sources")
    if overlapping_sources(split, "train", sources):
        raise ValueError("Checkpoint training and holdout sources overlap")
    if overlapping_sources(split, "validation", sources):
        raise ValueError("Checkpoint validation and holdout sources overlap")
    hashes, keys = split.get("source_sha256", {}), split.get("source_keys", {})
    if any(source not in hashes for source in sources):
        raise ValueError("Checkpoint holdout source checksums are missing")
    return [
        {
            "source": source,
            "source_key": keys.get(source, source),
            "source_sha256": hashes[source],
            "split": "evaluation",
        }
        for source in sources
    ]


def evaluation_records(model_metadata, split_metadata):
    """Allow a separate saved split while rejecting evaluated-model leakage."""
    from augmentation_families import excluded_evaluation_sources

    records = checkpoint_test_records(split_metadata)
    training_sources = model_metadata.get("dataset_split", {}).get("train_sources")
    model_split = model_metadata.get("dataset_split", {})
    if (
        model_metadata is not split_metadata
        and training_sources is None
        and not model_split.get("_has_train_membership")
    ):
        raise ValueError("Evaluated model must record its training sources")
    if overlapping_sources(
        model_split, "train", [record["source"] for record in records]
    ):
        raise ValueError(
            "Selected holdout sources overlap evaluated model training sources"
        )
    if overlapping_sources(
        model_split, "validation", [record["source"] for record in records]
    ):
        raise ValueError(
            "Selected holdout sources overlap evaluated model validation sources"
        )
    excluded = cached_family_exclusions(
        [record["source"] for record in records], ROOT, excluded_evaluation_sources
    )
    records = [record for record in records if record["source"] not in excluded]
    if not records:
        raise ValueError(
            "No independent holdout datasets remain after augmentation-family exclusion"
        )
    return records


def populate_test_schema(records, metadata, *, workers=8):
    """Read selected Parquet headers; use the model's original column policy."""
    data = metadata["pipeline_config"]["data"]
    policy = hashlib.sha256(json.dumps(data, sort_keys=True).encode()).hexdigest()
    cache_path = ROOT / ".cache" / "holdout_schemas" / f"{policy}.json"
    try:
        cached = json.loads(cache_path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        cached = {}
    pending = []
    for record in records:
        entry = cached.get(record["source"])
        if (
            entry
            and record.get("source_sha256")
            and entry["sha256"] == record["source_sha256"]
        ):
            record.update(entry["schema"])
        else:
            pending.append(record)
    logger.info(
        "Holdout schemas: {:,} cached; {:,} headers to read (workers={})",
        len(records) - len(pending),
        len(pending),
        workers,
    )

    def populate(record):
        path = ROOT / record["source"]
        if not path.is_file():
            raise FileNotFoundError(
                f"Saved holdout dataset is missing: {path}. Holdout sources come from "
                "the saved checkpoint split, not data.generated_dates in today's config. "
                "Restore the original dataset or select the current training checkpoint "
                "with holdout.checkpoint; changing dates cannot relocate a saved split."
            )
        source = pq.ParquetFile(path)
        columns = source.schema_arrow.names
        excluded = [
            name
            for name in columns
            if any(
                re.search(pattern, name) for pattern in data["excluded_column_patterns"]
            )
        ]
        targets = list(data["targets"])
        past = data["past_covariates"]
        if past == "auto":
            past = [
                name
                for name in columns
                if name not in targets + [data["time_column"]] + excluded
            ]
        if set(targets + list(past)) - set(columns) or set(targets + list(past)) & set(
            excluded
        ):
            raise ValueError(
                f"Holdout columns do not match saved model policy: {record['source']}"
            )
        record.update(
            targets=targets,
            roles={"targets": targets, "past": list(past), "known": []},
            source_rows=source.metadata.num_rows,
        )

    # WSL-to-Windows metadata reads have high latency. Bound concurrency instead
    # of opening every selected Parquet serially; no target values are loaded.
    started = reported = perf_counter()
    with ThreadPoolExecutor(max_workers=workers) as executor:
        for completed, _ in enumerate(executor.map(populate, pending), 1):
            now = perf_counter()
            if now - reported >= 30:
                logger.info(
                    "Holdout schemas {:,}/{:,} | elapsed={:.1f}s | ETA={:.1f}s",
                    completed,
                    len(pending),
                    now - started,
                    (now - started) / completed * (len(pending) - completed),
                )
                reported = now
    for record in pending:
        if record.get("source_sha256"):
            cached[record["source"]] = {
                "sha256": record["source_sha256"],
                "schema": {
                    key: record[key] for key in ("targets", "roles", "source_rows")
                },
            }
    if pending:
        cache_path.parent.mkdir(parents=True, exist_ok=True)
        atomic_json(cache_path, cached)
    return records


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
            "Checkpoint holdout sources missing or duplicated in patch index"
        )
    expected_hashes = split["source_sha256"]
    if any(r["source_sha256"] != expected_hashes[r["source"]] for r in records):
        raise ValueError(
            "Holdout record checksum differs from the trained model's saved split"
        )
    eligible = {r["source"] for r in evaluation_records(metadata, metadata)}
    return [r for r in records if r["source"] in eligible]


@logged_training("holdout_test")
def run(config, store=None):
    """Test the trained model; with a `store`, copy the finished run folder to it."""
    startup_started = perf_counter()
    benchmark_names(config)
    checkpoint = holdout_checkpoint(config)
    logger.info("HOLDOUT model={} output={}", checkpoint, holdout_directory(config))
    if not checkpoint.is_file():
        raise FileNotFoundError(
            f"No trained model at {checkpoint}; run train.py to save a checkpoint first"
        )
    torch.set_num_threads(config["training"].get("cpu_threads", 4))
    configure_cuda_memory(config["training"])
    with timed("holdout checkpoint metadata (no training corpus)"):
        metadata = load_holdout_metadata(checkpoint, config)
    split_option = config.get("holdout", {}).get("split_checkpoint")
    split_checkpoint = ROOT / split_option if split_option else checkpoint
    split_metadata = metadata
    if split_checkpoint.resolve() != checkpoint.resolve():
        logger.info("HOLDOUT split checkpoint={}", split_checkpoint)
        # A model trained on other tables (an annealing experiment) is tested on
        # the holdout of the run it came from. That run's checkpoint is checked
        # against that run's configuration, named by holdout.split_config.
        split_config_option = config.get("holdout", {}).get("split_config")
        split_config = (
            load_config(split_config_option) if split_config_option else config
        )
        with timed("load separate holdout split metadata"):
            split_metadata = load_holdout_metadata(split_checkpoint, split_config)
    records, selection = sample_holdout(
        evaluation_records(metadata, split_metadata), config
    )
    selection["augmentation_family_excluded_tables"] = (
        len(split_metadata["dataset_split"]["test_sources"])
        - selection["available_holdout_tables"]
    )
    with timed("read selected holdout schemas"):
        populate_test_schema(records, metadata)
    logger.info(
        "Holdout selection: {} / {} tables (requested {}%, seed={})",
        len(records),
        selection["available_holdout_tables"],
        selection["requested_sample_percent"],
        selection["sample_seed"],
    )
    summary = metadata.get("training_summary", {})
    with timed("fingerprint evaluated weights for holdout provenance"):
        recovery = metadata.get("checkpoint_weights") == "best_state_dict"
        checkpoint_digest = None if recovery else artifact_digest(checkpoint)
        selected_weights_digest = (
            weights_digest(checkpoint, metadata["state_dict"], field="best_state_dict")
            if recovery
            else None
        )
        split_digest = (
            checkpoint_digest
            if split_metadata is metadata
            else (
                None
                if split_metadata.get("checkpoint_weights") == "best_state_dict"
                else artifact_digest(split_checkpoint)
            )
        )
    model_identity = {
        "checkpoint_path": str(checkpoint),
        "checkpoint_sha256": checkpoint_digest,
        "evaluated_weights_sha256": selected_weights_digest,
        "checkpoint_digest_scope": "selected tensor names, dtypes, shapes and bytes; optimizer excluded"
        if recovery
        else "complete exported checkpoint file",
        "split_checkpoint_path": str(split_checkpoint),
        "split_checkpoint_sha256": split_digest,
        "selected_epoch": summary.get("exported_epoch", summary.get("best_epoch")),
        "selected_step": summary.get("exported_step", summary.get("best_step")),
        "test_tables": len(records),
        "checkpoint_weights": metadata.get("checkpoint_weights", "state_dict"),
    }
    if "training_split_path" in metadata:
        model_identity["training_split_path"] = metadata["training_split_path"]
        model_identity["training_split_sha256"] = metadata["dataset_split"].get(
            "_split_sha256"
        ) or artifact_digest(Path(metadata["training_split_path"]))
    load_started = perf_counter()
    with timed("load exported trained model", cuda=torch.cuda.is_available()):
        model = RWKVForecaster.load(checkpoint, checkpoint=metadata).eval()
        configure_variate_execution(model, config["training"], ROOT)
        configure_fast_inference(model, config["training"])
    model_identity["load_seconds"] = perf_counter() - load_started
    model_identity["inference_execution"] = inference_execution_summary(model)
    model_identity["startup_seconds_before_benchmarks"] = (
        perf_counter() - startup_started
    )
    model_identity["source_policy"] = (
        "model checkpoint column policy; split checkpoint saved test sources; "
        "evaluated-model training overlap rejected; selected raw checksums verified during evaluation"
    )
    del metadata, split_metadata
    # Each run writes into its own new folder; earlier runs are never touched.
    directory, config = new_holdout_run(
        config,
        epoch=model_identity["selected_epoch"],
        step=model_identity["selected_step"],
    )
    logger.info("HOLDOUT run folder={}", directory)
    write_json(directory / "model_used.json", model_identity)
    write_json(directory / "sample_selection.json", selection)
    emit_training_event("holdout_start", **model_identity)
    metrics = evaluate_holdout(model, records, config)
    metrics["rwkv_model_load_seconds"] = model_identity["load_seconds"]
    metrics["holdout_selection"] = {
        k: v for k, v in selection.items() if k != "selected_sources"
    }
    write_json(directory / "test_metrics.json", metrics)
    emit_training_event("out_of_sample_evaluation", out_of_sample_metrics=metrics)
    logger.success("Holdout test complete: {}", directory)
    if store is not None:
        # After everything is on the local disk: a failed copy loses nothing.
        emit_training_event("holdout_results_uploaded", **upload_run(store, directory))
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
            # Only this entry point uploads; find out now, not after the
            # test, that the container cannot be written.
            store = make_result_store(config)
            if store is not None:
                store.preflight()
            run(config, store)


if __name__ == "__main__":
    main()

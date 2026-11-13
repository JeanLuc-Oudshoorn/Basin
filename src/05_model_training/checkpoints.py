"""Atomic, self-contained epoch checkpoints for exact training continuation."""

import hashlib
import json
import os
import random
from dataclasses import asdict
from pathlib import Path
from time import perf_counter
from uuid import uuid4

import numpy as np
import torch
from loguru import logger
from training_log import emit_training_event
from transform import ROOT, location, signature

FORMAT = "rwkv_epoch_checkpoint_v1"
TIMED_FORMAT = "rwkv_timed_checkpoint_v1"


def atomic_torch_save(payload, path):
    """Flush beside the destination, then replace; interrupted writes preserve it."""
    path = Path(path)
    temporary = path.with_name(f".{path.name}.{uuid4().hex}.tmp")
    try:
        with temporary.open("wb") as stream:
            torch.save(payload, stream)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def checkpoint_identity(model, records, validation_records, config, optimizer):
    """Bind continuation to the corpus, model and optimization policy, not paths."""
    from augmentation_families import POLICY

    training = config["training"]
    policy = {
        key: training.get(key, default)
        for key, default in (
            ("learning_rate", None),
            ("weight_decay", None),
            ("gradient_clip", None),
            ("train_embeddings", True),
            ("batch_size", 1),
            ("gradient_accumulation_steps", 1),
            ("shuffle", False),
            ("bucket_by_length", True),
            ("holdout_fraction", 0.1),
            ("scheduler", {"name": "constant"}),
            ("fused_optimizer", True),
        )
    }
    early = training.get("early_stopping", {})
    if config["data"].get("validation_policy") == "sampled_training_origins":
        policy["validation_split_policy"] = "dataset_disjoint_v1"
    # Preserve compatibility with old checkpoints using the original order.
    if training.get("shape_run_batches", 1) != 1:
        policy["shape_run_batches"] = training["shape_run_batches"]
    if training.get("storage_locality", False):
        policy["storage_locality"] = True
    policy["monitor"] = early.get("monitor", "loss")
    policy["min_delta"] = early.get("min_delta", 0.0)
    digest = hashlib.sha256()
    from sharded_records import RecordSelection

    compact = isinstance(records, RecordSelection)
    for group_name, group in (("train", records), ("validation", validation_records)):
        digest.update(group_name.encode())
        logger.info(
            "Checkpoint identity: hashing {} records={:,} streaming={}",
            group_name,
            len(group),
            compact,
        )
        ordered = (
            group
            if compact
            else sorted(
                group, key=lambda r: (r.get("source_key", r["source"]), r.get("id", ""))
            )
        )
        for record in ordered:
            if hasattr(record, "materialize"):
                record = record.materialize()
            digest.update(
                json.dumps(
                    {
                        "source": record.get("source_key", record["source"]),
                        **{
                            k: record.get(k)
                            for k in (
                                "id",
                                "source_sha256",
                                "cutoff",
                                "context_rows",
                                "label_rows",
                            )
                        },
                    },
                    sort_keys=True,
                ).encode()
            )
    identity = {
        "evaluation_family_policy": POLICY,
        "input_signature": signature(config),
        "model": asdict(model.config),
        "dataset_sha256": digest.hexdigest(),
        "optimization": policy,
        "optimizer_class": type(optimizer).__qualname__,
    }
    if compact:
        identity["dataset_identity_policy"] = "ordered_stream_v1"
    return identity


def ordered_record_digest(records, *, include_storage=False):
    """Same SHA256 as newline-joining IDs, without a corpus-sized string."""
    digest = hashlib.sha256()
    first = True
    for record in records:
        storage_key = getattr(record, "storage_key", None) if include_storage else None
        if hasattr(record, "materialize"):
            record = record.materialize()
        if not first:
            digest.update(b"\n")
        digest.update(str(record.get("id", record["source"])).encode())
        if include_storage:
            digest.update(b"\0")
            digest.update(str(storage_key).encode())
        first = False
    return digest.hexdigest()


def rng_state():
    numpy_state = np.random.get_state()
    return {
        "torch": torch.get_rng_state(),
        "cuda": torch.cuda.get_rng_state_all() if torch.cuda.is_initialized() else [],
        "python": random.getstate(),
        "numpy": (numpy_state[0], numpy_state[1].tolist(), *numpy_state[2:]),
    }


def restore_rng(state):
    torch.set_rng_state(state["torch"])
    if state["cuda"]:
        if (
            not torch.cuda.is_available()
            or len(state["cuda"]) != torch.cuda.device_count()
        ):
            raise ValueError(
                "Checkpoint CUDA RNG state requires the same number of CUDA devices"
            )
        torch.cuda.set_rng_state_all(state["cuda"])
    random.setstate(state["python"])
    numpy_state = state["numpy"]
    np.random.set_state(
        (numpy_state[0], np.asarray(numpy_state[1], dtype=np.uint32), *numpy_state[2:])
    )


def resume_path(config):
    requested = config["training"].get("resume", "auto")
    if requested is False:
        return None
    if requested is True or (
        isinstance(requested, str) and requested.strip().lower() in ("true", "false")
    ):
        raise ValueError(
            "training.resume: use false (without quotes) to start fresh, "
            "'auto' to resume the latest checkpoint, or an explicit checkpoint path"
        )
    if requested == "auto":
        path = location(config, "training") / "latest_checkpoint.pt"
        return path if path.is_file() else None
    path = Path(requested)
    path = path if path.is_absolute() else ROOT / path
    if not path.is_file():
        raise FileNotFoundError(f"Training resume checkpoint does not exist: {path}")
    return path


def restore_checkpoint(
    path, model, optimizer, scheduler, identity, schedule_plan, best_path
):
    started = perf_counter()
    logger.info("Loading resumable epoch checkpoint {}", path)
    # Linux mmap keeps large optimizer checkpoints from being duplicated in RAM.
    # Windows cannot replace a file while its tensor storage is memory mapped.
    state = torch.load(
        path, map_location="cpu", weights_only=True, mmap=os.name != "nt"
    )
    if state.get("format") not in (FORMAT, TIMED_FORMAT):
        raise ValueError(
            "Resume requires a full epoch checkpoint; weight-only snapshots cannot restore training"
        )
    saved_identity = dict(state["identity"])
    # Older one-patch checkpoints predate direct-lead options. Only their exact
    # legacy defaults may be filled in; a 16-lead model still cannot resume them.
    saved_identity["model"] = dict(saved_identity["model"])
    saved_identity["model"].setdefault("output_patches", 1)
    saved_identity["model"].setdefault("multi_horizon_weight", 0.5)
    saved_identity["model"].setdefault("horizon_loss_weights", None)
    # Physical microbatch limits may need lowering after an activation OOM.
    # They do not change the logical batch cursor or point-weighted objective.
    # Different tensor sizes can change floating-point rounding/RNG assignment;
    # this is a safe continuation, not a promise of bitwise-identical results.
    compared_identity = dict(identity)
    changed_execution = {}
    # Deliberate one-way precision upgrade: the model, data, cursor and AdamW
    # hyperparameters still have to match. The optimizer promotes saved moments
    # and initializes masters from the model weights restored below.
    if (
        saved_identity.get("optimizer_class") == "AdamW"
        and compared_identity.get("optimizer_class") == "FP32MasterAdamW"
        and "fp32_master_weights" not in state["optimizer_state_dict"]
    ):
        saved_identity["optimizer_class"] = "FP32MasterAdamW"
        changed_execution["optimizer_precision"] = {
            "saved": "model parameter dtype",
            "current": "FP32 master weights and moments",
        }
    if state.get("format") == TIMED_FORMAT:
        for compared in (saved_identity, compared_identity):
            if "timed_policy" in compared:
                compared["timed_policy"] = dict(compared["timed_policy"])
        for key in (
            "microbatch_size",
            "microbatch_token_limit",
            "checkpoint_interval_seconds",
            "validation_interval_seconds",
        ):
            old = saved_identity.get("timed_policy", {}).pop(key, None)
            new = compared_identity.get("timed_policy", {}).pop(key, None)
            if old != new:
                changed_execution[key] = {"saved": old, "current": new}
    if saved_identity != compared_identity:
        raise ValueError(
            "Resume checkpoint model, datasets or optimization settings differ; use training.resume=false for a new run"
        )
    if changed_execution:
        logger.warning(
            "Resuming with adjusted execution/cadence settings={}; logical batch "
            "cursor, weights and optimizer state are preserved; numerical results may differ",
            changed_execution,
        )
    previous_schedule = state["schedule_plan"]
    if schedule_plan["name"] == "warmup_cosine" and previous_schedule != schedule_plan:
        raise ValueError("Cannot change a warmup_cosine update budget when resuming")
    model.load_state_dict(state["state_dict"])
    optimizer.load_state_dict(state["optimizer_state_dict"])
    scheduler.load_state_dict(state["scheduler_state_dict"])
    # Best weights travel inside the resume artifact: no dependency on a possibly
    # newer external best_weights.pt after an interrupted epoch-end write.
    atomic_torch_save({"state_dict": state["best_state_dict"]}, best_path)
    restore_rng(state["rng_state"])
    metadata = {
        k: state[k]
        for k in (
            "epoch",
            "step",
            "points_processed",
            "windows_processed",
            "best_step",
            "early_stopping",
            "history",
            "validation_history",
        )
    }
    metadata["loop_state"] = state.get("loop_state")
    metadata["execution_changes"] = changed_execution
    cursor = state.get("loop_state")
    completed_epoch = cursor["completed_epochs"] if cursor else state["epoch"]
    next_epoch = (
        (cursor["epoch"] + int(completed_epoch == cursor["epoch"]))
        if cursor
        else state["epoch"] + 1
    )
    logger.info(
        "RESUME checkpoint={} completed_epoch={} global_step={} next_epoch={} next_batch_offset={}",
        path,
        completed_epoch,
        state["step"],
        next_epoch,
        cursor["next_batch_offset"] if cursor else 0,
    )
    emit_training_event(
        "checkpoint_resumed",
        global_step=state["step"],
        checkpoint_path=str(path),
        completed_epoch=completed_epoch,
        next_epoch=next_epoch,
        next_batch_offset=cursor["next_batch_offset"] if cursor else 0,
        checkpoint_load_seconds=perf_counter() - started,
    )
    return metadata


def save_checkpoint(
    path,
    model,
    optimizer,
    scheduler,
    identity,
    schedule_plan,
    early,
    history,
    validation_history,
    points_processed,
    windows_processed,
    best_step,
    best_path,
    improved,
    stop_reason,
    *,
    loop_state=None,
):
    started = perf_counter()
    logger.info("Writing resumable training checkpoint {}", path)
    current = model.state_dict()
    # torch.save deduplicates shared storage when the current epoch is also best.
    best = (
        current
        if improved or not best_path.is_file()
        else torch.load(
            best_path, map_location="cpu", weights_only=True, mmap=os.name != "nt"
        )["state_dict"]
    )
    epoch, step = (
        (loop_state["epoch"], loop_state["step"])
        if loop_state is not None
        else (history[-1]["epoch"], history[-1]["step"])
    )
    atomic_torch_save(
        {
            "format": TIMED_FORMAT if loop_state is not None else FORMAT,
            "loop_state": loop_state,
            "identity": identity,
            "state_dict": current,
            "best_state_dict": best,
            "optimizer_state_dict": optimizer.state_dict(),
            "scheduler_state_dict": scheduler.state_dict(),
            "schedule_plan": schedule_plan,
            "epoch": epoch,
            "step": step,
            "points_processed": points_processed,
            "windows_processed": windows_processed,
            "best_step": best_step,
            "early_stopping": {
                "best": early.best,
                "best_epoch": early.best_epoch,
                "bad_epochs": early.bad_epochs,
            },
            "history": history,
            "validation_history": validation_history,
            "rng_state": rng_state(),
            "stop_reason": stop_reason,
        },
        path,
    )
    logger.info(
        "CHECKPOINT saved={} epoch={} global_step={} bytes={:,} seconds={:.2f}",
        path,
        epoch,
        step,
        path.stat().st_size,
        perf_counter() - started,
    )
    emit_training_event(
        "timed_training_checkpoint_saved"
        if loop_state is not None
        else "epoch_checkpoint_saved",
        global_step=step,
        epoch=epoch,
        checkpoint_path=str(path),
        checkpoint_bytes=path.stat().st_size,
        resumable=True,
        checkpoint_save_seconds=perf_counter() - started,
    )

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
    policy["monitor"] = early.get("monitor", "loss")
    policy["min_delta"] = early.get("min_delta", 0.0)
    digest = hashlib.sha256()
    for group_name, group in (("train", records), ("validation", validation_records)):
        digest.update(group_name.encode())
        for record in sorted(
            group, key=lambda r: (r.get("source_key", r["source"]), r.get("id", ""))
        ):
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
    return {
        "input_signature": signature(config),
        "model": asdict(model.config),
        "dataset_sha256": digest.hexdigest(),
        "optimization": policy,
        "optimizer_class": type(optimizer).__qualname__,
    }


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
    if state.get("format") != FORMAT:
        raise ValueError(
            "Resume requires a full epoch checkpoint; weight-only snapshots cannot restore training"
        )
    if state["identity"] != identity:
        raise ValueError(
            "Resume checkpoint model, datasets or optimization settings differ; use training.resume=false for a new run"
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
    logger.info(
        "RESUME checkpoint={} completed_epoch={} global_step={} next_epoch={}",
        path,
        state["epoch"],
        state["step"],
        state["epoch"] + 1,
    )
    emit_training_event(
        "checkpoint_resumed",
        global_step=state["step"],
        checkpoint_path=str(path),
        completed_epoch=state["epoch"],
        next_epoch=state["epoch"] + 1,
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
):
    started = perf_counter()
    logger.info("Writing resumable epoch checkpoint {}", path)
    current = model.state_dict()
    # torch.save deduplicates shared storage when the current epoch is also best.
    best = (
        current
        if improved
        else torch.load(
            best_path, map_location="cpu", weights_only=True, mmap=os.name != "nt"
        )["state_dict"]
    )
    epoch, step = history[-1]["epoch"], history[-1]["step"]
    atomic_torch_save(
        {
            "format": FORMAT,
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
        "epoch_checkpoint_saved",
        global_step=step,
        epoch=epoch,
        checkpoint_path=str(path),
        checkpoint_bytes=path.stat().st_size,
        resumable=True,
        checkpoint_save_seconds=perf_counter() - started,
    )

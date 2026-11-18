"""Hourly validation/checkpointing with exact optimizer-update-boundary resume."""

import math
from contextlib import nullcontext
from datetime import datetime, timezone
from time import perf_counter
from uuid import uuid4

import pandas as pd
import torch
from batching import BatchPlanner
from checkpoint_upload import make_checkpoint_store
from checkpoint_writer import CheckpointWriter
from checkpoints import (
    atomic_torch_save,
    checkpoint_identity,
    ordered_record_digest,
    restore_checkpoint,
    restore_rng,
    resume_path,
    rng_state,
    save_checkpoint,
)
from learning_rate import make_scheduler, scheduler_snapshot
from loguru import logger
from performance import prepared_updates
from rolling_validation import validate_origins
from training_log import emit_training_event
from training_progress import duration
from transform import location, write_json


def fit_timed(
    model,
    records,
    validation_records,
    config,
    optimizer,
    validation_fn=None,
    *,
    clock=None,
):
    """Train; whatever happens, let a checkpoint in flight reach the disk."""
    writers = []
    try:
        return _fit_timed(
            model,
            records,
            validation_records,
            config,
            optimizer,
            validation_fn,
            clock=clock,
            writers=writers,
        )
    except BaseException:
        # Training failed or was interrupted. The checkpoint taken before that
        # is still valid: finish writing it, and keep the original error.
        for writer in writers:
            try:
                writer.finish()
            except Exception as error:
                logger.error("Checkpoint writer failed while stopping: {}", error)
        raise


def _fit_timed(
    model,
    records,
    validation_records,
    config,
    optimizer,
    validation_fn=None,
    *,
    clock=None,
    writers=None,
):
    from epoch_training import (
        EarlyStopping,
        accumulated_update,
        monitored_value,
        training_options,
    )
    from weight_average import WeightAverage, weight_average_options

    clock = clock or perf_counter
    run_started = clock()
    if not records or not validation_records:
        raise ValueError("Need both training records and sampled validation origins")
    validation_sources = {r["source"] for r in validation_records}
    if any(r["source"] in validation_sources for r in records):
        raise ValueError(
            "Validation tables must be dataset-disjoint from gradient training"
        )
    t = config["training"]
    options = training_options(t)
    checkpoint_interval = t.get("checkpoint_interval_seconds", 3600)
    validation_interval = t.get("validation_interval_seconds", 3600)
    if checkpoint_interval is None or validation_interval is None:
        raise ValueError("Hourly training requires checkpoint/validation intervals")
    run_seconds = None if t.get("max_run_hours") is None else t["max_run_hours"] * 3600
    validation_fn = validation_fn or (lambda m, r: validate_origins(m, r, config))
    directory = location(config, "training")
    directory.mkdir(parents=True, exist_ok=True)
    # Checkpoints are written, and optionally copied to cloud storage, on a
    # background thread. A store that cannot be reached stops the run here,
    # before any training time is spent.
    store = make_checkpoint_store(config)
    if store is not None:
        if not t.get("background_checkpoints", True):
            raise ValueError(
                "training.checkpoint_upload requires training.background_checkpoints"
            )
        store.preflight()
    writer = CheckpointWriter(store) if t.get("background_checkpoints", True) else None
    if writer is not None and writers is not None:
        writers.append(writer)
    first_upload_pending = store is not None
    log_every_tables = t.get("log_every_tables")
    best_path, latest_path = (
        directory / "best_weights.pt",
        directory / "latest_checkpoint.pt",
    )
    for stale in directory.glob(f".{latest_path.name}.*.tmp"):
        # A run that was killed while writing leaves its unfinished file here.
        # The checkpoint itself is only ever replaced by a complete one.
        logger.warning(
            "Removing unfinished checkpoint file of an earlier run: {}", stale
        )
        stale.unlink(missing_ok=True)
    accumulation = options["gradient_accumulation_steps"]
    logger.info(
        "Batch planning: {:,} training windows; storage_locality={}",
        len(records),
        t.get("storage_locality", False),
    )
    planner = BatchPlanner(
        records,
        options["batch_size"],
        bucket_by_length=t.get("bucket_by_length", True),
        shape_run_batches=t.get("shape_run_batches", 1),
        storage_locality=t.get("storage_locality", False),
    )

    def batches_for(epoch):
        return planner.batches(
            config["seed"] + epoch - 1 if t.get("shuffle", False) else None
        )

    batches = batches_for(1)
    updates_per_epoch = math.ceil(len(batches) / accumulation)
    total_budget = options["steps"]
    if options["epochs"] is not None:
        total_budget = (
            min(total_budget, options["epochs"] * updates_per_epoch)
            if total_budget is not None
            else options["epochs"] * updates_per_epoch
        )
    scheduler, schedule_plan = make_scheduler(optimizer, t, total_budget)
    early = EarlyStopping(options["patience_epochs"], options["min_delta"])
    average_options = weight_average_options(t)
    # Starts as a copy of the initial weights; a resume replaces it below.
    average = (
        WeightAverage(model, optimizer, **average_options) if average_options else None
    )
    identity = checkpoint_identity(
        model, records, validation_records, config, optimizer
    )
    # The cursor relies on the same ordered corpus and microbatch execution.
    identity["timed_policy"] = {
        "version": 1,
        "normalization_policy": "positive_finite_std_v1",
        "validation_sampling": t.get("validation_sampling", {}),
        "microbatch_size": t.get("microbatch_size", 4),
        "microbatch_token_limit": t.get("microbatch_token_limit", 1536),
        "validation_batch_size": t.get("validation_batch_size", 16),
        "checkpoint_interval_seconds": checkpoint_interval,
        "validation_interval_seconds": validation_interval,
        "dtype": str(getattr(model, "compute_dtype", next(model.parameters()).dtype)),
        "device_type": next(model.parameters()).device.type,
        "torch_version": str(torch.__version__),
        "compile_variate": t.get("compile_variate", False),
        "compile_variate_inference": t.get("compile_variate_inference", False),
        "ordered_records": ordered_record_digest(
            records, include_storage=t.get("storage_locality", False)
        ),
    }
    state = {
        "epoch": 1,
        "step": 0,
        "next_batch_offset": 0,
        "completed_epochs": 0,
        "active_training_seconds": 0.0,
        "last_checkpoint_seconds": 0.0,
        "last_validation_seconds": 0.0,
        "epoch_loss_sum": 0.0,
        "epoch_mass": 0.0,
        "epoch_windows": 0,
        "epoch_target_points_processed": 0,
        "epoch_started_active_seconds": 0.0,
        "pending_validation": False,
        "validation_checks": 0,
        "last_validation_step": None,
    }
    state["embedding_gradient_updates"] = 0
    history, validation_history = [], []
    points_processed = windows_processed = 0
    best_step = None
    reason = None
    resumed_from = None
    previous = resume_path(config)
    if previous is not None:
        restored = restore_checkpoint(
            previous,
            model,
            optimizer,
            scheduler,
            identity,
            schedule_plan,
            best_path,
            weight_average=average,
        )
        if restored.get("loop_state") is None:
            raise ValueError(
                "Hourly training requires a cursor checkpoint; start a new run for this full-table policy"
            )
        state = restored["loop_state"]
        if "validation_interval_seconds" in restored.get("execution_changes", {}):
            previous_pending = state["pending_validation"]
            state["pending_validation"] = (
                state["active_training_seconds"] - state["last_validation_seconds"]
                >= validation_interval
            )
            logger.info(
                "Validation cadence changed: interval={} pending {} -> {}; "
                "completed validation history and training cursor preserved",
                duration(validation_interval),
                previous_pending,
                state["pending_validation"],
            )
        memory = state.get("memory_execution", {})
        model._microbatch_limits = dict(memory.get("microbatch_limits", {}))
        if memory.get("checkpoint_all_layers"):
            model.checkpoint_token_threshold = 0
        # Older hourly checkpoints still have an exact batch cursor, but no
        # explicit per-epoch point counter. Do not invent that historical count.
        state.setdefault("epoch_target_points_processed", None)
        history, validation_history = (
            restored["history"],
            restored["validation_history"],
        )
        points_processed, windows_processed = (
            restored["points_processed"],
            restored["windows_processed"],
        )
        best_step = restored["best_step"]
        early.best, early.best_epoch, early.bad_epochs = (
            restored["early_stopping"][k] for k in ("best", "best_epoch", "bad_epochs")
        )
        resumed_from = str(previous)
        if state["epoch"] != 1:
            batches = batches_for(state["epoch"])
        if not 0 <= state["next_batch_offset"] <= len(batches) or (
            state["next_batch_offset"] != len(batches)
            and state["next_batch_offset"] % accumulation
        ):
            raise ValueError("Invalid saved batch cursor")
        logger.info(
            "TIMED RESUME epoch={} next_batch={}/{} update={} active_training={} pending_validation={}",
            state["epoch"],
            state["next_batch_offset"],
            len(batches),
            state["step"],
            duration(state["active_training_seconds"]),
            state["pending_validation"],
        )
    plan = {
        "protocol": "hourly_full_table_v1",
        "updates_per_epoch": updates_per_epoch,
        "planned_updates": total_budget,
        "checkpoint_interval_seconds": checkpoint_interval,
        "validation_interval_seconds": validation_interval,
        "interval_clock": "active training including batch preparation, excluding validation/checkpoint writes",
        "max_run_hours": t.get("max_run_hours"),
        "duration_clock": "per invocation elapsed loop time including validation and checkpointing; safe-boundary soft limit",
        "validation_tables": len({r["source"] for r in validation_records}),
        "validation_forecasts": len(validation_records),
        "validation_scope": "dataset_disjoint_reserved_tables_five_origins",
        "patience_unit": "validation_checks",
        "batch_size": options["batch_size"],
        "gradient_accumulation_steps": accumulation,
        "storage_locality": t.get("storage_locality", False),
        "batch_shape_plan_reused_across_epochs": True,
        "scheduler": schedule_plan,
        "monitor": options["monitor"],
        "weight_average": average_options,
        "validated_weights": "exponential moving average"
        if average is not None
        else "current training weights",
        "history_policy": "latest 1000 updates in checkpoint/JSON; full stream in timestamped training JSONL"
        if log_every_tables is None
        else f"latest 1000 updates in checkpoint/JSON; one summary per {log_every_tables} tables in timestamped training JSONL",
        "log_every_tables": log_every_tables,
        "checkpoint_writes": "background thread" if writer is not None else "blocking",
        "checkpoint_upload": store.describe() if store is not None else None,
    }
    write_json(directory / "training_plan.json", plan)
    emit_training_event("training_plan", **plan)

    def expired():
        return run_seconds is not None and clock() - run_started >= run_seconds

    def persist(stop_reason=None):
        nonlocal first_upload_pending
        optimizer.zero_grad(set_to_none=True)
        state["memory_execution"] = {
            "microbatch_limits": getattr(model, "_microbatch_limits", {}),
            "checkpoint_all_layers": getattr(model, "checkpoint_token_threshold", None)
            == 0,
        }
        state["last_checkpoint_seconds"] = state["active_training_seconds"]
        save_checkpoint(
            latest_path,
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
            early.best is None,
            stop_reason,
            loop_state=dict(state),
            weight_average=average,
            writer=writer,
        )
        if first_upload_pending:
            # Prove the whole path once, with the real file, before relying on
            # it for the rest of the run: wait for this checkpoint to be
            # written, uploaded and verified, and stop if that fails.
            writer.finish(require_upload=True)
            first_upload_pending = False
        write_json(directory / "loss_history.json", history)
        write_json(
            directory / "learning_rate_scheduler.json",
            scheduler_snapshot(scheduler, schedule_plan),
        )
        emit_training_event(
            "timed_checkpoint_saved",
            global_step=state["step"],
            epoch=state["epoch"],
            next_batch_offset=state["next_batch_offset"],
            epoch_windows_processed=state["epoch_windows"],
            epoch_target_points_processed=state["epoch_target_points_processed"],
            active_training_seconds=state["active_training_seconds"],
            pending_validation=state["pending_validation"],
            stop_reason=stop_reason,
            checkpoint_write="background" if writer is not None else "blocking",
        )

    def evaluate(*, baseline=False):
        nonlocal best_step, reason
        # With averaging on, validate and save as best the averaged weights:
        # they are what gets exported. The training weights return afterwards.
        with average.applied() if average is not None else nullcontext():
            before = rng_state()
            was_training = model.training
            try:
                result = dict(validation_fn(model, validation_records))
            finally:
                model.train(was_training)
                restore_rng(before)
            if not all(math.isfinite(result[k]) for k in ("loss", "rmse")):
                raise FloatingPointError("Nonfinite timed validation metrics")
            monitored = monitored_value(result, options["monitor"])
            improved, stopped = early.observe(
                monitored, 0 if baseline else state["epoch"]
            )
            if improved:
                best_step = state["step"]
                if writer is not None:
                    # A checkpoint still being written may be reading this file.
                    writer.finish()
                atomic_torch_save({"state_dict": model.state_dict()}, best_path)
        rows = result.pop("tables", [])
        index = 0 if baseline else state["validation_checks"] + 1
        completed_at = datetime.now(timezone.utc)
        # Retries after a crash and intentional fresh runs in the same output
        # directory must never overwrite an earlier completed validation.
        name = (
            f"validation_check_{index:05d}_step_{state['step']:09d}_"
            f"{completed_at.strftime('%Y%m%dT%H%M%S_%fZ')}_{uuid4().hex[:8]}"
        )
        path = directory / f"{name}.parquet"
        pd.DataFrame(rows).to_parquet(path, index=False)
        if not baseline:
            state["validation_checks"] += 1
            if isinstance(scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau):
                scheduler.step(monitored)
        row = {
            "epoch": 0 if baseline else state["epoch"],
            "step": state["step"],
            "validation_check": index,
            "phase": "before_training" if baseline else "hourly",
            "completed_at_utc": completed_at.isoformat(),
            "completed_epoch": state["next_batch_offset"] == len(batches),
            "training_loss": state["epoch_loss_sum"] / state["epoch_mass"]
            if state["epoch_mass"]
            else None,
            **result,
            "monitor": options["monitor"],
            "monitored_value": monitored,
            "validated_weights": plan["validated_weights"],
            "table_metrics_path": str(path),
            "improved": improved,
            "bad_epochs": early.bad_epochs,
            "in_sample": False,
            "dataset_disjoint": True,
            "forecast_count": len(validation_records),
            "table_count": plan["validation_tables"],
            "active_training_seconds": state["active_training_seconds"],
            "epoch_windows_processed": state["epoch_windows"],
            "epoch_target_points_processed": state["epoch_target_points_processed"],
            "next_batch_offset": state["next_batch_offset"],
            "next_learning_rate": optimizer.param_groups[0]["lr"],
        }
        write_json(directory / f"{name}.json", row)
        validation_history.append(row)
        # Replace the cumulative plotting index atomically; every per-check
        # summary above remains independently available if the process dies.
        temporary_history = directory / f".validation_history_{uuid4().hex}.json"
        try:
            write_json(temporary_history, validation_history)
            temporary_history.replace(directory / "validation_history.json")
        finally:
            temporary_history.unlink(missing_ok=True)
        emit_training_event(
            "validation_baseline" if baseline else "validation_complete",
            global_step=state["step"],
            validation_loss=result["loss"],
            validation_rmse=result["rmse"],
            validation_seconds=result.get("seconds"),
            validation_tables=plan["validation_tables"],
            validation_forecasts=len(validation_records),
            validation_check=index,
            in_sample=False,
            dataset_disjoint=True,
        )
        logger.info(
            "VALIDATION check={} step={} {}={} | loss={} RMSE={} forecasts={} tables={} seconds={} improved={}",
            index,
            state["step"],
            options["monitor"],
            monitored,
            result["loss"],
            result["rmse"],
            len(validation_records),
            plan["validation_tables"],
            result.get("seconds"),
            improved,
        )
        state["last_validation_seconds"] = state["active_training_seconds"]
        state["last_validation_step"] = state["step"]
        state["pending_validation"] = False
        if stopped and not baseline:
            reason = "early_stopping"

    # With training.log_every_tables set, the journal and the console get one
    # summary per that many tables instead of one record per update.
    interval = {}

    def reset_interval():
        interval.clear()
        interval.update(
            tables=0,
            updates=0,
            loss_sum=0.0,
            mass=0.0,
            points=0,
            microbatches=0,
            oom_retries=0,
            preparation=0.0,
            optimizer=0.0,
            peak_gib=None,
            norm_max=0.0,
            last=None,
        )

    def add_to_interval(result, mass, learning_rate, progress):
        interval["tables"] += result["windows"]
        interval["updates"] += 1
        interval["loss_sum"] += result["loss"] * mass
        interval["mass"] += mass
        interval["points"] += result["target_points"]
        interval["microbatches"] += result["microbatches"]
        interval["oom_retries"] += result.get("oom_retries", 0)
        interval["preparation"] += result["preparation_wait_seconds"]
        interval["optimizer"] += result["optimizer_update_seconds"]
        interval["norm_max"] = max(interval["norm_max"], result["gradient_norm"])
        peak = result.get("cuda_peak_allocated_gib")
        if peak is not None:
            interval["peak_gib"] = max(interval["peak_gib"] or 0.0, peak)
        interval["last"] = (result, learning_rate, progress)

    def emit_interval():
        result, learning_rate, progress = interval["last"]
        emit_training_event(
            "metrics",
            global_step=state["step"],
            run_step=state["step"],
            epoch=state["epoch"],
            train_loss=interval["loss_sum"] / interval["mass"]
            if interval["mass"]
            else result["loss"],
            train_loss_scope="mean over the updates since the previous record",
            train_loss_last_update=result["loss"],
            learning_rate=learning_rate,
            gradient_norm=result["gradient_norm"],
            gradient_norm_max=interval["norm_max"],
            updates=interval["updates"],
            windows=interval["tables"],
            microbatches=interval["microbatches"],
            oom_retries=interval["oom_retries"],
            cuda_allocated_gib=result.get("cuda_allocated_gib"),
            cuda_reserved_gib=result.get("cuda_reserved_gib"),
            cuda_peak_allocated_gib=interval["peak_gib"],
            preparation_wait_seconds=interval["preparation"],
            optimizer_update_seconds=interval["optimizer"],
            embedding_shapes=result.get("embedding_shapes"),
            target_points=interval["points"],
            active_training_seconds=state["active_training_seconds"],
            training_target_points_processed=points_processed,
            log_every_tables=log_every_tables,
            **progress,
        )

    reset_interval()
    reported_first = False
    if early.bad_epochs >= early.patience:
        reason = "early_stopping"
    if previous is None and t.get("validate_before_training", True):
        evaluate(baseline=True)
        persist()
    model.train()
    while reason is None:
        if state["pending_validation"]:
            evaluate()
            persist(reason)
            if reason:
                break
        if expired():
            reason = "time_limit"
            break
        if total_budget is not None and state["step"] >= total_budget:
            reason = "step_limit"
            break
        if (
            options["epochs"] is not None
            and state["completed_epochs"] >= options["epochs"]
        ):
            reason = "epoch_limit"
            break
        if state["next_batch_offset"] == len(batches):
            state.update(
                epoch=state["epoch"] + 1,
                next_batch_offset=0,
                epoch_loss_sum=0.0,
                epoch_mass=0.0,
                epoch_windows=0,
                epoch_target_points_processed=0,
                epoch_started_active_seconds=state["active_training_seconds"],
            )
            batches = batches_for(state["epoch"])
        updates = prepared_updates(
            batches,
            accumulation,
            start_offset=state["next_batch_offset"],
            next_patch=model.config.objective == "next_patch",
            prefetch=t.get("prefetch", True),
            pin_memory=next(model.parameters()).is_cuda,
            microbatch_size=t.get("microbatch_size", 4),
            microbatch_token_limit=t.get("microbatch_token_limit", 1536),
        )
        try:
            while True:
                started = clock()
                try:
                    offset, group, prepared = next(updates)
                except StopIteration:
                    break
                prepared_at = clock()
                learning_rate = optimizer.param_groups[0]["lr"]
                result = accumulated_update(
                    model, group, optimizer, t["gradient_clip"], prepared
                )
                if average is not None:
                    average.update()
                result["preparation_wait_seconds"] = max(0.0, prepared_at - started)
                result["optimizer_update_seconds"] = max(0.0, clock() - prepared_at)
                state["active_training_seconds"] += max(0.0, clock() - started)
                state["step"] += 1
                state["next_batch_offset"] = offset + len(group)
                if state["next_batch_offset"] == len(batches):
                    state["completed_epochs"] = state["epoch"]
                if not isinstance(
                    scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau
                ):
                    scheduler.step()
                points_processed += result["target_points"]
                windows_processed += result["windows"]
                mass = result.get("target_loss_weight", result["target_points"])
                state["epoch_loss_sum"] += result["loss"] * mass
                state["epoch_mass"] += mass
                state["epoch_windows"] += result["windows"]
                if state["epoch_target_points_processed"] is not None:
                    state["epoch_target_points_processed"] += result["target_points"]
                state["embedding_gradient_updates"] += int(
                    result.get("embedding_skip_gradient_norm", 0) > 0
                )
                entry = {
                    "step": state["step"],
                    "epoch": state["epoch"],
                    **result,
                    "learning_rate": learning_rate,
                    "seconds": clock() - started,
                    "next_batch_offset": state["next_batch_offset"],
                }
                elapsed_epoch = (
                    state["active_training_seconds"]
                    - state["epoch_started_active_seconds"]
                )
                progress = {
                    "epoch_update": math.ceil(
                        state["next_batch_offset"] / accumulation
                    ),
                    "epoch_updates": updates_per_epoch,
                    "epoch_progress_percent": 100
                    * state["next_batch_offset"]
                    / len(batches),
                    "epoch_windows": state["epoch_windows"],
                    "epoch_total_windows": len(records),
                    "epoch_elapsed_seconds": elapsed_epoch,
                    "epoch_eta_seconds": elapsed_epoch
                    / max(1, state["epoch_windows"])
                    * (len(records) - state["epoch_windows"]),
                }
                entry.update(progress)
                history.append(entry)
                del history[:-1000]
                add_to_interval(result, mass, learning_rate, progress)
                if log_every_tables is None:
                    emit_training_event(
                        "metrics",
                        global_step=state["step"],
                        run_step=state["step"],
                        epoch=state["epoch"],
                        train_loss=result["loss"],
                        learning_rate=learning_rate,
                        gradient_norm=result["gradient_norm"],
                        windows=result["windows"],
                        microbatches=result["microbatches"],
                        oom_retries=result.get("oom_retries", 0),
                        cuda_allocated_gib=result.get("cuda_allocated_gib"),
                        cuda_reserved_gib=result.get("cuda_reserved_gib"),
                        cuda_peak_allocated_gib=result.get("cuda_peak_allocated_gib"),
                        preparation_wait_seconds=result["preparation_wait_seconds"],
                        optimizer_update_seconds=result["optimizer_update_seconds"],
                        embedding_shapes=result.get("embedding_shapes"),
                        target_points=result["target_points"],
                        window_ids=[r["id"] for batch in group for r in batch],
                        sources=[r["source"] for batch in group for r in batch],
                        active_training_seconds=state["active_training_seconds"],
                        training_target_points_processed=points_processed,
                        **progress,
                    )
                checkpoint_due = (
                    state["active_training_seconds"] - state["last_checkpoint_seconds"]
                    >= checkpoint_interval
                )
                state["pending_validation"] = (
                    state["active_training_seconds"] - state["last_validation_seconds"]
                    >= validation_interval
                )
                stopping = (
                    checkpoint_due
                    or state["pending_validation"]
                    or expired()
                    or (total_budget is not None and state["step"] >= total_budget)
                )
                if log_every_tables is None:
                    report = (
                        state["step"] == 1
                        or state["step"] % 100 == 0
                        or checkpoint_due
                        or state["pending_validation"]
                    )
                else:
                    # Also before a checkpoint, a validation or the end, so the
                    # journal's sums cover every update.
                    report = (
                        interval["tables"] >= log_every_tables
                        or not reported_first
                        or stopping
                    )
                    if report:
                        emit_interval()
                if report:
                    reported_first = True
                    if writer is not None:
                        writer.raise_if_failed()
                    logger.info(
                        "EPOCH {} | update {:,}/{:,} ({:.2f}%) | global {:,} | loss={:.6f} | epoch training ETA {} | checkpoint in {} | validation in {} | run remaining {}",
                        state["epoch"],
                        math.ceil(state["next_batch_offset"] / accumulation),
                        updates_per_epoch,
                        progress["epoch_progress_percent"],
                        state["step"],
                        result["loss"],
                        duration(progress["epoch_eta_seconds"]),
                        duration(
                            max(
                                0.0,
                                checkpoint_interval
                                - (
                                    state["active_training_seconds"]
                                    - state["last_checkpoint_seconds"]
                                ),
                            )
                        ),
                        duration(
                            max(
                                0.0,
                                validation_interval
                                - (
                                    state["active_training_seconds"]
                                    - state["last_validation_seconds"]
                                ),
                            )
                        ),
                        duration(max(0.0, run_seconds - (clock() - run_started)))
                        if run_seconds is not None
                        else "unlimited",
                    )
                    if "cuda_allocated_gib" in result:
                        logger.info(
                            "VRAM allocated={:.2f} GiB reserved={:.2f} GiB peak={:.2f} GiB | "
                            "physical microbatches={} shapes={} OOM retries={}",
                            result["cuda_allocated_gib"],
                            result["cuda_reserved_gib"],
                            result["cuda_peak_allocated_gib"],
                            result["microbatches"],
                            result["embedding_shapes"],
                            result["oom_retries"],
                        )
                    if log_every_tables is None:
                        recent = [
                            h
                            for h in history[-100:]
                            if h["epoch"] == state["epoch"]
                            and "preparation_wait_seconds" in h
                        ]
                        count = len(recent)
                        waited = sum(h["preparation_wait_seconds"] for h in recent)
                        worked = sum(h["optimizer_update_seconds"] for h in recent)
                        passes = sum(h["microbatches"] for h in recent)
                        tables = sum(h["windows"] for h in recent)
                    else:
                        count, tables = interval["updates"], interval["tables"]
                        waited, worked = interval["preparation"], interval["optimizer"]
                        passes = interval["microbatches"]
                    reset_interval()
                    logger.info(
                        "THROUGHPUT last {} updates | preparation wait {:.1f} ms/update | "
                        "model/optimizer {:.1f} ms/update | physical microbatches {:.2f}/update | "
                        "{:.1f} tables/s",
                        count,
                        1000 * waited / count,
                        1000 * worked / count,
                        passes / count,
                        tables / max(waited + worked, 1e-9),
                    )
                if stopping:
                    break
        finally:
            updates.close()
        # Persist before validation, which may be long or interrupted. A saved
        # pending flag makes a restart rerun that check before any next update.
        if (
            state["pending_validation"]
            or state["active_training_seconds"] - state["last_checkpoint_seconds"]
            >= checkpoint_interval
        ):
            persist()
    if (
        reason in ("step_limit", "epoch_limit")
        and schedule_plan["name"] == "warmup_cosine"
        and state["last_validation_step"] != state["step"]
    ):
        # A warm-up/cosine run ends at its lowest learning rate, usually with
        # its best weights. Validate them so they compete for selection instead
        # of exporting the last periodic check.
        budget_reason = reason
        evaluate()
        reason = budget_reason
    if log_every_tables is not None and interval["updates"]:
        emit_interval()  # updates since the last summary
        reset_interval()
    logger.info(
        "TIMED TRAIN stopping: reason={} update={} epoch={} next_batch={} active_training={} elapsed_run={}",
        reason,
        state["step"],
        state["epoch"],
        state["next_batch_offset"],
        duration(state["active_training_seconds"]),
        duration(clock() - run_started),
    )
    persist(reason)
    if writer is not None:
        # The final checkpoint (and its cloud copy) must be complete before
        # the run reports success.
        writer.finish()
        if store is not None:
            if writer.last_upload is None or writer.upload_failures:
                logger.warning(
                    "Checkpoint uploads this run: {} completed, {} failed, {} abandoned. "
                    "Check the container before relying on the cloud copy.",
                    writer.uploads,
                    writer.upload_failures,
                    writer.uploads_abandoned,
                )
            else:
                logger.info(
                    "Checkpoint uploads this run: {} completed, 0 failed, {} abandoned; "
                    "latest blob={}",
                    writer.uploads,
                    writer.uploads_abandoned,
                    writer.last_upload["blob"],
                )
    exported_step = state["step"]
    restored_best = best_step is not None and best_step > 0
    if restored_best:
        model.load_state_dict(
            torch.load(best_path, map_location="cpu", weights_only=True, mmap=True)[
                "state_dict"
            ]
        )
        exported_step = best_step
    summary = {
        "reason": reason,
        "completed_updates": state["step"],
        "completed_epochs": state["completed_epochs"],
        "epochs_started": state["epoch"],
        "last_epoch_complete": state["next_batch_offset"] == len(batches),
        "next_batch_offset": state["next_batch_offset"],
        "epoch_windows_processed": state["epoch_windows"],
        "epoch_target_points_processed": state["epoch_target_points_processed"],
        "active_training_seconds": state["active_training_seconds"],
        "run_elapsed_seconds": clock() - run_started,
        "validation_checks": state["validation_checks"],
        "best_epoch": early.best_epoch,
        "best_step": best_step,
        "best_metric": early.best,
        "exported_step": exported_step,
        "exported_validation_status": "validated_best"
        if restored_best
        else "not_yet_validated",
        "restored_best_weights": restored_best,
        "monitor": options["monitor"],
        "exported_epoch": early.best_epoch
        if restored_best
        else (state["epoch"] if state["step"] else 0),
        "embedding_gradient_updates": state["embedding_gradient_updates"],
        "epochs_without_improvement": early.bad_epochs,
        "patience_checks": early.patience,
        "patience_unit": "validation_checks",
        "training_target_points_processed": points_processed,
        "training_windows_processed": windows_processed,
        "full_training_corpus_pass_completed": state["completed_epochs"] > 0,
        "resumed_from": resumed_from,
        "resume_checkpoint": str(latest_path),
        "planned_steps": total_budget,
    }
    write_json(directory / "early_stopping.json", summary)
    emit_training_event(
        "checkpoint_selected", global_step=state["step"], selection=summary
    )
    return history, summary, scheduler_snapshot(scheduler, schedule_plan)

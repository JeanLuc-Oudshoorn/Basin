"""Gradient accumulation, full end-of-epoch validation and checkpoint selection."""

import gc
import math
from time import perf_counter

import pandas as pd
import torch
from batching import make_batches
from checkpoints import (
    atomic_torch_save,
    checkpoint_identity,
    restore_checkpoint,
    restore_rng,
    resume_path,
    rng_state,
    save_checkpoint,
)
from learning_rate import make_scheduler, scheduler_snapshot
from loguru import logger
from performance import (
    optimizer_options,
    prepare_update,
    prepared_updates,
    split_prepared_batches,
)
from training_log import emit_training_event
from training_progress import duration, epoch_progress
from training_protocol import validate
from transform import location, write_json

# Validation statistics that may drive best-weight selection, early stopping and
# a plateau schedule. The table statistics exist for sampled-origin validation.
MONITORS = (
    "loss",
    "rmse",
    "table_mase_profile",
    "trimmed_table_loss",
    "median_table_loss",
    "median_table_mase",
)


def monitored_value(validation, monitor):
    value = validation.get(monitor)
    if value is None:
        raise ValueError(
            f"Validation did not report early_stopping.monitor={monitor!r}; the table "
            "statistics need data.validation_policy=sampled_training_origins"
        )
    return value


def training_options(training):
    from weight_average import weight_average_options

    optimizer_options(training)
    weight_average_options(training)
    for key in (
        "checkpoint_interval_seconds",
        "validation_interval_seconds",
        "max_run_hours",
    ):
        value = training.get(key)
        if value is not None and (
            isinstance(value, bool)
            or not isinstance(value, (int, float))
            or not math.isfinite(value)
            or value <= 0
        ):
            raise ValueError(f"training.{key} must be positive and finite or null")
    for key, default in (("shape_run_batches", 1), ("training_graph_min_reuse", 8)):
        value = training.get(key, default)
        if isinstance(value, bool) or not isinstance(value, int) or value < 1:
            raise ValueError(f"training.{key} must be a positive integer")
    graph_tokens = training.get("training_graph_max_tokens", 32)
    if graph_tokens is not None and (
        isinstance(graph_tokens, bool)
        or not isinstance(graph_tokens, int)
        or graph_tokens < 1
    ):
        raise ValueError("training.training_graph_max_tokens must be positive or null")
    if training.get("shape_run_batches", 1) > 1 and not training.get(
        "bucket_by_length", True
    ):
        raise ValueError("training.shape_run_batches requires bucket_by_length")
    diagnostic = training.get("diagnostic_seconds")
    if diagnostic is not None and (
        isinstance(diagnostic, bool)
        or not isinstance(diagnostic, (int, float))
        or not math.isfinite(diagnostic)
        or diagnostic <= 0
    ):
        raise ValueError("training.diagnostic_seconds must be positive and finite")
    for key, default in (("microbatch_size", 4), ("microbatch_token_limit", 1536)):
        value = training.get(key, default)
        if value is not None and (
            isinstance(value, bool) or not isinstance(value, int) or value < 1
        ):
            raise ValueError(f"training.{key} must be a positive integer or null")
    limit = training.get("cuda_memory_limit_gib", 11.0)
    if limit is not None and (
        isinstance(limit, bool)
        or not isinstance(limit, (int, float))
        or not math.isfinite(limit)
        or limit <= 0
    ):
        raise ValueError(
            "training.cuda_memory_limit_gib must be positive and finite, or null"
        )
    headroom = training.get("cuda_memory_headroom_gib", 0.5)
    if (
        isinstance(headroom, bool)
        or not isinstance(headroom, (int, float))
        or not math.isfinite(headroom)
        or headroom < 0
    ):
        raise ValueError(
            "training.cuda_memory_headroom_gib must be finite and nonnegative"
        )
    resume = training.get("resume", "auto")
    if isinstance(resume, str) and resume.strip().lower() in ("true", "false"):
        raise ValueError(
            "training.resume: use false (without quotes) to start fresh, "
            "'auto' to resume the latest checkpoint, or an explicit checkpoint path"
        )
    if resume is not False and (not isinstance(resume, str) or not resume.strip()):
        raise ValueError("training.resume must be 'auto', a checkpoint path, or false")
    for key in ("validation_batch_size",):
        value = training.get(key, 8)
        if isinstance(value, bool) or not isinstance(value, int) or value < 1:
            raise ValueError(f"training.{key} must be a positive integer")
    threshold = training.get("checkpoint_token_threshold", 0)
    if isinstance(threshold, bool) or not isinstance(threshold, int) or threshold < 0:
        raise ValueError(
            "training.checkpoint_token_threshold must be a nonnegative integer"
        )
    for key in (
        "bucket_by_length",
        "storage_locality",
        "prefetch",
        "fused_optimizer",
        "compile_variate",
        "compile_variate_inference",
        "validate_before_training",
        "training_cuda_graphs",
        "auto_microbatch",
    ):
        if not isinstance(
            training.get(key, False if key == "training_cuda_graphs" else True), bool
        ):
            raise ValueError(f"training.{key} must be boolean")
    for key, default in (
        ("validation_cuda_graphs", False),
        ("validation_graph_retain", True),
    ):
        if not isinstance(training.get(key, default), bool):
            raise ValueError(f"training.{key} must be boolean")
    budget = training.get("validation_graph_memory_gib", 2.0)
    if (
        isinstance(budget, bool)
        or not isinstance(budget, (int, float))
        or not math.isfinite(budget)
        or budget <= 0
    ):
        raise ValueError(
            "training.validation_graph_memory_gib must be positive and finite"
        )
    shapes = training.get("training_graph_cache_shapes", 2)
    memory = training.get("training_graph_memory_gib", 0.75)
    if isinstance(shapes, bool) or not isinstance(shapes, int) or shapes < 1:
        raise ValueError(
            "training.training_graph_cache_shapes must be a positive integer"
        )
    if (
        isinstance(memory, bool)
        or not isinstance(memory, (int, float))
        or not math.isfinite(memory)
        or memory <= 0
    ):
        raise ValueError(
            "training.training_graph_memory_gib must be positive and finite"
        )
    cache = training.get("compile_cache_dir", ".cache/torchinductor")
    if not isinstance(cache, str) or not cache.strip():
        raise ValueError("training.compile_cache_dir must be a nonempty path")
    options = {
        "batch_size": training.get("batch_size", 1),
        "gradient_accumulation_steps": training.get("gradient_accumulation_steps", 1),
        "epochs": training.get("epochs", 1),
        "steps": training.get("steps"),
    }
    for key, value in options.items():
        if value is None and key in ("epochs", "steps"):
            continue
        if isinstance(value, bool) or not isinstance(value, int) or value < 1:
            raise ValueError(
                f"training.{key} must be a positive integer or null for limits"
            )
    early = dict(training.get("early_stopping", {}))
    if set(early) - {"patience_epochs", "patience_checks", "min_delta", "monitor"}:
        raise ValueError("Unknown early_stopping fields")
    patience = early.get("patience_checks", early.get("patience_epochs", 1))
    delta = early.get("min_delta", 0.0)
    monitor = early.get("monitor", "loss")
    if (
        isinstance(patience, bool)
        or not isinstance(patience, int)
        or patience < 1
        or not math.isfinite(delta)
        or delta < 0
        or monitor not in MONITORS
    ):
        raise ValueError("Invalid early stopping patience_epochs, min_delta or monitor")
    return {
        **options,
        "patience_epochs": patience,
        "min_delta": delta,
        "monitor": monitor,
    }


class EarlyStopping:
    def __init__(self, patience_epochs, min_delta):
        self.patience = patience_epochs
        self.delta = min_delta
        self.best = None
        self.best_epoch = None
        self.bad_epochs = 0

    def observe(self, value, epoch):
        if not math.isfinite(value):
            raise FloatingPointError("Nonfinite epoch validation metric")
        improved = self.best is None or value < self.best - self.delta
        if improved:
            self.best, self.best_epoch, self.bad_epochs = value, epoch, 0
        else:
            self.bad_epochs += 1
        return improved, self.bad_epochs >= self.patience


def accumulated_update(
    model, microbatches, optimizer, gradient_clip, prepared_batches=None
):
    """Retry activation OOMs before stepping; keep the same logical optimizer batch."""
    if prepared_batches is None:
        prepared_batches = prepare_update(
            microbatches, next_patch=model.config.objective == "next_patch"
        )
    device = next(model.parameters()).device
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)
    recover = getattr(model, "auto_microbatch", True)
    # Active graph captures need a separate lifetime protocol. Training graphs
    # are off by default; never try to recover an invalidated capture here.
    recover = recover and getattr(model, "training_graphs", None) is None
    recover = recover and model.config.objective == "next_patch"
    limits = getattr(model, "_microbatch_limits", {})
    key = (
        tuple(
            sorted(
                {(len(p["roles"]), p["padded_tokens"]) for p, _, _ in prepared_batches}
            )
        )
        if recover
        else ()
    )
    if recover and key in limits:
        prepared_batches = split_prepared_batches(prepared_batches, limits[key])
    initial_rng = rng_state() if recover else None
    retries = 0
    while True:
        try:
            result = _accumulated_gradients(
                model, microbatches, optimizer, gradient_clip, prepared_batches
            )
            break
        except torch.OutOfMemoryError:
            # Let the failed forward/backward frame unwind before collecting its
            # activations. A partially accumulated gradient must never be stepped.
            optimizer.zero_grad(set_to_none=True)
            largest = max(labels.shape[0] for _, labels, _ in prepared_batches)
            checkpoint_all = (
                largest == 1
                and getattr(model.config, "gradient_checkpointing", False)
                and getattr(model, "checkpoint_token_threshold", 0) > 0
            )
            if not recover or (largest == 1 and not checkpoint_all):
                logger.error(
                    "Training OOM cannot be reduced further: microbatch={} shapes={} "
                    "checkpoint_threshold={}; optimizer update was not applied",
                    largest,
                    key,
                    getattr(model, "checkpoint_token_threshold", None),
                )
                raise
        # Outside the except block: its traceback no longer retains GPU tensors.
        gc.collect()
        if device.type == "cuda":
            torch.cuda.empty_cache()
        restore_rng(initial_rng)
        retries += 1
        if checkpoint_all:
            model.checkpoint_token_threshold = 0
            logger.warning("Activation OOM: retrying with every layer checkpointed")
        else:
            maximum = max(1, largest // 2)
            limits[key] = maximum
            model._microbatch_limits = limits
            prepared_batches = split_prepared_batches(prepared_batches, maximum)
            logger.warning(
                "Activation OOM: retry={} shapes={} microbatch {} -> {}; "
                "discarded partial gradients, restored RNG, unchanged optimizer batch",
                retries,
                key,
                largest,
                maximum,
            )
    # An optimizer OOM can partially change parameters/state. It MUST propagate,
    # rather than replaying the update as if no weights had changed.
    if result.get("supervised", True):
        with torch.profiler.record_function("training.optimizer"):
            optimizer.step()
    result["oom_retries"] = retries
    if device.type == "cuda":
        result["cuda_allocated_gib"] = torch.cuda.memory_allocated(device) / 2**30
        result["cuda_reserved_gib"] = torch.cuda.memory_reserved(device) / 2**30
        result["cuda_peak_allocated_gib"] = (
            torch.cuda.max_memory_allocated(device) / 2**30
        )
    return result


def _accumulated_gradients(
    model, microbatches, optimizer, gradient_clip, prepared_batches
):
    """Point-weighted accumulation with one metrics transfer per optimizer update."""
    device = next(model.parameters()).device
    if prepared_batches is None:
        prepared_batches = prepare_update(
            microbatches, next_patch=model.config.objective == "next_patch"
        )
    total_points = sum(p for _, _, p in prepared_batches)
    weights = getattr(model.config, "horizon_loss_weights", None)
    if weights is not None:
        from multi_horizon import anchors_with_history, weighted_label_mass

        # The direct head leaves out anchors without any observed history;
        # count the same labels here so microbatch weights stay exact.
        direct = getattr(model.config, "head", "lead_offset") == "direct"
        masses = [
            float(
                weighted_label_mass(
                    labels,
                    weights,
                    anchors_with_history(
                        prepared["features"],
                        prepared["targets"],
                        model.config.patch_size,
                    )
                    if direct
                    else None,
                )
            )
            for prepared, labels, _ in prepared_batches
        ]
    else:
        masses = [points for _, _, points in prepared_batches]
    total_mass = sum(masses)
    optimizer.zero_grad(set_to_none=True)
    if total_mass <= 0:
        # Every target value this update could learn from is missing. There
        # is no gradient: report an empty update and leave the weights alone.
        logger.warning(
            "Optimizer update without any observed label ({} tables): skipped",
            sum(map(len, microbatches)),
        )
        return {
            "loss": 0.0,
            "target_points": 0,
            "target_loss_weight": 0.0,
            "microbatches": 0,
            "windows": sum(map(len, microbatches)),
            "gradient_norm": 0.0,
            "embedding_skip_gradient_norm": 0.0,
            "embedding_shapes": [],
            "embedding_shape": None,
            "device": str(device),
            "training_graphs": None,
            "supervised": False,
        }
    weighted_loss = torch.zeros((), device=device)
    losses, shapes = [], []
    graphs = getattr(model, "training_graphs", None)
    if graphs is not None:
        graphs.begin_update()
    for microstep, (prepared, labels, points) in enumerate(prepared_batches, 1):
        if masses[microstep - 1] <= 0:
            continue  # nothing observed in this microbatch; it has no gradient
        graphs = getattr(model, "training_graphs", None)
        if graphs is not None:
            graphs.prepare(prepared)
        with torch.profiler.record_function("training.transfer"):
            prepared = {
                k: v.to(device, non_blocking=True) if isinstance(v, torch.Tensor) else v
                for k, v in prepared.items()
            }
            labels = (
                prepared["labels"]
                if model.config.objective == "next_patch"
                else labels.to(device, non_blocking=True)
            )
        with torch.profiler.record_function("training.forward"):
            x = model.encode_tokens(prepared)
            output = model.forward_encoded(x, prepared, labels)
        if output.loss is None:
            raise FloatingPointError("Missing training loss")
        losses.append(output.loss.detach().float())
        fraction = masses[microstep - 1] / total_mass
        weighted_loss += losses[-1] * fraction
        with torch.profiler.record_function("training.backward"):
            (output.loss * fraction).backward()
        shapes.append(list(x.shape))
        logger.debug(
            "Accumulation microbatch {}/{} embedding_shape[B,V,T,D]={} max_real_patches={} kernel_patches={} observed_labels={}",
            microstep,
            len(prepared_batches),
            list(x.shape),
            prepared["tokens"],
            prepared["padded_tokens"],
            points,
        )
        del output, x
    embedding_grad = model.embedding.skip.weight.grad
    embedding_norm = (
        embedding_grad.float().norm()
        if embedding_grad is not None
        else torch.zeros((), device=device)
    )
    with torch.profiler.record_function("training.clip_and_metrics"):
        norm = torch.nn.utils.clip_grad_norm_(
            model.parameters(), gradient_clip, error_if_nonfinite=False, foreach=True
        )
        # One synchronization, after all backwards. Reject before optimizer.step.
        metrics = (
            torch.stack([weighted_loss, norm.float(), embedding_norm, *losses])
            .cpu()
            .tolist()
        )
        if not all(math.isfinite(value) for value in metrics):
            raise FloatingPointError("Nonfinite training loss or gradient norm")
    logger.debug("Accumulation microbatch losses={}", metrics[3:])
    return {
        "loss": metrics[0],
        "target_points": total_points,
        "target_loss_weight": total_mass,
        "microbatches": len(shapes),
        "windows": sum(map(len, microbatches)),
        "gradient_norm": metrics[1],
        "embedding_skip_gradient_norm": metrics[2],
        "embedding_shapes": shapes,
        "embedding_shape": shapes[-1],
        "device": str(device),
        "training_graphs": graphs.summary() if graphs is not None else None,
    }


def fit_epochs(
    model, records, validation_records, config, optimizer, validation_fn=None
):
    """One complete shuffled training pass precedes each full validation sweep."""
    if (
        config["data"].get("validation_policy") == "sampled_training_origins"
        and validation_records
        and "validation_origin" in validation_records[0]
    ):
        from timed_training import fit_timed

        return fit_timed(
            model,
            records,
            validation_records,
            config,
            optimizer,
            validation_fn=validation_fn,
        )
    validation_fn = validation_fn or (
        lambda model, records: validate(
            model,
            records,
            batch_size=config["training"].get("validation_batch_size", 8),
        )
    )
    t = config["training"]
    options = training_options(t)
    if t.get("weight_average") is not None:
        raise ValueError(
            "training.weight_average requires data.validation_policy="
            "sampled_training_origins (the timed trainer)"
        )
    if not records or not validation_records:
        raise ValueError("Need both training and validation records")
    directory = location(config, "training")
    directory.mkdir(parents=True, exist_ok=True)
    accumulation = options["gradient_accumulation_steps"]
    initial_batches = make_batches(
        records,
        options["batch_size"],
        bucket_by_length=t.get("bucket_by_length", True),
        storage_locality=t.get("storage_locality", False),
    )
    updates_per_epoch = math.ceil(len(initial_batches) / accumulation)
    del initial_batches
    total_budget = options["steps"]
    if options["epochs"] is not None:
        epoch_budget = options["epochs"] * updates_per_epoch
        total_budget = (
            epoch_budget if total_budget is None else min(total_budget, epoch_budget)
        )
    scheduler, schedule_plan = make_scheduler(optimizer, t, total_budget)
    early = EarlyStopping(options["patience_epochs"], options["min_delta"])
    plan = {
        **options,
        "planned_updates": total_budget,
        "updates_per_epoch": updates_per_epoch,
        "validation_scope": "all_training_tables_reserved_tail",
        "validation_horizon": config["data"]["horizon"],
        "horizon_loss_weights": getattr(model.config, "horizon_loss_weights", None),
        "validation_frequency": "epoch_end",
        "validation_tables": len(validation_records),
        "performance": {
            k: t.get(k)
            for k in (
                "validation_batch_size",
                "fused_optimizer",
                "bucket_by_length",
                "prefetch",
                "checkpoint_token_threshold",
                "microbatch_size",
                "microbatch_token_limit",
                "cuda_memory_limit_gib",
                "training_cuda_graphs",
                "training_graph_cache_shapes",
                "training_graph_memory_gib",
                "shape_run_batches",
                "training_graph_min_reuse",
                "training_graph_max_tokens",
            )
        },
        "scheduler": schedule_plan,
    }
    write_json(directory / "training_plan.json", plan)
    emit_training_event(
        "training_plan",
        planned_steps=total_budget,
        configured_steps=options["steps"],
        train_windows=len(records),
        train_tables=len({r["source"] for r in records}),
        validation_tables=len(validation_records),
        training_schedule=schedule_plan,
        accumulation_steps=accumulation,
        batch_size=options["batch_size"],
        microbatch_size=t.get("microbatch_size", 4),
        microbatch_token_limit=t.get("microbatch_token_limit", 1536),
        cuda_memory_limit_gib=t.get("cuda_memory_limit_gib", 8.0),
        early_stopping=options,
        validation_frequency="epoch_end",
        updates_per_epoch=updates_per_epoch,
    )
    logger.info(
        "Microbatch execution: max tables={}, max padded variable-patches={}; "
        "up to {} tables per optimizer update, weighted by observed target points",
        t.get("microbatch_size", 4),
        t.get("microbatch_token_limit", 1536),
        options["batch_size"] * accumulation,
    )
    history, validation_history = [], []
    step = epoch = points_processed = windows_processed = 0
    reason = None
    best_step = None
    best_path = directory / "best_weights.pt"
    latest_path = directory / "latest_checkpoint.pt"
    identity = checkpoint_identity(
        model, records, validation_records, config, optimizer
    )
    previous = resume_path(config)
    resumed_from = None
    completed_epoch = True
    if previous is not None:
        restored = restore_checkpoint(
            previous, model, optimizer, scheduler, identity, schedule_plan, best_path
        )
        epoch, step = restored["epoch"], restored["step"]
        points_processed, windows_processed = (
            restored["points_processed"],
            restored["windows_processed"],
        )
        best_step = restored["best_step"]
        early.best = restored["early_stopping"]["best"]
        early.best_epoch = restored["early_stopping"]["best_epoch"]
        early.bad_epochs = restored["early_stopping"]["bad_epochs"]
        history, validation_history = (
            restored["history"],
            restored["validation_history"],
        )
        resumed_from = str(previous)
        # A completed/stopped run can be retried after a failed final export or
        # holdout evaluation without silently training more epochs.
        if early.bad_epochs >= early.patience:
            reason = "early_stopping"
        elif options["epochs"] is not None and epoch >= options["epochs"]:
            reason = "epoch_limit"
        elif total_budget is not None and step >= total_budget:
            reason = "step_limit"
    if previous is None and t.get("validate_before_training", True):
        logger.info(
            "EPOCH 0 baseline: validating all {:,} tables before any optimizer update",
            len(validation_records),
        )
        before_baseline = rng_state()
        try:
            baseline = dict(validation_fn(model, validation_records))
        finally:
            restore_rng(before_baseline)
        if not all(math.isfinite(baseline[key]) for key in ("loss", "rmse")):
            raise FloatingPointError("Nonfinite epoch-zero validation metric")
        baseline_tables = baseline.pop("tables", [])
        baseline_path = directory / "validation_epoch_00000.parquet"
        pd.DataFrame(baseline_tables).to_parquet(baseline_path, index=False)
        baseline_row = {
            "epoch": 0,
            "step": 0,
            "completed_epoch": False,
            "phase": "before_training",
            "training_loss": None,
            **baseline,
            "table_count": len(validation_records),
            "table_metrics_path": str(baseline_path),
        }
        validation_history.append(baseline_row)
        write_json(directory / "validation_history.json", validation_history)
        emit_training_event(
            "validation_baseline",
            global_step=0,
            epoch=0,
            validation_loss=baseline["loss"],
            validation_rmse=baseline["rmse"],
            validation_seconds=baseline.get("seconds"),
            validation_tables=len(validation_records),
        )
        logger.info(
            "EPOCH 0 baseline: validation loss={} RMSE={} seconds={}",
            baseline["loss"],
            baseline["rmse"],
            baseline.get("seconds"),
        )
    elif previous is not None and not any(
        row["epoch"] == 0 for row in validation_history
    ):
        logger.info(
            "Resumed checkpoint has no epoch-0 baseline; cannot reconstruct the original untrained score"
        )
    model.train()
    diagnostic_started = perf_counter()
    while reason is None:
        epoch += 1
        started_epoch = perf_counter()
        epoch_loss_sum = epoch_points = 0
        epoch_windows = 0
        seed = config["seed"] + epoch - 1 if t.get("shuffle", False) else None
        batches = make_batches(
            records,
            options["batch_size"],
            seed,
            bucket_by_length=t.get("bucket_by_length", True),
            shape_run_batches=t.get("shape_run_batches", 1),
            storage_locality=t.get("storage_locality", False),
        )
        logger.info(
            "EPOCH {} start: {} windows, {} batch groups, {} updates, accumulate={}",
            epoch,
            len(records),
            len(batches),
            updates_per_epoch,
            accumulation,
        )
        completed_epoch = True
        updates = prepared_updates(
            batches,
            accumulation,
            next_patch=model.config.objective == "next_patch",
            prefetch=t.get("prefetch", True),
            pin_memory=next(model.parameters()).device.type == "cuda",
            microbatch_size=t.get("microbatch_size", 4),
            microbatch_token_limit=t.get("microbatch_token_limit", 1536),
        )
        try:
            for offset, group, prepared in updates:
                started = perf_counter()
                learning_rate = optimizer.param_groups[0]["lr"]
                update = accumulated_update(
                    model, group, optimizer, t["gradient_clip"], prepared
                )
                step += 1
                if not isinstance(
                    scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau
                ):
                    scheduler.step()
                points_processed += update["target_points"]
                windows_processed += update["windows"]
                epoch_windows += update["windows"]
                mass = update.get("target_loss_weight", update["target_points"])
                epoch_points += mass
                epoch_loss_sum += update["loss"] * mass
                progress = epoch_progress(
                    offset // accumulation + 1,
                    updates_per_epoch,
                    epoch_windows,
                    len(records),
                    perf_counter() - started_epoch,
                )
                progress["epoch_windows_per_second"] = epoch_windows / max(
                    progress["epoch_elapsed_seconds"], 1e-9
                )
                entry = {
                    "step": step,
                    "epoch": epoch,
                    **update,
                    **progress,
                    "learning_rate": learning_rate,
                    "next_learning_rate": optimizer.param_groups[0]["lr"],
                    "validation_loss": None,
                    "validation_rmse": None,
                    "source": group[0][0]["source"],
                    "window_ids": [r["id"] for batch in group for r in batch],
                    "sources": [r["source"] for batch in group for r in batch],
                    "seconds": perf_counter() - started,
                    "gpu_peak_allocated_gb": torch.cuda.max_memory_allocated() / 2**30
                    if next(model.parameters()).device.type == "cuda"
                    else None,
                }
                history.append(entry)
                logger.info(
                    "EPOCH {} | update {:,}/{:,} ({:.1f}%) | global {:,} | windows {:,}/{:,} | loss={:.6f} | training ETA {} | full validation follows",
                    epoch,
                    progress["epoch_update"],
                    updates_per_epoch,
                    progress["epoch_progress_percent"],
                    step,
                    epoch_windows,
                    len(records),
                    update["loss"],
                    duration(progress["epoch_eta_seconds"]),
                )
                logger.debug("TRAIN details {}", entry)
                emit_training_event(
                    "metrics",
                    global_step=step,
                    run_step=step,
                    train_loss=update["loss"],
                    learning_rate=learning_rate,
                    gradient_norm=update["gradient_norm"],
                    epoch=epoch,
                    microbatches=update["microbatches"],
                    windows=update["windows"],
                    target_points=update["target_points"],
                    target_loss_weight=mass,
                    train_loss_scope="accumulated_update",
                    training_target_points_processed=points_processed,
                    embedding_shape=update["embedding_shape"],
                    window_ids=entry["window_ids"],
                    sources=entry["sources"],
                    **progress,
                )
                if t.get("checkpoint_every") and step % t["checkpoint_every"] == 0:
                    atomic_torch_save(
                        {"state_dict": model.state_dict()},
                        directory / "latest_weights.pt",
                    )
                if total_budget is not None and step >= total_budget:
                    completed_epoch = offset + accumulation >= len(batches)
                    reason = "step_limit"
                    break
                if (
                    t.get("diagnostic_seconds") is not None
                    and perf_counter() - diagnostic_started >= t["diagnostic_seconds"]
                ):
                    completed_epoch = offset + accumulation >= len(batches)
                    reason = "diagnostic_time_limit"
                    logger.info(
                        "Diagnostic training time limit reached after {:.1f}s",
                        perf_counter() - diagnostic_started,
                    )
                    break
        finally:
            updates.close()
        optimizer.zero_grad(set_to_none=True)
        # Only an explicit debug step cap can end a partial epoch. The default
        # unlimited run validates exclusively after complete passes.
        logger.info(
            "EPOCH {} validation: all {:,} validation tables, full reserved tail per table",
            epoch,
            len(validation_records),
        )
        validation = dict(validation_fn(model, validation_records))
        table_metrics = validation.pop("tables", [])
        table_path = directory / f"validation_epoch_{epoch:05d}.parquet"
        pd.DataFrame(table_metrics).to_parquet(table_path, index=False)
        validation["table_count"] = len(validation_records)
        validation["table_metrics_path"] = str(table_path)
        value = monitored_value(validation, options["monitor"])
        improved, stop = early.observe(value, epoch)
        if isinstance(scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau):
            scheduler.step(value)
        if improved:
            best_step = step
            atomic_torch_save({"state_dict": model.state_dict()}, best_path)
        row = {
            "epoch": epoch,
            "step": step,
            "completed_epoch": completed_epoch,
            "training_loss": epoch_loss_sum / epoch_points,
            **validation,
            "improved": improved,
            "bad_epochs": early.bad_epochs,
            "next_learning_rate": optimizer.param_groups[0]["lr"],
            "epoch_seconds": perf_counter() - started_epoch,
        }
        baseline_loss = next(
            (r["loss"] for r in validation_history if r["epoch"] == 0), None
        )
        if baseline_loss is not None:
            row["epoch_zero_validation_loss"] = baseline_loss
            row["validation_loss_reduction_from_epoch_zero"] = (
                baseline_loss - validation["loss"]
            )
        validation_history.append(row)
        graphs = getattr(model, "training_graphs", None)
        if graphs is not None:
            row["training_graphs"] = graphs.summary()
            logger.info("Training CUDA graph cache: {}", row["training_graphs"])
        history[-1].update(
            validation_loss=validation["loss"], validation_rmse=validation["rmse"]
        )
        write_json(directory / "validation_history.json", validation_history)
        write_json(directory / "loss_history.json", history)
        emit_training_event(
            "epoch_end",
            global_step=step,
            epoch=epoch,
            train_loss=row["training_loss"],
            validation_loss=validation["loss"],
            validation_rmse=validation["rmse"],
            validation_tables=len(validation_records),
            validation_scope=plan["validation_scope"],
            completed_epoch=completed_epoch,
            best_epoch=early.best_epoch,
            epochs_without_improvement=early.bad_epochs,
            learning_rate=optimizer.param_groups[0]["lr"],
            training_graphs=row.get("training_graphs"),
        )
        logger.info(
            "EPOCH {} end: train={} validation={} best_epoch={} no_improvement={}/{} next_lr={}",
            epoch,
            row["training_loss"],
            value,
            early.best_epoch,
            early.bad_epochs,
            early.patience,
            optimizer.param_groups[0]["lr"],
        )
        if stop:
            reason = "early_stopping"
        elif options["epochs"] is not None and epoch >= options["epochs"]:
            reason = "epoch_limit"
        write_json(
            directory / "learning_rate_scheduler.json",
            scheduler_snapshot(scheduler, schedule_plan),
        )
        if completed_epoch:
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
                improved,
                reason,
            )
        else:
            logger.info(
                "Partial debug epoch: keeping the previous resumable epoch checkpoint"
            )
    optimizer.zero_grad(set_to_none=True)
    model.load_state_dict(
        torch.load(best_path, map_location="cpu", weights_only=True, mmap=True)[
            "state_dict"
        ]
    )
    summary = {
        "reason": reason,
        "completed_updates": step,
        "completed_epochs": epoch - int(not completed_epoch),
        "epochs_started": epoch,
        "last_epoch_complete": completed_epoch,
        "best_epoch": early.best_epoch,
        "best_step": best_step,
        "monitor": options["monitor"],
        "best_metric": early.best,
        "patience_epochs": early.patience,
        "min_delta": early.delta,
        "epochs_without_improvement": early.bad_epochs,
        "training_target_points_processed": points_processed,
        "training_windows_processed": windows_processed,
        "full_training_corpus_pass_completed": windows_processed >= len(records),
        "planned_steps": total_budget,
        "restored_best_weights": True,
        "resumed_from": resumed_from,
        "resume_checkpoint": str(latest_path) if latest_path.is_file() else None,
    }
    write_json(directory / "early_stopping.json", summary)
    emit_training_event("checkpoint_selected", global_step=step, selection=summary)
    return history, summary, scheduler_snapshot(scheduler, schedule_plan)

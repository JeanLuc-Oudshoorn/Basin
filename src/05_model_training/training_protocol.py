"""Dataset isolation and full last-token validation for the normal trainer."""

import hashlib
import math
from time import perf_counter

import torch
from batching import make_batches, pack_records
from loguru import logger
from training_log import emit_training_event
from training_progress import duration


def rolling_origins(length, patch_size):
    if length < max(48, 3 * patch_size):
        raise ValueError("Test datasets need at least 48 timesteps and three tokens")
    return range(2 * patch_size, length, patch_size)


def sampled_origins(length, patch_size, percentiles=(1, 25, 50, 75, 99)):
    """Nearest eligible patch starts to row percentiles, with forced endpoints.

    Ties choose the earlier origin. Reserve distinct origins for later cases;
    if fewer cases exist than requested, use every eligible origin once.
    """
    if (
        len(percentiles) < 2
        or percentiles[0] != 1
        or percentiles[-1] != 99
        or any(
            isinstance(p, bool)
            or not isinstance(p, (int, float))
            or not math.isfinite(p)
            for p in percentiles
        )
        or any(a >= b for a, b in zip(percentiles, percentiles[1:]))
    ):
        raise ValueError("Holdout percentiles must increase from 1 to 99")
    eligible = list(rolling_origins(length, patch_size))
    if len(eligible) <= len(percentiles):
        return eligible
    indices = [0]
    for position, percentile in enumerate(percentiles[1:-1], 1):
        remaining = len(percentiles) - position - 1
        candidates = range(indices[-1] + 1, len(eligible) - remaining)
        indices.append(
            min(
                candidates,
                key=lambda i: (abs(eligible[i] - length * percentile / 100), i),
            )
        )
    return [eligible[i] for i in indices + [len(eligible) - 1]]


def split_datasets(records, config):
    """Split whole admitted sources, never windows from the same source."""
    patch = config["model"]["patch_size"]
    minimum = max(48, 3 * patch)
    fraction = config["training"].get("holdout_fraction", 0.1)
    if isinstance(fraction, bool) or not 0 < fraction < 1:
        raise ValueError("training.holdout_fraction must be between zero and one")
    sources = sorted({r["source"] for r in records if r["source_rows"] >= minimum})
    if len(sources) < 2:
        raise ValueError(
            f"Need at least two admitted datasets with >= {minimum} timesteps"
        )
    # Hash ranking is independent of input/window order and Python hash randomization.
    keys = {r["source"]: r.get("source_key", r["source"]) for r in records}
    ranked = sorted(
        sources,
        key=lambda s: (
            hashlib.sha256(f"{config['seed']}:{keys[s]}".encode()).hexdigest(),
            s,
        ),
    )
    count = min(len(sources) - 1, math.ceil(len(sources) * fraction))
    test_sources = set(ranked[:count])
    train_sources = set(sources) - test_sources
    train = [
        r for r in records if r["source"] in train_sources and r["split"] == "train"
    ]
    validation = [
        r
        for r in records
        if r["source"] in train_sources and r["split"] == "evaluation"
    ]
    test = [
        r for r in records if r["source"] in test_sources and r["split"] == "evaluation"
    ]
    for name, group, expected in (
        ("training", train, train_sources),
        ("validation", validation, train_sources),
        ("test", test, test_sources),
    ):
        if {r["source"] for r in group} != expected:
            raise ValueError(f"Missing {name} records; rerun transform and embed")
        if name != "training" and len(group) != len(expected):
            raise ValueError(f"Expected exactly one {name} record per dataset")
    audit = {
        "schema": 1,
        "seed": config["seed"],
        "requested_test_fraction": fraction,
        "actual_test_fraction": count / len(sources),
        "minimum_timesteps": minimum,
        "rounding": "ceil, at least one test table and one training table",
        "admitted_eligible_tables": len(sources),
        "train_tables": len(train_sources),
        "test_tables": count,
        "train_windows": len(train),
        "train_sources": sorted(train_sources),
        "test_sources": sorted(test_sources),
        "source_keys": {s: keys[s] for s in sources},
        "source_sha256": {r["source"]: r["source_sha256"] for r in validation + test},
        "excluded_short_sources": sorted(
            {r["source"] for r in records if r["source_rows"] < minimum}
        ),
        "validation": "last full token of each training table; never used as a gradient label",
        "test": "after final optimizer update only; expanding actual context from two tokens",
    }
    logger.info(
        "Dataset split: seed={} eligible={} train_tables={} test_tables={} train_windows={} excluded_short={}",
        config["seed"],
        len(sources),
        len(train_sources),
        count,
        len(train),
        len(audit["excluded_short_sources"]),
    )
    return train, validation, test, audit


@torch.inference_mode()
def validate(model, records, batch_size=1):
    """Point-weighted normalized pinball loss over every training-table final token."""
    if (
        batch_size > 1
        and getattr(getattr(model, "config", None), "objective", None) == "next_patch"
    ):
        return validate_batched(model, records, batch_size)
    started = perf_counter()
    device = next(model.parameters()).device
    was_training = model.training
    model.eval()
    loss_sum = squared_sum = 0.0
    points = 0
    tables = []
    try:
        for completed, record in enumerate(records, 1):
            batch = {
                k: v.to(device) if isinstance(v, torch.Tensor) else v
                for k, v in record["batch"].items()
            }
            output = model(**batch)
            if output.loss is None or not torch.isfinite(output.loss):
                raise FloatingPointError(
                    f"Nonfinite validation loss: {record['source']}"
                )
            labels = batch["future_target"]
            valid = torch.isfinite(labels)
            count = int(valid.sum())
            sse = float((output.point[valid] - labels[valid]).float().square().sum())
            loss = float(output.loss)
            loss_sum += loss * count
            squared_sum += sse
            points += count
            tables.append(
                {
                    "source": record["source"],
                    "loss": loss,
                    "rmse": math.sqrt(sse / count),
                    "points": count,
                }
            )
            if completed % 1000 == 0 or completed == len(records):
                elapsed = perf_counter() - started
                eta = elapsed / completed * (len(records) - completed)
                logger.info(
                    "VALIDATION tables {:,}/{:,} ({:.1f}%) | ETA {}",
                    completed,
                    len(records),
                    100 * completed / len(records),
                    duration(eta),
                )
                emit_training_event(
                    "validation_progress",
                    validation_tables_completed=completed,
                    validation_total_tables=len(records),
                    validation_eta_seconds=eta,
                )
    finally:
        model.train(was_training)
    if not points:
        raise ValueError("No validation labels")
    result = {
        "loss": loss_sum / points,
        "rmse": math.sqrt(squared_sum / points),
        "points": points,
        "tables": tables,
        "seconds": perf_counter() - started,
    }
    logger.info(
        "Full validation: loss={} RMSE={} tables={} points={} seconds={}",
        result["loss"],
        result["rmse"],
        len(tables),
        points,
        result["seconds"],
    )
    return result


@torch.inference_mode()
def validate_batched(model, records, batch_size):
    """Exact full validation, batching only compatible independent tables."""
    started = perf_counter()
    device = next(model.parameters()).device
    was_training = model.training
    model.eval()
    tables = []
    loss_sum = squared_sum = 0.0
    points = completed = 0
    report_at = 1000
    try:
        for group in make_batches(records, batch_size, bucket_by_length=True):
            prepared = pack_records(group)
            # Each table has its own final real position, before right padding.
            indices = []
            for r in group:
                if "batch" in r and getattr(r, "encoder", None) is not None:
                    length = min(
                        r["batch"]["context"].shape[-1], model.config.context_length
                    )
                    indices.append(math.ceil(length / model.config.patch_size) - 1)
                else:
                    indices.append(r["prepared"]["tokens"] - 1)
            labels = torch.cat([r["batch"]["future_target"] for r in group])
            prepared.update(
                is_training=False,
                labels=None,
                last_token_indices=torch.tensor(indices),
                observed_points=int(torch.isfinite(labels).sum()),
            )
            prepared = {
                k: v.to(device) if isinstance(v, torch.Tensor) else v
                for k, v in prepared.items()
            }
            labels = labels.to(device)
            output = model.forward_encoded(
                model.encode_tokens(prepared), prepared, labels
            )
            valid = torch.isfinite(labels)
            counts = valid.flatten(1).sum(1)
            errors = torch.where(valid, output.point - labels, 0).float()
            sses = errors.square().flatten(1).sum(1)
            metrics = (
                torch.stack([output.per_sample_loss, sses, counts.float()], dim=1)
                .cpu()
                .tolist()
            )
            for record, (loss, sse, count) in zip(group, metrics):
                if count < 1 or not math.isfinite(loss) or not math.isfinite(sse):
                    raise FloatingPointError(
                        f"Invalid validation metrics: {record['source']}"
                    )
                count = int(count)
                loss_sum += loss * count
                squared_sum += sse
                points += count
                tables.append(
                    {
                        "source": record["source"],
                        "loss": loss,
                        "rmse": math.sqrt(sse / count),
                        "points": count,
                    }
                )
            completed += len(group)
            if completed >= report_at or completed == len(records):
                elapsed = perf_counter() - started
                eta = elapsed / completed * (len(records) - completed)
                logger.info(
                    "VALIDATION tables {:,}/{:,} ({:.1f}%) | ETA {}",
                    completed,
                    len(records),
                    100 * completed / len(records),
                    duration(eta),
                )
                emit_training_event(
                    "validation_progress",
                    validation_tables_completed=completed,
                    validation_total_tables=len(records),
                    validation_eta_seconds=eta,
                )
                report_at = completed + 1000
    finally:
        model.train(was_training)
    if not points:
        raise ValueError("No validation labels")
    result = {
        "loss": loss_sum / points,
        "rmse": math.sqrt(squared_sum / points),
        "points": points,
        "tables": tables,
        "seconds": perf_counter() - started,
    }
    logger.info(
        "Full batched validation: loss={} RMSE={} tables={} batch_size={} seconds={}",
        result["loss"],
        result["rmse"],
        len(tables),
        batch_size,
        result["seconds"],
    )
    return result

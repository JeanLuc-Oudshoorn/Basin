"""Fixed validation origins, streamed from dataset-disjoint source Parquets."""

import hashlib
import heapq
import math
from collections import defaultdict
from time import perf_counter

import pandas as pd
import torch
from batching import pack_records
from loguru import logger
from training_log import emit_training_event
from training_progress import duration
from transform import ROOT, file_digest, frame_window


def select_validation_tables(records, config, *, max_tables=None):
    """Select table metadata using a bounded heap; no raw tensors or origins."""
    settings = config["training"].get("validation_sampling", {})
    count, seed = settings.get("tables", 25000), settings.get("seed", config["seed"])
    percentiles = settings.get("percentiles", [1, 25, 50, 75, 99])
    if isinstance(count, bool) or not isinstance(count, int) or count < 1:
        raise ValueError("training.validation_sampling.tables must be positive")
    if isinstance(seed, bool) or not isinstance(seed, int):
        raise ValueError("training.validation_sampling.seed must be an integer")
    if len(percentiles) != 5:
        raise ValueError(
            "training.validation_sampling.percentiles must have five entries"
        )
    if max_tables is not None and max_tables < 1:
        raise ValueError(
            "Need separate tables for training, validation and test; no training table would remain"
        )
    patch = config["model"]["patch_size"]
    eligible_count = 0

    def candidates():
        nonlocal eligible_count
        for record in records:
            if record["source_rows"] >= 10 * patch:
                eligible_count += 1
                yield record

    selected = heapq.nsmallest(
        min(count, max_tables) if max_tables is not None else count,
        candidates(),
        key=lambda record: (
            hashlib.sha256(
                f"validation:{seed}:{record.get('source_key', record['source'])}".encode()
            ).hexdigest(),
            record["source"],
        ),
    )
    return selected, eligible_count, count, seed, percentiles


def select_validation_origins(records, config, *, max_tables=None):
    from training_protocol import sampled_origins

    selected, eligible_count, count, seed, percentiles = select_validation_tables(
        records, config, max_tables=max_tables
    )
    patch = config["model"]["patch_size"]
    chosen = [
        (
            record,
            sampled_origins(
                record["source_rows"], patch, percentiles, prediction_patches=4
            ),
        )
        for record in sorted(selected, key=lambda r: r["source"])
    ]
    if not chosen:
        raise ValueError(
            "Sampled validation requires tables with five distinct complete forecast origins (at least 160 points for patch_size=16)"
        )
    if len(chosen) < count:
        logger.warning(
            "Validation requested {} tables; selecting {} while retaining training data",
            count,
            len(chosen),
        )
    result = []
    for record, origins in chosen:
        metadata = {
            k: v
            for k, v in record.items()
            if k not in ("batch", "prepared", "initial_embeddings")
        }
        for origin in origins:
            result.append(
                dict(
                    metadata,
                    id=f"{record['id']}_monitor_{origin}",
                    validation_origin=origin,
                    cutoff=origin,
                    context_rows=[
                        max(0, origin - config["data"]["context_length"]),
                        origin,
                    ],
                    label_rows=[origin, origin + 4 * patch],
                    split="evaluation",
                )
            )
    audit = {
        "in_sample": False,
        "dataset_disjoint": True,
        "requested_tables": count,
        "eligible_tables": eligible_count,
        "selected_tables": len(chosen),
        "forecast_cases": len(result),
        "origins_per_table": 5,
        "forecast_tokens": 4,
        "seed": seed,
        "percentiles": percentiles,
        "selection": "fixed seeded SHA256 ranking of independent non-test sources with five distinct eligible origins; selected tables excluded from training",
        "source_digest": hashlib.sha256(
            "\n".join(r["source"] for r, _ in chosen).encode()
        ).hexdigest(),
    }
    logger.info(
        "Dataset-disjoint validation: {} reserved tables, {} forecast requests; no gradient training on these tables",
        len(chosen),
        len(result),
    )
    return result, audit


def validation_windows(records, config, *, encoder=None):
    """Load one source at a time; never retain a corpus of context tensors."""
    previous, frame = None, None
    patch = config["model"]["patch_size"]
    for descriptor in records:
        source = descriptor["source"]
        if source != previous:
            path = ROOT / source
            if file_digest(path) != descriptor["source_sha256"]:
                raise ValueError(f"Validation source changed: {path}")
            columns = descriptor["targets"] + descriptor["roles"]["past"]
            frame = pd.read_parquet(path, columns=columns)
            previous = source
        origin = descriptor["validation_origin"]
        batch = frame_window(
            frame,
            cutoff=origin,
            context_length=min(origin, config["data"]["context_length"]),
            horizon=4 * patch,
            targets=descriptor["targets"],
            past_covariates=descriptor["roles"]["past"],
        )
        record = dict(descriptor, batch=batch)
        if encoder is not None:
            from embed import LazyPatchRecord

            yield LazyPatchRecord(record, encoder)
        else:
            yield record


@torch.inference_mode()
def validate_origins(model, records, config):
    """Batched Basin-only validation; same weighted pinball objective as training."""
    started = perf_counter()
    was_training = model.training
    model.eval()
    size = config["training"].get("validation_batch_size", 16)
    buckets = defaultdict(list)
    rows = []
    numerator = loss_mass = squared = absolute = scaled = scaled_mass = 0.0
    points = 0
    device = next(model.parameters()).device
    patch = model.config.patch_size
    weights = torch.tensor(
        model.config.horizon_loss_weights, device=device
    ).repeat_interleave(patch)[None, None]

    def consume(group):
        nonlocal numerator, loss_mass, squared, absolute, scaled, scaled_mass, points
        prepared = pack_records(group)
        prepared.update(
            is_training=False,
            labels=None,
            prediction_patches=4,
            last_token_indices=torch.tensor(
                [r["prepared"]["tokens"] - 1 for r in group]
            ),
        )
        prepared = {
            k: v.to(device) if isinstance(v, torch.Tensor) else v
            for k, v in prepared.items()
        }
        labels = torch.cat([r["batch"]["future_target"] for r in group]).to(device)
        output = model.forward_encoded(model.encode_tokens(prepared), prepared, labels)
        valid = torch.isfinite(labels)
        mass = (valid * weights).flatten(1).sum(1)
        errors = torch.where(valid, output.point.float() - labels, 0)
        sses = (errors.square() * weights).flatten(1).sum(1)
        maes = (errors.abs() * weights).flatten(1).sum(1)
        counts = valid.flatten(1).sum(1)
        denominators = []
        for r in group:
            context = r["batch"]["context"].float()
            pairs = torch.isfinite(context[..., 1:]) & torch.isfinite(context[..., :-1])
            changes = torch.where(
                pairs, (context[..., 1:] - context[..., :-1]).abs(), 0
            )
            denominator = changes.sum(-1) / pairs.sum(-1).clamp_min(1)
            denominators.append(denominator)
        scales = torch.cat(denominators).to(device)[..., None]
        defined = torch.isfinite(scales) & (scales > 0)
        ase = torch.where(defined, errors.abs() / torch.where(defined, scales, 1), 0)
        ase_sum = (ase * valid * weights).flatten(1).sum(1)
        ase_mass = (defined * valid * weights).flatten(1).sum(1)
        values = (
            torch.stack(
                [output.per_sample_loss, mass, sses, maes, ase_sum, ase_mass, counts],
                dim=1,
            )
            .cpu()
            .tolist()
        )
        for record, (loss, weight, sse, mae, ase_value, ase_weight, count) in zip(
            group, values
        ):
            if (
                not all(
                    math.isfinite(value)
                    for value in (loss, weight, sse, mae, ase_value)
                )
                or weight <= 0
            ):
                raise FloatingPointError(f"Invalid validation forecast: {record['id']}")
            numerator += loss * weight
            loss_mass += weight
            squared += sse
            absolute += mae
            scaled += ase_value
            scaled_mass += ase_weight
            points += int(count)
            rows.append(
                {
                    "source": record["source"],
                    "origin": record["validation_origin"],
                    "loss": loss,
                    "loss_mass": weight,
                    "rmse": math.sqrt(sse / weight),
                    "mae": mae / weight,
                    "mase": ase_value / ase_weight if ase_weight else None,
                    "points": int(count),
                }
            )

    report_at = 500
    try:
        for record in validation_windows(records, config, encoder=model):
            record["prepared"] = record["prepared"]  # bounded, per-bucket reuse only
            key = (
                len(record["targets"]),
                len(record["roles"]["past"]),
                record["prepared"]["padded_tokens"],
            )
            bucket = buckets[key]
            bucket.append(record)
            if len(bucket) == size:
                consume(bucket)
                bucket.clear()
            if len(rows) >= report_at:
                elapsed = perf_counter() - started
                logger.info(
                    "VALIDATION forecasts {:,}/{:,} | ETA {}",
                    len(rows),
                    len(records),
                    duration(elapsed / max(1, len(rows)) * (len(records) - len(rows))),
                )
                emit_training_event(
                    "validation_progress",
                    validation_forecasts_completed=len(rows),
                    validation_total_forecasts=len(records),
                )
                report_at = len(rows) + 500
        for bucket in buckets.values():
            if bucket:
                consume(bucket)
    finally:
        model.train(was_training)
    if not loss_mass:
        raise ValueError("No validation labels")
    return {
        "loss": numerator / loss_mass,
        "rmse": math.sqrt(squared / loss_mass),
        "mae": absolute / loss_mass,
        "mase": scaled / scaled_mass if scaled_mass else None,
        "points": points,
        "loss_mass": loss_mass,
        "tables": rows,
        "table_count": len({r["source"] for r in records}),
        "forecast_count": len(rows),
        "in_sample": False,
        "dataset_disjoint": True,
        "seconds": perf_counter() - started,
    }

"""Dataset isolation and full reserved-tail validation for the normal trainer."""

import hashlib
import heapq
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


def sampled_origins(
    length, patch_size, percentiles=(1, 25, 50, 75, 99), *, prediction_patches=1
):
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
    if (
        isinstance(prediction_patches, bool)
        or not isinstance(prediction_patches, int)
        or prediction_patches < 1
    ):
        raise ValueError("prediction_patches must be a positive integer")
    # Preserve patch-aligned starts and two context patches, while reserving
    # the entire requested future block (including when length is not aligned).
    eligible = (
        list(
            range(
                2 * patch_size, length - prediction_patches * patch_size + 1, patch_size
            )
        )
        if prediction_patches > 1
        else list(rolling_origins(length, patch_size))
    )
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


def holdout_request(config):
    """(tables, fraction): a fixed table count takes precedence over a fraction.

    The saved split records the same pair, so the holdout stage can refuse a
    split that was made under a different policy.
    """
    training = config["training"]
    tables = training.get("holdout_tables")
    if tables is not None:
        if isinstance(tables, bool) or not isinstance(tables, int) or tables < 1:
            raise ValueError("training.holdout_tables must be a positive integer")
        return tables, None
    fraction = training.get("holdout_fraction", 0.1)
    if isinstance(fraction, bool) or not 0 < fraction < 1:
        raise ValueError("training.holdout_fraction must be between zero and one")
    return None, fraction


def split_datasets(records, config):
    """Split whole admitted sources, never windows from the same source."""
    from augmentation_families import POLICY, excluded_evaluation_sources
    from startup_index import fetch, scan

    patch = config["model"]["patch_size"]
    minimum = max(48, 3 * patch)
    tables, fraction = holdout_request(config)
    logger.info(
        "Split startup: scanning {:,} record metadata entries for eligible sources",
        len(records),
    )
    keys, hashes, source_rows, short_sources = {}, {}, {}, set()
    # A sharded corpus is passed over through its start-up index: both loops
    # over all records use a few small fields of each (see startup_index.py).
    for record in scan(records):
        source = record["source"]
        if record["source_rows"] >= minimum:
            keys[source] = record.get("source_key", source)
            hashes[source] = record["source_sha256"]
            source_rows[source] = record["source_rows"]
        else:
            short_sources.add(source)
    sources = keys.keys()
    if len(sources) < 2:
        raise ValueError(
            f"Need at least two admitted datasets with >= {minimum} timesteps"
        )
    # Hash ranking is independent of input/window order and Python hash randomization.
    logger.info(
        "Split startup: screening augmentation families for {:,} sources", len(sources)
    )
    family_sources = excluded_evaluation_sources(sources)
    independent_sources = set(sources)
    independent_sources.difference_update(family_sources)
    independent_eligible_count = len(independent_sources)
    if len(independent_sources) < 2:
        raise ValueError(
            "Need at least two independent datasets for validation/test; augmentation families are training-only"
        )
    if tables is None:
        count = min(
            len(independent_sources) - 1,
            math.ceil(len(independent_sources) * fraction),
        )
    elif tables >= len(independent_sources):
        # A fixed test set is never silently shrunk: a smaller one would not
        # be comparable with runs that report the configured size.
        raise ValueError(
            f"training.holdout_tables={tables:,} needs more than {tables:,} "
            f"independent tables, but only {len(independent_sources):,} of "
            f"{len(sources):,} admitted tables have no coarse, aggregate or "
            "sign-inverted copy; generate more tables or lower the setting"
        )
    else:
        count = tables
    logger.info(
        "Split startup: selecting {:,} test tables from {:,} independent sources using bounded ranking",
        count,
        len(independent_sources),
    )
    ranked = heapq.nsmallest(
        count,
        independent_sources,
        key=lambda s: (
            hashlib.sha256(f"{config['seed']}:{keys[s]}".encode()).hexdigest(),
            s,
        ),
    )
    test_sources = set(ranked)
    train_sources = set(sources)
    train_sources.difference_update(test_sources)
    independent_sources.difference_update(test_sources)
    independent_train_sources = independent_sources
    sampled = config["data"].get("validation_policy") == "sampled_training_origins"
    validation_sources = independent_train_sources
    if sampled:
        from rolling_validation import select_validation_tables

        logger.info(
            "Split startup: selecting reserved validation table IDs before materializing windows"
        )
        selected, validation_eligible_count, _, _, _ = select_validation_tables(
            (
                {"source": s, "source_key": keys[s], "source_rows": source_rows[s]}
                for s in independent_train_sources
            ),
            config,
            max_tables=len(train_sources) - 1,
        )
        validation_sources = {r["source"] for r in selected}
        train_sources -= validation_sources
    del source_rows, independent_train_sources, ranked, independent_sources
    logger.info(
        "Split startup: selecting training/evaluation windows from metadata (no embeddings computed)"
    )
    from sharded_records import EncodedRecords, RecordSelection

    compact = isinstance(records, EncodedRecords)
    train = RecordSelection(records) if compact else []
    validation, test = [], []
    missing_train_sources = set(train_sources)
    # A compact corpus is scanned, which yields indexed fields only. The
    # selected evaluation records are needed whole: their positions are kept
    # and the records fetched below.
    for position, record in enumerate(scan(records) if compact else records):
        source = record["source"]
        if record["split"] == "train" and source in train_sources:
            if compact:
                train.indices.append(position)
            else:
                train.append(record)
            missing_train_sources.discard(source)
        elif record["split"] == "evaluation":
            if source in validation_sources:
                validation.append(position if compact else record)
            elif source in test_sources:
                test.append(position if compact else record)
    if compact:
        fetched = fetch(records, validation + test)
        validation, test = fetched[: len(validation)], fetched[len(validation) :]
    if missing_train_sources:
        raise ValueError(
            "Missing training records; rerun transform.py and index_patches.py"
        )
    del missing_train_sources
    for name, group, expected in (
        ("validation", validation, validation_sources),
        ("test", test, test_sources),
    ):
        if {r["source"] for r in group} != expected:
            raise ValueError(
                f"Missing {name} records; rerun transform.py and index_patches.py"
            )
        if name != "training" and len(group) != len(expected):
            raise ValueError(f"Expected exactly one {name} record per dataset")
    audit = {
        "schema": 1,
        "seed": config["seed"],
        "requested_test_fraction": fraction,
        "requested_test_tables": tables,
        "actual_test_fraction": count / len(sources),
        "minimum_timesteps": minimum,
        "rounding": "ceil, at least one test table and one training table"
        if tables is None
        else "fixed table count; error when too few independent tables exist",
        "admitted_eligible_tables": len(sources),
        "train_tables": len(train_sources),
        "test_tables": count,
        "train_windows": len(train),
        "train_sources": sorted(train_sources),
        "test_sources": sorted(test_sources),
        "source_keys": keys,
        "source_sha256": hashes,
        "evaluation_family_policy": POLICY,
        "evaluation_excluded_family_sources": sorted(family_sources),
        "evaluation_excluded_family_tables": len(family_sources),
        "independent_eligible_tables": independent_eligible_count,
        "test_fraction_denominator": "independent_eligible_tables",
        "excluded_short_sources": sorted(short_sources),
        "validation": f"last {config['data']['horizon'] // patch} full tokens of each training table; never used as gradient labels",
        "test": "after final optimizer update only; expanding actual context from two tokens",
    }
    if sampled:
        from rolling_validation import select_validation_origins

        logger.info("Split startup: reserving dataset-disjoint validation tables")
        validation, sampling = select_validation_origins(validation, config)
        sampling["eligible_tables"] = validation_eligible_count
        validation_sources = {r["source"] for r in validation}
        if (
            not train
            or train_sources & validation_sources
            or test_sources & validation_sources
        ):
            raise ValueError(
                "Training, validation and test tables must be nonempty and disjoint"
            )
        audit["schema"] = 3
        audit["validation_split_policy"] = "dataset_disjoint_v1"
        audit["validation_sources"] = sorted(validation_sources)
        audit["validation_tables"] = len(validation_sources)
        audit["actual_validation_fraction"] = len(validation_sources) / len(sources)
        audit["validation"] = (
            f"dataset-disjoint: fixed reserved tables at five complete {sampling['forecast_tokens']}-token forecast origins; no windows or labels from these tables enter gradient training"
        )
        audit["validation_sampling"] = sampling
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
    """Weighted normalized pinball over every training table's reserved tail."""
    if (
        batch_size > 1
        and getattr(getattr(model, "config", None), "objective", None) == "next_patch"
    ):
        return validate_batched(model, records, batch_size)
    started = perf_counter()
    device = next(model.parameters()).device
    was_training = model.training
    model.eval()
    loss_sum = squared_sum = loss_weight = 0.0
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
            mass = (
                float(output.loss_mass.sum())
                if getattr(output, "loss_mass", None) is not None
                else count
            )
            loss_sum += loss * mass
            loss_weight += mass
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
        "loss": loss_sum / loss_weight,
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
    loss_sum = squared_sum = loss_weight = 0.0
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
                prediction_patches=math.ceil(
                    labels.shape[-1] / model.config.patch_size
                ),
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
            masses = (
                output.loss_mass
                if getattr(output, "loss_mass", None) is not None
                else counts.float()
            )
            metrics = (
                torch.stack(
                    [output.per_sample_loss, sses, counts.float(), masses], dim=1
                )
                .cpu()
                .tolist()
            )
            for record, (loss, sse, count, mass) in zip(group, metrics):
                if count < 1 or not math.isfinite(loss) or not math.isfinite(sse):
                    raise FloatingPointError(
                        f"Invalid validation metrics: {record['source']}"
                    )
                count = int(count)
                loss_sum += loss * mass
                loss_weight += mass
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
        "loss": loss_sum / loss_weight,
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

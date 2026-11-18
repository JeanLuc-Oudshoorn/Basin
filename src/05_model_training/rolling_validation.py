"""Fixed validation origins, streamed from dataset-disjoint source Parquets."""

import hashlib
import heapq
import json
import math
import os
from collections import defaultdict
from pathlib import Path
from time import perf_counter
from uuid import uuid4

import numpy as np
import pandas as pd
import torch
from batching import pack_records
from loguru import logger
from training_log import emit_training_event
from training_progress import duration
from transform import ROOT, file_digest, frame_window, location


def forecast_patches(config):
    """Forecast tokens per validation request: the model's full direct block."""
    return config["model"].get("output_patches", 4)


def table_mase_profile(p10, median, p90):
    """One number for "did the 10th, 50th and 90th percentile improve on average".

    The geometric mean of the three falls exactly when their average relative
    change is an improvement, so a 2% gain at the 10th percentile counts as much
    as a 2% gain at the 90th. A plain average would follow the 90th percentile,
    which is several times larger than the other two.
    """
    values = (p10, median, p90)
    if any(value is None for value in values):
        return None
    return float(np.exp(np.mean(np.log(np.maximum(values, 1e-12)))))


def robust_table_summary(rows, *, trim=0.05):
    """Validation statistics that a handful of extreme forecasts cannot move.

    The mean loss over all forecasts is dominated by a few of them (five of
    125,000 made up 54% of it on the 500M run), so it is a poor signal for
    choosing weights. Each table's origins are first combined into one score;
    the summaries are then taken across tables. A table's MASE is the mean over
    its origins with a defined scale; tables without one are left out.
    """
    frame = pd.DataFrame(rows)
    weighted = frame["loss"] * frame["loss_mass"]
    grouped = pd.DataFrame(
        {"source": frame["source"], "weighted": weighted, "mass": frame["loss_mass"]}
    ).groupby("source", sort=True)
    table_loss = np.sort(
        (grouped["weighted"].sum() / grouped["mass"].sum()).to_numpy(dtype=float)
    )
    drop = int(len(table_loss) * trim)
    central = table_loss[drop : len(table_loss) - drop] if drop else table_loss
    table_mase = (
        frame.assign(mase=pd.to_numeric(frame["mase"], errors="coerce"))
        .groupby("source", sort=True)["mase"]
        .mean()
        .dropna()
        .to_numpy(dtype=float)
    )
    p10, median, p90 = (
        (float(value) for value in np.quantile(table_mase, [0.1, 0.5, 0.9]))
        if len(table_mase)
        else (None, None, None)
    )
    return {
        "table_mase_profile": table_mase_profile(p10, median, p90),
        "table_mase_p10": p10,
        "median_table_mase": median,
        "table_mase_p90": p90,
        "tables_with_defined_mase": len(table_mase),
        "trimmed_table_loss": float(central.mean()),
        "median_table_loss": float(np.median(table_loss)),
        "table_loss_p90": float(np.quantile(table_loss, 0.9)),
        "table_loss_p99": float(np.quantile(table_loss, 0.99)),
        "trimmed_fraction_each_side": trim,
    }


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
    # Five distinct origins, each after two context tokens and before a full
    # forecast block (ten tokens for the four-token models).
    minimum_rows = (forecast_patches(config) + 6) * patch
    eligible_count = 0

    def candidates():
        nonlocal eligible_count
        for record in records:
            if record["source_rows"] >= minimum_rows:
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
    leads = forecast_patches(config)
    chosen = [
        (
            record,
            sampled_origins(
                record["source_rows"], patch, percentiles, prediction_patches=leads
            ),
        )
        for record in sorted(selected, key=lambda r: r["source"])
    ]
    if not chosen:
        raise ValueError(
            f"Sampled validation requires tables with five distinct complete forecast origins (at least {(leads + 6) * patch} points)"
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
                    label_rows=[origin, origin + leads * patch],
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
        "forecast_tokens": leads,
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


PANEL_FORMAT = "validation_panel_v1"
PANEL_VALUES = "validation_panel.bin"
PANEL_INDEX = "validation_panel.pt"


def panel_key(records, config):
    """Identity of the validation panel: every request and every source file."""
    digest = hashlib.sha256()
    digest.update(
        json.dumps(
            {
                "format": PANEL_FORMAT,
                "patch_size": config["model"]["patch_size"],
                "forecast_patches": forecast_patches(config),
                "context_length": config["data"]["context_length"],
            },
            sort_keys=True,
        ).encode()
    )
    for descriptor in records:
        digest.update(
            json.dumps(
                [
                    descriptor["source"],
                    descriptor["source_sha256"],
                    descriptor["validation_origin"],
                    descriptor["targets"],
                    descriptor["roles"]["past"],
                ]
            ).encode()
        )
    return digest.hexdigest()


class _PanelWriter:
    """Appends each request's float32 windows to one file while they are read."""

    def __init__(self, directory, key, horizon):
        self.directory = Path(directory)
        self.directory.mkdir(parents=True, exist_ok=True)
        self.key, self.horizon = key, horizon
        self.temporary = self.directory / f".{PANEL_VALUES}.{uuid4().hex}.tmp"
        self.stream = self.temporary.open("wb")
        self.offsets, self.lengths, self.targets, self.covariates = [0], [], [], []

    def add(self, batch):
        context, future = batch["context"], batch["future_target"]
        past = batch.get("past_covariates")
        if future.shape[-1] != self.horizon:
            raise ValueError("Validation window has an unexpected horizon")
        count = 0
        for tensor in (context, future, past):
            if tensor is not None:
                values = tensor.contiguous().numpy()
                if values.dtype != np.float32:
                    raise ValueError("Validation windows must be float32")
                self.stream.write(values.tobytes())
                count += values.size
        self.offsets.append(self.offsets[-1] + count)
        self.lengths.append(context.shape[-1])
        self.targets.append(context.shape[1])
        self.covariates.append(0 if past is None else past.shape[1])

    def abandon(self):
        self.stream.close()
        self.temporary.unlink(missing_ok=True)

    def finish(self):
        from checkpoints import atomic_torch_save

        self.stream.flush()
        os.fsync(self.stream.fileno())
        self.stream.close()
        # The index is written last and names the size of the values file, so
        # a half-written panel is never mistaken for a complete one.
        os.replace(self.temporary, self.directory / PANEL_VALUES)
        atomic_torch_save(
            {
                "format": PANEL_FORMAT,
                "key": self.key,
                "horizon": self.horizon,
                "bytes": 4 * self.offsets[-1],
                "offsets": torch.tensor(self.offsets, dtype=torch.int64),
                "lengths": torch.tensor(self.lengths, dtype=torch.int32),
                "targets": torch.tensor(self.targets, dtype=torch.int32),
                "covariates": torch.tensor(self.covariates, dtype=torch.int32),
            },
            self.directory / PANEL_INDEX,
        )
        logger.info(
            "Validation panel saved: {:,} forecast requests, {:.2f} GiB in {}; "
            "later checks read it instead of the source tables",
            len(self.lengths),
            4 * self.offsets[-1] / 2**30,
            self.directory / PANEL_VALUES,
        )


def open_panel(directory, key, count):
    """The saved panel for exactly these requests, or None."""
    values_path, index_path = (
        Path(directory) / PANEL_VALUES,
        Path(directory) / PANEL_INDEX,
    )
    if not values_path.is_file() or not index_path.is_file():
        return None
    index = torch.load(index_path, map_location="cpu", weights_only=True)
    if (
        index.get("format") != PANEL_FORMAT
        or index.get("key") != key
        or len(index["lengths"]) != count
        or values_path.stat().st_size != index["bytes"]
    ):
        return None
    # Copy-on-write mapping: pages come from the file on demand, stay
    # evictable, and nothing written to the arrays can reach the file.
    values = np.memmap(values_path, dtype=np.float32, mode="c")
    return {
        "values": values,
        "horizon": int(index["horizon"]),
        "offsets": index["offsets"].tolist(),
        "lengths": index["lengths"].tolist(),
        "targets": index["targets"].tolist(),
        "covariates": index["covariates"].tolist(),
    }


def panel_batch(panel, position):
    """The window `frame_window` returned when the panel was built."""
    start = panel["offsets"][position]
    length = panel["lengths"][position]
    targets = panel["targets"][position]
    covariates = panel["covariates"][position]
    horizon = panel["horizon"]
    values = panel["values"]

    def take(rows, columns):
        nonlocal start
        tensor = torch.from_numpy(values[start : start + rows * columns]).view(
            1, rows, columns
        )
        start += rows * columns
        return tensor

    batch = {
        "context": take(targets, length),
        "horizon": horizon,
        "future_target": take(targets, horizon),
    }
    if covariates:
        batch["past_covariates"] = take(covariates, length)
    return batch


def panel_directory(config):
    """Where `training.validation_cache` keeps the panel; None when it is off.

    `true` means the training output folder. A path names another folder, for
    example one on WSL's own disk: reading a large file through WSL's mount of
    a Windows drive was measured at 5 MiB/s here, against 106 MiB/s for the
    same drive read natively. Each training folder gets its own subfolder.
    """
    setting = config["training"].get("validation_cache", False)
    if setting is True:
        return location(config, "training")
    if isinstance(setting, str) and setting.strip():
        folder = hashlib.sha256(str(location(config, "training")).encode()).hexdigest()[
            :16
        ]
        return Path(setting).expanduser() / folder
    if setting in (False, None):
        return None
    raise ValueError("training.validation_cache must be true, false or a folder path")


def validation_windows(records, config, *, encoder=None, cache_directory=None):
    """Load one source at a time; never retain a corpus of context tensors.

    With `cache_directory`, the windows of the first pass are saved there and
    later passes over the same requests read them back instead of opening,
    hashing and slicing every source table again (about 22 ms per table through
    WSL's Windows-drive mount). The saved values are the float32 arrays the
    first pass produced, so every later check scores exactly the same numbers.
    Source files are verified against their recorded SHA-256 when the panel is
    built; a changed request list or source hash builds a new panel.
    """
    previous, frame = None, None
    patch = config["model"]["patch_size"]
    leads = forecast_patches(config)
    panel = writer = None
    if cache_directory is not None:
        key = panel_key(records, config)
        panel = open_panel(cache_directory, key, len(records))
        if panel is None:
            writer = _PanelWriter(cache_directory, key, leads * patch)
            logger.info(
                "Validation panel: reading {:,} forecast requests from their source "
                "tables and saving them for later checks",
                len(records),
            )
        else:
            logger.info(
                "Validation panel: {:,} forecast requests from {} (source tables "
                "were verified when it was saved)",
                len(records),
                Path(cache_directory) / PANEL_VALUES,
            )
    try:
        for position, descriptor in enumerate(records):
            if panel is not None:
                batch = panel_batch(panel, position)
            else:
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
                    horizon=leads * patch,
                    targets=descriptor["targets"],
                    past_covariates=descriptor["roles"]["past"],
                )
                if writer is not None:
                    writer.add(batch)
            record = dict(descriptor, batch=batch)
            if encoder is not None:
                from index_patches import LazyPatchRecord

                yield LazyPatchRecord(record, encoder)
            else:
                yield record
        if writer is not None:
            writer.finish()
            writer = None
    finally:
        if writer is not None:
            writer.abandon()  # interrupted before the last request


def validation_engine(model, config):
    """CUDA-graph replay for full validation batches; None keeps the plain forward.

    Kept on the model between checks. It is separate from `model.fast_inference`,
    so forecasting behaviour and the holdout cache are unaffected.
    """
    options = config["training"]
    if (
        not options.get("validation_cuda_graphs", False)
        or not next(model.parameters()).is_cuda
        or model.config.objective != "next_patch"
    ):
        return None
    engine = getattr(model, "validation_inference", None)
    if engine is None:
        from fast_inference import FastInference

        engine = model.validation_inference = FastInference(
            model,
            max_graphs=options.get("validation_graph_cache_size", 256),
            graph_memory_gib=options.get("validation_graph_memory_gib", 2.0),
        )
    return engine


@torch.inference_mode()
def validate_origins(model, records, config):
    """Batched Basin-only validation; same weighted pinball objective as training."""
    started = perf_counter()
    was_training = model.training
    model.eval()
    size = config["training"].get("validation_batch_size", 16)
    engine = validation_engine(model, config)
    replay_before = (
        (engine.graphs.requests, engine.graphs.replay_requests, engine.graphs.captures)
        if engine is not None
        else None
    )
    median = model.config.quantiles.index(0.5)
    from multi_horizon import weighted_forecast_loss

    buckets = defaultdict(list)
    rows = []
    numerator = loss_mass = squared = absolute = scaled = scaled_mass = 0.0
    points = without_labels = without_history = 0
    device = next(model.parameters()).device
    patch = model.config.patch_size
    leads = forecast_patches(config)
    if leads != model.config.output_patches:
        raise ValueError(
            "Validation forecasts the model's full block: config model.output_patches "
            "must match the model"
        )
    weights = torch.tensor(
        model.config.horizon_loss_weights, device=device
    ).repeat_interleave(patch)[None, None]

    def consume(group):
        nonlocal numerator, loss_mass, squared, absolute, scaled, scaled_mass
        nonlocal points, without_labels, without_history
        prepared = pack_records(group)
        prepared.update(
            is_training=False,
            labels=None,
            prediction_patches=leads,
            last_token_indices=torch.tensor(
                [r["prepared"]["tokens"] - 1 for r in group]
            ),
        )
        prepared = {
            k: v.to(device) if isinstance(v, torch.Tensor) else v
            for k, v in prepared.items()
        }
        labels = torch.cat([r["batch"]["future_target"] for r in group]).to(device)
        if engine is not None and len(group) == size:
            # Only full batches: a bucket's leftover batch has a size of its
            # own and would cost a capture for a single use.
            block = engine.scored_block(prepared)
            prediction, loc, scale = block[0], block[-2], block[-1]
            steps = labels.shape[-1]
            point = prediction[..., :steps, median]
            per_sample_loss = weighted_forecast_loss(
                model,
                prediction[..., :steps, :],
                labels,
                loc,
                scale,
                normalized=block[1][..., :steps, :] if len(block) == 4 else None,
            )[1]
        else:
            output = model.forward_encoded(
                model.encode_tokens(prepared), prepared, labels
            )
            point, per_sample_loss = output.point, output.per_sample_loss
        valid = torch.isfinite(labels)
        mass = (valid * weights).flatten(1).sum(1)
        errors = torch.where(valid, point.float() - labels, 0)
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
                [per_sample_loss, mass, sses, maes, ase_sum, ase_mass, counts],
                dim=1,
            )
            .cpu()
            .tolist()
        )
        for record, (loss, weight, sse, mae, ase_value, ase_weight, count) in zip(
            group, values
        ):
            # Missing values are never scored. The forecast above was still
            # made; a request is only left out when nothing can be scored at
            # all: every label is missing, or the target has no observed
            # history (then neither a scale nor a scaled error exists).
            if count == 0:
                without_labels += 1
                continue
            if not torch.isfinite(record["batch"]["context"]).any():
                without_history += 1
                continue
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
        cache_directory = panel_directory(config)
        for record in validation_windows(
            records, config, encoder=model, cache_directory=cache_directory
        ):
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
            if len(rows) + without_labels + without_history >= report_at:
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
                report_at = len(rows) + without_labels + without_history + 500
        for bucket in buckets.values():
            if bucket:
                consume(bucket)
    finally:
        model.train(was_training)
    replay = None
    if engine is not None:
        graphs = engine.graphs
        batches = graphs.requests - replay_before[0]
        replay = {
            "full_batches": batches,
            "replayed_batches": graphs.replay_requests - replay_before[1],
            "captures": graphs.captures - replay_before[2],
            "cached_shapes": len(graphs.entries),
            "cache_gib": graphs.retained_bytes / 2**30,
            "fallback_reasons": dict(graphs.fallback_reasons),
        }
        if not config["training"].get("validation_graph_retain", True):
            # Hand the graph memory back to training until the next check.
            engine.clear()
            torch.cuda.empty_cache()
        logger.info("VALIDATION graph replay: {}", replay)
    if not loss_mass:
        raise ValueError("No validation labels")
    return {
        **({"graph_replay": replay} if replay is not None else {}),
        "loss": numerator / loss_mass,
        "rmse": math.sqrt(squared / loss_mass),
        "mae": absolute / loss_mass,
        "mase": scaled / scaled_mass if scaled_mass else None,
        **robust_table_summary(rows),
        "points": points,
        "loss_mass": loss_mass,
        "tables": rows,
        "table_count": len({r["source"] for r in records}),
        "forecast_count": len(rows),
        "unscored_forecasts_without_observed_labels": without_labels,
        "unscored_forecasts_without_observed_history": without_history,
        "in_sample": False,
        "dataset_disjoint": True,
        "seconds": perf_counter() - started,
    }

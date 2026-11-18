"""Index the patch inputs of every window; patch embeddings are learned in training."""

import argparse
import math
import multiprocessing
import os
import queue
import shutil
import signal
import sys
from bisect import bisect_right
from concurrent.futures import ProcessPoolExecutor
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path
from threading import Event, Thread
from time import perf_counter
from uuid import uuid4

import torch
from loguru import logger
from torch import nn
from torch.nn import functional as F

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src/03_transforming_input_data"))
from patch_index_progress import (
    PatchIndexCheckpoint,
    PatchIndexProgress,
    detail_line,
    patch_index_options,
)
from sharded_records import ShardedRecords
from transform import (
    file_digest,
    load_config,
    location,
    read_artifact,
    run_context,
    save_artifact,
    timed,
)
from transform_progress import atomic_json, atomic_write, digest


@dataclass(frozen=True)
class ForecastConfig:
    patch_size: int = 32
    width: int = 128
    layers: int = 2
    variate_heads: int = 4
    max_patches: int = 512
    quantiles: tuple[float, ...] = (0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9)
    objective: str = "masked_horizon"
    context_length: int = 2048
    gradient_checkpointing: bool = False
    output_patches: int = 1
    multi_horizon_weight: float = 0.5
    horizon_loss_weights: tuple[float, ...] | None = None
    # Every field below defaults to the architecture of the archived 447M/500M
    # models, so their checkpoints load unchanged. See
    # docs/model_training/ARCHITECTURE_V2_2026-11-15.md for the reasons.
    head: str = "lead_offset"  # "direct": one projection emits every lead
    variate_every: int = 1  # cross-variable mixing after every Nth RWKV block
    temporal_every: int = 0  # causal temporal attention after every Nth block
    gated_mixing: bool = False  # mixing branches start closed (zero output init)
    value_transform: str = "none"  # "asinh" after causal standardisation
    fp32_io: bool = False  # float32 patch embedding, final norm and head
    rwkv_init: str = "constructor"  # "upstream": RWKV-7 training-recipe init

    def __post_init__(self):
        if self.horizon_loss_weights == "equal":
            # Every forecast timestep carries the same weight. The count is
            # validated below, after the weights have their final form.
            object.__setattr__(
                self,
                "horizon_loss_weights",
                (1.0 / self.output_patches,) * self.output_patches
                if isinstance(self.output_patches, int) and self.output_patches > 0
                else None,
            )
        if self.head not in ("lead_offset", "direct"):
            raise ValueError("head must be 'lead_offset' or 'direct'")
        if self.head == "direct" and (
            self.objective != "next_patch" or self.horizon_loss_weights is None
        ):
            raise ValueError(
                "The direct head requires next_patch and horizon_loss_weights"
            )
        for name in ("variate_every", "temporal_every"):
            value = getattr(self, name)
            if (
                isinstance(value, bool)
                or not isinstance(value, int)
                or value < (1 if name == "variate_every" else 0)
            ):
                raise ValueError(f"{name} must be a nonnegative integer layer interval")
        if self.temporal_every and self.objective != "next_patch":
            raise ValueError("Temporal attention requires the next_patch objective")
        for name in ("gated_mixing", "fp32_io"):
            if not isinstance(getattr(self, name), bool):
                raise ValueError(f"{name} must be boolean")
        if self.value_transform not in ("none", "asinh"):
            raise ValueError("value_transform must be 'none' or 'asinh'")
        if self.value_transform != "none" and self.head != "direct":
            # Only the direct head's loss and decoding work in transformed space.
            raise ValueError("value_transform requires head='direct'")
        if self.rwkv_init not in ("constructor", "upstream"):
            raise ValueError("rwkv_init must be 'constructor' or 'upstream'")
        if self.horizon_loss_weights is not None:
            weights = tuple(self.horizon_loss_weights)
            object.__setattr__(self, "horizon_loss_weights", weights)
            if (
                self.objective != "next_patch"
                or len(weights) != self.output_patches
                or any(
                    isinstance(w, bool)
                    or not isinstance(w, (int, float))
                    or not math.isfinite(w)
                    or w <= 0
                    for w in weights
                )
                or not math.isclose(sum(weights), 1.0, abs_tol=1e-8)
            ):
                raise ValueError(
                    "horizon_loss_weights must be positive, sum to one and match next_patch output_patches"
                )
        if (
            isinstance(self.output_patches, bool)
            or not isinstance(self.output_patches, int)
            or self.output_patches < 1
        ):
            raise ValueError("output_patches must be a positive integer")
        if not 0 <= self.multi_horizon_weight <= 1:
            raise ValueError("multi_horizon_weight must be between zero and one")
        if self.output_patches > 1 and self.objective != "next_patch":
            raise ValueError("Multiple output patches require next_patch")
        if self.objective not in ("masked_horizon", "next_patch"):
            raise ValueError("Unknown forecasting objective")
        if self.patch_size < 1 or self.width < 64 or self.width % 64:
            raise ValueError(
                "patch_size must be positive; width must be a multiple of 64"
            )
        if (
            self.context_length < self.patch_size
            or self.context_length % self.patch_size
        ):
            raise ValueError("context_length must be a positive multiple of patch_size")
        if self.layers < 2 or self.max_patches < 16 or self.max_patches % 16:
            raise ValueError(
                "layers >= 2 and max_patches a positive multiple of 16 required"
            )
        if self.variate_heads < 1 or self.width % self.variate_heads:
            raise ValueError("variate_heads must divide width")
        if (
            not self.quantiles
            or tuple(sorted(set(self.quantiles))) != self.quantiles
            or not all(0 < q < 1 for q in self.quantiles)
            or 0.5 not in self.quantiles
        ):
            raise ValueError(
                "quantiles must be unique, increasing, inside (0, 1), including 0.5"
            )


class ResidualProjection(nn.Module):
    def __init__(self, inputs, outputs, hidden):
        super().__init__()
        self.skip = nn.Linear(inputs, outputs)
        self.mlp = nn.Sequential(
            nn.Linear(inputs, hidden), nn.GELU(), nn.Linear(hidden, outputs)
        )

    def forward(self, x):
        return self.skip(x) + self.mlp(x)


class PatchEncoder(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.embedding = ResidualProjection(
            3 * config.patch_size, config.width, 4 * config.width
        )
        self.roles = nn.Embedding(3, config.width)

    @staticmethod
    def _validate(x, name, batch=None, length=None):
        if x.ndim != 3 or min(x.shape) < 1 or not x.is_floating_point():
            raise ValueError(
                f"{name} must be a nonempty floating [batch, variable, time] tensor"
            )
        if batch is not None and x.shape[0] != batch:
            raise ValueError(f"{name} batch size mismatch")
        if length is not None and x.shape[-1] != length:
            raise ValueError(f"{name} time length mismatch")
        if torch.isinf(x).any():
            raise ValueError(
                f"{name} contains infinity; use NaN for missing observations"
            )

    def tokenize(
        self,
        context,
        horizon,
        *,
        past_covariates=None,
        known_covariates=None,
        future_covariates=None,
        future_target=None,
    ):
        """Predict without feeding labels into the encoder.

        known_covariates contains the historical segment of future_covariates.
        Both are required together and must have matching variable order/count.
        NaNs denote missing data. Entirely missing series use location=0, scale=1.
        """
        if self.config.objective == "next_patch":
            if known_covariates is not None or future_covariates is not None:
                raise ValueError(
                    "Next-patch mode accepts historical covariates, not future covariates"
                )
            return self.tokenize_next(context, past_covariates=past_covariates)
        self._validate(context, "context")
        if not isinstance(horizon, int) or isinstance(horizon, bool) or horizon < 1:
            raise ValueError("horizon must be a positive integer")
        b, targets, length = context.shape
        p = self.config.patch_size
        cp, hp = (length + p - 1) // p, (horizon + p - 1) // p
        tokens = cp + hp
        padded_tokens = (tokens + 15) // 16 * 16
        logger.debug(
            "Patching batch={} targets={} context={} horizon={} patch_size={} context_patches={} future_patches={} kernel_padded_patches={}",
            b,
            targets,
            length,
            horizon,
            p,
            cp,
            hp,
            padded_tokens,
        )
        if padded_tokens > self.config.max_patches:
            raise ValueError("context + horizon exceeds max_patches")
        histories, futures, roles = (
            [context],
            [context.new_full((b, targets, horizon), float("nan"))],
            [0] * targets,
        )
        if past_covariates is not None:
            self._validate(past_covariates, "past_covariates", b, length)
            histories.append(past_covariates)
            futures.append(
                context.new_full((b, past_covariates.shape[1], horizon), float("nan"))
            )
            roles += [1] * past_covariates.shape[1]
        if (known_covariates is None) != (future_covariates is None):
            raise ValueError(
                "known_covariates and future_covariates must be supplied together"
            )
        if known_covariates is not None:
            self._validate(known_covariates, "known_covariates", b, length)
            self._validate(future_covariates, "future_covariates", b, horizon)
            if known_covariates.shape[1] != future_covariates.shape[1]:
                raise ValueError("known/future covariate counts differ")
            histories.append(known_covariates)
            futures.append(future_covariates)
            roles += [2] * known_covariates.shape[1]
        history = torch.cat(histories, dim=1).float()
        future = torch.cat(futures, dim=1).float()
        # Context-only statistics: no target or future covariate leakage.
        observed = torch.isfinite(history)
        count = observed.sum(-1, keepdim=True).clamp_min(1)
        clean = history.nan_to_num()
        loc = clean.sum(-1, keepdim=True) / count
        scale = (
            ((clean - loc).square() * observed).sum(-1, keepdim=True) / count
        ).sqrt()
        # Preserve every positive finite standard deviation, regardless of units.
        # Only degenerate/unavailable scales need the division-by-zero fallback.
        scale = torch.where(
            torch.isfinite(scale) & (scale > 0), scale, torch.ones_like(scale)
        )
        history = F.pad(history, (cp * p - length, 0), value=float("nan"))
        future = F.pad(future, (0, hp * p - horizon), value=float("nan"))
        values = torch.cat([history, future], dim=-1)
        mask = torch.isfinite(values)
        values = ((values - loc) / scale).nan_to_num().clamp(-1e4, 1e4)
        time = torch.arange(-cp * p, hp * p, device=context.device).float() / (cp * p)
        time = time.expand_as(values)
        features = torch.cat(
            [z.reshape(b, len(roles), tokens, p) for z in (values, mask.float(), time)],
            dim=-1,
        )
        return {
            "features": features,
            "roles": torch.tensor(roles, device=context.device),
            "loc": loc,
            "scale": scale,
            "b": b,
            "targets": targets,
            "horizon": horizon,
            "cp": cp,
            "hp": hp,
            "tokens": tokens,
            "padded_tokens": padded_tokens,
        }

    def tokenize_next(self, context, *, past_covariates=None, training=False):
        """Causal patch features and shifted labels; no future-dependent normalization."""
        p, limit = self.config.patch_size, self.config.context_length
        self._validate(context if training else context[..., -limit:], "context")
        b, targets, length = context.shape
        if past_covariates is not None:
            if past_covariates.shape[-1] != length:
                raise ValueError("past_covariates time length mismatch")
            self._validate(
                past_covariates if training else past_covariates[..., -limit:],
                "past_covariates",
                b,
                length if training else min(length, limit),
            )
        if training and length > limit + p:
            raise ValueError("Training chunk exceeds context plus one target patch")
        if not training and length > limit:
            logger.info(
                "Truncating {} old observations; using final {}", length - limit, limit
            )
            context = context[..., -limit:]
            if past_covariates is not None:
                past_covariates = past_covariates[..., -limit:]
        history = (
            context
            if past_covariates is None
            else torch.cat([context, past_covariates], dim=1)
        )
        roles = [0] * targets + [1] * (history.shape[1] - targets)
        length = history.shape[-1]
        history = F.pad(history.float(), ((-length) % p, 0), value=float("nan"))
        patches = history.reshape(b, len(roles), -1, p)
        valid = torch.isfinite(patches)
        clean = patches.nan_to_num().double()
        count = valid.sum(-1, keepdim=True).cumsum(-2).clamp_min(1)
        loc = clean.sum(-1, keepdim=True).cumsum(-2) / count
        variance = (
            clean.square().sum(-1, keepdim=True).cumsum(-2) / count - loc.square()
        )
        scale = variance.clamp_min(0).sqrt()
        scale = torch.where(
            torch.isfinite(scale) & (scale > 0), scale, torch.ones_like(scale)
        ).float()
        loc = loc.float()
        values = ((patches - loc) / scale).nan_to_num().clamp(-1e4, 1e4)
        if self.config.value_transform == "asinh":
            # Near-linear around zero, logarithmic for extreme standardised
            # values. Labels and quantile outputs use the same space.
            values = torch.asinh(values)
        time = (
            torch.arange(history.shape[-1], device=history.device)
            .float()
            .reshape(1, 1, -1, p)
            / limit
        )
        features = torch.cat([values, valid.float(), time.expand_as(values)], dim=-1)
        labels = None
        if training:
            if patches.shape[-2] < 2:
                raise ValueError(
                    "Need at least two observed patches for next-patch training"
                )
            labels = patches[:, :targets, 1:].contiguous()
            features, loc, scale = features[:, :, :-1], loc[:, :, :-1], scale[:, :, :-1]
        tokens = features.shape[-2]
        padded_tokens = (tokens + 15) // 16 * 16
        if padded_tokens > self.config.max_patches:
            raise ValueError("Token count exceeds configured kernel context")
        # Once per table, in training and in validation: TRACE keeps it out of
        # the DEBUG log file, which is written through the Windows-drive mount.
        logger.trace(
            "Next-patch tokens={} padded={} variables={} training={}",
            tokens,
            padded_tokens,
            len(roles),
            training,
        )
        return {
            "features": features,
            "roles": torch.tensor(roles, device=context.device),
            "loc": loc,
            "scale": scale,
            "b": b,
            "targets": targets,
            "tokens": tokens,
            "padded_tokens": padded_tokens,
            "is_training": training,
            "labels": labels,
            "horizon": p,
        }

    def encode_tokens(self, prepared):
        x = self.embedding(prepared["features"].to(self.embedding.skip.weight.dtype))
        roles = self.roles(prepared["roles"])[None, :, None, :]
        # With fp32_io the projection reads the raw features in float32; the
        # sum is rounded once, to the backbone's dtype.
        x = x.to(roles.dtype) + roles
        return F.pad(x, (0, 0, 0, prepared["padded_tokens"] - prepared["tokens"]))


class LazyPatchRecord(dict):
    """A raw window with disposable, per-access features; never caches projections."""

    def __init__(self, record, encoder):
        super().__init__(record)
        self.encoder = encoder

    def __missing__(self, key):
        if key not in ("prepared", "initial_embeddings"):
            raise KeyError(key)
        # Unlike inference_mode, no_grad tensors can subsequently participate in
        # a differentiable embedding forward during training.
        with torch.no_grad():
            if self.encoder.config.objective == "next_patch":
                prepared = self.encoder.tokenize_next(
                    self["batch"]["context"],
                    past_covariates=self["batch"].get("past_covariates"),
                    training=self["split"] == "train",
                )
            else:
                prepared = self.encoder.tokenize(**self["batch"])
            if key == "prepared":
                return prepared
            # Compatibility for explicit inspections of old snapshot APIs only.
            # The normal trainer computes with its own current encoder on GPU.
            return self.encoder.encode_tokens(prepared)

    def get(self, key, default=None):
        try:
            return self[key]
        except KeyError:
            return default


def open_patch_index(index, config):
    """Expose legacy record APIs while keeping raw tensors memory mapped."""
    path = location(config, "processed") / "windows.pt"
    if file_digest(path) != index["processed_sha256"]:
        raise ValueError(
            "Processed data changed: rerun index_patches.py to rebuild its index"
        )
    processed = read_artifact(path, config)
    if len(processed["records"]) != index["record_count"]:
        raise ValueError("Patch index window count mismatch; rerun index_patches.py")
    settings = dict(config["model"], quantiles=tuple(config["model"]["quantiles"]))
    # Initialization is reproducible without advancing the caller's RNG stream.
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(config["seed"])
        encoder = PatchEncoder(ForecastConfig(**settings)).eval()
    encoder.load_state_dict(index["encoder_state"])
    if processed.get("format") == "sharded_windows_v1":
        from sharded_records import EncodedRecords

        records = EncodedRecords(processed["records"], encoder, LazyPatchRecord)
    else:
        records = [LazyPatchRecord(r, encoder) for r in processed["records"]]
    return {
        **index,
        "records": records,
    }


def record_summary(record, settings):
    """Inspect one raw window without tokenization or learned projections."""
    if "batch" not in record:
        if record["split"] != "evaluation":
            raise ValueError(f"Missing raw training batch: {record['id']}")
        return {
            "id": record["id"],
            "source": record["source"],
            "split": record["split"],
            "metadata_only": True,
        }, 0
    batch = record["batch"]
    context = batch["context"]
    if context.ndim != 3 or min(context.shape) < 1:
        raise ValueError(
            f"Invalid context shape for {record['id']}: {list(context.shape)}"
        )
    variables = context.shape[1] + sum(
        batch[name].shape[1]
        for name in ("past_covariates", "known_covariates")
        if batch.get(name) is not None
    )
    p = settings.patch_size
    length = context.shape[-1]
    if settings.objective == "next_patch":
        length = (
            length
            if record["split"] == "train"
            else min(length, settings.context_length)
        )
        tokens = (length + p - 1) // p - int(record["split"] == "train")
    else:
        tokens = (length + p - 1) // p + (batch["horizon"] + p - 1) // p
    padded = (tokens + 15) // 16 * 16
    if tokens < 1 or padded > settings.max_patches:
        raise ValueError(f"Invalid token count for {record['id']}: {tokens}")
    shape = [context.shape[0], variables, padded, settings.width]
    return {
        "id": record["id"],
        "source": record["source"],
        "split": record["split"],
        "context_shape": list(context.shape),
        "input_tokens": tokens,
        "padded_tokens": padded,
        "embedding_shape_when_used": shape,
    }, math.prod(shape) * 4


def window_detail(record, settings, policy):
    """Summarize one record, after checking that it holds what its policy needs."""
    if "batch" not in record and policy != "sampled_training_origins":
        raise ValueError(f"Missing raw training batch: {record['id']}")
    return record_summary(record, settings)


# Starting worker processes costs a few seconds; smaller jobs are inspected in place.
WORKER_MINIMUM_SHARDS = 8

_worker = {}


def worker_count(options, records, start):
    """Worker processes for the records from `start` on; 0 inspects them in place."""
    if not isinstance(records, ShardedRecords) or start >= len(records):
        return 0
    if options["workers"] is not None:
        return options["workers"]
    remaining = len(records.shards) - bisect_right(records.ends, start)
    if remaining < WORKER_MINIMUM_SHARDS:
        return 0
    return min(8, (os.cpu_count() or 2) // 2)


def start_worker(processed_path, config):
    """Pool initializer: open the shard index once and leave Ctrl+C to the parent."""
    signal.signal(signal.SIGINT, signal.SIG_IGN)
    logger.remove()
    logger.add(sys.stderr, level="WARNING")
    torch.set_num_threads(1)
    settings = dict(config["model"], quantiles=tuple(config["model"]["quantiles"]))
    _worker.update(
        records=read_artifact(Path(processed_path), config)["records"],
        settings=ForecastConfig(**settings),
        policy=config["data"].get("validation_policy"),
    )


def summarize_shard(task):
    """Output lines and size estimates for records [start, stop) of one shard."""
    shard, start, stop = task
    records = _worker["records"]
    # The parent process checked this shard's checksum just before sending it.
    records.verified.add(shard)
    lines, estimates = [], []
    for position in range(start, stop):
        detail, estimated_bytes = window_detail(
            records[position], _worker["settings"], _worker["policy"]
        )
        lines.append(detail_line(detail))
        estimates.append(estimated_bytes)
    records.cached = None
    records.cached_index = None
    return lines, estimates


@contextmanager
def shard_summaries(records, start, workers, processed_path, config):
    """Iterate `summarize_shard` results for the records from `start` on, in order.

    One thread checks the shards' checksums in file order, so that a hard disk
    reads them as a single stream, and hands each verified shard to a worker
    process, which opens it and inspects its records.
    """
    pending = queue.Queue(maxsize=2 * workers)
    stop = Event()

    def offer(item):
        while not stop.is_set():
            try:
                return pending.put(item, timeout=0.2)
            except queue.Full:
                pass

    def verify_ahead(pool):
        try:
            for shard in range(bisect_right(records.ends, start), len(records.shards)):
                if stop.is_set():
                    return
                path = records.directory / records.shards[shard]["path"]
                if digest(path) != records.shards[shard]["sha256"]:
                    raise ValueError(f"Corrupt processed shard: {path}")
                first = max(start, records.ends[shard - 1] if shard else 0)
                task = shard, first, records.ends[shard]
                offer(pool.submit(summarize_shard, task))
            offer(None)
        except BaseException as error:  # raised again where results are consumed
            offer(error)

    def results():
        while True:
            item = pending.get()
            if item is None:
                return
            if isinstance(item, BaseException):
                raise item
            yield item.result()

    logger.info("Inspecting shards in {} worker processes", workers)
    # Unlike multiprocessing.Pool, this reports a worker that died instead of
    # waiting forever for its result.
    pool = ProcessPoolExecutor(
        workers,
        mp_context=multiprocessing.get_context("spawn"),
        initializer=start_worker,
        initargs=(str(processed_path), config),
    )
    reader = Thread(
        target=verify_ahead, args=(pool,), daemon=True, name="shard-verifier"
    )
    try:
        reader.start()
        yield results()
    finally:
        stop.set()
        pool.shutdown(wait=True, cancel_futures=True)
        reader.join()


def build_patch_index(config):
    """Index windows in bounded batches and resume from a durable JSONL prefix."""
    import psutil

    options = patch_index_options(config.get("patch_index", {}))
    threads = options["cpu_threads"]
    previous_threads = torch.get_num_threads()
    torch.set_num_threads(threads)
    directory = location(config, "patch_index")
    temporary = directory / f".patch_index_{uuid4().hex}.tmp"
    started = perf_counter()
    process = psutil.Process()
    try:
        directory.mkdir(parents=True, exist_ok=True)
        processed_path = location(config, "processed") / "windows.pt"
        logger.info("Checking processed artifact identity: {}", processed_path)
        processed_sha256 = file_digest(processed_path)
        processed = read_artifact(processed_path, config)
        settings = dict(config["model"], quantiles=tuple(config["model"]["quantiles"]))
        with torch.random.fork_rng(devices=[]):
            torch.manual_seed(config["seed"])
            encoder = PatchEncoder(ForecastConfig(**settings)).eval()
        records = processed["records"]
        count = len(records)
        if not count:
            raise ValueError("Processed artifact has no windows")
        implementation = file_digest(Path(__file__)) + file_digest(
            Path(__file__).with_name("patch_index_progress.py")
        )
        with PatchIndexCheckpoint(
            directory,
            processed["signature"],
            processed_sha256,
            count,
            implementation,
            options,
        ) as checkpoint:
            checkpoint.restore()
            initial = checkpoint.position
            progress = PatchIndexProgress(count, initial, options["progress_every"])
            logger.info(
                "Patch preparation: windows={} batch={} CPU threads={} RSS_GiB={:.3f}; "
                "learned projections run only during training",
                count,
                options["batch_size_records"],
                threads,
                process.memory_info().rss / 2**30,
            )
            policy = config["data"].get("validation_policy")
            workers = worker_count(options, records, initial)
            if workers:
                # Worker processes open the shards. Their results are written
                # here in record order, with the same checkpoints as below.
                with shard_summaries(
                    records, initial, workers, processed_path, config
                ) as summaries:
                    for lines, estimates in summaries:
                        for line, estimated_bytes in zip(lines, estimates):
                            checkpoint.append_line(line, estimated_bytes)
                            checkpoint.save()
                            progress.update(checkpoint.position)
                if checkpoint.position != count:
                    raise RuntimeError(
                        f"Shard inspection stopped at record {checkpoint.position}"
                    )
            else:
                for batch_start in range(initial, count, options["batch_size_records"]):
                    batch_stop = min(count, batch_start + options["batch_size_records"])
                    for position in range(batch_start, batch_stop):
                        record = records[position]
                        detail, estimated_bytes = window_detail(
                            record, encoder.config, policy
                        )
                        checkpoint.append(detail, estimated_bytes)
                        checkpoint.save()
                        progress.update(checkpoint.position)
                        del record
                    # Keep at most one shard mapped. Do not repeatedly deserialize
                    # the same large shard just because a batch ends inside it.
                    if isinstance(records, ShardedRecords) and (
                        batch_stop == count
                        or batch_stop >= records.ends[records.cached_index]
                    ):
                        records.cached = None
                        records.cached_index = None
            checkpoint.save(force=True)
            checkpoint.status("finalizing")
            estimated_old_bytes = checkpoint.estimated_old_bytes
            logger.info(
                "All records checkpointed. Publishing patch index and streamed window details."
            )
            payload = {
                "format": "lazy_patch_index_v1",
                "signature": processed["signature"],
                "processed_sha256": processed_sha256,
                "record_count": count,
                "source_selection": processed.get("source_selection"),
                "encoder_state": encoder.state_dict(),
            }

            # Copy the completed metadata stream in bounded blocks. Keep the recovery
            # copy so interrupted finalization can retry without indexing again.
            def publish_details(stream):
                with checkpoint.details_path.open("rb") as source:
                    shutil.copyfileobj(source, stream, length=1024 * 1024)

            atomic_write(directory / "windows.jsonl", publish_details)
            save_artifact(temporary, payload)
            with temporary.open("r+b") as stream:
                os.fsync(stream.fileno())
            os.replace(temporary, directory / "patch_index.pt")
            report = {
                "config": config,
                "format": payload["format"],
                "window_count": count,
                "source_selection": processed.get("source_selection"),
                "window_details": "windows.jsonl",
                "cpu_threads": threads,
                "batch_size_records": options["batch_size_records"],
                "resumed_records": initial,
                "checkpoint": str(checkpoint.path),
                "cached_embedding_vectors": False,
                "prepared_features": "computed per accessed record",
                "avoided_float32_snapshot_bytes": estimated_old_bytes,
                "artifact_bytes": (directory / "patch_index.pt").stat().st_size,
                "rss_bytes_at_completion": process.memory_info().rss,
                "available_ram_bytes": psutil.virtual_memory().available,
                "output_disk_free_bytes": shutil.disk_usage(directory).free,
                "seconds": perf_counter() - started,
                "note": "Preparation only. Model training jointly optimizes the patch projection; no offline embedding fitting.",
            }
            atomic_json(directory / "manifest.json", report)
            checkpoint.status("complete")
            logger.success(
                "Patch index complete: {} windows, resumed={}, artifact_MiB={:.2f}, elapsed={:.2f}s",
                count,
                initial,
                report["artifact_bytes"] / 2**20,
                report["seconds"],
            )
            return report
    finally:
        temporary.unlink(missing_ok=True)
        torch.set_num_threads(previous_threads)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default=str(ROOT / "configs/config.json"))
    args = parser.parse_args()
    config = load_config(args.config)
    with run_context(config, "index_patches"):
        with timed("INDEX PATCHES stage"):
            build_patch_index(config)


if __name__ == "__main__":
    main()

"""Stage 2: index patch inputs; create learned embeddings inside model training."""

import argparse
import json
import os
import shutil
import sys
from dataclasses import dataclass
from pathlib import Path
from time import perf_counter
from uuid import uuid4

import torch
from loguru import logger
from torch import nn
from torch.nn import functional as F

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src/03_transforming_input_data"))
from transform import (
    file_digest,
    load_config,
    location,
    read_artifact,
    run_context,
    save_artifact,
    timed,
    write_json,
)


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

    def __post_init__(self):
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
        scale = torch.where(scale > 1e-5, scale, torch.ones_like(scale))
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
        scale = torch.where(scale > 1e-5, scale, torch.ones_like(scale)).float()
        loc = loc.float()
        values = ((patches - loc) / scale).nan_to_num().clamp(-1e4, 1e4)
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
        logger.debug(
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
        x = x + self.roles(prepared["roles"])[None, :, None, :]
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
        raise ValueError("Processed data changed: rerun embed.py to rebuild its index")
    processed = read_artifact(path, config)
    if len(processed["records"]) != index["record_count"]:
        raise ValueError("Patch index window count mismatch; rerun embed.py")
    settings = dict(config["model"], quantiles=tuple(config["model"]["quantiles"]))
    # Initialization is reproducible without advancing the caller's RNG stream.
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(config["seed"])
        encoder = PatchEncoder(ForecastConfig(**settings)).eval()
    encoder.load_state_dict(index["encoder_state"])
    return {
        **index,
        "records": [LazyPatchRecord(r, encoder) for r in processed["records"]],
    }


def embed(config):
    """Prepare an index, not a corpus-sized collection of random embeddings."""
    import psutil

    options = config.get("embedding", {})
    threads = options.get("cpu_threads", 2)
    interval = options.get("progress_every", 1000)
    if any(
        isinstance(v, bool) or not isinstance(v, int) or v < 1
        for v in (threads, interval)
    ):
        raise ValueError(
            "embedding.cpu_threads and progress_every must be positive integers"
        )
    previous_threads = torch.get_num_threads()
    torch.set_num_threads(threads)
    directory = location(config, "embedded")
    directory.mkdir(parents=True, exist_ok=True)
    temporary = directory / f".embeddings_{uuid4().hex}.tmp"
    detail_temp = directory / f".windows_{uuid4().hex}.tmp"
    started = perf_counter()
    process = psutil.Process()
    try:
        processed_path = location(config, "processed") / "windows.pt"
        processed = read_artifact(processed_path, config)
        settings = dict(config["model"], quantiles=tuple(config["model"]["quantiles"]))
        with torch.random.fork_rng(devices=[]):
            torch.manual_seed(config["seed"])
            encoder = PatchEncoder(ForecastConfig(**settings)).eval()
        count = len(processed["records"])
        if not count:
            raise ValueError("Processed artifact has no windows")
        estimated_old_bytes = 0
        logger.info(
            "Patch preparation: windows={} CPU threads={} RSS_GiB={:.3f}; "
            "raw tensors memory mapped; learned projections run only during training",
            count,
            threads,
            process.memory_info().rss / 2**30,
        )
        with detail_temp.open("w", encoding="utf-8") as details:
            for position, record in enumerate(processed["records"], 1):
                batch = record["batch"]
                context = batch["context"]
                context_shape = list(context.shape)
                if context.ndim != 3 or min(context.shape) < 1:
                    raise ValueError(
                        f"Invalid context shape for {record['id']}: {context.shape}"
                    )
                variables = context.shape[1] + sum(
                    batch[name].shape[1]
                    for name in ("past_covariates", "known_covariates")
                    if batch.get(name) is not None
                )
                p = encoder.config.patch_size
                length = context.shape[-1]
                if encoder.config.objective == "next_patch":
                    length = (
                        length
                        if record["split"] == "train"
                        else min(length, encoder.config.context_length)
                    )
                    tokens = (length + p - 1) // p - int(record["split"] == "train")
                else:
                    tokens = (length + p - 1) // p + (batch["horizon"] + p - 1) // p
                padded = (tokens + 15) // 16 * 16
                if tokens < 1 or padded > encoder.config.max_patches:
                    raise ValueError(
                        f"Invalid token count for {record['id']}: {tokens}"
                    )
                shape = [context.shape[0], variables, padded, encoder.config.width]
                estimated_old_bytes += (
                    context.shape[0] * variables * padded * encoder.config.width * 4
                )
                details.write(
                    json.dumps(
                        {
                            "id": record["id"],
                            "source": record["source"],
                            "split": record["split"],
                            "context_shape": context_shape,
                            "input_tokens": tokens,
                            "padded_tokens": padded,
                            "embedding_shape_when_used": shape,
                        }
                    )
                    + "\n"
                )
                if position % interval == 0 or position == count:
                    logger.info(
                        "Indexed {}/{} windows elapsed={:.2f}s RSS_GiB={:.3f} free_RAM_GiB={:.3f}",
                        position,
                        count,
                        perf_counter() - started,
                        process.memory_info().rss / 2**30,
                        psutil.virtual_memory().available / 2**30,
                    )
        payload = {
            "format": "lazy_patch_index_v1",
            "signature": processed["signature"],
            "processed_sha256": file_digest(processed_path),
            "record_count": count,
            "source_selection": processed.get("source_selection"),
            "encoder_state": encoder.state_dict(),
        }
        # Write beside the destination on E:, never in C:'s temporary directory.
        # A failed/interrupted write cannot replace the prior completed artifact.
        save_artifact(temporary, payload)
        os.replace(temporary, directory / "embeddings.pt")
        os.replace(detail_temp, directory / "windows.jsonl")
        report = {
            "config": config,
            "format": payload["format"],
            "window_count": count,
            "source_selection": processed.get("source_selection"),
            "window_details": "windows.jsonl",
            "cpu_threads": threads,
            "cached_embedding_vectors": False,
            "prepared_features": "computed per accessed record",
            "avoided_float32_snapshot_bytes": estimated_old_bytes,
            "artifact_bytes": (directory / "embeddings.pt").stat().st_size,
            "rss_bytes_at_completion": process.memory_info().rss,
            "available_ram_bytes": psutil.virtual_memory().available,
            "output_disk_free_bytes": shutil.disk_usage(directory).free,
            "seconds": perf_counter() - started,
            "note": "Preparation only. Model training jointly optimizes the patch projection; no offline embedding fitting.",
        }
        write_json(directory / "manifest.json", report)
        logger.success(
            "Patch index complete: {} windows, artifact_MiB={:.2f}, "
            "avoided random embeddings_GiB={:.2f}, elapsed={:.2f}s",
            count,
            report["artifact_bytes"] / 2**20,
            estimated_old_bytes / 2**30,
            report["seconds"],
        )
        return report
    finally:
        temporary.unlink(missing_ok=True)
        detail_temp.unlink(missing_ok=True)
        torch.set_num_threads(previous_threads)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default=str(ROOT / "configs/config.json"))
    args = parser.parse_args()
    config = load_config(args.config)
    with run_context(config, "embed"):
        with timed("EMBED stage"):
            embed(config)


if __name__ == "__main__":
    main()

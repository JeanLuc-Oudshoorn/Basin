"""Independent-table minibatches with masked time padding."""

import math
import random
from collections import defaultdict

import torch
from torch.nn import functional as F


def pack_records(records):
    """Pad time only; never mix padded/different variable sets in attention."""
    examples = [r["prepared"] for r in records]
    roles = examples[0]["roles"]
    if any(not torch.equal(e["roles"], roles) for e in examples):
        raise ValueError("Batch must have matching variable counts and roles")
    n = max(e["tokens"] for e in examples)
    packed = dict(
        examples[0],
        b=len(examples),
        tokens=n,
        padded_tokens=math.ceil(n / 16) * 16,
        is_training=True,
    )
    if len(records) == 1 and records[0]["split"] == "train":
        return packed
    for name, fill in [("features", 0), ("loc", 0), ("scale", 1)]:
        packed[name] = torch.cat(
            [
                e[name]
                if e["tokens"] == n
                else F.pad(e[name], (0, 0, 0, n - e["tokens"]), value=fill)
                for e in examples
            ]
        )
    labels = []
    for record, e in zip(records, examples):
        if record["split"] == "train":
            target = e["labels"]
        else:
            target = torch.full(
                (1, e["targets"], e["tokens"], e["horizon"]), float("nan")
            )
            # Validation labels are supplied separately after packing. Only
            # legacy one-patch records fit this placeholder training tensor.
            if record["batch"]["future_target"].shape[-1] == e["horizon"]:
                target[:, :, -1] = record["batch"]["future_target"]
        labels.append(F.pad(target, (0, 0, 0, n - e["tokens"]), value=float("nan")))
    packed["labels"] = torch.cat(labels)
    return packed


def _context_length(record):
    if hasattr(record, "materialize"):
        record = record.materialize()
    if "context_rows" in record:
        start, stop = record["context_rows"]
        return stop - start
    return record["batch"]["context"].shape[-1]


def _storage_key(record):
    key = getattr(record, "storage_key", None)
    if key is not None:
        return key
    batch = record.get("batch")
    owner = getattr(batch, "owner", None)
    return owner.shards[batch.shard]["sha256"] if owner is not None else None


class BatchPlanner:
    """Sort shape buckets once; shuffle lightweight handles each epoch."""

    def __init__(
        self,
        records,
        batch_size,
        *,
        bucket_by_length=False,
        shape_run_batches=1,
        storage_locality=False,
    ):
        if (
            isinstance(shape_run_batches, bool)
            or not isinstance(shape_run_batches, int)
            or shape_run_batches < 1
        ):
            raise ValueError("shape_run_batches must be a positive integer")
        if shape_run_batches > 1 and not bucket_by_length:
            raise ValueError("shape runs require bucket_by_length")
        self.batch_size = batch_size
        self.shape_run_batches = shape_run_batches
        self.storage_locality = storage_locality
        buckets = defaultdict(list)
        for record in records:
            details = record.materialize() if hasattr(record, "materialize") else record
            if (
                "batch" in details
                and "roles" in details
                and isinstance(details["roles"], dict)
            ):
                roles = details["roles"]
                key = (0,) * len(roles["targets"])
                key += (1,) * len(roles["past"])
                key += (2,) * len(roles["known"])
            elif "batch" in details:
                batch = details["batch"]
                key = (0,) * batch["context"].shape[1]
                key += (1,) * (
                    batch["past_covariates"].shape[1]
                    if batch.get("past_covariates") is not None
                    else 0
                )
                key += (2,) * (
                    batch["known_covariates"].shape[1]
                    if batch.get("known_covariates") is not None
                    else 0
                )
            else:
                key = tuple(details["prepared"]["roles"].tolist())
            if bucket_by_length:
                encoder = getattr(record, "encoder", None)
                if "batch" in details and encoder is not None:
                    length = _context_length(details)
                    if details["split"] != "train":
                        length = min(length, encoder.config.context_length)
                    tokens = math.ceil(length / encoder.config.patch_size) - int(
                        details["split"] == "train"
                    )
                else:
                    tokens = details["prepared"]["tokens"]
                key = (key, math.ceil(tokens / 16))
            storage = _storage_key(record) if storage_locality else None
            buckets[(storage, key)].append(record)
        self.groups = [
            (
                storage,
                sorted(
                    group,
                    key=lambda r: (
                        _context_length(r) if "batch" in r else r["prepared"]["tokens"]
                    ),
                ),
            )
            for (storage, _), group in buckets.items()
        ]

    def batches(self, seed=None):
        # Reproducible training shuffle, not a security-sensitive RNG.
        rng = random.Random(seed)  # nosec B311
        batches, runs = [], []
        storage_runs = defaultdict(list)
        for storage, original in self.groups:
            group = list(original)
            if seed is not None:
                rng.shuffle(group)
            grouped = [
                group[i : i + self.batch_size]
                for i in range(0, len(group), self.batch_size)
            ]
            batches.extend(grouped)
            pieces = [
                grouped[i : i + self.shape_run_batches]
                for i in range(0, len(grouped), self.shape_run_batches)
            ]
            runs.extend(pieces)
            storage_runs[storage].extend(pieces)
        if self.storage_locality and any(
            storage is not None for storage in storage_runs
        ):
            # A shard is visited once per epoch. Shapes and table order are still
            # shuffled within it; observations inside tables stay chronological.
            sections = list(storage_runs.values())
            if seed is not None:
                for section in sections:
                    rng.shuffle(section)
                rng.shuffle(sections)
            return [batch for section in sections for run in section for batch in run]
        if seed is not None:
            if self.shape_run_batches == 1:
                rng.shuffle(batches)
            else:
                rng.shuffle(runs)
                batches = [batch for run in runs for batch in run]
        return batches


def make_batches(
    records,
    batch_size,
    seed=None,
    *,
    bucket_by_length=False,
    shape_run_batches=1,
    storage_locality=False,
):
    return BatchPlanner(
        records,
        batch_size,
        bucket_by_length=bucket_by_length,
        shape_run_batches=shape_run_batches,
        storage_locality=storage_locality,
    ).batches(seed)

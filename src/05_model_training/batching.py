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
            target[:, :, -1] = record["batch"]["future_target"]
        labels.append(F.pad(target, (0, 0, 0, n - e["tokens"]), value=float("nan")))
    packed["labels"] = torch.cat(labels)
    return packed


def make_batches(records, batch_size, seed=None, *, bucket_by_length=False):
    buckets = defaultdict(list)
    for record in records:
        # Only time is padded. Exact variate counts prevent cross-table leakage.
        if "batch" in record:
            batch = record["batch"]
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
            key = tuple(record["prepared"]["roles"].tolist())
        if bucket_by_length:
            if "batch" in record:
                # Raw lazy records carry their encoder; eager records have tokens.
                encoder = getattr(record, "encoder", None)
                if encoder is not None:
                    length = record["batch"]["context"].shape[-1]
                    p = encoder.config.patch_size
                    if record["split"] != "train":
                        length = min(length, encoder.config.context_length)
                    tokens = math.ceil(length / p) - int(record["split"] == "train")
                else:
                    tokens = record["prepared"]["tokens"]
            else:
                tokens = record["prepared"]["tokens"]
            key = (key, math.ceil(tokens / 16))
        buckets[key].append(record)
    # Reproducible minibatch order, not a security-sensitive random generator.
    rng = random.Random(seed)  # nosec B311
    batches = []
    for group in buckets.values():
        group = sorted(
            group,
            key=lambda r: (
                r["batch"]["context"].shape[-1]
                if "batch" in r
                else r["prepared"]["tokens"]
            ),
        )
        if seed is not None:
            rng.shuffle(group)
        batches.extend(
            group[i : i + batch_size] for i in range(0, len(group), batch_size)
        )
    if seed is not None:
        rng.shuffle(batches)
    return batches

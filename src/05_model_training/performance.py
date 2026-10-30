"""Bounded CPU preparation and configurable CUDA optimizer selection."""

from concurrent.futures import ThreadPoolExecutor

import torch
from batching import pack_records
from loguru import logger


def configure_cuda_memory(training):
    """Bound the caching allocator as well as live activations; reserve device headroom."""
    limit = training.get("cuda_memory_limit_gib", 8.0)
    if limit is None or not torch.cuda.is_available():
        return
    device = torch.cuda.current_device()
    total = torch.cuda.get_device_properties(device).total_memory
    budget = min(float(limit) * 2**30, total * 0.8)
    torch.cuda.set_per_process_memory_fraction(float(budget / total), device)
    logger.info(
        "CUDA allocator budget={:.2f} GiB / {:.2f} GiB dedicated capacity; "
        "includes cached reservations, excludes external CUDA/desktop allocations",
        budget / 2**30,
        total / 2**30,
    )


def make_optimizer(model, training):
    parameters = [p for p in model.parameters() if p.requires_grad]
    fused = (
        training.get("fused_optimizer", True) and parameters[0].device.type == "cuda"
    )
    return torch.optim.AdamW(
        parameters,
        lr=training["learning_rate"],
        weight_decay=training["weight_decay"],
        fused=fused,
        foreach=False,
    )


def prepare_update(
    microbatches,
    *,
    next_patch=True,
    pin_memory=False,
    microbatch_size=None,
    microbatch_token_limit=None,
):
    """Prepare only one optimizer update; labels and normalization stay on CPU."""
    batches = []
    for records in microbatches:
        if next_patch:
            prepared = pack_records(records)
            labels = prepared["labels"]
        else:
            if len(records) != 1:
                raise ValueError("Legacy masked-horizon mode supports batch_size=1")
            prepared = dict(records[0]["prepared"])
            labels = records[0]["batch"]["future_target"]
        size = len(records)
        if next_patch:
            size = min(size, microbatch_size or size)
            if microbatch_token_limit:
                tokens_per_table = len(prepared["roles"]) * prepared["padded_tokens"]
                size = min(size, max(1, microbatch_token_limit // tokens_per_table))
        for start in range(0, len(records), size):
            # Keep exactly the original batch's padding and ordering. Only the
            # physical execution is split; optimizer groups remain unchanged.
            chunk = dict(prepared)
            chunk_labels = labels[start : start + size]
            if next_patch:
                for key in ("features", "loc", "scale", "labels"):
                    chunk[key] = prepared[key][start : start + size]
                chunk["b"] = chunk_labels.shape[0]
            points = int(torch.isfinite(chunk_labels).sum())
            if points < 1:
                raise ValueError("Microbatch has no observed training labels")
            chunk["observed_points"] = points
            if pin_memory:
                chunk = {
                    k: v.pin_memory() if isinstance(v, torch.Tensor) else v
                    for k, v in chunk.items()
                }
                chunk_labels = (
                    chunk["labels"] if next_patch else chunk_labels.pin_memory()
                )
            batches.append((chunk, chunk_labels, points))
    return batches


def prepared_updates(
    batches,
    accumulation,
    *,
    next_patch=True,
    prefetch=True,
    pin_memory=False,
    microbatch_size=None,
    microbatch_token_limit=None,
):
    """At most current + one future update; no corpus-sized token cache/workers."""

    def prepare(offset):
        group = batches[offset : offset + accumulation]
        return group, prepare_update(
            group,
            next_patch=next_patch,
            pin_memory=pin_memory,
            microbatch_size=microbatch_size,
            microbatch_token_limit=microbatch_token_limit,
        )

    offsets = iter(range(0, len(batches), accumulation))
    if not prefetch:
        for offset in offsets:
            yield offset, *prepare(offset)
        return
    with ThreadPoolExecutor(
        max_workers=1, thread_name_prefix="patch-preparation"
    ) as pool:
        first = next(offsets, None)
        if first is None:
            return
        pending = pool.submit(prepare, first)
        offset = first
        while pending is not None:
            group, prepared = pending.result()
            following = next(offsets, None)
            pending = pool.submit(prepare, following) if following is not None else None
            yield offset, group, prepared
            offset = following

"""Bounded CPU preparation and configurable CUDA optimizer selection."""

from concurrent.futures import ThreadPoolExecutor

import torch
from batching import pack_records
from loguru import logger
from master_optimizer import FP32MasterAdamW


def configure_cuda_memory(training):
    """Bound the caching allocator as well as live activations; reserve device headroom."""
    limit = training.get("cuda_memory_limit_gib", 11.0)
    if limit is None or not torch.cuda.is_available():
        return
    device = torch.cuda.current_device()
    total = torch.cuda.get_device_properties(device).total_memory
    free, _ = torch.cuda.mem_get_info(device)
    headroom = float(training.get("cuda_memory_headroom_gib", 0.5)) * 2**30
    # Include this allocator's existing reservation when reconfiguring a live
    # process, but leave external CUDA/desktop allocations out of its budget.
    available = free + torch.cuda.memory_reserved(device)
    budget = min(float(limit) * 2**30, total - headroom, available - headroom)
    if budget <= 0:
        raise RuntimeError("Insufficient free dedicated VRAM after reserving headroom")
    torch.cuda.set_per_process_memory_fraction(float(budget / total), device)
    logger.info(
        "CUDA allocator budget={:.2f} GiB / {:.2f} GiB dedicated capacity; "
        "free before allocation={:.2f} GiB, external headroom={:.2f} GiB; "
        "budget includes cached reservations",
        budget / 2**30,
        total / 2**30,
        free / 2**30,
        headroom / 2**30,
    )


def make_optimizer(model, training):
    parameters = [p for p in model.parameters() if p.requires_grad]
    fused = (
        training.get("fused_optimizer", True) and parameters[0].device.type == "cuda"
    )
    optimizer_class = (
        FP32MasterAdamW
        if any(p.dtype in (torch.bfloat16, torch.float16) for p in parameters)
        else torch.optim.AdamW
    )
    logger.info(
        "Optimizer={} model_dtype={} update_precision={}",
        optimizer_class.__name__,
        parameters[0].dtype,
        "FP32 master weights and moments"
        if optimizer_class is FP32MasterAdamW
        else str(parameters[0].dtype),
    )
    return optimizer_class(
        parameters,
        lr=training["learning_rate"],
        weight_decay=training["weight_decay"],
        fused=fused,
        foreach=False,
    )


def _pin_prepared_batches(batches, *, next_patch):
    """Pin on the consuming training thread, before it can begin graph capture.

    Host pinning uses CUDA APIs even though the tensors are on CPU. Doing it in
    the prefetch worker races with process-wide CUDA graph capture restrictions.
    """
    pinned = []
    for chunk, labels, points in batches:
        chunk = {
            k: v.pin_memory() if isinstance(v, torch.Tensor) else v
            for k, v in chunk.items()
        }
        labels = chunk["labels"] if next_patch else labels.pin_memory()
        pinned.append((chunk, labels, points))
    return pinned


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
            batches.append((chunk, chunk_labels, points))
    return (
        _pin_prepared_batches(batches, next_patch=next_patch) if pin_memory else batches
    )


def split_prepared_batches(batches, maximum_size):
    """Split CPU tensors along tables, preserving padding, labels and their order."""
    result = []
    for prepared, labels, points in batches:
        count = labels.shape[0]
        if count <= maximum_size:
            result.append((prepared, labels, points))
            continue
        for start in range(0, count, maximum_size):
            chunk = dict(prepared)
            chunk_labels = labels[start : start + maximum_size]
            for key in ("features", "loc", "scale", "labels"):
                if key in prepared:
                    chunk[key] = prepared[key][start : start + maximum_size]
            chunk["b"] = chunk_labels.shape[0]
            chunk["observed_points"] = int(torch.isfinite(chunk_labels).sum())
            result.append((chunk, chunk_labels, chunk["observed_points"]))
    return result


def prepared_updates(
    batches,
    accumulation,
    *,
    next_patch=True,
    prefetch=True,
    pin_memory=False,
    microbatch_size=None,
    microbatch_token_limit=None,
    start_offset=0,
):
    """At most current + one future update; no corpus-sized token cache/workers."""

    def prepare(offset):
        group = batches[offset : offset + accumulation]
        return group, prepare_update(
            group,
            next_patch=next_patch,
            # The worker must remain CPU-only while the consumer captures graphs.
            pin_memory=False,
            microbatch_size=microbatch_size,
            microbatch_token_limit=microbatch_token_limit,
        )

    offsets = iter(range(start_offset, len(batches), accumulation))
    if not prefetch:
        for offset in offsets:
            group, prepared = prepare(offset)
            if pin_memory:
                prepared = _pin_prepared_batches(prepared, next_patch=next_patch)
            yield offset, group, prepared
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
            # Main-thread pinning cannot overlap its own capture. CPU preparation
            # of the following update still overlaps this pinning and GPU work.
            if pin_memory:
                prepared = _pin_prepared_batches(prepared, next_patch=next_patch)
            yield offset, group, prepared
            offset = following

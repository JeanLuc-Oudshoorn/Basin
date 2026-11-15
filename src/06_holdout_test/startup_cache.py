"""Bounded split loading and invalidated caches for immutable run artifacts."""

import hashlib
import json
import os
from pathlib import Path
from time import perf_counter
from uuid import uuid4

import ijson
import numpy as np
from loguru import logger

VERSION = 1


def weights_digest(path, weights, *, field):
    """Fingerprint precisely the evaluated tensors, never recovery optimizer data."""
    import torch

    path = Path(path)
    before = stamp(path)
    cache = path.with_name(f".{path.name}.{field}.sha256.json")
    try:
        saved = json.loads(cache.read_text(encoding="utf-8"))
        if saved["stamp"] == before:
            logger.info("Selected model weights checksum cache HIT")
            return saved["sha256"]
    except (OSError, ValueError, KeyError):
        pass
    digest = hashlib.sha256()
    tensors = sorted(weights.items())
    total = sum(t.numel() * t.element_size() for _, t in tensors)
    logger.info("Fingerprinting selected model weights only: {:.2f} GiB", total / 2**30)
    started = reported = perf_counter()
    completed = 0
    for name, tensor in tensors:
        description = [name, str(tensor.dtype), list(tensor.shape)]
        header = json.dumps(description, separators=(",", ":")).encode()
        digest.update(len(header).to_bytes(8, "little"))
        digest.update(header)
        raw = tensor.detach().cpu().contiguous().reshape(-1).view(torch.uint8).numpy()
        digest.update(memoryview(raw))
        completed += tensor.numel() * tensor.element_size()
        now = perf_counter()
        if now - reported >= 30:
            logger.info(
                "Selected weights checksum: {:.1f}% elapsed={:.1f}s",
                completed / max(1, total) * 100,
                now - started,
            )
            reported = now
    if stamp(path) != before:
        raise ValueError("Checkpoint changed while fingerprinting selected weights")
    value = digest.hexdigest()
    atomic_json(cache, {"stamp": before, "sha256": value})
    return value


def stamp(path):
    value = Path(path).stat()
    return [value.st_size, value.st_mtime_ns, value.st_ctime_ns]


def atomic_json(path, value):
    temporary = path.with_name(f"{path.name}.{uuid4().hex}.tmp")
    try:
        temporary.write_text(json.dumps(value), encoding="utf-8")
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def artifact_digest(path):
    """Reuse a SHA256 only while size and both file timestamps remain unchanged."""
    path = Path(path)
    before = stamp(path)
    cache = path.with_name(f".{path.name}.sha256.json")
    try:
        saved = json.loads(cache.read_text(encoding="utf-8"))
        if saved["stamp"] == before:
            logger.info("Artifact checksum cache HIT: {}", path.name)
            return saved["sha256"]
    except (OSError, ValueError, KeyError):
        pass
    logger.info("Hashing artifact once: {} ({:.2f} GiB)", path, before[0] / 2**30)
    started = reported = perf_counter()
    digest = hashlib.sha256()
    read = 0
    with path.open("rb") as stream:
        while block := stream.read(8 * 2**20):
            digest.update(block)
            read += len(block)
            now = perf_counter()
            if now - reported >= 30:
                logger.info(
                    "Artifact checksum {}: {:.1f}% elapsed={:.1f}s",
                    path.name,
                    read / max(1, before[0]) * 100,
                    now - started,
                )
                reported = now
    if stamp(path) != before:
        raise ValueError(f"Artifact changed while hashing: {path}")
    value = digest.hexdigest()
    atomic_json(cache, {"stamp": before, "sha256": value})
    return value


def source_digest(source):
    return hashlib.blake2b(source.encode(), digest_size=16).digest()


def overlapping_sources(split, kind, sources):
    """Compact 128-bit membership preserves rejection of any actual overlap.

    A hash collision can only reject an independent source, never admit leakage.
    """
    index = split.get(f"_{kind}_index")
    if index is None:
        return set(sources) & set(split.get(f"{kind}_sources", []))
    if not len(index):
        return set()
    result = set()
    for source in sources:
        value = np.frombuffer(source_digest(source), dtype="V16")[0]
        position = np.searchsorted(index, value)
        if position < len(index) and index[position] == value:
            result.add(source)
    return result


class HashingReader:
    def __init__(self, stream):
        self.stream, self.digest = stream, hashlib.sha256()

    def read(self, size=-1):
        block = self.stream.read(size)
        self.digest.update(block)
        return block

    def readinto(self, buffer):
        count = self.stream.readinto(buffer)
        self.digest.update(memoryview(buffer)[:count])
        return count


def load_split(path):
    """Never materialize millions of training paths or irrelevant source maps."""
    path = Path(path)
    before = stamp(path)
    directory = path.parent / ".holdout_split_cache"
    manifest = directory / "manifest.json"
    try:
        saved = json.loads(manifest.read_text(encoding="utf-8"))
        if saved["version"] == VERSION and saved["stamp"] == before:
            result = saved["split"]
            for kind in ("train", "validation"):
                index_path = directory / saved[f"{kind}_file"]
                if artifact_digest(index_path) != saved[f"{kind}_sha256"]:
                    raise ValueError("Membership cache checksum changed")
                result[f"_{kind}_index"] = np.load(
                    index_path,
                    mmap_mode="r" if os.name != "nt" else None,
                    allow_pickle=False,
                )
            logger.info(
                "Holdout split cache HIT: {} test sources", len(result["test_sources"])
            )
            return result
    except (OSError, ValueError, KeyError):
        pass
    logger.info("Streaming split once (bounded memory): {}", path)
    started = reported = perf_counter()
    with path.open("rb") as stream:
        sources = next(ijson.items(stream, "test_sources"), [])
    wanted = set(sources)
    result = {"test_sources": sources, "source_keys": {}, "source_sha256": {}}
    memberships = {"train": bytearray(), "validation": bytearray()}
    with path.open("rb") as stream:
        reader = HashingReader(stream)
        for prefix, event, value in ijson.parse(reader, use_float=True):
            if prefix == "train_sources" and event == "start_array":
                result["_has_train_membership"] = True
            if (
                prefix in ("train_sources.item", "validation_sources.item")
                and event == "string"
            ):
                kind = prefix.split("_", 1)[0]
                memberships[kind].extend(source_digest(value))
            elif prefix in (
                "seed",
                "requested_test_fraction",
                "requested_test_tables",
                "input_signature",
            ):
                result[prefix] = value
            elif event == "string":
                for key in ("source_keys", "source_sha256"):
                    if prefix.startswith(key + "."):
                        source = prefix[len(key) + 1 :]
                        if source in wanted:
                            result[key][source] = value
                        break
            now = perf_counter()
            if now - reported >= 30:
                logger.info(
                    "Streaming split: {:.1f}% elapsed={:.1f}s",
                    stream.tell() / max(1, before[0]) * 100,
                    now - started,
                )
                reported = now
    if stamp(path) != before:
        raise ValueError("Training split changed while reading")
    result["_split_sha256"] = reader.digest.hexdigest()
    directory.mkdir(exist_ok=True)
    saved = {"version": VERSION, "stamp": before, "split": dict(result)}
    for kind, data in memberships.items():
        index = np.frombuffer(data, dtype="V16").copy()
        index.sort()
        name = f"{kind}_{uuid4().hex}.npy"
        index_path = directory / name
        np.save(index_path, index, allow_pickle=False)
        saved[f"{kind}_file"] = name
        saved[f"{kind}_sha256"] = artifact_digest(index_path)
        result[f"_{kind}_index"] = index
    atomic_json(manifest, saved)
    logger.info("Holdout split cache built in {:.1f}s", perf_counter() - started)
    return result


def cached_family_exclusions(sources, root, scanner):
    """Recheck lineage when any scanned directory or requested membership changes."""
    key = hashlib.sha256("\n".join(sorted(sources)).encode()).hexdigest()
    directory = root / ".cache" / "holdout_lineage"
    path = directory / f"{key}.json"
    try:
        saved = json.loads(path.read_text(encoding="utf-8"))
        if saved["version"] == VERSION and all(
            stamp(Path(name)) == fingerprint
            for name, fingerprint in saved["directories"].items()
        ):
            logger.info("Holdout lineage cache HIT: {} sources", len(sources))
            return set(saved["excluded"])
    except (OSError, ValueError, KeyError):
        pass
    audit = {}
    excluded = scanner(sources, audit=audit)
    if not audit:
        return excluded
    if any(stamp(Path(name)) != fingerprint for name, fingerprint in audit.items()):
        raise ValueError(
            "Raw directory changed during lineage screening; retry holdout"
        )
    directory.mkdir(parents=True, exist_ok=True)
    atomic_json(
        path,
        {"version": VERSION, "excluded": sorted(excluded), "directories": audit},
    )
    return excluded

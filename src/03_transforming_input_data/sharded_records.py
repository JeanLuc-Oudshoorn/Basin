"""Read processed shards through a bounded cache; retained records hold locators."""

import gzip
import json
import os
import uuid
import zlib
from array import array
from bisect import bisect_right
from collections.abc import Mapping, Sequence
from pathlib import Path
from time import perf_counter

import torch
from loguru import logger
from torch.utils.serialization import config as serialization_config
from transform_progress import atomic_json, digest


def write_record_metadata(directory, shard, records):
    """Publish tensor-free metadata, bound to the verified tensor shard identity."""
    metadata = []
    for record in records:
        item = {k: v for k, v in record.items() if k != "batch"}
        if "batch" in record:
            item["_shard_batch_keys"] = list(record["batch"])
        metadata.append(item)
    path = Path(directory) / ".record_metadata" / f"{shard['sha256']}.json"
    atomic_json(path, {"shard_sha256": shard["sha256"], "records": metadata})
    return metadata


def table_metadata_path(directory, shard):
    return Path(directory) / ".table_metadata" / f"{shard['sha256']}.json.gz"


def write_table_metadata(directory, shard, tables, excluded):
    """Keep a shard's table entries beside it so the manifest need not load tensors.

    A convenience copy: readers fall back to the shard itself when it is absent
    or unreadable, so it is replaced atomically but not synced to disk.
    """
    content = json.dumps(
        {"shard_sha256": shard["sha256"], "tables": tables, "excluded": excluded},
        allow_nan=False,
    ).encode("utf-8")
    path = table_metadata_path(directory, shard)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{uuid.uuid4().hex}.tmp")
    try:
        temporary.write_bytes(gzip.compress(content, compresslevel=1, mtime=0))
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def read_table_metadata(directory, shard, shard_path):
    """Return a shard's (tables, excluded) entries, from its copy when available."""
    try:
        content = table_metadata_path(directory, shard).read_bytes()
        payload = json.loads(gzip.decompress(content))
        if payload["shard_sha256"] == shard["sha256"]:
            return payload["tables"], payload["excluded"]
    except (OSError, EOFError, ValueError, KeyError, TypeError, zlib.error):
        pass
    batch = load_shard(shard_path)
    return batch["tables"], batch["excluded"]


def load_shard(path):
    previous = serialization_config.load.calculate_storage_offsets
    try:
        serialization_config.load.calculate_storage_offsets = True
        return torch.load(path, map_location="cpu", weights_only=True, mmap=True)
    finally:
        serialization_config.load.calculate_storage_offsets = previous


class ShardBatch(Mapping):
    """A batch reference that never retains a shard's tensor storage."""

    def __init__(self, owner, shard, record, keys):
        self.owner, self.shard, self.record, self.keys_saved = (
            owner,
            shard,
            record,
            keys,
        )

    def __getitem__(self, key):
        return self.owner.load(self.shard)["records"][self.record]["batch"][key]

    def __iter__(self):
        return iter(self.keys_saved)

    def __len__(self):
        return len(self.keys_saved)


class ShardedRecords(Sequence):
    """Sequence of metadata dictionaries backed by at most one mapped shard."""

    def __init__(self, index, directory):
        self.directory = Path(directory).resolve()
        self.shards = index["shards"]
        self.ends = []
        total = 0
        for shard in self.shards:
            path = (self.directory / shard["path"]).resolve()
            if not path.is_relative_to(self.directory):
                raise ValueError(
                    "Processed shard must remain inside its artifact directory"
                )
            total += shard["record_count"]
            self.ends.append(total)
        if total != index["record_count"]:
            raise ValueError("Processed shard record count mismatch")
        self.cached_index, self.cached = None, None
        self.verified = set()
        self.metadata_index, self.metadata = None, None
        self.last_report = perf_counter()
        logger.info(
            "Sharded corpus: {:,} records / {:,} shards; metadata-only startup; "
            "older artifacts build a reusable metadata cache on first access",
            total,
            len(self.shards),
        )

    def record_metadata(self, index):
        if self.metadata_index == index:
            return self.metadata
        shard = self.shards[index]
        path = self.directory / ".record_metadata" / f"{shard['sha256']}.json"
        if path.is_file():
            payload = json.loads(path.read_text(encoding="utf-8"))
            if payload["shard_sha256"] != shard["sha256"]:
                raise ValueError(f"Wrong metadata shard identity: {path}")
            metadata = payload["records"]
        else:
            # Compatibility: existing artifacts require one verified shard read,
            # not another tensor deserialization for every split/audit pass.
            if index == 0:
                logger.info(
                    "Building metadata cache for shard {}/{} (one-time tensor read)",
                    index + 1,
                    len(self.shards),
                )
            metadata = write_record_metadata(
                self.directory, shard, self.load(index)["records"]
            )
        if len(metadata) != shard["record_count"]:
            raise ValueError(f"Processed metadata record count mismatch: {path}")
        self.metadata_index, self.metadata = index, metadata
        now = perf_counter()
        if now - self.last_report >= 30 or index == len(self.shards) - 1:
            logger.info(
                "Corpus metadata: shard {:,}/{:,} | records through {:,}/{:,}",
                index + 1,
                len(self.shards),
                self.ends[index],
                len(self),
            )
            self.last_report = now
        return metadata

    def __len__(self):
        return self.ends[-1] if self.ends else 0

    def load(self, index):
        if self.cached_index != index:
            shard = self.shards[index]
            path = self.directory / shard["path"]
            if index not in self.verified:
                if digest(path) != shard["sha256"]:
                    raise ValueError(f"Corrupt processed shard: {path}")
                self.verified.add(index)
            self.cached = None
            self.cached_index = None
            batch = load_shard(path)
            if len(batch["records"]) != shard["record_count"]:
                raise ValueError(f"Processed shard record count mismatch: {path}")
            self.cached, self.cached_index = batch, index
        return self.cached

    def __getitem__(self, index):
        if isinstance(index, slice):
            return [self[i] for i in range(*index.indices(len(self)))]
        if index < 0:
            index += len(self)
        if not 0 <= index < len(self):
            raise IndexError(index)
        shard = bisect_right(self.ends, index)
        local = index - (self.ends[shard - 1] if shard else 0)
        record = dict(self.record_metadata(shard)[local])
        keys = record.pop("_shard_batch_keys", None)
        if keys is not None:
            record["batch"] = ShardBatch(self, shard, local, tuple(keys))
        return record

    def __iter__(self):
        for index in range(len(self)):
            yield self[index]


class EncodedRecords(Sequence):
    """Wrap raw records on demand instead of allocating a second corpus list."""

    def __init__(self, records, encoder, factory):
        self.records, self.encoder, self.factory = records, encoder, factory

    def __len__(self):
        return len(self.records)

    def __getitem__(self, index):
        if isinstance(index, slice):
            return [self[i] for i in range(*index.indices(len(self)))]
        return self.factory(self.records[index], self.encoder)


class RecordRef(Mapping):
    """A small record handle; metadata and tensors remain owned by shard caches."""

    __slots__ = ("owner", "index")

    def __init__(self, owner, index):
        self.owner, self.index = owner, index

    @property
    def encoder(self):
        return self.owner.encoder

    @property
    def storage_key(self):
        raw = self.owner.records
        shard = bisect_right(raw.ends, self.index)
        return raw.shards[shard]["sha256"]

    def __getitem__(self, key):
        return self.owner[self.index][key]

    def materialize(self):
        """Fetch one disposable metadata dictionary for a multi-field inspection."""
        return self.owner[self.index]

    def __iter__(self):
        return iter(self.owner[self.index])

    def __len__(self):
        return len(self.owner[self.index])


class RecordSelection(Sequence):
    """Eight bytes per selected index, instead of retaining millions of dicts."""

    def __init__(self, owner, indices=()):
        self.owner = owner
        self.indices = array("Q", indices)

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, index):
        if isinstance(index, slice):
            return RecordSelection(self.owner, self.indices[index])
        return RecordRef(self.owner, self.indices[index])

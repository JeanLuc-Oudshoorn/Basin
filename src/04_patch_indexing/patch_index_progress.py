"""Bounded metadata output with durable byte offsets for patch indexing."""

import hashlib
import json
import math
import os
import sys
from time import perf_counter

from loguru import logger
from transform_progress import atomic_json


def patch_index_options(options):
    values = {
        "cpu_threads": options.get("cpu_threads", 2),
        "batch_size_records": options.get("batch_size_records", 1000),
        "progress_every": options.get("progress_every", 1000),
        "checkpoint_every_records": options.get("checkpoint_every_records", 100_000),
        "checkpoint_interval_seconds": options.get("checkpoint_interval_seconds"),
        # null: choose from the number of shards; 0: inspect in this process.
        "workers": options.get("workers"),
    }
    for name, value in values.items():
        optional = name.startswith("checkpoint_") or name == "workers"
        if value is None and optional:
            continue
        kinds = (int, float) if name == "checkpoint_interval_seconds" else (int,)
        if (
            isinstance(value, bool)
            or not isinstance(value, kinds)
            or not math.isfinite(value)
            or (value < 0 if name == "workers" else value <= 0)
        ):
            raise ValueError(
                f"patch_index.{name} must be "
                + ("non-negative" if name == "workers" else "positive")
                + (" or null" if optional else "")
            )
    return values


def detail_line(detail):
    """One window's metadata as a line of the JSONL output."""
    return (json.dumps(detail, allow_nan=False) + "\n").encode("utf-8")


class PatchIndexProgress:
    def __init__(self, total, initial, interval):
        self.total, self.initial, self.interval = total, initial, interval
        self.started = self.last_report = perf_counter()
        self.last_position = initial
        self.update(initial, force=True)

    def update(self, position, *, force=False):
        now = perf_counter()
        if not (
            force
            or position == self.total
            or position // self.interval > self.last_position // self.interval
            or now - self.last_report >= 5
        ):
            return
        elapsed = now - self.started
        rate = (position - self.initial) / elapsed if elapsed else 0
        fraction = position / self.total
        filled = min(24, int(24 * fraction))
        eta = self.duration((self.total - position) / rate) if rate else "--"
        logger.info(
            "Index [{}{}] {}/{} records ({:.1f}%) | {:.1f} records/s | elapsed={} | ETA={}",
            "#" * filled,
            "-" * (24 - filled),
            position,
            self.total,
            100 * fraction,
            rate,
            self.duration(elapsed),
            eta,
        )
        self.last_report, self.last_position = now, position

    @staticmethod
    def duration(seconds):
        seconds = int(seconds)
        return f"{seconds // 3600:02d}:{seconds // 60 % 60:02d}:{seconds % 60:02d}"


class PatchIndexCheckpoint:
    """Commit an fsynced JSONL prefix before publishing its recovery position."""

    def __init__(
        self, directory, signature, processed_sha256, count, implementation, options
    ):
        identity = hashlib.sha256(
            json.dumps(
                [signature, processed_sha256, count, implementation], sort_keys=True
            ).encode()
        ).hexdigest()
        self.directory = directory / ".patch_index_checkpoints" / identity
        self.directory.mkdir(parents=True, exist_ok=True)
        self.path = self.directory / "progress.json"
        self.details_path = self.directory / "windows.jsonl"
        self.options = options
        self.state = {
            "schema": 1,
            "signature": signature,
            "processed_sha256": processed_sha256,
            "record_count": count,
            "completed_records": 0,
            "details_bytes": 0,
            "estimated_old_bytes": 0,
            "chunks": [],
            "status": "indexing",
        }
        self.file = None

    def __enter__(self):
        self.lock = (self.directory.parent / "run.lock").open("a+b")
        try:
            if sys.platform == "win32":
                import msvcrt

                self.lock.seek(0)
                if not self.lock.read(1):
                    self.lock.write(b"0")
                    self.lock.flush()
                self.lock.seek(0)
                msvcrt.locking(self.lock.fileno(), msvcrt.LK_NBLCK, 1)
            else:
                import fcntl

                fcntl.flock(self.lock.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        except OSError as error:
            self.lock.close()
            raise RuntimeError(
                f"Another index_patches run is using {self.directory.parent}"
            ) from error
        return self

    def __exit__(self, *args):
        try:
            if self.file is not None:
                self.file.close()
        finally:
            self.lock.close()

    def restore(self):
        if self.path.exists():
            saved = json.loads(self.path.read_text(encoding="utf-8"))
            for key in ("schema", "signature", "processed_sha256", "record_count"):
                if saved[key] != self.state[key]:
                    raise ValueError(
                        f"Incompatible patch index checkpoint: {self.path}"
                    )
            self.state = saved
        mode = "r+b" if self.details_path.exists() else "w+b"
        self.file = self.details_path.open(mode)
        if self.state["details_bytes"]:
            logger.info(
                "Verifying saved patch index metadata: {} bytes | {}",
                self.state["details_bytes"],
                self.details_path,
            )
        last_report = perf_counter()
        expected = 0
        verified_records = 0
        for chunk in self.state["chunks"]:
            if (
                chunk["start"] != expected
                or chunk["stop"] <= expected
                or not verified_records
                < chunk["completed_records"]
                <= self.state["record_count"]
            ):
                raise ValueError(
                    f"Invalid patch index checkpoint sequence: {self.path}"
                )
            hasher = hashlib.sha256()
            remaining = chunk["stop"] - chunk["start"]
            while remaining:
                block = self.file.read(min(1024 * 1024, remaining))
                if not block:
                    raise ValueError(
                        f"Truncated patch index checkpoint output: {self.details_path}"
                    )
                hasher.update(block)
                remaining -= len(block)
                if perf_counter() - last_report >= 5:
                    logger.info(
                        "Checking patch index checkpoint output: {}/{} bytes",
                        self.file.tell(),
                        self.state["details_bytes"],
                    )
                    last_report = perf_counter()
            if hasher.hexdigest() != chunk["sha256"]:
                raise ValueError(
                    f"Corrupt patch index checkpoint output: {self.details_path}"
                )
            expected = chunk["stop"]
            verified_records = chunk["completed_records"]
        if (
            verified_records != self.state["completed_records"]
            or expected != self.state["details_bytes"]
        ) or not (0 <= self.state["completed_records"] <= self.state["record_count"]):
            raise ValueError(f"Invalid patch index checkpoint position: {self.path}")
        self.file.truncate(expected)
        self.file.seek(expected)
        self.position = self.state["completed_records"]
        self.estimated_old_bytes = self.state["estimated_old_bytes"]
        self.pending_hash = hashlib.sha256()
        self.last_save = perf_counter()
        logger.info(
            "Patch index checkpoint: {} | resuming at {}/{} records",
            self.path,
            self.position,
            self.state["record_count"],
        )
        logger.info(
            "Patch index policy: batch={} records, checkpoint_every_records={}, interval_seconds={}",
            self.options["batch_size_records"],
            self.options["checkpoint_every_records"],
            self.options["checkpoint_interval_seconds"],
        )

    def append(self, detail, estimated_bytes):
        self.append_line(detail_line(detail), estimated_bytes)

    def append_line(self, data, estimated_bytes):
        self.file.write(data)
        self.pending_hash.update(data)
        self.position += 1
        self.estimated_old_bytes += estimated_bytes

    def save(self, *, force=False):
        previous = self.state["completed_records"]
        if self.position <= previous:
            return
        records = self.options["checkpoint_every_records"]
        seconds = self.options["checkpoint_interval_seconds"]
        due = records is not None and self.position - previous >= records
        due |= seconds is not None and perf_counter() - self.last_save >= seconds
        if not (force or due):
            return
        self.file.flush()
        os.fsync(self.file.fileno())
        offset = self.file.tell()
        chunk = {
            "start": self.state["details_bytes"],
            "stop": offset,
            "sha256": self.pending_hash.hexdigest(),
            "completed_records": self.position,
        }
        state = {
            **self.state,
            "completed_records": self.position,
            "details_bytes": offset,
            "estimated_old_bytes": self.estimated_old_bytes,
            "chunks": [*self.state["chunks"], chunk],
            "status": "indexing",
        }
        atomic_json(self.path, state)
        self.state = state
        self.pending_hash = hashlib.sha256()
        self.last_save = perf_counter()
        logger.info(
            "Patch index checkpoint saved: {}/{} records | {}",
            self.position,
            self.state["record_count"],
            self.path,
        )

    def status(self, status):
        state = {**self.state, "status": status}
        atomic_json(self.path, state)
        self.state = state

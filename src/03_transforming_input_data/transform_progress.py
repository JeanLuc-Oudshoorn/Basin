"""Durable, incremental checkpoints and terminal progress for transformation."""

import hashlib
import json
import math
import os
import shutil
import sys
import uuid
from array import array
from pathlib import Path
from time import perf_counter

import torch
from loguru import logger


def atomic_write(path, writer):
    """Publish only a fully written and synced file, beside its destination."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{uuid.uuid4().hex}.tmp")
    try:
        with temporary.open("wb") as stream:
            writer(stream)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def atomic_json(path, value):
    content = json.dumps(value, indent=2, allow_nan=False).encode("utf-8")
    atomic_write(path, lambda stream: stream.write(content))


def digest(path):
    with Path(path).open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def examine(paths, root):
    """Yield (key relative to root, size, modification time) for each path."""
    for path in paths:
        stat = path.stat()
        yield path.relative_to(root).as_posix(), stat.st_size, stat.st_mtime_ns


class TransformProgress:
    """A bar in both terminals and captured WSL logs; no extra dependency."""

    def __init__(self, label, total, initial=0):
        self.label, self.total, self.initial = label, total, initial
        # Every 100 tables up to a million; coarser beyond, so that a corpus of
        # millions reports about every five seconds instead of flooding the log.
        self.step = max(100, 10 ** (len(str(total)) - 4))
        self.started = self.last_report = perf_counter()
        self.last_completed = initial
        self.update(initial, force=True)

    def update(self, completed, *, force=False):
        if not (
            force
            or completed == self.total
            or completed // self.step > self.last_completed // self.step
            or perf_counter() - self.last_report >= 5
        ):
            return
        now = perf_counter()
        elapsed = now - self.started
        rate = (completed - self.initial) / elapsed if elapsed > 0 else 0
        eta = (self.total - completed) / rate if rate > 0 else None
        fraction = completed / self.total if self.total else 1
        filled = min(24, int(24 * fraction))
        logger.info(
            "{} [{}{}] {}/{} ({:.1f}%) | {:.1f} tables/s | elapsed={} | ETA={}",
            self.label,
            "#" * filled,
            "-" * (24 - filled),
            completed,
            self.total,
            100 * fraction,
            rate,
            self.duration(elapsed),
            self.duration(eta) if eta is not None else "--",
        )
        self.last_report, self.last_completed = now, completed

    @staticmethod
    def duration(seconds):
        seconds = int(seconds)
        return f"{seconds // 3600:02d}:{seconds // 60 % 60:02d}:{seconds % 60:02d}"


class TransformCheckpoint:
    """Append completed table batches, never rewrite the growing tensor corpus."""

    def __init__(
        self, directory, signature, paths, root, implementation, *, options=None
    ):
        options = options or {}
        self.every_tables = options.get("checkpoint_every_tables", 100_000)
        self.interval_seconds = options.get("checkpoint_interval_seconds")
        for name, value, kinds in (
            ("checkpoint_every_tables", self.every_tables, (int,)),
            ("checkpoint_interval_seconds", self.interval_seconds, (int, float)),
        ):
            if value is not None and (
                isinstance(value, bool)
                or not isinstance(value, kinds)
                or not math.isfinite(value)
                or value <= 0
            ):
                raise ValueError(f"transform.{name} must be positive or null")
        self.signature = signature
        self.total = len(paths)
        identity = hashlib.sha256()
        identity.update(
            json.dumps([signature, implementation, str(root.resolve())]).encode()
        )
        inventory = TransformProgress("Checking source inventory", self.total)
        # A filesystem scan has already asked the disk about every file; only
        # sources that arrive as plain paths are examined here.
        recorded = getattr(paths, "inventory", None)
        entries = recorded(root) if recorded else None
        if entries is None:
            entries = examine(paths, root)
        self.source_bytes = array("q")
        for index, (key, size, modified) in enumerate(entries, 1):
            identity.update(json.dumps([key, size, modified]).encode())
            self.source_bytes.append(size)
            inventory.update(index)
        self.directory = (
            Path(directory) / ".transform_checkpoints" / identity.hexdigest()
        )
        self.directory.mkdir(parents=True, exist_ok=True)
        self.path = self.directory / "progress.json"
        self.state = {
            "schema": 1,
            "signature": signature,
            "total_tables": self.total,
            "completed_tables": 0,
            "shards": [],
            "status": "transforming",
        }
        self.counts = (0, 0, 0)
        self.last_save = perf_counter()

    def __enter__(self):
        # OS-owned locks are released even after a killed process; no stale PID files.
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
                f"Another transform is using {self.directory}"
            ) from error
        return self

    def __exit__(self, *args):
        self.lock.close()

    def restore(self):
        records, tables, excluded = [], [], []
        if self.path.exists():
            self.state = json.loads(self.path.read_text(encoding="utf-8"))
            if (
                self.state["schema"] != 1
                or self.state["signature"] != self.signature
                or self.state["total_tables"] != self.total
            ):
                raise ValueError(f"Incompatible transform checkpoint: {self.path}")
            expected = 0
            bar = TransformProgress(
                "Restoring saved tables", self.state["completed_tables"]
            )
            for shard in self.state["shards"]:
                path = self.directory / shard["name"]
                if path.parent != self.directory or digest(path) != shard["sha256"]:
                    raise ValueError(f"Corrupt transform checkpoint: {path}")
                batch = torch.load(
                    path, weights_only=True, mmap=True, map_location="cpu"
                )
                if batch["start"] != expected or batch["stop"] != shard["stop"]:
                    raise ValueError(f"Invalid transform checkpoint sequence: {path}")
                records.extend(batch["records"])
                tables.extend(batch["tables"])
                excluded.extend(batch["excluded"])
                expected = batch["stop"]
                bar.update(expected)
            if expected != self.state["completed_tables"]:
                raise ValueError(f"Incomplete transform checkpoint: {self.path}")
        self.counts = (len(records), len(tables), len(excluded))
        self.last_save = perf_counter()
        logger.info(
            "Transform checkpoint: {} | resuming at {}/{} tables",
            self.path,
            self.state["completed_tables"],
            self.total,
        )
        logger.info(
            "Checkpoint policy: every_tables={} interval_seconds={} (null disables a trigger)",
            self.every_tables,
            self.interval_seconds,
        )
        return records, tables, excluded

    def save(self, completed, records, tables, excluded, *, force=False):
        previous = self.state["completed_tables"]
        if completed <= previous:
            return
        if not (
            force
            or (
                self.every_tables is not None
                and completed - previous >= self.every_tables
            )
            or (
                self.interval_seconds is not None
                and perf_counter() - self.last_save >= self.interval_seconds
            )
        ):
            return
        r, t, e = self.counts
        name = f"tables_{previous:09d}_{completed:09d}.pt"
        path = self.directory / name
        batch = {
            "start": previous,
            "stop": completed,
            "records": records[r:],
            "tables": tables[t:],
            "excluded": excluded[e:],
        }
        atomic_write(path, lambda stream: torch.save(batch, stream))
        self.state["shards"].append(
            {"name": name, "stop": completed, "sha256": digest(path)}
        )
        self.state.update(completed_tables=completed, status="transforming")
        atomic_json(self.path, self.state)
        self.counts = (len(records), len(tables), len(excluded))
        self.last_save = perf_counter()
        logger.info(
            "Checkpoint saved: {}/{} tables | {}", completed, self.total, self.path
        )

    def status(self, status):
        self.state["status"] = status
        atomic_json(self.path, self.state)


class ShardedTransformCheckpoint(TransformCheckpoint):
    """Spill bounded buffers independently of the durable checkpoint interval."""

    def __init__(self, *args, options=None, **kwargs):
        options = options or {}
        self.batch_tables = options.get("batch_size_tables", 1000)
        for name, value in (
            ("batch_size_tables", self.batch_tables),
            ("batch_memory_limit_mb", options.get("batch_memory_limit_mb", 64)),
        ):
            if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
                raise ValueError(f"transform.{name} must be a positive integer")
        self.max_bytes = options.get("batch_memory_limit_mb", 64) * 1024 * 1024
        super().__init__(*args, options=options, **kwargs)
        self.state.update(
            schema=2,
            record_count=0,
            table_count=0,
            excluded_count=0,
            training_transitions=0,
        )
        self.records, self.tables, self.excluded = [], [], []
        self.buffer_bytes = 0
        self.shards = []
        self.written_bytes = 0
        self.last_spill = 0
        self.counts_total = dict(
            record_count=0, table_count=0, excluded_count=0, training_transitions=0
        )

    def restore(self):
        if self.path.exists():
            state = json.loads(self.path.read_text(encoding="utf-8"))
            if (
                state["schema"] != 2
                or state["signature"] != self.signature
                or state["total_tables"] != self.total
            ):
                raise ValueError(f"Incompatible transform checkpoint: {self.path}")
            bar = TransformProgress("Checking saved shards", len(state["shards"]))
            for index, shard in enumerate(state["shards"], 1):
                path = self.directory / shard["name"]
                if path.parent != self.directory or digest(path) != shard["sha256"]:
                    raise ValueError(f"Corrupt transform checkpoint: {path}")
                bar.update(index)
            self.state = state
        self.shards = list(self.state["shards"])
        self.written_bytes = sum(
            path.stat().st_size
            for shard in self.shards
            for path in self.shard_files(shard)
            if path.exists()
        )
        committed = {shard["name"] for shard in self.shards}
        # Only uncommitted files in this run's locked directory are disposable.
        for path in self.directory.glob("batch_*.pt"):
            if path.name not in committed:
                path.unlink()
        for path in self.directory.glob(".batch_*.pt.*.tmp"):
            path.unlink()
        self.counts_total = {key: self.state[key] for key in self.counts_total}
        self.last_spill = self.state["completed_tables"]
        self.last_save = perf_counter()
        logger.info(
            "Transform checkpoint: {} | resuming at {}/{} tables | buffer={} tables / {} MiB",
            self.path,
            self.last_spill,
            self.total,
            self.batch_tables,
            self.max_bytes // (1024 * 1024),
        )

    def append_record(self, record):
        self.records.append(record)
        self.counts_total["record_count"] += 1
        self.buffer_bytes += sum(
            t.numel() * t.element_size()
            for t in record.get("batch", {}).values()
            if isinstance(t, torch.Tensor)
        )
        if self.buffer_bytes >= self.max_bytes or len(self.records) >= 10_000:
            self.spill()

    def append_table(self, table):
        self.tables.append(table)
        self.counts_total["table_count"] += 1
        self.counts_total["training_transitions"] += table["training_transitions"]

    def append_excluded(self, table):
        self.excluded.append(table)
        self.counts_total["excluded_count"] += 1

    def spill(self):
        if not (self.records or self.tables or self.excluded):
            return
        name = f"batch_{len(self.shards):09d}.pt"
        path = self.directory / name
        batch = {
            "records": self.records,
            "tables": self.tables,
            "excluded": self.excluded,
        }
        atomic_write(path, lambda stream: torch.save(batch, stream))
        shard = {
            "name": name,
            "sha256": digest(path),
            "record_count": len(self.records),
        }
        from sharded_records import write_record_metadata, write_table_metadata

        write_record_metadata(self.directory.parent.parent, shard, self.records)
        write_table_metadata(
            self.directory.parent.parent, shard, self.tables, self.excluded
        )
        self.shards.append(shard)
        self.written_bytes += sum(p.stat().st_size for p in self.shard_files(shard))
        self.records, self.tables, self.excluded = [], [], []
        self.buffer_bytes = 0

    def shard_files(self, shard):
        """The tensor file of a shard and its tensor-free record metadata."""
        metadata = self.directory.parent.parent / ".record_metadata"
        return self.directory / shard["name"], metadata / f"{shard['sha256']}.json"

    def report_space(self, completed):
        """Say early if the remaining tables will not fit on the output disk."""
        if not completed:
            return
        gib = 2**30
        remaining = self.written_bytes / completed * (self.total - completed)
        free = shutil.disk_usage(self.directory).free
        logger.log(
            "WARNING" if remaining > free else "INFO",
            "Output so far {:.1f} GiB; about {:.1f} GiB more expected for the "
            "remaining tables; {:.1f} GiB free on the output disk{}",
            self.written_bytes / gib,
            remaining / gib,
            free / gib,
            ": NOT ENOUGH SPACE at this rate; free some up (the run can resume)"
            if remaining > free
            else "",
        )

    def save_progress(self, completed, *, force=False):
        if completed - self.last_spill >= self.batch_tables:
            self.spill()
            self.last_spill = completed
        previous = self.state["completed_tables"]
        if completed <= previous:
            return
        due = (
            self.every_tables is not None and completed - previous >= self.every_tables
        )
        due |= (
            self.interval_seconds is not None
            and perf_counter() - self.last_save >= self.interval_seconds
        )
        if not (force or due):
            return
        self.spill()
        state = {
            **self.state,
            **self.counts_total,
            "completed_tables": completed,
            "shards": list(self.shards),
            "status": "transforming",
        }
        atomic_json(self.path, state)
        self.state = state
        self.last_save = perf_counter()
        self.last_spill = completed
        logger.info(
            "Checkpoint saved: {}/{} tables | {} shards | {}",
            completed,
            self.total,
            len(self.shards),
            self.path,
        )
        self.report_space(completed)

    def artifact_index(self):
        directory = self.directory.parent.parent
        return {
            "format": "sharded_windows_v1",
            "signature": self.signature,
            "record_count": self.state["record_count"],
            "shards": [
                {
                    "path": (self.directory / s["name"])
                    .relative_to(directory)
                    .as_posix(),
                    "sha256": s["sha256"],
                    "record_count": s["record_count"],
                }
                for s in self.state["shards"]
            ],
        }

    def write_manifest(self, stream, summary, screening_excluded):
        """Stream table metadata instead of constructing a corpus-sized JSON string."""
        from sharded_records import read_table_metadata

        def write(value):
            stream.write(value.encode("utf-8"))

        write(json.dumps(summary, allow_nan=False)[:-1])
        for key in ("tables", "excluded_tables"):
            write(f', "{key}": [')
            first = True
            if key == "excluded_tables":
                for item in screening_excluded:
                    write(("" if first else ",") + json.dumps(item, allow_nan=False))
                    first = False
            for shard in self.state["shards"]:
                entries = read_table_metadata(
                    self.directory.parent.parent, shard, self.directory / shard["name"]
                )
                for item in entries[0 if key == "tables" else 1]:
                    write(("" if first else ",") + json.dumps(item, allow_nan=False))
                    first = False
            write("]")
        write("}\n")

"""Stage 1: raw parquet -> explicit, leakage-free train/evaluation windows."""

import argparse
import hashlib
import json
import multiprocessing
import os
import queue
import re
import sys
import uuid
from collections.abc import Sequence
from concurrent.futures import ProcessPoolExecutor
from contextlib import contextmanager, nullcontext
from datetime import date, datetime, timezone
from pathlib import Path, PurePosixPath
from threading import Event, Thread
from time import perf_counter

import numpy as np
import pandas as pd
import torch
from loguru import logger
from source_io import list_directory, read_files, visit
from table_decoder import column_roles, decode_in_worker, decode_table, start_worker
from torch.utils.serialization import config as serialization_config
from transform_progress import (
    ShardedTransformCheckpoint,
    TransformProgress,
    atomic_write,
)

ROOT = Path(__file__).resolve().parents[2]
TABLE_FILE = "data.parquet"
PROBE_BLOCK = 65536  # neighbouring folders examined together during discovery


def load_config(path=None):
    path = Path(path) if path else ROOT / "configs/config.json"
    if not path.is_absolute():
        path = ROOT / path
    config = json.loads(path.read_text(encoding="utf-8"))
    renamed = [
        f"{old} is now {new}"
        for old, new, used in (
            (
                "paths.embedded",
                "paths.patch_index",
                "embedded" in config.get("paths", {}),
            ),
            ("embedding", "patch_index", "embedding" in config),
        )
        if used
    ]
    if renamed:
        # Otherwise the old settings would be ignored without a word.
        raise ValueError(
            f"{path} names settings from before embed.py became index_patches.py: "
            + "; ".join(renamed)
        )
    generated_dates(config)
    d = config["data"]
    for key in (
        "context_length",
        "horizon",
        "train_windows_per_table",
        "stride",
    ):
        if not isinstance(d[key], int) or isinstance(d[key], bool) or d[key] < 1:
            raise ValueError(f"data.{key} must be a positive integer")
    if not d["targets"] or len(set(d["targets"] + d["known_covariates"])) != len(
        d["targets"] + d["known_covariates"]
    ):
        raise ValueError(
            "targets and known_covariates must be nonempty/disjoint as applicable"
        )
    if d["max_tables"] is not None and d["max_tables"] < 1:
        raise ValueError("max_tables must be null (all tables) or positive")
    steps = config["training"]["steps"]
    if (steps is not None and steps < 1) or config["training"]["learning_rate"] <= 0:
        raise ValueError("training steps and learning_rate must be positive")
    return config


def location(config, name):
    return ROOT / config["paths"][name]


def generated_dates(config):
    """Explicit dates are reproducible; absence retains legacy raw-path behavior."""
    if "generated_dates" not in config["data"]:
        return None
    dates = config["data"]["generated_dates"]
    if not isinstance(dates, list) or not dates:
        raise ValueError(
            "data.generated_dates must be a nonempty list of YYYY-MM-DD folder names"
        )
    for value in dates:
        if not isinstance(value, str) or not re.fullmatch(r"\d{4}-\d{2}-\d{2}", value):
            raise ValueError(
                f"Invalid generated date folder: {value!r}; expected YYYY-MM-DD"
            )
        try:
            date.fromisoformat(value)
        except ValueError as error:
            raise ValueError(f"Invalid generated date folder: {value!r}") from error
    if len(set(dates)) != len(dates):
        raise ValueError("data.generated_dates must not contain duplicates")
    return sorted(dates)


def source_selection(config):
    return {
        "raw": str(config["paths"]["raw"]).replace("\\", "/").rstrip("/"),
        "generated_dates": generated_dates(config),
    }


def discover_source_tables(config, *, manifest=None):
    """Use a run-local native file index when crossing the Windows/WSL boundary."""
    manifest = manifest or os.environ.get("BASIN_SOURCE_MANIFEST")
    if not manifest:
        return _scan_source_tables(config)
    started = perf_counter()
    logger.info("Loading source file index: {}", manifest)
    payload = json.loads(Path(manifest).read_text(encoding="utf-8"))
    if payload.get("schema") != 1 or payload.get(
        "source_selection"
    ) != source_selection(config):
        raise ValueError(
            "Source file index does not match the configured root and dataset dates"
        )
    root = location(config, "raw")
    dates = generated_dates(config)
    if not root.is_dir():
        raise FileNotFoundError(f"Generated-data root not found: {root}")
    for day in dates or ():
        if not (root / day).is_dir():
            raise FileNotFoundError(
                f"Selected generated date folder not found: {root / day}"
            )
    keys = payload.get("paths")
    if not isinstance(keys, list) or not keys:
        raise ValueError("Source file index must contain a nonempty list of paths")
    selected, seen = [], set()
    for key in keys:
        if not isinstance(key, str) or "\\" in key or ":" in key:
            raise ValueError("Source index paths must be portable relative paths")
        relative = PurePosixPath(key)
        if (
            relative.is_absolute()
            or ".." in relative.parts
            or relative.name != "data.parquet"
        ):
            raise ValueError(
                "Source index paths must stay inside the selected data root"
            )
        if dates is not None and relative.parts[0] not in dates:
            raise ValueError("Source index contains an unselected dataset date")
        canonical = relative.as_posix()
        if canonical in seen:
            raise ValueError("Duplicate source paths in file index")
        seen.add(canonical)
        # Deliberately avoid resolve/is_file here: each would incur remote metadata
        # requests on /mnt/e. The validator opens and hashes files as it uses them.
        selected.append(canonical)
    logger.info(
        "Loaded {} source tables from file index in {:.2f}s",
        len(selected),
        perf_counter() - started,
    )
    return SourceTables(root, selected)


def write_source_manifest(config):
    """Snapshot file membership natively on Windows before launching a GPU stage."""
    paths = _scan_source_tables(config)
    directory = ROOT / ".cache/source_indexes"
    directory.mkdir(parents=True, exist_ok=True)
    path = directory / (
        datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S_")
        + uuid.uuid4().hex
        + ".json"
    )
    write_json(
        path,
        {
            "schema": 1,
            "created_at_utc": datetime.now(timezone.utc).isoformat(),
            "source_selection": source_selection(config),
            "paths": paths.keys,
        },
    )
    logger.info("Prepared native source file index: {} tables, {}", len(paths), path)
    return path


class SourceTables(Sequence):
    """Sorted source tables, stored as keys relative to the raw-data root.

    Indexing yields the same `Path` objects a list of paths would, but builds
    them on access: millions of live `Path` objects need many gigabytes. A
    filesystem scan also keeps what it learned about each file, so later steps
    need not ask the disk again. `sizes` and `modified` are what `os.stat`
    reported; `records` orders the tables by their place in the volume's file
    table (see `source_io`). Each is None when unknown.
    """

    def __init__(self, root, keys, sizes=None, modified=None, records=None):
        self.root, self.keys = Path(root), keys
        self.sizes, self.modified, self.records = sizes, modified, records

    def __len__(self):
        return len(self.keys)

    def __getitem__(self, index):
        if isinstance(index, slice):
            known = (self.sizes, self.modified, self.records)
            return SourceTables(
                self.root,
                self.keys[index],
                *(None if values is None else values[index] for values in known),
            )
        return self.root / self.keys[index]

    def __eq__(self, other):
        return (
            isinstance(other, Sequence)
            and len(self) == len(other)
            and all(mine == theirs for mine, theirs in zip(self, other))
        )

    def files(self, start, stop):
        """Path strings for tables [start, stop), without building Path objects."""
        prefix = str(self.root) + os.sep
        return [prefix + key.replace("/", os.sep) for key in self.keys[start:stop]]

    def inventory(self, root):
        """(key, size, modification time) of every table if the scan recorded them."""
        if self.sizes is None or Path(root) != self.root:
            return None

        def entries(step=65536):
            for start in range(0, len(self.keys), step):
                block = slice(start, start + step)
                yield from zip(
                    self.keys[block],
                    self.sizes[block].tolist(),
                    self.modified[block].tolist(),
                )

        return entries()


def _scan_source_tables(config):
    """Shared recursive selection, with a heartbeat even before the first match."""
    root = location(config, "raw")
    if not root.is_dir():
        raise FileNotFoundError(f"Generated-data root not found: {root}")
    dates = generated_dates(config)
    keys, sizes, modified, records = [], [], [], []
    for day in dates if dates is not None else [None]:
        directory = root / day if day else root
        if not directory.is_dir():
            raise FileNotFoundError(
                f"Selected generated date folder not found: {directory}"
            )
        started = perf_counter()
        progress = {"listed": 0, "found": 0}
        stop = Event()

        def report_progress():
            while not stop.wait(5):
                logger.info(
                    "Discovering source tables: {} | listed {} | found {} | elapsed {:.1f}s",
                    directory,
                    progress["listed"],
                    progress["found"],
                    perf_counter() - started,
                )

        logger.info("Discovering source tables in {} (filesystem scan)", directory)
        heartbeat = Thread(target=report_progress, daemon=True)
        heartbeat.start()
        try:
            tables = _find_tables(root, f"{day}/" if day else "", progress)
        finally:
            stop.set()
            heartbeat.join()
        if not tables[0]:
            raise ValueError(f"No data.parquet tables in selected folder: {directory}")
        logger.info(
            "Selected generated folder={} tables={} ({:.1f}s)",
            directory,
            len(tables[0]),
            perf_counter() - started,
        )
        keys.extend(tables[0])
        sizes.append(tables[1])
        modified.append(tables[2])
        records.append(tables[3])
    logger.info(
        "Source selection={} total_tables={}", source_selection(config), len(keys)
    )
    return SourceTables(
        root,
        keys,
        np.concatenate(sizes),
        np.concatenate(modified),
        None if any(r is None for r in records) else np.concatenate(records),
    )


def _find_tables(root, prefix, progress, threads=8):
    """Locate every data.parquet below root/prefix; keys are relative to root.

    A folder that holds a data.parquet is a table and is not searched further;
    any other folder is searched recursively. Telling them apart takes one
    metadata lookup per folder rather than listing each one, made in file-table
    order where the platform reports it. What the lookup returns (size and
    modification time) is kept, so the checkpoint need not ask again.
    """
    base = str(root)
    keys, sizes, modified, records = [], [], [], []
    ordered = True
    pending = [prefix]
    while pending:
        prefix = pending.pop()
        folder = base + os.sep + prefix.replace("/", os.sep) if prefix else base
        names, places = [], []
        try:
            for name, is_directory, record in list_directory(folder):
                progress["listed"] += 1
                if _is_table(name) or is_directory:
                    names.append(name)
                    places.append(record)
        except OSError:
            continue  # unreadable folders are skipped, as pathlib's rglob does
        if not names:
            continue
        ordered = ordered and places[0] is not None
        places = np.array(places if ordered else [0] * len(names), dtype=np.int64)
        order = range(len(names))
        if ordered:
            # File-table order within blocks of neighbouring names: the sweep
            # through the file table stays sequential, while only a small part
            # of this folder's own index (huge for millions of tables) has to
            # stay in memory at any time.
            order = np.concatenate(
                [
                    start
                    + np.argsort(places[start : start + PROBE_BLOCK], kind="stable")
                    for start in range(0, len(names), PROBE_BLOCK)
                ]
            ).tolist()
        found_size = np.full(len(names), -1, dtype=np.int64)
        found_modified = np.zeros(len(names), dtype=np.int64)

        def probe(index):
            name = names[index]
            target = name if _is_table(name) else name + os.sep + TABLE_FILE
            try:
                status = os.stat(folder + os.sep + target)
            except OSError:
                return
            found_size[index] = status.st_size
            found_modified[index] = status.st_mtime_ns
            progress["found"] += 1

        visit(probe, order, threads)
        for name, size in zip(names, found_size.tolist()):
            if _is_table(name):
                if size >= 0:
                    keys.append(prefix + name)
            elif size >= 0:
                keys.append(f"{prefix}{name}/{TABLE_FILE}")
            else:
                pending.append(f"{prefix}{name}/")
        found = found_size >= 0
        sizes.append(found_size[found])
        modified.append(found_modified[found])
        records.append(places[found])
    sizes, modified, records = (
        np.concatenate(values) if values else np.zeros(0, dtype=np.int64)
        for values in (sizes, modified, records)
    )
    order = _path_order(keys)
    if order is not None:
        keys = [keys[index] for index in order]
        sizes, modified, records = sizes[order], modified[order], records[order]
    return keys, sizes, modified, records if ordered else None


def _is_table(name):
    """Whether a directory entry is a table; pathlib ignores case on Windows."""
    return (name.lower() if os.name == "nt" else name) == TABLE_FILE


def _path_order(keys):
    """Positions that sort relative keys exactly as pathlib orders the paths.

    Paths compare part by part, ignoring case on Windows. Joining the parts
    with a character that sorts below every other makes plain string comparison
    agree with that. Returns None when the keys are already in order, which a
    directory listing usually is.
    """
    fold = str.lower if os.name == "nt" else str
    previous = ""
    for key in keys:
        rank = fold(key).replace("/", "\0")
        if rank < previous:
            break
        previous = rank
    else:
        return None
    ranks = [fold(key).replace("/", "\0") for key in keys]
    return sorted(range(len(keys)), key=ranks.__getitem__)


def signature(config):
    content = {k: config[k] for k in ("seed", "data", "model")}
    if "validation" in config:
        content["validation"] = config["validation"]
    if "generated_dates" in config["data"]:
        # Canonical ordering makes date-list order irrelevant. Bind the root too:
        # the same date names under another source root are a different cohort.
        content["data"] = dict(config["data"], generated_dates=generated_dates(config))
        content["source_selection"] = source_selection(config)
    return hashlib.sha256(json.dumps(content, sort_keys=True).encode()).hexdigest()


def file_digest(path):
    with Path(path).open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def setup_logging(config, stage):
    run_id = (
        datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S_%fZ")
        + f"_{stage}_{uuid.uuid4().hex[:8]}"
    )
    directory = location(config, "logs") / run_id
    directory.mkdir(parents=True, exist_ok=False)
    logger.remove()
    logger.add(sys.stderr, level=config["logging"]["level"])
    logger.add(
        str(directory / "events.log"), level="DEBUG", rotation="10 MB", diagnose=False
    )
    logger.info(
        "Stage={} interpreter={} torch={} CUDA={} root={}",
        stage,
        sys.executable,
        torch.__version__,
        torch.cuda.is_available(),
        ROOT,
    )
    logger.info("Input configuration:\n{}", json.dumps(config, indent=2))
    write_json(directory / "config.json", config)
    sources = [
        ROOT / "rwkv7_backend.py",
        *ROOT.glob("src/02_data_validation/*.py"),
        *ROOT.glob("src/03_transforming_input_data/*.py"),
        *ROOT.glob("src/04_patch_indexing/*.py"),
        *ROOT.glob("src/05_model_training/*.py"),
        *ROOT.glob("src/06_holdout_test/*.py"),
        *ROOT.glob("src/07_forecasting/*.py"),
    ]
    write_json(
        directory / "provenance.json",
        {
            "run_id": run_id,
            "stage": stage,
            "interpreter": sys.executable,
            "torch": torch.__version__,
            "source_sha256": {
                p.relative_to(ROOT).as_posix(): file_digest(p) for p in sources
            },
        },
    )
    return directory


@contextmanager
def run_context(config, stage):
    if stage in ("train", "capacity_test", "holdout_test"):
        # Training owns the JSONL journal; preprocessing needs only ordinary logs.
        sys.path.insert(0, str(ROOT / "src/05_model_training"))
        from training_log import training_run

        context = training_run(config, stage)
    else:
        context = nullcontext()
    with context as journal:
        directory = setup_logging(config, stage)
        if journal is not None:
            logger.info("Training JSONL: {}", journal.path)
            write_json(
                directory / "training_jsonl.json",
                {
                    "path": str(journal.path),
                    "run_id": journal.run_id,
                    "description": "Pointer to the live JSONL journal; open path for training events. This file is not the journal.",
                    "journal_filename": journal.path.name,
                    "journal_relative_to_this_file": "../" + journal.path.name,
                    "metrics_visibility": "Every event is flushed; epoch and checkpoint events are also synced to disk.",
                    "epoch_metrics_event": "epoch_end",
                },
            )
        started = perf_counter()
        write_json(directory / "status.json", {"status": "running"})
        try:
            yield directory
        except BaseException as error:
            logger.exception("Stage failed: {}", stage)
            write_json(
                directory / "status.json",
                {
                    "status": "failed",
                    "error": str(error),
                    "seconds": perf_counter() - started,
                },
            )
            raise
        else:
            write_json(
                directory / "status.json",
                {"status": "complete", "seconds": perf_counter() - started},
            )


@contextmanager
def timed(label, cuda=False, level="INFO"):
    if cuda:
        torch.cuda.synchronize()
    start = perf_counter()
    logger.log(level, "START {}", label)
    try:
        yield
    except Exception:
        logger.exception("FAILED {} after {:.4f}s", label, perf_counter() - start)
        raise
    finally:
        if cuda:
            torch.cuda.synchronize()
        logger.log(level, "END {} elapsed={:.4f}s", label, perf_counter() - start)


def describe(name, tensor, preview=8):
    x = tensor.detach().cpu()
    finite = torch.isfinite(x)
    logger.debug(
        "{} shape={} dtype={} finite={}/{} preview={}",
        name,
        list(x.shape),
        x.dtype,
        int(finite.sum()),
        x.numel(),
        x.flatten()[:preview].tolist(),
    )
    if not finite.all():
        logger.warning(
            "{} contains {} missing/nonfinite values", name, int((~finite).sum())
        )
    if finite.any():
        values = x[finite].float()
        logger.debug(
            "{} finite statistics min={} max={} mean={}",
            name,
            float(values.min()),
            float(values.max()),
            float(values.mean()),
        )
    return {
        "shape": list(x.shape),
        "dtype": str(x.dtype),
        "finite": int(finite.sum()),
        "elements": x.numel(),
    }


def save_artifact(path, payload):
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(payload, path)
    logger.info("Saved artifact={} bytes={}", path, path.stat().st_size)


def read_artifact(path, config):
    logger.info("Loading artifact={} bytes={}", path, path.stat().st_size)
    # Map tensor storage from the artifact instead of copying the whole corpus
    # into anonymous RAM. Only accessed pages need to become resident.
    # Calculate offsets from the modern checkpoint layout instead of seeking to
    # every tensor header. Large corpora otherwise cause hundreds of thousands
    # of small reads on the project drive even with memory mapping enabled.
    previous_offsets = serialization_config.load.calculate_storage_offsets
    try:
        serialization_config.load.calculate_storage_offsets = True
        artifact = torch.load(path, map_location="cpu", weights_only=True, mmap=True)
    finally:
        serialization_config.load.calculate_storage_offsets = previous_offsets
    if artifact["signature"] != signature(config):
        raise ValueError(
            f"Stale artifact {path}: seed/data/model configuration changed; rerun upstream stages"
        )
    if artifact.get("format") == "sharded_windows_v1":
        from sharded_records import ShardedRecords

        return {**artifact, "records": ShardedRecords(artifact, path.parent)}
    if artifact.get("format") == "lazy_patch_index_v1":
        sys.path.insert(0, str(ROOT / "src/04_patch_indexing"))
        from index_patches import open_patch_index

        return open_patch_index(artifact, config)
    return artifact


def write_json(path, value, *, level="INFO"):
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{uuid.uuid4().hex}.tmp")
    logger.log(level, "Writing JSON={}", path)
    heartbeat = None
    stop = Event()
    keys = value.get("source_keys") if isinstance(value, dict) else None
    if isinstance(keys, dict) and len(keys) >= 100_000:
        started = perf_counter()

        def report_progress():
            while not stop.wait(30):
                try:
                    written = temporary.stat().st_size
                except OSError:
                    written = 0
                logger.log(
                    level,
                    "Writing JSON={} bytes_written={:,} elapsed={:.1f}s",
                    path,
                    written,
                    perf_counter() - started,
                )

        heartbeat = Thread(target=report_progress, daemon=True)
        heartbeat.start()
    try:
        with temporary.open(
            "w", encoding="utf-8", newline="\n", buffering=1024 * 1024
        ) as stream:
            json.dump(value, stream, indent=2, allow_nan=False)
            stream.write("\n")
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
    finally:
        stop.set()
        if heartbeat is not None:
            heartbeat.join()
        temporary.unlink(missing_ok=True)
    logger.log(level, "Saved JSON={}", path)


def frame_window(
    frame,
    *,
    cutoff,
    context_length,
    horizon,
    targets=("outcome",),
    past_covariates=(),
    known_covariates=(),
    device="cpu",
):
    columns = list(targets) + list(past_covariates) + list(known_covariates)
    if not targets or len(set(columns)) != len(columns):
        raise ValueError("Column roles must be disjoint and targets nonempty")
    if (
        context_length < 1
        or horizon < 1
        or cutoff < context_length
        or cutoff + horizon > len(frame)
    ):
        raise ValueError("Window lies outside table")

    def tensor(names, start, stop):
        return torch.as_tensor(
            frame[list(names)].iloc[start:stop].to_numpy(dtype=np.float32).T.copy(),
            device=device,
        ).unsqueeze(0)

    batch = {
        "context": tensor(targets, cutoff - context_length, cutoff),
        "horizon": horizon,
        "future_target": tensor(targets, cutoff, cutoff + horizon),
    }
    if past_covariates:
        batch["past_covariates"] = tensor(
            past_covariates, cutoff - context_length, cutoff
        )
    if known_covariates:
        batch["known_covariates"] = tensor(
            known_covariates, cutoff - context_length, cutoff
        )
        batch["future_covariates"] = tensor(known_covariates, cutoff, cutoff + horizon)
    return batch


def validation_run_root(config):
    """Separate cohorts by canonical dates and source root; each run is immutable."""
    selection = source_selection(config)
    dates = generated_dates(config)
    label = "_".join(dates) if dates else "selected_tables"
    if len(label) > 70:
        label = f"{dates[0]}_to_{dates[-1]}"
    identity = hashlib.sha256(
        json.dumps(selection, sort_keys=True).encode()
    ).hexdigest()[:12]
    return (ROOT / config["validation"]["metrics_path"]).parent / f"{label}_{identity}"


def validation_report_path(config):
    """Locate this cohort's most recent fully written report."""
    policy = config["validation"]
    if not policy.get("run_folders", True):
        return ROOT / policy["metrics_path"]
    directory = validation_run_root(config).resolve()
    pointer = directory / "latest.json"
    if not pointer.is_file():
        raise FileNotFoundError(
            f"No validation report for {source_selection(config)}. Run src/02_data_validation/validate_generated_data.py first."
        )
    metadata = json.loads(pointer.read_text(encoding="utf-8"))
    path = (directory / metadata["report"]).resolve()
    if not path.is_relative_to(directory):
        raise ValueError(
            "Validation report pointer must remain inside its cohort folder"
        )
    return path


def validated_sources(config, paths):
    """Admit current successful multivariate forecasts no worse than persistence."""
    policy = config.get("validation", {})
    if policy.get("enabled", False) and policy.get("engine", "chronos_bolt") in (
        "chronos2",
        "chronos_bolt",
    ):
        return foundation_validated_sources(config, paths)
    if not policy.get("enabled", False):
        logger.info(
            "Data screening skipped: validation.enabled=false; admitting all {} candidate tables",
            len(paths),
        )
        audit = {
            "enabled": False,
            "screening_skipped": True,
            "reason": "validation.enabled=false",
            "source_count": len(paths),
            "admitted_count": len(paths),
            "excluded": [],
            "source_selection": source_selection(config),
        }
        write_json(location(config, "processed") / "validation_selection.json", audit)
        return paths, audit
    if (
        config["data"]["targets"] != ["outcome"]
        or config["data"]["time_column"] != "time_index"
    ):
        raise ValueError(
            "Prophet screening currently requires target=outcome and time_column=time_index"
        )
    report_path = validation_report_path(config)
    if not report_path.is_file():
        raise FileNotFoundError(
            f"Missing validation report {report_path}. Run src/02_data_validation/validate_generated_data.py first."
        )
    scores = pd.read_parquet(report_path)
    required = {
        "source_key",
        "source_sha256",
        "status",
        "baseline_mae",
        "multivariate_mae",
        "reserved_tail_rows",
        "covariate_mode",
        "screening_signature",
    }
    if required - set(scores):
        raise ValueError(
            f"Validation report lacks columns: {sorted(required - set(scores))}"
        )
    if scores.source_key.duplicated().any():
        raise ValueError("Duplicate source keys in validation report")
    expected = screening_signature(config)
    indexed = scores.set_index("source_key")
    admitted, excluded = [], []
    for path in paths:
        key = path.relative_to(location(config, "raw")).as_posix()
        reason = "missing validation row"
        if key in indexed.index:
            row = indexed.loc[key]
            reason = "Multivariate Prophet did not produce a valid MAE <= last-observation MAE"
            if row.screening_signature != expected:
                reason = "screening configuration mismatch; rerun validation"
            elif row.source_sha256 != file_digest(path):
                reason = "source content changed; rerun validation"
            elif row.reserved_tail_rows != config["data"]["horizon"]:
                reason = "final test horizon was not reserved"
            elif row.covariate_mode != policy["covariate_mode"]:
                reason = "covariate availability mismatch"
            elif row.status == "ok":
                baseline = row.baseline_mae
                score = row.multivariate_mae
                fit_error = row.get("multivariate_error", "")
                if (
                    np.isfinite(baseline)
                    and baseline >= 0
                    and np.isfinite(score)
                    and 0 <= score <= baseline
                    and (pd.isna(fit_error) or not str(fit_error).strip())
                ):
                    admitted.append(path)
                    logger.info(
                        "ADMIT {} baseline_mae={} multivariate_mae={}",
                        key,
                        baseline,
                        score,
                    )
                    continue
            else:
                reason = "validation failed"
        excluded.append({"source": key, "reason": reason})
        logger.warning("REJECT {}: {}", key, reason)
    audit = {
        "enabled": True,
        "report": str(report_path),
        "report_sha256": file_digest(report_path),
        "screening_signature": expected,
        "source_selection": source_selection(config),
        "source_count": len(paths),
        "admitted_count": len(admitted),
        "excluded": excluded,
    }
    if "smoke_test_only" in scores and scores.smoke_test_only.fillna(False).any():
        audit.update(
            smoke_test_only=True,
            actual_scoring_metric="rmse",
            metric_alias="RMSE values exposed as MAE for the authorized smoke test",
        )
        logger.info(
            "Smoke-test screening: legacy RMSE scores are being used via MAE aliases"
        )
    write_json(location(config, "processed") / "validation_selection.json", audit)
    logger.info(
        "Learnability selection: {} admitted / {} candidates", len(admitted), len(paths)
    )
    if not admitted:
        raise ValueError(
            "No datasets passed Prophet-vs-naive validation; inspect validation_selection.json"
        )
    return admitted, audit


def foundation_validated_sources(config, paths):
    """Use current foundation-model decisions; automatically screen stale cohorts."""
    sys.path.insert(0, str(ROOT / "src/02_data_validation"))
    from bolt_validator import bolt_admitted

    policy = config["validation"]
    engine = policy.get("engine", "chronos_bolt")
    label = "Chronos-2 Small" if engine == "chronos2" else "Chronos-Bolt"
    admission = bolt_admitted
    if engine == "chronos2":
        from chronos_validator import chronos_admitted

        admission = chronos_admitted
    tolerance = policy.get(
        "max_relative_degradation", 0.15 if engine == "chronos2" else 0.1
    )
    expected = screening_signature(config)
    digests = {}

    def source_digest(path):
        if path not in digests:
            digests[path] = file_digest(path)
        return digests[path]

    try:
        report = validation_report_path(config)
        scores = pd.read_parquet(report)
    except FileNotFoundError:
        report, scores = None, None
    required = {
        "source_key",
        "source_sha256",
        "screening_signature",
        "status",
        "reserved_tail_rows",
        "mase_scale",
        "baseline_mase",
        f"{engine}_mase",
        "baseline_mae",
        f"{engine}_mae",
    }
    current = (
        scores is not None
        and not (required - set(scores))
        and scores.screening_signature.eq(expected).all()
    )
    keys = [path.relative_to(location(config, "raw")).as_posix() for path in paths]
    if current:
        if scores.source_key.duplicated().any():
            raise ValueError("Duplicate source keys in validation report")
        indexed = scores.set_index("source_key")
        current = all(
            key in indexed.index
            and indexed.loc[key, "source_sha256"] == source_digest(path)
            for path, key in zip(paths, keys)
        )
    if not current:
        if not policy.get("auto_run", True):
            raise ValueError(
                f"Missing or stale {label} screening report; run src/02_data_validation/validate_generated_data.py"
            )
        from validate_generated_data import validate

        logger.info(
            "{} screening missing/stale; screening selected cohort before transformation",
            label,
        )
        scores = validate(config, retry_errors=report is not None)
        digests.clear()
        report = Path(scores.attrs["report_path"])
    if required - set(scores):
        raise ValueError(f"{label} screening report lacks required columns")
    if scores.source_key.duplicated().any():
        raise ValueError("Duplicate source keys in validation report")
    decisions = admission(scores, tolerance)
    indexed = scores.assign(accepted=decisions).set_index("source_key")
    admitted, excluded = [], []
    for path, key in zip(paths, keys):
        row = indexed.loc[key] if key in indexed.index else None
        valid = (
            row is not None
            and row.screening_signature == expected
            and row.source_sha256 == source_digest(path)
            and row.reserved_tail_rows == config["data"]["horizon"]
        )
        if valid and bool(row.accepted):
            admitted.append(path)
        else:
            excluded.append(
                {
                    "source": key,
                    "reason": f"{label} error more than tolerance worse than naive"
                    if valid and row.status == "ok"
                    else "missing, stale or failed screening",
                }
            )
    audit = {
        "enabled": True,
        "engine": engine,
        "admission_metric": "mae" if engine == "chronos2" else "mase",
        "max_relative_degradation": tolerance,
        "report": str(report),
        "report_sha256": file_digest(report),
        "screening_signature": expected,
        "source_selection": source_selection(config),
        "source_count": len(paths),
        "admitted_count": len(admitted),
        "undefined_mase_retained": int((scores.mase_scale.eq(0) & decisions).sum()),
        "excluded": excluded,
    }
    write_json(location(config, "processed") / "validation_selection.json", audit)
    logger.info(
        "{} selection: {} admitted / {} candidates", label, len(admitted), len(paths)
    )
    if not admitted:
        raise ValueError(
            f"No datasets passed {label} screening; inspect validation_selection.json"
        )
    return admitted, audit


def screening_signature(config):
    """Separate screening settings from training/embedding hyperparameters."""
    policy = {
        k: v
        for k, v in config["validation"].items()
        if k not in ("enabled", "metrics_path", "workers", "run_folders", "auto_run")
    }
    content = {
        "validation": policy,
        "seed": config["seed"],
        "reserved_tail_rows": config["data"]["horizon"],
        "schema": 5
        if policy.get("engine") == "chronos2"
        else 4
        if policy.get("engine", "chronos_bolt") == "chronos_bolt"
        else 3,
        "admission_metric": "mase"
        if policy.get("engine", "chronos_bolt") == "chronos_bolt"
        else "mae",
    }
    if policy.get("engine") == "chronos2":
        content["input_roles"] = {
            key: config["data"][key]
            for key in (
                "targets",
                "time_column",
                "past_covariates",
                "known_covariates",
                "excluded_column_patterns",
            )
        }
    return hashlib.sha256(json.dumps(content, sort_keys=True).encode()).hexdigest()


def transform(config):
    if config["model"].get("objective") == "next_patch":
        return transform_next_patch(config)
    d = config["data"]
    directory = location(config, "processed")
    directory.mkdir(parents=True, exist_ok=True)
    paths = discover_source_tables(config)
    paths, selection = validated_sources(config, paths)
    records, manifest = [], []
    selected = 0
    for path in paths:
        if d["max_tables"] is not None and selected >= d["max_tables"]:
            break
        with timed(f"read/validate {path}"):
            frame = pd.read_parquet(path)
            logger.info("Raw shape={} columns={}", frame.shape, frame.columns.tolist())
            if len(frame) < d["context_length"] + 2 * d["horizon"]:
                logger.warning("Skipping short table={} rows={}", path, len(frame))
                continue
            if d["time_column"] in frame:
                time = frame[d["time_column"]]
                if (
                    time.isna().any()
                    or not time.is_monotonic_increasing
                    or time.duplicated().any()
                ):
                    raise ValueError(f"Non-unique/non-increasing time in {path}")
            else:
                logger.warning("No time column: using existing row order for {}", path)
            excluded = [
                c
                for c in frame
                if any(
                    re.search(pattern, c) for pattern in d["excluded_column_patterns"]
                )
            ]
            past = d["past_covariates"]
            if past == "auto":
                past = [
                    c
                    for c in frame
                    if c
                    not in d["targets"]
                    + d["known_covariates"]
                    + [d["time_column"]]
                    + excluded
                ]
            columns = d["targets"] + list(past) + d["known_covariates"]
            if set(columns) & set(excluded):
                raise ValueError(
                    f"Explicit input roles include excluded lag columns: {set(columns) & set(excluded)}"
                )
            logger.info("Excluded lag-derived columns={}", excluded)
            values = frame[columns].to_numpy(dtype=np.float32)
            if np.isinf(values).any():
                raise ValueError(
                    f"Infinity in {path}; use NaN to mark missing observations"
                )
            logger.info(
                "Roles targets={} past={} known_future={}; missing={}",
                d["targets"],
                past,
                d["known_covariates"],
                int(np.isnan(values).sum()),
            )
        split = len(frame) - d["horizon"]
        train_cutoffs = list(
            range(d["context_length"], split - d["horizon"] + 1, d["stride"])
        )
        train_cutoffs = train_cutoffs[-d["train_windows_per_table"] :]
        source_path = (
            path.relative_to(ROOT).as_posix()
            if path.is_relative_to(ROOT)
            else str(path)
        )
        table_metadata = {
            "table_index": selected,
            "source": source_path,
            "source_sha256": file_digest(path),
            "source_rows": len(frame),
            "evaluation_cutoff": split,
            "excluded_columns": excluded,
            "variable_names": columns,
            "roles": {
                "targets": d["targets"],
                "past": list(past),
                "known": d["known_covariates"],
            },
        }
        for role, cutoff in [("train", c) for c in train_cutoffs] + [
            ("evaluation", split)
        ]:
            with timed(f"window table={selected} split={role} cutoff={cutoff}"):
                batch = frame_window(
                    frame,
                    cutoff=cutoff,
                    context_length=d["context_length"],
                    horizon=d["horizon"],
                    targets=d["targets"],
                    past_covariates=past,
                    known_covariates=d["known_covariates"],
                )
                if not torch.isfinite(batch["future_target"]).any():
                    raise ValueError(f"No observed target labels in {path} at {cutoff}")
                key = f"table_{selected:03d}_{role}_{cutoff}"
                summary = {
                    k: describe(k, v, config["logging"]["preview_values"])
                    for k, v in batch.items()
                    if isinstance(v, torch.Tensor)
                }
                records.append(
                    {
                        **table_metadata,
                        "id": key,
                        "split": role,
                        "batch": batch,
                        "source": (
                            str(path.relative_to(ROOT))
                            if path.is_relative_to(ROOT)
                            else str(path)
                        ),
                        "cutoff": cutoff,
                        "context_rows": [cutoff - d["context_length"], cutoff],
                        "label_rows": [cutoff, cutoff + d["horizon"]],
                        "targets": d["targets"],
                    }
                )
                manifest.append(
                    {
                        **table_metadata,
                        "id": key,
                        "source": (
                            str(path.relative_to(ROOT))
                            if path.is_relative_to(ROOT)
                            else str(path)
                        ),
                        "cutoff": cutoff,
                        "context_rows": [cutoff - d["context_length"], cutoff],
                        "label_rows": [cutoff, cutoff + d["horizon"]],
                        "split": role,
                        "roles": {
                            "targets": d["targets"],
                            "past": past,
                            "known": d["known_covariates"],
                        },
                        "tensors": summary,
                    }
                )
                if d["export_processed_tables"]:
                    export_columns = (
                        [d["time_column"]] if d["time_column"] in frame else []
                    ) + columns
                    frame.loc[:, export_columns].iloc[
                        cutoff - d["context_length"] : cutoff + d["horizon"]
                    ].to_parquet(
                        directory / f"{key}.parquet", index=False, compression="zstd"
                    )
                    logger.info(
                        "Saved inspection Parquet={}", directory / f"{key}.parquet"
                    )
        selected += 1
    if not records:
        raise ValueError(
            "No eligible raw tables. Check paths.raw and context/horizon lengths"
        )
    payload = {
        "signature": signature(config),
        "records": records,
        "source_selection": source_selection(config),
    }
    save_artifact(directory / "windows.pt", payload)
    write_json(
        directory / "manifest.json",
        {"config": config, "windows": manifest, "validation": selection},
    )
    logger.success("Transform complete: {} tables, {} windows", selected, len(records))
    return payload


def array_window(targets, past, *, cutoff, context_length, horizon):
    """`frame_window` for a table already converted to float32 arrays."""
    if (
        context_length < 1
        or horizon < 1
        or cutoff < context_length
        or cutoff + horizon > len(targets)
    ):
        raise ValueError("Window lies outside table")

    def tensor(values, start, stop):
        return torch.as_tensor(values[start:stop].T.copy(), device="cpu").unsqueeze(0)

    batch = {
        "context": tensor(targets, cutoff - context_length, cutoff),
        "horizon": horizon,
        "future_target": tensor(targets, cutoff, cutoff + horizon),
    }
    if past is not None:
        batch["past_covariates"] = tensor(past, cutoff - context_length, cutoff)
    return batch


# Starting worker processes costs a few seconds; smaller jobs decode in place.
WORKER_MINIMUM_TABLES = 4096


def throughput_options(config):
    """Settings that change how fast tables are read and decoded, never the result."""
    options = config.get("transform", {})
    values = {
        "workers": options.get("workers"),
        "read_threads": options.get("read_threads", 8),
        "read_window_tables": options.get("read_window_tables", 16384),
        "read_window_mb": options.get("read_window_mb", 512),
    }
    for name, value in values.items():
        optional = name == "workers"
        if not (optional and value is None) and (
            isinstance(value, bool)
            or not isinstance(value, int)
            or value < (0 if optional else 1)
        ):
            raise ValueError(
                f"transform.{name} must be "
                + (
                    "null or a non-negative integer"
                    if optional
                    else "a positive integer"
                )
            )
    return values


def source_windows(sizes, start, stop, max_tables, max_bytes):
    """Split tables [start, stop) into consecutive ranges bounded in count and bytes."""
    first, total = start, 0
    for index in range(start, stop):
        if index > first and (
            index - first >= max_tables or total + sizes[index] > max_bytes
        ):
            yield first, index
            first, total = index, 0
        total += sizes[index]
    if first < stop:
        yield first, stop


@contextmanager
def decoded_tables(paths, start, sizes, settings, options):
    """Iterate `decode_table` results for tables [start, len(paths)), in order.

    One thread reads the next window of source files in on-disk order while the
    previous window is decoded, in worker processes when the job is large
    enough to repay starting them. Reading ahead only changes when the work is
    done: the caller still receives every table strictly in order.
    """
    workers = options["workers"]
    if workers is None:
        large = len(paths) - start >= WORKER_MINIMUM_TABLES
        workers = min(8, (os.cpu_count() or 2) // 2) if large else 0
    windows = queue.Queue(maxsize=1)
    stop = Event()

    def offer(item):
        while not stop.is_set():
            try:
                return windows.put(item, timeout=0.2)
            except queue.Full:
                pass

    def read_ahead():
        try:
            ranges = source_windows(
                sizes,
                start,
                len(paths),
                options["read_window_tables"],
                options["read_window_mb"] * 1024 * 1024,
            )
            for first, last in ranges:
                if stop.is_set():
                    return
                order = None
                if isinstance(paths, SourceTables):
                    files = paths.files(first, last)
                    if paths.records is not None:
                        order = paths.records[first:last].tolist()
                else:
                    files = [str(path) for path in paths[first:last]]
                offer((first, read_files(files, options["read_threads"], order)))
            offer(None)
        except BaseException as error:  # raised again where the tables are consumed
            offer(error)

    def results(pool):
        while True:
            window = windows.get()
            if window is None:
                return
            if isinstance(window, BaseException):
                raise window
            first, contents = window
            tasks = zip(range(first, first + len(contents)), contents)
            if pool is None:
                for task in tasks:
                    yield decode_table(task, settings)
            else:
                yield from pool.map(decode_in_worker, tasks, chunksize=32)

    reader = Thread(target=read_ahead, daemon=True, name="source-reader")
    pool = None
    try:
        if workers:
            logger.info("Decoding tables in {} worker processes", workers)
            # Unlike multiprocessing.Pool, this reports a worker that died
            # instead of waiting forever for its result.
            pool = ProcessPoolExecutor(
                workers,
                mp_context=multiprocessing.get_context("spawn"),
                initializer=start_worker,
                initargs=(settings,),
            )
        reader.start()
        yield results(pool)
    finally:
        stop.set()
        if pool is not None:
            pool.shutdown(wait=True, cancel_futures=True)
        if reader.is_alive():
            reader.join()


def transform_next_patch(config):
    """Build causal transitions using full tables or the legacy reserved-tail policy."""
    d, p = config["data"], config["model"]["patch_size"]
    policy = d.get("validation_policy", "reserved_tail")
    if policy not in ("reserved_tail", "sampled_training_origins"):
        raise ValueError("Unknown data.validation_policy")
    validation_patches = d["horizon"] // p
    if (
        d["horizon"] % p
        or validation_patches < 1
        or validation_patches > config["model"].get("output_patches", 1)
        or d["context_length"] != config["model"]["context_length"]
    ):
        raise ValueError(
            "Next-patch validation requires whole output patches and matching context limits"
        )
    if d["known_covariates"]:
        raise ValueError(
            "Next-patch objective currently uses observed, time-aligned covariates only"
        )
    throughput_options(config)  # reject a bad setting before the long source scan
    paths = discover_source_tables(config)
    paths, selection = validated_sources(config, paths)
    if d["max_tables"] is not None:
        paths = paths[: d["max_tables"]]
    if not paths:
        raise ValueError("No source tables")
    implementation = "".join(
        file_digest(Path(__file__).with_name(name))
        for name in (
            "transform.py",
            "transform_progress.py",
            "sharded_records.py",
            "table_decoder.py",
            "source_io.py",
        )
    )
    with ShardedTransformCheckpoint(
        location(config, "processed"),
        signature(config),
        paths,
        location(config, "raw"),
        implementation,
        options=config.get("transform", {}),
    ) as checkpoint:
        return _transform_next_patch_tables(config, paths, selection, checkpoint)


def _transform_next_patch_tables(config, paths, selection, checkpoint):
    d, p = config["data"], config["model"]["patch_size"]
    policy = d.get("validation_policy", "reserved_tail")
    full_training = policy == "sampled_training_origins"
    validation_patches = d["horizon"] // p
    max_tokens = d["context_length"] // p
    checkpoint.restore()
    initial = checkpoint.state["completed_tables"]
    progress = TransformProgress("Transform", len(paths), initial)

    def report_progress(completed):
        checkpoint.save_progress(completed)
        progress.update(completed)

    raw = location(config, "raw")
    minimum = max(48, 2 * p if full_training else (validation_patches + 2) * p)
    export_directory = None
    if d["export_processed_tables"]:
        export_directory = location(config, "processed")
        export_directory.mkdir(parents=True, exist_ok=True)
    settings = {
        "data": d,
        "minimum_rows": minimum,
        "export_directory": export_directory and str(export_directory),
    }
    with decoded_tables(
        paths, initial, checkpoint.source_bytes, settings, throughput_options(config)
    ) as tables:
        for index in range(initial, len(paths)):
            table = next(tables, None)
            if table is None:
                raise RuntimeError(f"Source reading stopped before table {index}")
            path = paths[index]
            rows, problem = table["rows"], table.get("problem")
            if problem == "short":
                reason = (
                    f"need at least {minimum} timesteps for causal full-table training"
                    if full_training
                    else f"need at least {minimum} timesteps (two training tokens plus {validation_patches} validation tokens)"
                )
                checkpoint.append_excluded({"source": str(path), "reason": reason})
                logger.warning(
                    "Skipping short table {} rows={}: {}", path, rows, reason
                )
                report_progress(index + 1)
                continue
            if problem == "time":
                raise ValueError(f"Invalid time ordering: {path}")
            if problem == "roles":
                raise ValueError("Explicit roles contain excluded lag columns")
            if problem == "infinity":
                raise ValueError(f"Infinity in {path}")
            excluded, past, columns = column_roles(table["names"], d)
            targets, covariates = table["targets"], table.get("past")
            evaluation_horizon = (
                min(d["horizon"], rows - 2 * p) if full_training else d["horizon"]
            )
            cutoff = rows - evaluation_horizon
            training_stop = rows if full_training else cutoff
            left_pad = (-rows) % p
            train_tokens = (training_stop + left_pad) // p
            source = (
                path.relative_to(ROOT).as_posix()
                if path.is_relative_to(ROOT)
                else str(path)
            )
            metadata = {
                "table_index": index,
                "screening_smoke_test_only": selection.get("smoke_test_only", False),
                "source": source,
                "source_key": path.relative_to(raw).as_posix(),
                "source_sha256": table["sha256"],
                "source_rows": rows,
                "evaluation_cutoff": cutoff,
                "validation_policy": policy,
                "training_stop": training_stop,
                "targets": d["targets"],
                "excluded_columns": excluded,
                "variable_names": columns,
                "roles": {"targets": d["targets"], "past": list(past), "known": []},
                "left_padding_points": left_pad,
                "training_transitions": train_tokens - 1,
            }
            logger.debug(
                "Table={} rows={} variables={} train_transitions={} training_stop={} policy={} evaluation_preview=[{}, {}) excluded={}",
                index,
                rows,
                len(columns),
                train_tokens - 1,
                training_stop,
                policy,
                cutoff,
                rows,
                excluded,
            )
            # A chunk has at most C input points plus one target patch. Adjacent chunks
            # share one patch as context, never duplicate supervised transitions.
            for token_start in range(0, train_tokens - 1, max_tokens):
                start = max(0, token_start * p - left_pad)
                stop = min(training_stop, (token_start + max_tokens + 1) * p - left_pad)
                batch = {
                    "context": torch.tensor(targets[start:stop].T.copy())[None],
                    "horizon": p,
                }
                if past:
                    batch["past_covariates"] = torch.tensor(
                        covariates[start:stop].T.copy()
                    )[None]
                record = {
                    **metadata,
                    "id": f"table_{index:03d}_train_tokens_{token_start}",
                    "split": "train",
                    "batch": batch,
                    "context_rows": [start, stop],
                    "cutoff": stop,
                    "token_start": token_start,
                }
                checkpoint.append_record(record)
            evaluation_record = {
                **metadata,
                "id": f"table_{index:03d}_evaluation_{cutoff}",
                "split": "evaluation",
                "cutoff": cutoff,
                "context_rows": [max(0, cutoff - d["context_length"]), cutoff],
                "label_rows": [cutoff, rows],
            }
            if not full_training:
                if len(set(columns)) != len(columns):
                    raise ValueError(
                        "Column roles must be disjoint and targets nonempty"
                    )
                evaluation_record["batch"] = array_window(
                    targets,
                    covariates,
                    cutoff=cutoff,
                    context_length=min(cutoff, d["context_length"]),
                    horizon=evaluation_horizon,
                )
            checkpoint.append_record(evaluation_record)
            checkpoint.append_table(metadata)
            report_progress(index + 1)
    checkpoint.save_progress(len(paths), force=True)
    if not checkpoint.state["table_count"]:
        raise ValueError(
            "No admitted datasets meet the selected full-table/validation-tail length policy"
        )
    payload = {
        **checkpoint.artifact_index(),
        "objective": "next_patch",
        "source_selection": source_selection(config),
    }
    checkpoint.status("finalizing")
    logger.info(
        "All tables checkpointed. Publishing shard index; no corpus-sized merge."
    )
    atomic_write(
        location(config, "processed") / "windows.pt",
        lambda stream: torch.save(payload, stream),
    )
    summary = {
        "config": config,
        "source_selection": source_selection(config),
        "format": "sharded_windows_v1",
        "table_count": checkpoint.state["table_count"],
        "validation": selection,
        "training_transitions": checkpoint.state["training_transitions"],
        "split_rule": (
            "right-aligned patches; full training-table labels; validation tables reserved by trainer before any gradient updates"
            if full_training
            else f"right-aligned patches; final {validation_patches} full patches held out; all earlier causal next-patch pairs trained"
        ),
    }
    atomic_write(
        location(config, "processed") / "manifest.json",
        lambda stream: checkpoint.write_manifest(
            stream, summary, selection["excluded"]
        ),
    )
    checkpoint.status("complete")
    logger.success(
        "Transform complete: {} tables, {} records in {} shards",
        checkpoint.state["table_count"],
        checkpoint.state["record_count"],
        len(checkpoint.shards),
    )
    from sharded_records import ShardedRecords

    return {
        **payload,
        "records": ShardedRecords(payload, location(config, "processed")),
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default=str(ROOT / "configs/config.json"))
    parser.add_argument(
        "--source-manifest", help="Run-local native source index passed to WSL"
    )
    args = parser.parse_args()
    config = load_config(args.config)
    if args.source_manifest:
        os.environ["BASIN_SOURCE_MANIFEST"] = str(ROOT / args.source_manifest)
    with run_context(config, "transform"):
        if config.get("validation", {}).get("enabled") and config["validation"].get(
            "engine", "chronos_bolt"
        ) in ("chronos2", "chronos_bolt"):
            sys.path.insert(0, str(ROOT / "src/05_model_training"))
            from train import launch_wsl

            extra_args = ()
            if os.name == "nt":
                manifest = (
                    Path(os.environ["BASIN_SOURCE_MANIFEST"])
                    if args.source_manifest
                    else write_source_manifest(config)
                )
                extra_args = (
                    "--source-manifest",
                    manifest.relative_to(ROOT).as_posix(),
                )
            if launch_wsl(__file__, args.config, extra_args=extra_args):
                return
        with timed("TRANSFORM stage"):
            transform(config)


if __name__ == "__main__":
    main()

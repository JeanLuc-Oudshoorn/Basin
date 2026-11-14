"""Keep generated augmentation families in training only, including originals."""

import os
import re
from pathlib import Path
from time import perf_counter

from loguru import logger

ROOT = Path(__file__).resolve().parents[2]
POLICY = "augmentation_families_train_only_v2"
_SUFFIX = re.compile(
    r"(?:_coarse_[2-5]_[0-4]|_sign_inverted|_aggregate_(?:min|max|mean)_[2-5])$"
)


class _Progress:
    def __init__(self, phase, total=None):
        self.phase, self.total = phase, total
        self.started = self.last_report = perf_counter()
        logger.info("Augmentation screening: {} started | total={}", phase, total)

    def update(self, completed, *, force=False):
        if not force and completed % 1000:
            return
        now = perf_counter()
        if not force and now - self.last_report < 30:
            return
        elapsed = now - self.started
        rate = completed / elapsed if elapsed else 0
        eta = (
            (self.total - completed) / rate if self.total is not None and rate else None
        )
        logger.info(
            "Augmentation screening: {} | processed={:,} total={} | elapsed={:.1f}s | ETA={}",
            self.phase,
            completed,
            self.total,
            elapsed,
            f"{max(0, eta):.1f}s" if eta is not None else "unknown",
        )
        self.last_report = now


def _original(name):
    while (match := _SUFFIX.search(name)) is not None:
        name = name[: match.start()]
    return name


def excluded_evaluation_sources(sources, *, audit=None):
    """Scan each raw date directory once; never load table or metadata files.

    Copies are deliberately discovered before admission filtering. Generated
    names also protect families whose provenance file was lost or not yet saved.
    No global cache: a later run must notice newly generated copies.
    """
    if not hasattr(sources, "__len__"):
        sources = tuple(sources)
    # Generated tables have a fixed date/table/data.parquet layout. Resolve the
    # shared date prefix once, not every file (millions of WSL stat/readlink calls).
    parents = {}
    progress = _Progress("index date folders", len(sources))
    for completed, source in enumerate(sources, 1):
        table, name = os.path.split(source)
        if name == "data.parquet":
            parent = os.path.dirname(table)
            if parent not in parents:
                parents[parent] = (ROOT / parent).resolve()
        progress.update(completed)
    progress.update(len(sources), force=True)
    families = {}
    aliases = {}
    for parent in sorted(set(parents.values())):
        if audit is not None and parent.exists():
            value = parent.stat()
            audit[str(parent)] = [value.st_size, value.st_mtime_ns, value.st_ctime_ns]
        roots = set()
        links = {}
        progress = _Progress(f"scan raw directory {parent}")
        completed = 0
        if parent.is_dir():
            with os.scandir(parent) as entries:
                for entry in entries:
                    completed += 1
                    progress.update(completed)
                    if entry.is_symlink():
                        links[entry.name] = Path(entry.path).resolve()
                    original = _original(entry.name)
                    if original == entry.name or not entry.is_dir():
                        continue
                    roots.add(original)
        families[parent] = roots
        aliases[parent] = links
        progress.update(completed, force=True)
    excluded = set()
    progress = _Progress("classify source families", len(sources))
    for completed, source in enumerate(sources, 1):
        table, name = os.path.split(source)
        if name == "data.parquet":
            parent = parents[os.path.dirname(table)]
            table_name = os.path.basename(table)
            if table_name in aliases[parent]:
                # Resolve only actual directory links, found during the scan.
                target = aliases[parent][table_name]
                table_name, parent = target.name, target.parent
                if parent not in families:
                    if audit is not None and parent.exists():
                        value = parent.stat()
                        audit[str(parent)] = [
                            value.st_size,
                            value.st_mtime_ns,
                            value.st_ctime_ns,
                        ]
                    families[parent] = (
                        {
                            _original(entry.name)
                            for entry in parent.iterdir()
                            if _original(entry.name) != entry.name and entry.is_dir()
                        }
                        if parent.is_dir()
                        else set()
                    )
            original = _original(table_name)
            if original != table_name or original in families[parent]:
                excluded.add(source)
        progress.update(completed)
    progress.update(len(sources), force=True)
    logger.info(
        "Evaluation lineage screening: sources={} excluded={} raw_directories={} policy={}",
        len(sources),
        len(excluded),
        len(set(parents.values())),
        POLICY,
    )
    return excluded

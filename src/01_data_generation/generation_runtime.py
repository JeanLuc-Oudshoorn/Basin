"""Bounded generation workers and rate-limited console progress."""

import copy
import math
from time import perf_counter

import numpy as np


def invert_frame(frame):
    """Reflect recorded values around zero; the clock and row order stay fixed."""
    result = frame.copy()
    for name in result:
        if name != "time_index":
            result[name] = -frame[name]
    return result


def coarsen_frame(frame, window_size, offset):
    """One synchronized observation per complete non-overlapping window.

    Every column uses the same fixed phase; this is decimation, not averaging.
    Reindex the recorded clock so downstream code sees the coarser sample rate.
    """
    if (
        isinstance(window_size, bool)
        or not isinstance(window_size, int)
        or window_size not in (2, 3, 4, 5)
    ):
        raise ValueError("window_size must be one of 2, 3, 4, 5")
    if (
        isinstance(offset, bool)
        or not isinstance(offset, int)
        or not 0 <= offset < window_size
    ):
        raise ValueError("offset must be an integer inside the window")
    stop = len(frame) // window_size * window_size
    result = frame.iloc[offset:stop:window_size].reset_index(drop=True).copy()
    result["time_index"] = np.arange(len(result))
    return result


def duration(seconds):
    """Compact elapsed/ETA display, including multi-day generation runs."""
    seconds = max(0, int(seconds))
    days, seconds = divmod(seconds, 86400)
    hours, seconds = divmod(seconds, 3600)
    minutes, seconds = divmod(seconds, 60)
    clock = f"{hours:02d}:{minutes:02d}:{seconds:02d}"
    return f"{days}d {clock}" if days else clock


class GenerationProgress:
    def __init__(self, total, workers, interval=1.0, *, clock=perf_counter):
        self.total, self.workers, self.interval = total, workers, interval
        self.clock = clock
        self.started = clock()
        self.last_report = -math.inf
        self.extra_tables = 0
        self.mirrored_tables = 0

    def record_copies(self, groups):
        for group in groups:
            for path in group[1:]:
                if path.name.endswith("_sign_inverted"):
                    self.mirrored_tables += 1
                else:
                    self.extra_tables += 1

    def update(self, completed, *, force=False):
        now = self.clock()
        if not force and now - self.last_report < self.interval:
            return
        elapsed = max(0.0, now - self.started)
        rate = completed / elapsed if elapsed > 0 else 0.0
        eta = duration((self.total - completed) / rate) if rate else "estimating"
        if completed == self.total:
            eta = "00:00:00"
        print(
            f"[{completed:,}/{self.total:,} {100 * completed / self.total:.1f}%] "
            f"elapsed {duration(elapsed)} | {rate:.2f} tables/s | ETA {eta} | "
            f"workers {self.workers}"
            + (f" | +{self.extra_tables:,} coarse copies" if self.extra_tables else "")
            + (
                f" | +{self.mirrored_tables:,} sign-inverted copies"
                if self.mirrored_tables
                else ""
            ),
            # Keep progress/ETA denominators tied to original SCMs, not extras.
            flush=True,
        )
        self.last_report = now


_worker_generator = None


def initialize_worker(generator):
    """One generator per process; never spawn nested render pools.

    Arrow's own thread pool is unnecessary for tiny individual tables, and can
    otherwise multiply thread counts across all generation processes.
    """
    import pyarrow

    pyarrow.set_cpu_count(1)
    pyarrow.set_io_thread_count(1)
    global _worker_generator
    _worker_generator = generator


def generate_batch(generator, jobs):
    """Write a bounded batch; return only paths, never large frames or models.

    Every table gets an independent seed fixed by the parent. Completion order,
    scheduling, rejection counts in other tables and batch sizes cannot change
    its sampled SCM. Copy RNG ownership so this also works in serial fallback.
    """
    generator = copy.copy(generator)
    paths = []
    for seed, trend_free, directory in jobs:
        generator.rng = np.random.default_rng(seed)
        model, frame = generator._sample_accepted_model(trend_free)
        generator._write_table(model, frame, directory)
        if generator.save_plot:
            generator._plot(frame, directory / "series.png")
        if generator.save_dag:
            generator._plot_dag(model, directory / "dag.png")
        extras = generator._write_extra_copies(model, frame, directory)
        # Keep one result group per requested SCM. Extra copies must not inflate
        # completion counts, overwrite later path slots, or make ETA reach zero early.
        paths.append([directory] + extras)
    return paths


def generate_worker_batch(jobs):
    return generate_batch(_worker_generator, jobs)

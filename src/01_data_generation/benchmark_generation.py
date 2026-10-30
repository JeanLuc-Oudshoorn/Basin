"""Repeatable speed/equality audit. Edit settings, then run directly."""

import importlib
import io
import json
import platform
import statistics
import sys
import tempfile
import time
import types
from contextlib import redirect_stdout
from pathlib import Path

import numpy as np
import pandas as pd
from scm_generator import OperationalSCMGenerator

# Optional directory with an older scm_generator.py and its sibling modules.
# None benchmarks the current simulator and serial vs parallel rendering only.
REFERENCE_SOURCE = None
REPEATS = 3
BATCH_REPEATS = 2
BATCH_TABLES = 8
PARALLEL_WORKERS = 4
OUTPUT = (
    Path(__file__).resolve().parents[2]
    / "results"
    / "data_generation"
    / "generation_benchmark.json"
)


def _reference_class():
    if REFERENCE_SOURCE is None:
        return None
    package = types.ModuleType("_generation_benchmark_reference")
    package.__path__ = [str(Path(REFERENCE_SOURCE).resolve())]
    sys.modules[package.__name__] = package
    return importlib.import_module(
        package.__name__ + ".scm_generator"
    ).OperationalSCMGenerator


def _training_columns(frame):
    return frame.loc[
        :,
        [
            c
            for c in frame
            if c in ("time_index", "outcome") or c.startswith("covariate_")
        ],
    ]


def _verify_tables(left, right):
    for source, candidate in zip(left, right, strict=True):
        pd.testing.assert_frame_equal(
            _training_columns(pd.read_parquet(source / "data.parquet")),
            pd.read_parquet(candidate / "data.parquet"),
            check_exact=True,
        )
        before = json.loads((source / "scm.json").read_text())
        after = json.loads((candidate / "scm.json").read_text())
        # Export-description text changed intentionally. Every sampled parameter,
        # observation seed, quality score and rejection count must still match.
        before["equations"].pop("initialization", None)
        after["equations"].pop("initialization", None)
        if before != after:
            raise AssertionError("Sampling or quality-screening results changed")
        for name in ("series.png", "dag.png"):
            if (source / name).read_bytes() != (candidate / name).read_bytes():
                raise AssertionError(f"Plot output changed: {name}")


def main():
    baseline = _reference_class()
    factory = baseline or OperationalSCMGenerator
    models = [
        factory(
            "unused",
            seed=400 + i,
            min_rows=rows,
            max_rows=rows,
            min_covariates=8,
            max_covariates=8,
            max_lag=512,
            max_lag_count=4,
        ).sample_model(trend_free=bool(i % 2))
        for i, rows in enumerate((48, 256, 1024, 2056, 512, 2056))
    ]
    variants = {"optimized": OperationalSCMGenerator("unused")}
    if baseline:
        variants = {"reference": baseline("unused"), **variants}
    timings = {name: [] for name in variants}
    for repeat in range(REPEATS):
        # Reverse order on alternating repeats to reduce warm-cache bias.
        for name in list(variants)[:: 1 if repeat % 2 == 0 else -1]:
            started = time.perf_counter()
            for model in models:
                variants[name].simulate(model)
            timings[name].append(time.perf_counter() - started)
    if baseline:
        for model in models:
            pd.testing.assert_frame_equal(
                _training_columns(variants["reference"].simulate(model)),
                variants["optimized"].simulate(model),
                check_exact=True,
            )
    print(
        "Simulation medians:",
        {k: round(statistics.median(v), 3) for k, v in timings.items()},
        flush=True,
    )
    batch_times = {"optimized_serial": [], "optimized_parallel": []}
    if baseline:
        batch_times = {"reference_serial": [], **batch_times}
    with tempfile.TemporaryDirectory(prefix="basin_generation_benchmark_") as temp:
        for repeat in range(BATCH_REPEATS):
            paths = {}
            for name in batch_times:
                cls = (
                    baseline if name == "reference_serial" else OperationalSCMGenerator
                )
                options = dict(
                    seed=519,
                    min_rows=512,
                    max_rows=512,
                    min_covariates=6,
                    max_covariates=6,
                    max_lag=512,
                    max_lag_count=4,
                )
                if name != "reference_serial":
                    options["render_workers"] = (
                        PARALLEL_WORKERS if name.endswith("parallel") else 1
                    )
                generator = cls(Path(temp) / f"{repeat}_{name}", **options)
                started = time.perf_counter()
                with redirect_stdout(io.StringIO()):
                    paths[name] = generator.generate(BATCH_TABLES)
                elapsed = time.perf_counter() - started
                batch_times[name].append(elapsed)
                print(
                    f"Batch {repeat + 1}/{BATCH_REPEATS}, {name}: {elapsed:.3f}s",
                    flush=True,
                )
            _verify_tables(paths["optimized_serial"], paths["optimized_parallel"])
            if baseline:
                _verify_tables(paths["reference_serial"], paths["optimized_serial"])
    report = {
        "python": sys.version,
        "platform": platform.platform(),
        "numpy": np.__version__,
        "reference_source": str(REFERENCE_SOURCE) if REFERENCE_SOURCE else None,
        "simulation_rows": [m["rows"] for m in models],
        "simulation_covariates": 8,
        "simulation_repeats": REPEATS,
        "simulation_seconds": timings,
        "simulation_median_seconds": {
            k: statistics.median(v) for k, v in timings.items()
        },
        "batch_tables": BATCH_TABLES,
        "batch_rows": 512,
        "batch_covariates": 6,
        "batch_repeats": BATCH_REPEATS,
        "parallel_workers": PARALLEL_WORKERS,
        "batch_seconds": batch_times,
        "batch_median_seconds": {
            k: statistics.median(v) for k, v in batch_times.items()
        },
        "checks": "Retained values exact; sampled parameters and screening identical; series/DAG PNG bytes identical",
    }
    if baseline:
        report["simulation_speedup"] = statistics.median(
            timings["reference"]
        ) / statistics.median(timings["optimized"])
        report["batch_speedup"] = statistics.median(
            batch_times["reference_serial"]
        ) / statistics.median(batch_times["optimized_parallel"])
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    OUTPUT.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(OUTPUT)


if __name__ == "__main__":
    main()

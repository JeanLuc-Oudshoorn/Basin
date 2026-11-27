"""Add a classical model to a finished holdout run, on the CPU.

A classical model needs neither the GPU nor Basin's weights, so it does not
have to be part of the long run. This reads the points a finished run saved
for every table, fits the model to the same histories at the same origins,
and scores its forecasts on the same points with the same code. The run's own
files are left as they are; everything is written to a folder named after
the model inside the run folder.

    python src/06_holdout_test/add_classical.py <run folder> --model holtwinters

Latency is measured as in the stage: the run's timed tables are forecast
first, one request at a time, in this process, while nothing else runs. The
other tables are then forecast by several processes at once and not timed.
An interrupted run continues where it stopped when started again.
"""

import argparse
import json
import os
import sys
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path
from time import perf_counter

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
for folder in (
    "03_transforming_input_data",
    "04_patch_indexing",
    "05_model_training",
    "06_holdout_test",
):
    sys.path.insert(0, str(ROOT / "src" / folder))

PERCENTILES = {
    "p5": 5,
    "p10": 10,
    "p25": 25,
    "median": 50,
    "p75": 75,
    "p90": 90,
    "p95": 95,
}
TABLES_PER_PART = 250


def forecast_table(path, model, timer=None):
    """Forecast every request of one saved table; its points and its scores."""
    from holdout import classical_request, summarize

    frame = pd.read_parquet(path)
    source = pd.read_parquet(ROOT / frame.source.iloc[0])
    horizon = int(frame.forecast_horizon.iloc[0])
    table_index = int(path.stem.rsplit("_", 1)[1])
    forecasts = np.full(len(frame), np.nan)
    config = {"holdout": {}}
    for origin, rows in frame.groupby("origin", sort=False):
        targets = list(dict.fromkeys(rows.target))
        points = int(rows.context_points.iloc[0])
        history = source.iloc[origin - points : origin]

        def request():
            return classical_request(model, history, targets, [], horizon, config)[0]

        if timer is None:
            values = request()
        else:
            values = timer.measure(
                model,
                request,
                device="cpu",
                request={
                    "table_index": table_index,
                    "source": frame.source.iloc[0],
                    "origin": int(origin),
                    "context_points": points,
                    "targets": len(targets),
                    "horizon": horizon,
                },
            )
        for target, row in zip(targets, values, strict=True):
            place = frame.index.get_indexer(rows.index[rows.target == target])
            if len(place) != horizon:
                raise ValueError(f"{path} does not hold {horizon} points per request")
            forecasts[place] = row
    if not np.isfinite(forecasts).all():
        raise FloatingPointError(f"Nonfinite or missing {model} forecast for {path}")
    frame[model] = forecasts
    scores = None
    if np.isfinite(frame.actual).any():
        summary = summarize(frame)
        scores = {
            "source": frame.source.iloc[0],
            "table_index": table_index,
            "origins": int(frame.origin.nunique()),
            "observations": summary["observations"],
            model: summary[model],
        }
    keep = ["source", "target", "origin", "time_index", model]
    return frame[keep], scores


def forecast_part(paths, model, output, timed=False):
    """One group of tables: forecasts to `output`, scores next to it."""
    from loguru import logger

    # One warning per request with a missing value in its history otherwise.
    logger.disable("holdout")
    timer = None
    if timed:
        from latency import RequestTimer

        timer = RequestTimer({"holdout": {"latency": {"warmup_requests": 1}}})
    frames, scores = [], []
    for path in paths:
        frame, table_scores = forecast_table(Path(path), model, timer)
        frames.append(frame)
        if table_scores is not None:
            scores.append(table_scores)
    result = {"tables": len(paths), "scores": scores}
    if timer is not None:
        result["latency_rows"] = timer.rows
        result["latency"] = timer.summary()[model]
        result["latency_protocol"] = timer.metadata()
    partial = output.with_name(output.name + ".partial")
    pd.concat(frames, ignore_index=True).to_parquet(partial, index=False)
    partial.replace(output)
    # Written last: a part with scores is complete.
    output.with_suffix(".json").write_text(json.dumps(result), encoding="utf-8")
    return output


def summarise(directory, model, run, seconds):
    parts = sorted(directory.glob("part_*.json"))
    results = [json.loads(path.read_text(encoding="utf-8")) for path in parts]
    scores = [row for result in results for row in result["scores"]]
    mase = np.array(
        [np.nan if row[model]["mase"] is None else row[model]["mase"] for row in scores]
    )
    defined = mase[np.isfinite(mase)]
    timed = next(result for result in results if "latency" in result)
    summary = {
        "model": model,
        "holdout_run": run.name,
        "tables": int(sum(result["tables"] for result in results)),
        "scored_tables": len(scores),
        "tables_with_defined_mase": int(len(defined)),
        "table_mase": {
            name: float(np.percentile(defined, level))
            for name, level in PERCENTILES.items()
        },
        "latency": timed["latency"],
        "latency_protocol": dict(
            timed["latency_protocol"],
            scope="resident CPU history to CPU forecast, one process and one "
            "request at a time; includes fitting the model",
            gpu=None,
        ),
        "seconds_this_session": seconds,
    }
    (directory / "table_metrics.json").write_text(json.dumps(scores), encoding="utf-8")
    (directory / "metrics.json").write_text(
        json.dumps(summary, indent=2), encoding="utf-8"
    )
    return summary


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("run", type=Path, help="a finished holdout run folder")
    parser.add_argument("--model", default="holtwinters", choices=["holtwinters"])
    parser.add_argument("--workers", type=int, default=max(1, os.cpu_count() - 2))
    parser.add_argument(
        "--limit", type=int, help="only this many of the untimed tables (a trial)"
    )
    args = parser.parse_args()
    run = args.run.resolve()
    if not (run / "test_metrics.json").is_file():
        raise SystemExit(f"{run} is not a finished holdout run")
    directory = run / (args.model if args.limit is None else f"{args.model}_trial")
    directory.mkdir(exist_ok=True)
    timed_indices = {
        json.loads(line)["table_index"]
        for line in (run / "request_latencies.jsonl").read_text("utf-8").splitlines()
        if line.strip()
    }
    tables = {
        int(path.stem.rsplit("_", 1)[1]): path
        for path in run.glob("test_table_*.parquet")
    }
    timed = [tables[i] for i in sorted(timed_indices) if i in tables]
    others = [tables[i] for i in sorted(set(tables) - timed_indices)]
    if args.limit is not None:
        others = others[:: max(1, len(others) // args.limit)][: args.limit]
    print(f"{len(timed)} timed tables, {len(others)} others, output {directory}")
    started = perf_counter()
    first = directory / "part_timed.parquet"
    if not first.with_suffix(".json").is_file():
        forecast_part(timed, args.model, first, timed=True)
        print(f"timed tables done in {perf_counter() - started:.0f} s", flush=True)
    jobs = {}
    for number, start in enumerate(range(0, len(others), TABLES_PER_PART)):
        output = directory / f"part_{number:04d}.parquet"
        if not output.with_suffix(".json").is_file():
            jobs[output] = others[start : start + TABLES_PER_PART]
    # One thread per process: the processes are the parallelism.
    for name in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS"):
        os.environ[name] = "1"
    parallel_started, done = perf_counter(), 0
    with ProcessPoolExecutor(args.workers) as pool:
        futures = [
            pool.submit(forecast_part, [str(p) for p in paths], args.model, output)
            for output, paths in jobs.items()
        ]
        for future in as_completed(futures):
            future.result()
            done += 1
            elapsed = perf_counter() - parallel_started
            print(
                f"{done}/{len(jobs)} parts | elapsed {elapsed / 60:.1f} min | "
                f"remaining about {elapsed / done * (len(jobs) - done) / 60:.1f} min",
                flush=True,
            )
    summary = summarise(directory, args.model, run, perf_counter() - started)
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()

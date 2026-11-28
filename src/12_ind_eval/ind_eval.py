"""Stage 12: IND-Eval, forecasting on public industrial time series.

Runs Basin, the naive forecast, Holt-Winters and the holdout stage's
foundation models on the evaluation windows of IND-Eval
(cognitedata/IND-Eval, arXiv 2610.16652) and scores them as the benchmark
does: MASE and CRPS against the seasonal-naive forecast, per task, per source
and overall. Every channel of a window is forecast on its own, from its own
history; every model gets at most 2,048 points of it and forecasts at most
112 points. Nothing here touches training data or the holdout.
"""
# ruff: noqa: E402

import argparse
import json
import sys
from pathlib import Path
from time import perf_counter

import numpy as np
import pandas as pd
import torch
from loguru import logger

ROOT = Path(__file__).resolve().parents[2]
HOLDOUT = ROOT / "src" / "06_holdout_test"
for folder in (ROOT / "src" / "09_fakebench", Path(__file__).resolve().parent):
    sys.path.insert(0, str(folder))
import fakebench
from holt_winters import HoltWintersPool
from ind_eval_data import (
    RECORD,
    data_directory,
    duplicate_index,
    list_tasks,
    model_context,
    read_task,
    settings,
)
from ind_eval_scoring import (
    LEVELS,
    REFERENCE,
    leaderboard,
    read_scores,
    score_series,
    summarise,
)
from performance import configure_cuda_memory
from train import launch_wsl
from transform import load_config, run_context, timed, write_json

COMPLETED = "results.json"  # written last, so it marks a scored run
# Models that need no GPU; a run of only these stays where it was started.
CPU_MODELS = ("naive", REFERENCE, "holtwinters")
INPUTS = "the target channel's own history; other channels are not used"


def results_root(config):
    return ROOT / config["paths"].get("ind_eval", "results/model/ind_eval")


def as_quantiles(points):
    """A point forecast [series, horizon] as every quantile, [series, 9, horizon]."""
    points = np.asarray(points, dtype=np.float32)
    return np.repeat(points[:, None, :], len(LEVELS), axis=1)


def naive(contexts, horizon, seasonalities):
    return as_quantiles([np.full(horizon, c[-1]) for c in contexts])


def seasonal_naive(contexts, horizon, seasonalities):
    """Repeat the last cycle; with a cycle of 1 this is the naive forecast."""
    return as_quantiles(
        [
            np.resize(c[-min(m, len(c)) :], horizon)
            for c, m in zip(contexts, seasonalities)
        ]
    )


class HoltWinters:
    """Holt-Winters fitted to every context, by several processes at once."""

    def __init__(self, workers):
        self.pool = HoltWintersPool(workers)

    def __call__(self, contexts, horizon, seasonalities):
        return as_quantiles(self.pool(contexts, horizon)[0])

    def close(self):
        self.pool.close()


def build(name, config, options):
    """A function (contexts, horizon, seasonalities) -> quantiles [n, 9, horizon]."""
    if name == "naive":
        return naive
    if name == REFERENCE:
        return seasonal_naive
    if name == "holtwinters":
        return HoltWinters(options["holt_winters_workers"])
    batch_size = options["batch_size"]
    forecaster = (
        fakebench.basin(config, batch_size=batch_size, quantiles=True)
        if name == "basin"
        else fakebench.build(name, config, batch_size=batch_size)
    )

    def forecast(contexts, horizon, seasonalities):
        return forecaster(contexts, horizon)

    forecast.details = getattr(forecaster, "details", {})
    return forecast


def score_path(output, model, task):
    return output / "scores" / model / f"{task.key}.parquet"


def forecast_task(forecaster, model, task, options, output, skip=None):
    """Forecast and score one task; its per-series scores are written whole."""
    points = options["context_points"]
    batch_size = options["batch_size"]
    if model == "holtwinters":
        batch_size = max(batch_size, 8 * options["holt_winters_workers"])
    rows, skipped, seconds = [], 0, 0.0
    for group in read_task(task, options, skip=skip):
        layouts = {}
        for series in group:
            context = model_context(series.context, points)
            if context is None:
                skipped += 1  # no observed history: nothing to forecast from
                continue
            layouts.setdefault((len(context), series.horizon), []).append(
                (series, context)
            )
        for (_, horizon), members in layouts.items():
            for start in range(0, len(members), batch_size):
                chunk = members[start : start + batch_size]
                started = perf_counter()
                quantiles = forecaster(
                    [context for _, context in chunk],
                    horizon,
                    [series.seasonality for series, _ in chunk],
                )
                seconds += perf_counter() - started
                quantiles = np.asarray(quantiles)
                if quantiles.shape != (len(chunk), len(LEVELS), horizon):
                    raise ValueError(
                        f"{model} returned shape {quantiles.shape} for "
                        f"{len(chunk)} series and {horizon} steps"
                    )
                if not np.isfinite(quantiles).all():
                    raise FloatingPointError(f"{model}: a forecast is not finite")
                rows += [
                    score_series(series, quantiles[i], len(context))
                    for i, (series, context) in enumerate(chunk)
                ]
    path = score_path(output, model, task)
    path.parent.mkdir(parents=True, exist_ok=True)
    partial = path.with_name(path.name + ".partial")
    pd.DataFrame(rows).to_parquet(partial, index=False)
    partial.replace(path)
    # Written last: a task with this file is complete.
    record = {"series": len(rows), "skipped_series": skipped, "seconds": seconds}
    path.with_suffix(".json").write_text(json.dumps(record), encoding="utf-8")
    return record


def model_speed(output, model):
    records = [
        json.loads(path.read_text(encoding="utf-8"))
        for path in (output / "scores" / model).glob("*.json")
    ]
    series = sum(r["series"] for r in records)
    seconds = sum(r["seconds"] for r in records)
    return {
        "series": series,
        "skipped_series": sum(r["skipped_series"] for r in records),
        "forecast_seconds": seconds,
        "series_per_second": series / seconds if seconds > 0 else None,
    }


def score_run(output, record):
    """Summarise every finished task in `output` and write the result files."""
    scores = read_scores(output)
    speed = {model: model_speed(output, model) for model in scores}
    results = summarise(scores, speed)
    facts = {
        "Basin": record.get("basin") or "not run",
        "Data": record["data"],
        "Inputs": INPUTS,
        "History given": f"the latest {record['context_points']:,} points at most",
        "Forecast": f"the first {record['horizon']} points of each target at most",
        "Repeated series left out": f"{record['repeated_series_left_out']:,}, and "
        f"{len(record['tasks_left_out_as_repeats'])} tasks that only repeat others",
    }
    (output / "leaderboard.md").write_text(leaderboard(results, facts), "utf-8")
    write_json(output / COMPLETED, {"run": record, **results})
    return results


def run(config, args):
    options = settings(config)
    if args.sources:
        options["sources"] = args.sources
    trial = args.limit is not None
    if trial:
        options["max_series_per_task"] = args.limit
    tasks = list_tasks(options)
    # Series that repeat one of an earlier task of their source are left out,
    # and with them a task that has no other series.
    repeats = duplicate_index(tasks, options, log=logger.info)
    skip = {key: set(entry["duplicates"]) for key, entry in repeats.items()}
    emptied = [t.key for t in tasks if repeats[t.key]["series"] == len(skip[t.key])]
    tasks = [task for task in tasks if task.key not in emptied]
    download = data_directory(options) / RECORD
    data = json.loads(download.read_text("utf-8")) if download.is_file() else {}
    # Basin first: its checkpoint names the run folder. The reference always.
    names = list(dict.fromkeys([*(args.models or options["models"]), REFERENCE]))
    names.sort(key=lambda name: name != "basin")
    cuda = torch.cuda.is_available()
    if cuda:
        torch.set_num_threads(config["training"].get("cpu_threads", 4))
        configure_cuda_memory(config["training"])
    record = {
        "data": f"{data.get('repository', 'IND-Eval')} at {data.get('revision', '?')}",
        "max_series_per_task": options["max_series_per_task"],
        "context_points": options["context_points"],
        "horizon": options["horizon"],
        "sources": options["sources"],
        "repeated_series_left_out": sum(len(s) for s in skip.values()),
        "tasks_left_out_as_repeats": emptied,
        "models": {},
    }
    loaded = {}
    if args.resume:
        output = results_root(config) / args.resume
        earlier = json.loads((output / "run.json").read_text(encoding="utf-8"))
        for key in ("data", "max_series_per_task", "context_points", "horizon"):
            if earlier[key] != record[key]:
                raise ValueError(
                    f"{output} was made with {key}={earlier[key]!r}, not "
                    f"{record[key]!r}; it cannot be continued with other settings"
                )
        record = {**earlier, "sources": record["sources"]}
    else:
        if "basin" in names:
            with timed("load basin", cuda=cuda):
                loaded["basin"] = build("basin", config, options)
            record["basin"] = loaded["basin"].details
        output = fakebench.new_run(
            config,
            epoch=record.get("basin", {}).get("epoch"),
            step=record.get("basin", {}).get("step"),
            root=results_root(config) / "trials" if trial else results_root(config),
        )
    logger.info("IND-EVAL {} tasks, models {}; output={}", len(tasks), names, output)
    for name in names:
        todo = [
            task
            for task in tasks
            if not score_path(output, name, task).with_suffix(".json").is_file()
        ]
        record["models"][name] = "point forecast" if name in CPU_MODELS else "quantiles"
        write_json(output / "run.json", record)
        if not todo:
            logger.info("{}: all {} tasks are already scored", name, len(tasks))
            continue
        if name not in loaded:
            with timed(f"load {name}", cuda=cuda and name not in CPU_MODELS):
                loaded[name] = build(name, config, options)
        forecaster = loaded.pop(name)
        if name == "basin":
            record["basin"] = forecaster.details
        started = perf_counter()
        for number, task in enumerate(todo, 1):
            done = forecast_task(
                forecaster, name, task, options, output, skip[task.key]
            )
            elapsed = perf_counter() - started
            logger.info(
                "{} {}/{} {}: {} series in {:.0f} s | elapsed {:.1f} min, "
                "remaining tasks {}",
                name,
                number,
                len(todo),
                task.key,
                done["series"],
                done["seconds"],
                elapsed / 60,
                len(todo) - number,
            )
        if hasattr(forecaster, "close"):
            forecaster.close()
        del forecaster
        if cuda:
            torch.cuda.empty_cache()
    write_json(output / "run.json", record)
    results = score_run(output, record)
    for model, row in results["overall"].items():
        logger.success(
            "IND-EVAL {} nMASE {} nCRPS {}", model, row["nmase"], row["ncrps"]
        )
    logger.success("IND-Eval complete: {}", output / "leaderboard.md")
    return output


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default=str(ROOT / "configs/config.json"))
    parser.add_argument(
        "--models",
        nargs="+",
        help="default: ind_eval.models; the seasonal-naive reference is always added",
    )
    parser.add_argument("--sources", nargs="+", help="default: ind_eval.sources")
    parser.add_argument(
        "--limit",
        type=int,
        help="an even subset of this many series per task, written under "
        "trials/ (a quick trial)",
    )
    parser.add_argument(
        "--resume",
        metavar="RUN",
        help="continue this run folder: finish its tasks, or add models to it",
    )
    args = parser.parse_args()
    config = load_config(args.config)
    names = args.models or settings(config)["models"]
    extra = []
    for flag, values in (("--models", args.models), ("--sources", args.sources)):
        extra += [flag, *values] if values else []
    extra += ["--limit", str(args.limit)] if args.limit is not None else []
    extra += ["--resume", args.resume] if args.resume else []
    with run_context(config, "ind_eval"):
        if set(names) <= set(CPU_MODELS):
            with timed("IND-EVAL stage"):
                run(config, args)
            return
        if launch_wsl(__file__, args.config, extra_args=extra):
            return
        if not torch.cuda.is_available() or not torch.cuda.is_bf16_supported():
            raise RuntimeError("Use the configured WSL CUDA bf16 runtime")
        with timed("IND-EVAL stage", cuda=True):
            run(config, args)


if __name__ == "__main__":
    main()

"""IND-Eval scores: MASE and CRPS per task, against the seasonal-naive forecast.

Follows the benchmark's paper (arXiv 2610.16652) as far as it states them:

- MASE of a series: the mean absolute error of the point forecast (the 50%
  quantile) divided by the mean absolute seasonal difference of the series'
  context. A task's MASE is the mean over its series.
- CRPS of a task: the quantile loss at the levels 0.1 to 0.9, summed over all
  series and steps of the task and divided by the sum of the absolute target
  values (the weighted quantile loss, the usual approximation of the CRPS).
- nMASE and nCRPS: a model's MASE and CRPS of a task divided by those of the
  seasonal-naive forecast on the same series. Below 1 is better than seasonal
  naive.
- A source's score is the geometric mean over its tasks, the overall score
  the geometric mean over the sources, so every source counts the same.

The paper's own code was not available when this was written, so details may
differ from the published leaderboard; compare models within one run.
"""

import numpy as np
import pandas as pd

LEVELS = np.round(np.arange(1, 10) / 10, 1)
MEDIAN = 4
REFERENCE = "seasonal_naive"
# Per-series columns every model's score file has.
SCORE_COLUMNS = (
    "series",
    "horizon",
    "context_points",
    "scored_points",
    "absolute_error",
    "scale",
    "quantile_loss",
    "absolute_target",
)


def seasonal_scale(context, seasonality):
    """Mean absolute change over one cycle within the context; NaN if undefined."""
    context = np.asarray(context, dtype=float)
    if len(context) <= seasonality:
        return np.nan
    change = np.abs(context[seasonality:] - context[:-seasonality])
    change = change[np.isfinite(change)]
    scale = change.mean() if len(change) else np.nan
    return float(scale) if scale > 0 else np.nan


def score_series(series, quantiles, context_points):
    """One row of sums for one series; `quantiles` is [9 levels, horizon]."""
    target = series.target.astype(float)
    observed = np.isfinite(target)
    quantiles = np.asarray(quantiles, dtype=float)[:, observed]
    target = target[observed]
    error = target - quantiles
    pinball = np.maximum(LEVELS[:, None] * error, (LEVELS[:, None] - 1) * error)
    return {
        "series": series.key,
        "horizon": series.horizon,
        "context_points": context_points,
        "scored_points": int(observed.sum()),
        "absolute_error": float(np.abs(error[MEDIAN]).sum()),
        "scale": seasonal_scale(series.context, series.seasonality),
        # Twice the pinball loss, averaged over the levels, summed over steps.
        "quantile_loss": float(2 * pinball.mean(axis=0).sum()),
        "absolute_target": float(np.abs(target).sum()),
    }


def task_scores(frame):
    """A task's MASE and CRPS from its per-series rows."""
    scored = frame[frame.scored_points > 0]
    mase = (scored.absolute_error / scored.scored_points / scored.scale).dropna()
    weight = scored.absolute_target.sum()
    return {
        "series": int(len(scored)),
        "series_without_scale": int(len(scored) - len(mase)),
        "mase": float(mase.mean()) if len(mase) else None,
        "crps": float(scored.quantile_loss.sum() / weight) if weight > 0 else None,
    }


def geometric_mean(values):
    values = [v for v in values if v is not None and np.isfinite(v) and v > 0]
    return float(np.exp(np.mean(np.log(values)))) if values else None


def summarise(scores, speed=None):
    """Scores by task, source and overall.

    `scores` maps model -> {(source, task): per-series frame}. Every model
    is compared on the series that all models of a task have scores for, and
    on the tasks every model has.
    """
    models = list(scores)
    if REFERENCE not in models:
        raise ValueError(f"The {REFERENCE} reference has no scores")
    shared = set.intersection(*(set(scores[m]) for m in models))
    tasks, left_out = {}, []
    for source, task in sorted(shared):
        frames = {m: scores[m][(source, task)].set_index("series") for m in models}
        common = frames[REFERENCE].index
        for frame in frames.values():
            common = common.intersection(frame.index)
        rows = {m: task_scores(frames[m].loc[common]) for m in models}
        reference = rows[REFERENCE]
        if not reference["mase"] or not reference["crps"]:
            left_out.append(f"{source}/{task}")
            continue
        for row in rows.values():
            for metric in ("mase", "crps"):
                row[f"n{metric}"] = (
                    None if row[metric] is None else row[metric] / reference[metric]
                )
        tasks[f"{source}/{task}"] = {"source": source, "models": rows}
    sources = sorted({task["source"] for task in tasks.values()})
    by_source = {
        source: {
            m: {
                metric: geometric_mean(
                    task["models"][m][metric]
                    for task in tasks.values()
                    if task["source"] == source
                )
                for metric in ("nmase", "ncrps")
            }
            for m in models
        }
        for source in sources
    }
    overall = {
        m: {
            metric: geometric_mean(by_source[s][m][metric] for s in sources)
            for metric in ("nmase", "ncrps")
        }
        for m in models
    }
    return {
        "models": models,
        "overall": overall,
        "by_source": by_source,
        "tasks": tasks,
        "tasks_per_source": {
            s: sum(task["source"] == s for task in tasks.values()) for s in sources
        },
        "series": int(
            sum(task["models"][REFERENCE]["series"] for task in tasks.values())
        ),
        "tasks_left_out": left_out,
        "tasks_not_run_by_every_model": sorted(
            f"{s}/{t}" for m in models for s, t in set(scores[m]) - shared
        ),
        "speed": speed or {},
    }


def _number(value):
    return "-" if value is None else f"{value:.3f}"


NAMES = {
    "basin": "Basin",
    "naive": "Naive (last value)",
    "seasonal_naive": "Seasonal naive",
    "holtwinters": "Holt-Winters",
    "chronos2": "Chronos-2",
    "timestral": "Timestral",
    "timeh2o": "TimeH2O",
}
POINT_ONLY = ("naive", "seasonal_naive", "holtwinters")


def leaderboard(results, facts=None):
    """Markdown: the overall table, then nMASE and nCRPS by source."""
    models = sorted(
        results["models"], key=lambda m: results["overall"][m]["nmase"] or np.inf
    )
    sources = list(results["by_source"])
    lines = [
        "# IND-Eval results",
        "",
        f"{results['series']:,} series in {len(results['tasks'])} tasks from "
        f"{len(sources)} sources. nMASE and nCRPS are a model's error divided by "
        "that of the seasonal-naive forecast on the same series: below 1 is "
        "better than seasonal naive. Tasks are combined by geometric mean "
        "within a source, and the sources by geometric mean, so every source "
        "counts the same.",
        "",
    ]
    lines += [f"- **{name}:** {value}" for name, value in (facts or {}).items()]
    lines += [
        "",
        "## All sources",
        "",
        "| Model | nMASE | nCRPS | Series per second |",
        "|---|---|---|---|",
    ]
    for model in models:
        row = results["overall"][model]
        speed = results["speed"].get(model, {}).get("series_per_second")
        lines.append(
            f"| {NAMES.get(model, model)} | {_number(row['nmase'])} | "
            f"{_number(row['ncrps'])} | {'-' if speed is None else f'{speed:,.1f}'} |"
        )
    lines += [
        "",
        "Naive, seasonal naive and Holt-Winters give a point forecast only. "
        "Their nCRPS scores that point as if it were every quantile, which "
        "favours the models that give quantiles; compare them on nMASE.",
    ]
    for metric, title in (("nmase", "nMASE"), ("ncrps", "nCRPS")):
        lines += [
            "",
            f"## {title} by source",
            "",
            "| Model | " + " | ".join(sources) + " |",
            "|---|" + "---|" * len(sources),
        ]
        for model in models:
            cells = [_number(results["by_source"][s][model][metric]) for s in sources]
            lines.append(f"| {NAMES.get(model, model)} | " + " | ".join(cells) + " |")
    lines += [
        "",
        "Tasks per source: "
        + ", ".join(f"{s} {n}" for s, n in results["tasks_per_source"].items())
        + ".",
    ]
    for key, text in (
        ("tasks_left_out", "Left out, because seasonal naive has no defined score"),
        ("tasks_not_run_by_every_model", "Left out, because not every model ran them"),
    ):
        if results[key]:
            lines += ["", f"{text}: " + ", ".join(results[key]) + "."]
    return "\n".join(lines) + "\n"


def read_scores(directory):
    """{model: {(source, task): frame}} from a run folder's score files."""
    scores = {}
    for folder in sorted(p for p in (directory / "scores").iterdir() if p.is_dir()):
        for path in sorted(folder.glob("*.parquet")):
            if not path.with_suffix(".json").is_file():
                continue  # still being written, or cut off
            source, _, task = path.stem.partition("__")
            scores.setdefault(folder.name, {})[(source, task)] = pd.read_parquet(path)
    return scores

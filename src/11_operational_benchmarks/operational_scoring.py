"""Score quantile forecasts of the prepared cases, for both tasks.

Every model gives nine quantiles (10% to 90%) for each of the 112 steps. A
point is outside the interval of its step when it lies below the lower or
above the upper of two of them (`anomaly.quantiles`, 10% and 90%). No
distribution is assumed for that. With `anomaly.interval` set to "normal" the
interval is instead the median plus and minus `interval_sigmas` standard
deviations of the normal distribution that has those two quantiles.

Forecast task: the median is the point forecast. MASE is the mean absolute
error divided by the mean absolute step of the context (the error of repeating
the last value one step ahead), over the first 32, 64 and all steps.

Anomaly task: a window is flagged when its judged steps hold a point outside
the interval (or, with `anomaly.min_share_outside`, more than that share of
them). A model with narrow intervals flags normal data too, so every anomaly
case is compared with the forecast cases of the same series and context length
(its controls): the false alarm rate is how often the same steps of a control
are flagged. Precision, recall and F1 count windows, with one normal window
for every anomaly window.
"""

import math

import numpy as np
import pandas as pd
from scipy.special import ndtri


def scored_steps(horizon):
    """The forecast lengths MASE is reported for: 32, 64 and the whole horizon."""
    return sorted({steps for steps in (32, 64) if steps < horizon} | {horizon})


def intervals(quantiles, levels, anomaly):
    """(median, lower, upper), each [cases, steps], from quantiles [cases, levels, steps].

    `anomaly` is the `anomaly` part of the settings.
    """
    low, high = anomaly["quantiles"]
    levels = [round(float(level), 6) for level in levels]
    try:
        positions = [levels.index(round(level, 6)) for level in (low, 0.5, high)]
    except ValueError:
        raise ValueError(
            f"The forecasts have quantiles {levels}; the interval needs {low}, 0.5 "
            f"and {high}"
        ) from None
    lower, median, upper = (quantiles[:, p].astype(np.float64) for p in positions)
    if anomaly["interval"] == "normal":
        sigma = (upper - lower) / float(ndtri(high) - ndtri(low))
        reach = float(anomaly["interval_sigmas"]) * np.maximum(sigma, 0.0)
        lower, upper = median - reach, median + reach
    return median, lower, upper


def surprise(actual, median, lower, upper):
    """How far a point is from the median, in units of the distance to the bound.

    Above 1 is outside the interval. Infinite off an interval without width.
    """
    distance = np.abs(actual - median)
    width = np.where(actual >= median, upper - median, median - lower)
    with np.errstate(divide="ignore", invalid="ignore"):
        scaled = distance / width
    return np.where(width > 0, scaled, np.where(distance > 0, np.inf, 0.0))


def flagged(scores, share):
    """Whether windows of scores [..., steps] count as flagged."""
    outside = (scores > 1).sum(axis=-1)
    return outside > share * scores.shape[-1]


def score_model(cases, quantiles, levels, options):
    """One row per case for one model's forecasts, in the order of `cases`."""
    horizon = options["horizon"]
    share = float(options["anomaly"]["min_share_outside"])
    median, lower, upper = intervals(quantiles, levels, options["anomaly"])
    rows, control_scores = [], {}
    values = [np.asarray(v, dtype=np.float64) for v in cases["values"]]
    scores = []
    for position, case in enumerate(cases.itertuples(index=False)):
        context = values[position][: case.context_length]
        actual = values[position][case.context_length :]
        score = surprise(actual, median[position], lower[position], upper[position])
        scores.append(score)
        scale = np.mean(np.abs(np.diff(context)))
        error = np.abs(actual - median[position])
        row = {
            "case": case.case,
            "outside": float(np.mean(score > 1)),
            "mae": float(error.mean()),
        }
        for steps in scored_steps(horizon):
            row[f"mase_{steps}"] = (
                float(error[:steps].mean() / scale) if scale > 0 else math.nan
            )
        rows.append(row)
        if case.task == "forecast":
            key = (case.dataset, case.series, case.context_length)
            control_scores.setdefault(key, []).append(score)
    controls = {key: np.stack(rows_) for key, rows_ in control_scores.items()}
    for row, score, case in zip(rows, scores, cases.itertuples(index=False)):
        if case.task != "anomaly":
            continue
        span = slice(case.span_start, case.span_start + case.span_length)
        peak = float(score[span].max())
        row["detected"] = float(flagged(score[span], share))
        row["peak"] = peak
        row["flagged_before"] = (
            float((score[: case.span_start] > 1).any()) if case.span_start else math.nan
        )
        matched = controls.get((case.dataset, case.series, case.context_length))
        if matched is None:
            row["false_alarm"] = row["separation"] = math.nan
            continue
        control_peaks = matched[:, span].max(axis=1)
        row["false_alarm"] = float(np.mean(flagged(matched[:, span], share)))
        # How often this anomaly surprises the model more than a control does.
        row["separation"] = float(
            np.mean(peak > control_peaks) + 0.5 * np.mean(peak == control_peaks)
        )
    return pd.DataFrame(rows)


def _number(value):
    return None if value is None or not math.isfinite(value) else float(value)


def _order(value):
    """Numbers in numeric order (also when they are text), other text after them."""
    text = str(value)
    return (0, float(text), "") if text.lstrip("-").isdigit() else (1, 0.0, text)


def forecast_summary(frame):
    summary = {"cases": int(len(frame))}
    columns = sorted(
        (c for c in frame if c.startswith("mase_")), key=lambda c: int(c[5:])
    )
    for column in columns:
        finite = frame[column][np.isfinite(frame[column])]
        summary[f"{column}_median"] = _number(finite.median())
        if column == columns[-1]:
            summary["mase_undefined_cases"] = int(len(frame) - len(finite))
            positive = finite[finite > 0]
            summary[f"{column}_geometric_mean"] = _number(
                math.exp(np.log(positive).mean()) if len(positive) else None
            )
    if "beats_naive" in frame:
        summary["beats_naive"] = _number(frame["beats_naive"].mean())
    summary["inside_interval"] = _number(1 - frame["outside"].mean())
    return summary


def anomaly_summary(frame):
    """Precision, recall and F1 over windows, one normal window per anomaly window.

    Each anomaly case is set against one normal window of the same series and
    context length, judged on the same steps. Which normal window would be
    chance, so its outcome is the average over all of them (`false_alarm`).
    """
    judged = frame[np.isfinite(frame["false_alarm"])]
    hits = judged["detected"].sum()  # anomaly windows flagged
    alarms = judged["false_alarm"].sum()  # normal windows flagged
    misses = len(judged) - hits
    return {
        "cases": int(len(frame)),
        "anomalies": int(frame["anomaly_key"].nunique()),
        "f1": _number(2 * hits / (2 * hits + alarms + misses) if len(judged) else None),
        "precision": _number(hits / (hits + alarms) if hits + alarms > 0 else None),
        "recall": _number(hits / len(judged) if len(judged) else None),
        "false_alarm": _number(alarms / len(judged) if len(judged) else None),
        "separation": _number(frame["separation"].mean()),
        "flagged_before_anomaly": _number(frame["flagged_before"].mean()),
        "median_points_judged": _number(frame["span_length"].median()),
    }


BREAKDOWNS = {
    "forecast": ("context_length", "category", "series"),
    "anomaly": (
        "context_length",
        "lead",
        "role",
        "category",
        "kind",
        "earlier_anomaly",
    ),
}


def summarise(cases, scored, options):
    """All numbers of a run: task -> dataset -> model -> overall and breakdowns.

    `scored` maps a model name to the table `score_model` returned for it.
    """
    details = cases.drop(columns=["values", "covariate_values"], errors="ignore")
    details = details.assign(
        anomaly_key=details["dataset"] + "/" + details["anomaly"],
    )
    reference = scored.get("naive")
    column = f"mase_{options['horizon']}"
    results = {"forecast": {}, "anomaly": {}}
    for model, table in scored.items():
        frame = details.merge(table, on="case", validate="one_to_one")
        if reference is not None and model != "naive":
            other = reference.set_index("case")[column].reindex(frame["case"]).values
            frame["beats_naive"] = np.where(
                np.isfinite(frame[column]) & np.isfinite(other),
                (frame[column] < other).astype(float),
                np.nan,
            )
        for task, summary in (
            ("forecast", forecast_summary),
            ("anomaly", anomaly_summary),
        ):
            for dataset, part in frame[frame["task"] == task].groupby("dataset"):
                entry = {"overall": summary(part)}
                for by in BREAKDOWNS[task]:
                    if by in part and part[by].nunique() > 1:
                        groups = sorted(part.groupby(by), key=lambda g: _order(g[0]))
                        entry[f"by_{by}"] = {
                            str(value): summary(group) for value, group in groups
                        }
                results[task].setdefault(dataset, {})[model] = entry
    return results


def _cell(value, style):
    if value is None:
        return "n/a"
    if style == "count":
        return f"{int(value):,}"
    return f"{value:.1%}" if style == "share" else f"{value:.3f}"


def forecast_columns(horizon):
    return (
        ("cases", "Cases", "count"),
        *(
            (f"mase_{steps}_median", f"MASE, {steps} steps", "number")
            for steps in scored_steps(horizon)
        ),
        (
            f"mase_{horizon}_geometric_mean",
            f"MASE, {horizon} steps (geometric mean)",
            "number",
        ),
        ("beats_naive", "Better than Naive", "share"),
        ("inside_interval", "Points inside the interval", "share"),
    )


ANOMALY_COLUMNS = (
    ("cases", "Cases", "count"),
    ("f1", "F1", "number"),
    ("precision", "Precision", "share"),
    ("recall", "Recall", "share"),
    ("false_alarm", "False alarms", "share"),
    ("separation", "Separation", "number"),
    ("flagged_before_anomaly", "Flagged before the anomaly", "share"),
)
TITLES = {
    "cats": "CATS",
    "ucr": "UCR (operational series)",
    "cloud": "Microsoft cloud monitoring",
    "caster": "Micromill caster",
}
NAMES = {
    "earlier_anomaly": "earlier anomaly in the context",
    "kind": "anomaly type",
    "role": "role of the channel",
    "lead": "lead points",
}


def _table(rows, columns, first):
    """`rows` maps a row label to a summary."""
    used = [c for c in columns if any(c[0] in summary for summary in rows.values())]
    lines = [
        "| " + " | ".join([first, *(title for _, title, _ in used)]) + " |",
        "|" + "---|" * (len(used) + 1),
    ]
    for label, summary in rows.items():
        cells = [_cell(summary.get(key), style) for key, _, style in used]
        lines.append("| " + " | ".join([str(label), *cells]) + " |")
    return "\n".join(lines)


def leaderboard(results, options, record):
    """The results as a Markdown page: the grid first, the breakdowns after it."""
    anomaly = options["anomaly"]
    low, high = anomaly["quantiles"]
    if anomaly["interval"] == "normal":
        sigmas = anomaly["interval_sigmas"]
        nominal = math.erf(sigmas / math.sqrt(2))
        interval = (
            f"the median plus and minus {sigmas:g} standard deviations of the "
            f"normal distribution that has the model's {low:.0%} and {high:.0%} "
            "quantiles"
        )
    else:
        nominal = high - low
        interval = f"from the model's {low:.0%} quantile to its {high:.0%} quantile"
    share = anomaly["min_share_outside"]
    rule = (
        f"more than {share:.0%} of its judged points lie"
        if share
        else "a judged point of it lies"
    )
    text = [
        "# Operational benchmarks",
        "",
        f"Contexts of {', '.join(map(str, options['context_lengths']))} points; "
        f"every forecast is {options['horizon']} steps. The interval is {interval}. "
        f"A calibrated model keeps {nominal:.1%} of normal points inside it. A "
        f"window is flagged when {rule} outside the interval.",
        "",
    ]
    for key, value in (record or {}).items():
        text.append(f"- {key}: {value}")
    for task, title, columns, note in (
        (
            "forecast",
            "Forecasting normal data",
            forecast_columns(options["horizon"]),
            "Median MASE over the cases; lower is better, 1 is the error of "
            "repeating the last value one step ahead.",
        ),
        (
            "anomaly",
            "Detecting anomalies",
            ANOMALY_COLUMNS,
            "Recall: the share of anomalies that are flagged. False alarms: the "
            "share of normal windows of the same series that are flagged on the "
            "same steps. Precision: of the flagged windows, the share that hold "
            "an anomaly, with one normal window for every anomaly. F1 combines "
            "precision and recall; a model that flags every window gets 0.667 "
            "(precision 50%, recall 100%), one that flags nothing gets 0. "
            "Separation: how often an anomaly lies further outside the interval "
            "than a normal window (0.5 is chance, 1 is always).",
        ),
    ):
        text += ["", f"## {title}", "", note]
        for dataset, models in results[task].items():
            text += ["", f"### {TITLES.get(dataset, dataset)}", ""]
            text.append(
                _table({m: e["overall"] for m, e in models.items()}, columns, "Model")
            )
        for dataset, models in results[task].items():
            breakdowns = sorted({k for e in models.values() for k in e} - {"overall"})
            for by in breakdowns:
                if by == "by_series":
                    continue  # in results.json; too long for the page
                name = NAMES.get(by[3:], by[3:].replace("_", " "))
                text += ["", f"### {TITLES.get(dataset, dataset)} by {name}", ""]
                rows = {
                    f"{model}, {value}": summary
                    for model, entry in models.items()
                    for value, summary in entry.get(by, {}).items()
                }
                text.append(_table(rows, columns, f"Model, {name}"))
    return "\n".join(text) + "\n"

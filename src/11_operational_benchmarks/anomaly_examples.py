"""A page of example anomaly cases: what each model expected and what came."""

import hashlib

import numpy as np
import plotly.graph_objects as go
from operational_scoring import intervals
from plotly.subplots import make_subplots

SHOWN_CONTEXT = 224  # points of the context drawn before the forecast


def pick(cases, per_dataset, seed=7):
    """A seeded choice of anomaly cases with the longest context, per dataset."""
    anomalies = cases[cases["task"] == "anomaly"]
    chosen = []
    for _, part in anomalies.groupby("dataset"):
        part = part[part["context_length"] == part["context_length"].max()]
        ranked = sorted(
            part.index,
            key=lambda i: hashlib.sha256(f"{seed}:{part['case'][i]}".encode()).digest(),
        )
        chosen += ranked[:per_dataset]
    return chosen


def write_examples(path, cases, forecasts, options, per_dataset=4):
    """One row per case, one panel per model: actual values, median and interval."""
    saved = {p.stem: np.load(p) for p in sorted(forecasts.glob("*.npz"))}
    rows = pick(cases, per_dataset)
    if not saved or not rows:
        return
    bands = {
        model: intervals(data["quantiles"], data["levels"], options["anomaly"])
        for model, data in saved.items()
    }
    figure = make_subplots(
        rows=len(rows),
        cols=len(saved),
        shared_yaxes=True,
        horizontal_spacing=0.01,
        vertical_spacing=0.25 / len(rows),
        column_titles=list(saved),
        row_titles=[
            f"{cases['dataset'][i]} {cases['series'][i]}<br>{cases['anomaly'][i]}"
            for i in rows
        ],
    )
    for row, index in enumerate(rows, start=1):
        case = cases.loc[index]
        values = np.asarray(case["values"], dtype=float)
        context = values[: case["context_length"]][-SHOWN_CONTEXT:]
        actual = values[case["context_length"] :]
        before = np.arange(-len(context), 0)
        steps = np.arange(len(actual))
        for column, model in enumerate(saved, start=1):
            median, lower, upper = (part[index] for part in bands[model])

            def line(x, y, **style):
                figure.add_trace(
                    go.Scatter(x=x, y=y, mode="lines", showlegend=False, **style),
                    row=row,
                    col=column,
                )

            line(steps, upper, line={"width": 0}, hoverinfo="skip")
            line(
                steps,
                lower,
                line={"width": 0},
                fill="tonexty",
                fillcolor="rgba(31,119,180,0.25)",
                name="interval",
            )
            line(steps, median, line={"color": "#1f77b4"}, name="median")
            line(
                np.concatenate([before, steps]),
                np.concatenate([context, actual]),
                line={"color": "black", "width": 1},
                name="actual",
            )
            figure.add_vrect(
                x0=case["span_start"],
                x1=case["span_start"] + case["span_length"] - 1,
                fillcolor="red",
                opacity=0.12,
                line_width=0,
                row=row,
                col=column,
            )
    figure.update_layout(
        template="plotly_white",
        height=260 * len(rows) + 80,
        title="Anomaly cases: actual values (black), median forecast and interval "
        "(blue); the labelled anomaly is shaded red. Step 0 is the first forecast step.",
    )
    figure.write_html(path, include_plotlyjs=True)

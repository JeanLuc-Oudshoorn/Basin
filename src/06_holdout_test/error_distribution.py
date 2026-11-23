"""Equal-table error summaries and offline interactive kernel density plots."""

import argparse
import json
import math
import sys
from pathlib import Path

import numpy as np
import plotly.graph_objects as go
from plotly.colors import qualitative
from plotly.subplots import make_subplots

METRICS = ("mase", "rmse", "mae", "normalized_rmse", "normalized_pinball")


def table_values(tables, method, metric):
    values = np.asarray(
        [table.get(method, {}).get(metric, np.nan) for table in tables], dtype=float
    )
    return values[np.isfinite(values)]


def summarize_tables(tables, methods):
    """Every finite table score gets one vote, irrespective of its point count."""
    models = {}
    for method in methods:
        models[method] = {}
        for metric in METRICS:
            if not any(metric in table.get(method, {}) for table in tables):
                continue
            values = table_values(tables, method, metric)
            models[method][metric] = {
                "mean": float(values.mean()) if len(values) else None,
                "median": float(np.median(values)) if len(values) else None,
                "p10": float(np.quantile(values, 0.1)) if len(values) else None,
                "p90": float(np.quantile(values, 0.9)) if len(values) else None,
                "valid_tables": len(values),
                "undefined_tables": len(tables) - len(values),
            }
    return {
        "aggregation": "equal weight per table; finite scores only; linear-interpolated percentiles",
        "table_count": len(tables),
        "models": models,
    }


def bandwidth(values):
    """Silverman's robust bandwidth; a degenerate distribution has no KDE."""
    if len(values) < 2 or np.ptp(values) == 0:
        return None
    std = np.std(values, ddof=1)
    iqr = np.subtract(*np.quantile(values, [0.75, 0.25])) / 1.34
    return float(0.9 * (min(std, iqr) if iqr > 0 else std) * len(values) ** -0.2)


def reflected_density(values, grid, width):
    """Gaussian KDE reflected at zero, evaluated in bounded-memory chunks."""
    density = np.zeros(len(grid))
    for start in range(0, len(values), 512):
        chunk = values[start : start + 512]
        for signed in (chunk, -chunk):
            z = (grid[:, None] - signed[None, :]) / width
            density += np.exp(-0.5 * z**2).sum(axis=1)
    return density / (len(values) * width * np.sqrt(2 * np.pi))


def write_error_distribution(result, directory, config, *, tables=None):
    """Keep raw metric values intact; transformation affects only the KDE view."""
    tables, methods = result["tables"] if tables is None else tables, result["models"]
    result["table_summary"] = summarize_tables(tables, methods)
    settings = config.get("plot", {}).get("error_distribution", {})
    if not settings.get("enabled", True):
        result.pop("error_distribution_plot", None)
        return
    axis = settings.get("axis", "log1p")
    points = settings.get("grid_points", 512)
    if (
        axis not in ("log1p", "linear")
        or isinstance(points, bool)
        or not isinstance(points, int)
        or points < 32
    ):
        raise ValueError(
            "Error density requires axis=log1p or linear and grid_points >= 32"
        )
    metrics = [
        m
        for m in METRICS
        if any(m in t.get(name, {}) for t in tables for name in methods)
    ]
    figure = make_subplots(
        rows=math.ceil(len(metrics) / 2), cols=2, subplot_titles=metrics
    )
    for index, metric in enumerate(metrics):
        row, col = index // 2 + 1, index % 2 + 1
        values = {name: table_values(tables, name, metric) for name in methods}
        if any(np.any(v < 0) for v in values.values()):
            raise ValueError("Forecast error distributions require nonnegative metrics")
        transformed = {
            name: np.log1p(v) if axis == "log1p" else v for name, v in values.items()
        }
        widths = {name: bandwidth(v) for name, v in transformed.items()}
        upper = max(
            (
                v.max() + 4 * (widths[name] or 0)
                for name, v in transformed.items()
                if len(v)
            ),
            default=1.0,
        )
        grid = np.linspace(0, upper if upper > 0 else 1, points)
        for method_index, name in enumerate(methods):
            v, width = transformed[name], widths[name]
            if not len(v):
                continue
            color = qualitative.Plotly[method_index % len(qualitative.Plotly)]
            if width is None:
                figure.add_trace(
                    go.Scatter(
                        x=[float(v[0])],
                        y=[0],
                        mode="markers",
                        name=f"{name} (constant; no KDE)",
                        legendgroup=name,
                        showlegend=index == 0,
                        marker={"color": color, "size": 10},
                    ),
                    row=row,
                    col=col,
                )
                continue
            rgb = tuple(int(color[i : i + 2], 16) for i in (1, 3, 5))
            figure.add_trace(
                go.Scatter(
                    x=grid,
                    y=reflected_density(v, grid, width),
                    customdata=np.expm1(grid) if axis == "log1p" else grid,
                    name=name,
                    legendgroup=name,
                    showlegend=index == 0,
                    mode="lines",
                    line={"color": color},
                    fill="tozeroy",
                    fillcolor=f"rgba({rgb[0]},{rgb[1]},{rgb[2]},0.16)",
                    hovertemplate="Error=%{customdata:.6g}<br>Density=%{y:.4g}<extra>%{fullData.name}</extra>",
                ),
                row=row,
                col=col,
            )
        figure.update_xaxes(
            title_text=f"log(1 + {metric})" if axis == "log1p" else metric,
            row=row,
            col=col,
        )
        figure.update_yaxes(
            title_text="Density in transformed space" if axis == "log1p" else "Density",
            row=row,
            col=col,
        )
    figure.update_layout(
        title="Per-table forecast errors — equal table weights; no tail clipping",
        template="plotly_white",
        height=350 * math.ceil(len(metrics) / 2),
        legend={"groupclick": "togglegroup"},
    )
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    filename = "error_distributions.html"
    figure.write_html(
        directory / filename,
        include_plotlyjs=True,
        config={"scrollZoom": True, "displaylogo": False},
    )
    result["error_distribution_plot"] = {
        "file": filename,
        "axis": axis,
        "grid_points": points,
        "estimator": "Gaussian KDE, Silverman robust bandwidth, reflection at zero; constant samples shown as markers",
        "weighting": "one score per table, independently normalized density per model; all finite values included",
        "undefined": "excluded from KDE and summaries; counts in table_summary",
    }


def write_latency_distribution(result, requests, directory, config):
    """Plot scored request latency, with the same transparent KDE as errors."""
    settings = config.get("plot", {}).get("latency_distribution", {})
    if not settings.get("enabled", True):
        return
    axis = settings.get("axis", "log1p")
    points = settings.get("grid_points", 512)
    if (
        axis not in ("log1p", "linear")
        or isinstance(points, bool)
        or not isinstance(points, int)
        or points < 32
    ):
        raise ValueError(
            "Latency density requires axis=log1p or linear and grid_points >= 32"
        )
    figure = go.Figure()
    for index, name in enumerate(result["models"]):
        values = (
            requests.loc[
                (requests.model == name) & (requests.phase == "scored"), "seconds"
            ].to_numpy()
            * 1000
        )
        values = values[np.isfinite(values) & (values >= 0)]
        if not len(values):
            continue
        transformed = np.log1p(values) if axis == "log1p" else values
        width = bandwidth(transformed)
        color = qualitative.Plotly[index % len(qualitative.Plotly)]
        if width is None:
            figure.add_trace(
                go.Scatter(
                    x=[float(transformed[0])],
                    y=[0],
                    mode="markers",
                    name=f"{name} (constant)",
                    marker={"color": color},
                )
            )
            continue
        grid = np.linspace(0, max(transformed.max() + 4 * width, 1e-10), points)
        rgb = tuple(int(color[i : i + 2], 16) for i in (1, 3, 5))
        figure.add_trace(
            go.Scatter(
                x=grid,
                y=reflected_density(transformed, grid, width),
                customdata=np.expm1(grid) if axis == "log1p" else grid,
                name=name,
                mode="lines",
                line={"color": color},
                fill="tozeroy",
                fillcolor=f"rgba({rgb[0]},{rgb[1]},{rgb[2]},0.16)",
                hovertemplate="Latency=%{customdata:.4g} ms<br>Density=%{y:.4g}<extra>%{fullData.name}</extra>",
            )
        )
    figure.update_layout(
        title="Scored request latency distributions (warmups excluded; capture costs included)",
        template="plotly_white",
        xaxis_title="log(1 + latency in ms)" if axis == "log1p" else "Latency (ms)",
        yaxis_title="Density in transformed space" if axis == "log1p" else "Density",
    )
    filename = "latency_distributions.html"
    figure.write_html(
        Path(directory) / filename,
        include_plotlyjs=True,
        config={"scrollZoom": True, "displaylogo": False},
    )
    result["latency_distribution_plot"] = {
        "file": filename,
        "axis": axis,
        "grid_points": points,
        "population": "all scored request repeats; warmups excluded; no tail clipping; independently normalized KDE per model",
    }


def main():
    root = Path(__file__).resolve().parents[2]
    sys.path.insert(0, str(root / "src/03_transforming_input_data"))
    from holdout_runs import latest_holdout_run
    from result_upload import upload_changed_run
    from transform import load_config, run_context, write_json

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default=str(root / "configs/config.json"))
    parser.add_argument(
        "--run", help="holdout run folder name; default: most recent completed run"
    )
    args = parser.parse_args()
    config = load_config(args.config)
    with run_context(config, "holdout_error_distribution"):
        directory = latest_holdout_run(config, args.run)
        path = directory / "test_metrics.json"
        result = json.loads(path.read_text(encoding="utf-8"))
        tables = result.get("tables")
        if tables is None:
            tables = json.loads(
                (directory / result["detail_files"]["tables"]).read_text(
                    encoding="utf-8"
                )
            )
        write_error_distribution(result, directory, config, tables=tables)
        import pandas as pd

        write_latency_distribution(
            result,
            pd.read_parquet(directory / "request_latencies.parquet"),
            directory,
            config,
        )
        if "tables" in result:
            write_json(
                directory / "table_metrics.json", result["tables"], level="DEBUG"
            )
            result.setdefault("detail_files", {})["tables"] = "table_metrics.json"
        if "origins" in result:
            write_json(
                directory / "origin_metrics.json", result["origins"], level="DEBUG"
            )
            result.setdefault("detail_files", {})["origins"] = "origin_metrics.json"
        result.pop("tables", None)
        result.pop("origins", None)
        write_json(path, result)
        upload_changed_run(config, directory)


if __name__ == "__main__":
    main()

"""Interactive holdout comparisons; render saved predictions without model inference."""

import argparse
import sys
from pathlib import Path

import pandas as pd
import plotly.graph_objects as go
from plotly.colors import qualitative

ROOT = Path(__file__).resolve().parents[2]


def write_holdout_plot(
    frame, predictions, target, methods, output, config, table_index
):
    """One trace per model, with gaps between independent forecast origins."""
    selected = predictions[predictions.target == target]
    figure = go.Figure()
    figure.add_trace(
        go.Scatter(
            x=frame.index.tolist(),
            y=frame[target].tolist(),
            name="Actual",
            mode="lines",
            line={"color": "black", "width": 1.5},
        )
    )
    for index, method in enumerate(methods):
        x, y, origins = [], [], []
        for origin, segment in selected.groupby("origin", sort=True):
            x.extend(segment.time_index.tolist() + [None])
            y.extend(segment[method].tolist() + [None])
            origins.extend([int(origin)] * len(segment) + [None])
        figure.add_trace(
            go.Scatter(
                x=x,
                y=y,
                customdata=origins,
                name=method,
                mode="lines",
                connectgaps=False,
                line={
                    "color": qualitative.Plotly[index % len(qualitative.Plotly)],
                    "width": 2,
                },
                hovertemplate="Timestep %{x}<br>Forecast %{y:.6g}<br>Origin %{customdata}<extra>%{fullData.name}</extra>",
            )
        )
    figure.add_vline(
        x=int(selected.origin.min()),
        line_dash="dash",
        line_color="gray",
        annotation_text="First forecast",
        annotation_position="top left",
    )
    figure.update_layout(
        title=f"Held-out table {table_index}: {target} ({selected.origin.nunique()} one-token forecasts)",
        template="plotly_white",
        xaxis_title="Timestep",
        yaxis_title=target,
        height=max(350, round(config["plot"].get("height", 5) * 100)),
        autosize=True,
        dragmode="zoom",
        hovermode="closest",
        legend={"groupclick": "toggleitem"},
    )
    figure.update_xaxes(rangeslider_visible=True)
    figure.update_yaxes(fixedrange=False)
    output = Path(output)
    output.parent.mkdir(parents=True, exist_ok=True)
    # Embed Plotly itself so the plots work offline, without a CDN or server.
    figure.write_html(
        output,
        include_plotlyjs=True,
        full_html=True,
        config={"scrollZoom": True, "displaylogo": False, "responsive": True},
    )
    return figure


def regenerate_saved_plots(config):
    """Use the saved random selection; never load model weights or forecast again."""
    import json

    from holdout import holdout_directory
    from transform import write_json

    directory = holdout_directory(config)
    selection = json.loads(
        (directory / "plot_selection.json").read_text(encoding="utf-8")
    )
    metrics_path = directory / "test_metrics.json"
    metrics = json.loads(metrics_path.read_text(encoding="utf-8"))
    methods = metrics["models"]
    count = 0
    for record in selection["tables"]:
        index = record["table_index"]
        if (
            index >= len(metrics["tables"])
            or metrics["tables"][index]["source"] != record["source"]
        ):
            raise ValueError(
                "Plot selection does not match completed metrics; finish the holdout run first"
            )
        predictions = pd.read_parquet(directory / f"test_table_{index:03d}.parquet")
        if set(predictions.source) != {record["source"]}:
            raise ValueError("Saved prediction table does not match the plot selection")
        frame = pd.read_parquet(ROOT / record["source"])
        filenames = []
        for target_index, target in enumerate(predictions.target.drop_duplicates()):
            filename = f"test_table_{index:03d}_target_{target_index}.html"
            write_holdout_plot(
                frame, predictions, target, methods, directory / filename, config, index
            )
            filenames.append(filename)
            count += 1
        record["plots"] = filenames
    selection["format"] = "plotly_html"
    write_json(directory / "plot_selection.json", selection, level="DEBUG")
    metrics["plot_selection"] = selection
    write_json(metrics_path, metrics, level="DEBUG")
    return count


def main():
    for stage in ("03_transforming_input_data", "05_model_training", "06_holdout_test"):
        sys.path.insert(0, str(ROOT / "src" / stage))
    from loguru import logger
    from transform import load_config, run_context

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default=str(ROOT / "configs/config.json"))
    args = parser.parse_args()
    config = load_config(args.config)
    with run_context(config, "holdout_plots"):
        count = regenerate_saved_plots(config)
        logger.success("Saved {} interactive holdout HTML plots", count)


if __name__ == "__main__":
    main()

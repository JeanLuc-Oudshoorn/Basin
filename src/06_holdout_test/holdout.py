"""Final, dataset-disjoint, one-token rolling evaluation for train and forecast."""

import logging
import math
import tempfile
from contextlib import contextmanager
from time import perf_counter

import numpy as np
import pandas as pd
import torch
from benchmarks import FoundationBenchmarks, benchmark_names, select_plot_tables
from inference import forecast_contexts
from loguru import logger
from training_log import emit_training_event
from training_progress import duration
from training_protocol import sampled_origins
from transform import ROOT, file_digest, location, timed, write_json


@contextmanager
def quiet_baseline_info():
    """Keep Prophet/Stan warnings and errors, suppress repetitive fit chatter."""
    from cmdstanpy.utils.logging import get_logger

    loggers = [get_logger(), logging.getLogger("prophet")]
    levels = [item.level for item in loggers]
    try:
        for item in loggers:
            item.setLevel(max(item.getEffectiveLevel(), logging.WARNING))
        yield
    finally:
        for item, level in zip(loggers, levels):
            item.setLevel(level)


def baseline_forecasts(
    history, horizon, config, *, past_covariates=None, regressors=None
):
    """Fit Prophet and carry forward the last observed target, using history only."""
    observed = np.asarray(history)[np.isfinite(history)]
    if not len(observed):
        raise ValueError("Cannot fit baselines to an entirely missing target context")
    naive = np.full(horizon, observed[-1], dtype=float)
    names = benchmark_names(config)
    result = {"naive": naive} if "naive" in names else {}
    if "prophet" not in names:
        return result

    if not np.isfinite(history).all():
        logger.warning(
            "Imputing missing historical target values for baseline fitting only"
        )
    history = (
        pd.Series(history).interpolate(limit_direction="both").to_numpy(dtype=float)
    )
    if not np.isfinite(history).all():
        raise ValueError("Cannot fit baselines to an entirely missing target context")
    if regressors is None:
        regressors = prophet_regressor_inputs(past_covariates, horizon, config)
    result["prophet"] = prophet_forecast(history, horizon, config, regressors)
    if any(not np.isfinite(v).all() for v in result.values()):
        raise FloatingPointError("Nonfinite baseline forecast")
    return result


def prophet_regressor_inputs(history, horizon, config):
    """Build future regressors from past covariates only, once per origin."""
    settings = config["prophet"]
    enabled = settings.get("use_covariates", True)
    method = settings.get("covariate_forecast", "prophet")
    if not isinstance(enabled, bool) or method not in ("prophet", "naive"):
        raise ValueError(
            "prophet.use_covariates must be boolean and covariate_forecast must be prophet or naive"
        )
    bundle = {
        "training": {},
        "future": {},
        "columns": [],
        "skipped": [],
        "method": method,
    }
    if not enabled or history is None:
        return bundle
    for index, column in enumerate(history.columns):
        values = history[column].to_numpy(dtype=float)
        if np.isinf(values).any():
            raise ValueError(f"Infinite historical Prophet covariate: {column}")
        values = pd.Series(values).interpolate(limit_direction="both").to_numpy()
        if not np.isfinite(values).all() or np.unique(values).size < 2:
            bundle["skipped"].append(column)
            continue
        future = (
            prophet_forecast(values, horizon, config)
            if method == "prophet"
            else np.full(horizon, values[-1], dtype=float)
        )
        name = f"regressor_{index:03d}"
        bundle["training"][name] = values
        bundle["future"][name] = future
        bundle["columns"].append(column)
    return bundle


def prophet_forecast(history, horizon, config, regressors=None):
    """Fit one series with history-derived regressors and clean up Stan outputs."""
    from prophet import Prophet

    settings = config["prophet"]
    dates = pd.date_range(
        settings["start_date"],
        periods=len(history) + horizon,
        freq=settings["frequency"],
    )
    options = {
        k: settings[k]
        for k in (
            "yearly_seasonality",
            "weekly_seasonality",
            "daily_seasonality",
            "seasonality_mode",
            "changepoint_prior_scale",
            "seasonality_prior_scale",
            "uncertainty_samples",
        )
    }
    with quiet_baseline_info(), timed("holdout Prophet fit and predict", level="DEBUG"):
        prophet = Prophet(**options)
        prophet.add_seasonality(
            name="configured_period",
            period=settings["seasonality_period"],
            fourier_order=settings["fourier_order"],
        )
        training = pd.DataFrame({"ds": dates[: len(history)], "y": history})
        future = pd.DataFrame({"ds": dates[len(history) :]})
        if regressors:
            for name, values in regressors["training"].items():
                if len(values) != len(history):
                    raise ValueError(
                        "Prophet target and covariate history lengths differ"
                    )
                prophet.add_regressor(name)
                training[name] = values
                future[name] = regressors["future"][name]
        temp_root = ROOT / config.get("holdout", {}).get(
            "prophet_temp_dir", ".cache/prophet_holdout"
        )
        temp_root.mkdir(parents=True, exist_ok=True)
        with tempfile.TemporaryDirectory(prefix="fit_", dir=temp_root) as directory:
            prophet.fit(training, seed=config["seed"], output_dir=directory)
            predictions = prophet.predict(future)["yhat"].to_numpy().copy()
    if predictions.shape != (horizon,) or not np.isfinite(predictions).all():
        raise FloatingPointError("Nonfinite or invalid Prophet forecast")
    return predictions


def summarize(frame):
    """Pool squared errors over observations, not averages of per-origin RMSE."""
    result = {"observations": len(frame)}
    for method in ("rwkv", "prophet", "naive", "chronos", "timestral"):
        if method not in frame:
            continue
        error = frame[method].to_numpy() - frame.actual.to_numpy()
        result[method] = {
            "rmse": float(np.sqrt(np.mean(error**2))),
            "mae": float(np.mean(np.abs(error))),
            "normalized_rmse": float(
                np.sqrt(np.mean((error / frame.context_scale) ** 2))
            ),
        }
    result["rwkv"]["normalized_pinball"] = float(frame.normalized_pinball.mean())
    return result


@torch.inference_mode()
def evaluate_holdout(model, records, config):
    """Never fit the neural model on these sources, including their initial context."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    settings = config.get("holdout", {})
    percentiles = settings.get("percentiles", [1, 25, 50, 75, 99])
    batch_size = settings.get("inference_batch_size", 5)
    progress_every = settings.get("progress_every_tables", 100)
    if (
        isinstance(progress_every, bool)
        or not isinstance(progress_every, int)
        or progress_every < 1
    ):
        raise ValueError("holdout.progress_every_tables must be a positive integer")
    directory = holdout_directory(config)
    directory.mkdir(parents=True, exist_ok=True)
    plot_indices, plot_selection = select_plot_tables(records, config)
    write_json(directory / "plot_selection.json", plot_selection, level="DEBUG")
    device = next(model.parameters()).device
    patch = model.config.patch_size
    cap = min(model.config.context_length, config["data"]["context_length"])
    benchmarks = FoundationBenchmarks(config, cap)
    methods = ["rwkv"] + benchmarks.names
    rows, origin_metrics, table_metrics = [], [], []
    was_training = model.training
    model.eval()
    started = perf_counter()
    completed_origins = 0
    logger.info(
        "HOLDOUT start: {} tables, up to {} origins per table, models={}, output={}",
        len(records),
        len(percentiles),
        methods,
        directory,
    )
    try:
        for table_index, record in enumerate(records):
            path = ROOT / record["source"]
            if file_digest(path) != record["source_sha256"]:
                raise ValueError(f"Holdout source changed since transformation: {path}")
            frame = pd.read_parquet(path)
            targets, covariates = record["targets"], record["roles"]["past"]
            table_rows = []
            origins = sampled_origins(len(frame), patch, percentiles)
            contexts = []
            for origin in origins:
                history = frame.iloc[max(0, origin - cap) : origin]
                contexts.append(
                    {
                        "context": torch.tensor(
                            history[targets].to_numpy(dtype=np.float32).T.copy()
                        )[None],
                        "past_covariates": None
                        if not covariates
                        else torch.tensor(
                            history[covariates].to_numpy(dtype=np.float32).T.copy()
                        )[None],
                    }
                )
            with timed(
                f"holdout RWKV table={table_index} origins={origins}",
                cuda=device.type == "cuda",
                level="DEBUG",
            ):
                predictions = list(forecast_contexts(model, contexts, batch_size))
            foundation_predictions = benchmarks.predict(contexts, patch)
            for origin, prediction, foundation in zip(
                origins, predictions, foundation_predictions
            ):
                stop = min(origin + patch, len(frame))
                history = frame.iloc[max(0, origin - cap) : origin]
                quantiles = prediction[:, : stop - origin].numpy()
                point = quantiles[..., model.config.quantiles.index(0.5)]
                if not np.isfinite(quantiles).all():
                    raise FloatingPointError("Nonfinite holdout neural forecast")
                regressors = (
                    prophet_regressor_inputs(history[covariates], stop - origin, config)
                    if "prophet" in benchmarks.names
                    else None
                )
                for target_index, target in enumerate(targets):
                    baseline = (
                        baseline_forecasts(
                            history[target].to_numpy(),
                            stop - origin,
                            config,
                            regressors=regressors,
                        )
                        if any(
                            name in benchmarks.names for name in ("prophet", "naive")
                        )
                        else {}
                    )
                    actual = frame[target].iloc[origin:stop].to_numpy(dtype=float)
                    scale = float(np.nanstd(history[target].to_numpy(dtype=float)))
                    scale = scale if math.isfinite(scale) and scale > 1e-5 else 1.0
                    error = (actual[:, None] - quantiles[target_index]) / scale
                    levels = np.asarray(model.config.quantiles)
                    pinball = np.maximum(levels * error, (levels - 1) * error).mean(-1)
                    current_rows = []
                    for offset, value in enumerate(actual):
                        if not np.isfinite(value):
                            continue
                        current_rows.append(
                            {
                                "source": record["source"],
                                "target": target,
                                "origin": origin,
                                "time_index": origin + offset,
                                "context_points": len(history),
                                "actual": value,
                                "rwkv": float(point[target_index, offset]),
                                **{
                                    name: float(baseline[name][offset])
                                    if name in ("prophet", "naive")
                                    else float(foundation[name][target_index, offset])
                                    for name in benchmarks.names
                                },
                                "context_scale": scale,
                                "normalized_pinball": float(pinball[offset]),
                            }
                        )
                    if not current_rows:
                        raise ValueError(
                            f"No observed holdout labels: {record['source']} origin={origin}"
                        )
                    table_rows.extend(current_rows)
                    metric = {
                        "source": record["source"],
                        "target": target,
                        "origin": origin,
                        **summarize(pd.DataFrame(current_rows)),
                    }
                    origin_metrics.append(metric)
                    if regressors is not None:
                        metric["prophet_regressors"] = regressors["columns"]
                        metric["prophet_skipped_covariates"] = regressors["skipped"]
                    logger.debug("OUT OF SAMPLE {}", metric)
                    emit_training_event(
                        "out_of_sample_progress", out_of_sample_origin=metric
                    )
                completed_origins += 1
            table_frame = pd.DataFrame(table_rows)
            rows.extend(table_rows)
            metric = {
                "source": record["source"],
                "origins": len(origins),
                **summarize(table_frame),
            }
            table_metrics.append(metric)
            table_frame.to_parquet(
                directory / f"test_table_{table_index:03d}.parquet", index=False
            )
            if table_index in plot_indices:
                for target in targets:
                    selected = table_frame[table_frame.target == target]
                    fig, axis = plt.subplots(
                        figsize=(config["plot"]["width"], config["plot"]["height"])
                    )
                    axis.plot(frame.index, frame[target], color="black", label="Actual")
                    for method_index, method in enumerate(methods):
                        for number, (_, segment) in enumerate(
                            selected.groupby("origin")
                        ):
                            axis.plot(
                                segment.time_index,
                                segment[method],
                                label=method if number == 0 else None,
                                color=f"C{method_index % 10}",
                            )
                    axis.axvline(
                        2 * patch, color="gray", linestyle="--", label="First forecast"
                    )
                    axis.set(
                        title=f"Held-out table {table_index}: {target} ({len(origins)} sampled one-token forecasts)",
                        xlabel="Timestep",
                        ylabel=target,
                    )
                    axis.legend()
                    fig.tight_layout()
                    target_index = targets.index(target)
                    fig.savefig(
                        directory
                        / f"test_table_{table_index:03d}_target_{target_index}.png",
                        dpi=config["plot"]["dpi"],
                    )
                    plt.close(fig)
            elapsed = perf_counter() - started
            completed = len(table_metrics)
            progress = {
                "completed_tables": completed,
                "total_tables": len(records),
                "completed_origins": completed_origins,
                "elapsed_seconds": elapsed,
                "eta_seconds": elapsed / completed * (len(records) - completed),
                "progress_percent": 100 * completed / len(records),
                "last_table": metric,
            }
            if completed % progress_every == 0 or completed == len(records):
                logger.info(
                    "HOLDOUT tables {:,}/{:,} ({:.1f}%) | origins={:,} | elapsed {} | ETA {}",
                    completed,
                    len(records),
                    progress["progress_percent"],
                    completed_origins,
                    duration(elapsed),
                    duration(progress["eta_seconds"]),
                )
            emit_training_event("holdout_table_complete", **progress)
            write_json(
                directory / "test_metrics_progress.json",
                progress,
                level="DEBUG",
            )
        if not rows:
            raise ValueError("No holdout observations")
        all_predictions = pd.DataFrame(rows)
        all_predictions.to_parquet(directory / "test_predictions.parquet", index=False)
        result = {
            "protocol": "dataset-disjoint, sampled patch origins, one token ahead",
            "percentiles": percentiles,
            "origin_selection": "first eligible, nearest row percentiles with distinct ordered origins, final eligible; all origins for short tables",
            "patch_size": patch,
            "context_limit": cap,
            "initial_context_tokens": 2,
            "partial_final_patch": "score real observations only",
            "models": methods,
            "benchmark_metadata": benchmarks.metadata,
            "plot_selection": plot_selection,
            "baseline_covariates": "Naive uses historical targets; foundation models use past covariates directly; Prophet uses configured historical regressors with forecast-derived future values, no observed future covariates",
            "table_count": len(records),
            "origin_target_count": len(origin_metrics),
            "seconds": perf_counter() - started,
            "pooled": summarize(all_predictions),
            "tables": table_metrics,
            "origins": origin_metrics,
        }
        write_json(directory / "test_metrics.json", result)
        logger.success("Final out-of-sample comparison: {}", result["pooled"])
        return result
    finally:
        model.train(was_training)


def holdout_directory(config):
    """Dedicated results beside the training directory, also for older configs."""
    if "holdout_test" in config["paths"]:
        return location(config, "holdout_test")
    return location(config, "training").parent / "holdout_test"

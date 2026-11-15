"""Dataset-disjoint full-block evaluation with training-weighted lead scores."""

import json
import logging
import math
import tempfile
from contextlib import contextmanager
from time import perf_counter

import numpy as np
import pandas as pd
import pyarrow.parquet as pq
import torch
from benchmarks import FoundationBenchmarks, benchmark_names, select_plot_tables
from error_distribution import write_error_distribution, write_latency_distribution
from fast_inference import inference_execution_summary
from graph_profiles import configure_holdout_graph_profiles
from horizon_policy import evaluation_weights
from inference import forecast_contexts
from latency import RequestTimer
from loguru import logger
from plotting import write_holdout_plot
from rolling_validation import table_mase_profile
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


def mase_scale(history, lag=1):
    """Historical naive MAE; never bridge missing observations or use future labels."""
    values = np.asarray(history, dtype=float)
    if len(values) <= lag:
        return np.nan
    left, right = values[lag:], values[:-lag]
    valid = np.isfinite(left) & np.isfinite(right)
    if not valid.any():
        return np.nan
    scale = float(np.abs(left[valid] - right[valid]).mean())
    return scale if scale > 0 else np.nan


def summarize(frame):
    """Lead-weighted scores; unobserved future values never enter metrics."""
    frame = frame[np.isfinite(frame.actual)].copy()
    result = {"observations": len(frame)}
    weights = frame.get("metric_weight", pd.Series(1.0, index=frame.index)).to_numpy()

    def mean(values, mask=None):
        if mask is None:
            mask = np.ones(len(values), dtype=bool)
        return (
            float(np.average(np.asarray(values)[mask], weights=weights[mask]))
            if mask.any()
            else None
        )

    scales = frame.get("mase_scale", pd.Series(np.nan, index=frame.index)).to_numpy()
    valid = np.isfinite(scales) & (scales > 0)
    for method in ("rwkv", "prophet", "naive", "chronos", "timestral"):
        if method not in frame:
            continue
        error = frame[method].to_numpy() - frame.actual.to_numpy()
        result[method] = {
            "mase": mean(np.abs(error) / np.where(valid, scales, 1), valid)
            if valid.any()
            else None,
            "mase_observations": int(valid.sum()),
            "mase_undefined_observations": int((~valid).sum()),
            "rmse": float(np.sqrt(mean(error**2))),
            "mae": mean(np.abs(error)),
            "normalized_rmse": float(np.sqrt(mean((error / frame.context_scale) ** 2))),
        }
        values = np.abs(error[valid]) / scales[valid]
        for label, q in (
            ("median_absolute_scaled_error", 0.5),
            ("absolute_scaled_error_p10", 0.1),
            ("absolute_scaled_error_p90", 0.9),
        ):
            result[method][label] = weighted_quantile(values, weights[valid], q)
    result["rwkv"]["normalized_pinball"] = mean(frame.normalized_pinball)
    return result


def weighted_quantile(values, weights, quantile):
    """Inverse weighted empirical CDF; no invented observations between points."""
    if not len(values):
        return None
    order = np.argsort(values)
    cumulative = np.cumsum(weights[order])
    index = min(np.searchsorted(cumulative, quantile * cumulative[-1]), len(order) - 1)
    return float(values[order[index]])


def classical_request(method, history, targets, covariates, horizon, config):
    """One complete classical-model request, including any regressor fitting."""
    options = dict(
        config, holdout=dict(config.get("holdout", {}), benchmark_models=[method])
    )
    regressors = (
        prophet_regressor_inputs(history[covariates], horizon, options)
        if method == "prophet"
        else None
    )
    points = np.stack(
        [
            baseline_forecasts(
                history[target].to_numpy(), horizon, options, regressors=regressors
            )[method]
            for target in targets
        ]
    )
    return points, regressors


@torch.inference_mode()
def evaluate_holdout(model, records, config):
    """Never fit the neural model on these sources, including their initial context."""
    settings = config.get("holdout", {})
    probabilities = evaluation_weights(config)
    percentiles = settings.get("percentiles", [1, 25, 50, 75, 99])
    lag = settings.get("mase_seasonality", 1)
    if isinstance(lag, bool) or not isinstance(lag, int) or lag < 1:
        raise ValueError("holdout.mase_seasonality must be a positive integer")
    timer = RequestTimer(config)
    progress_every = settings.get("progress_every_tables", 100)
    if (
        isinstance(progress_every, bool)
        or not isinstance(progress_every, int)
        or progress_every < 1
    ):
        raise ValueError("holdout.progress_every_tables must be a positive integer")
    directory = holdout_directory(config)
    directory.mkdir(parents=True, exist_ok=True)
    device = next(model.parameters()).device
    patch = model.config.patch_size
    requested_tables = len(records)
    minimum = (2 + len(probabilities)) * patch
    records = [
        record
        for record in records
        if (
            record["source_rows"]
            if "source_rows" in record
            else pq.ParquetFile(ROOT / record["source"]).metadata.num_rows
        )
        >= minimum
    ]
    if len(records) != requested_tables:
        logger.warning(
            "Skipping {} short holdout tables: need at least {} points for two context tokens plus {} forecast tokens",
            requested_tables - len(records),
            minimum,
            len(probabilities),
        )
    if not records:
        raise ValueError(
            f"No holdout tables have at least {minimum} points for a full {len(probabilities)}-token forecast"
        )
    plot_indices, plot_selection = select_plot_tables(records, config)
    write_json(directory / "plot_selection.json", plot_selection, level="DEBUG")
    cap = min(model.config.context_length, config["data"]["context_length"])
    benchmarks = FoundationBenchmarks(config, cap)
    benchmarks.metadata["rwkv"] = {
        "parameters": sum(p.numel() for p in model.parameters()),
        "dtype": str(getattr(model, "compute_dtype", next(model.parameters()).dtype)),
        "device": str(device),
        "point_forecast": "median of configured quantiles",
        "inputs": "historical targets and time-matched past covariates",
    }
    methods = ["rwkv"] + benchmarks.names
    graph_plan = configure_holdout_graph_profiles(model, records, config)
    if graph_plan is not None:
        logger.info("Holdout graph profile admission: {}", graph_plan)
    rows, origin_metrics, table_metrics = [], [], []
    origins_without_history = origins_without_labels = unscored_tables = 0
    was_training = model.training
    model.eval()
    started = perf_counter()
    completed_origins = 0
    latency_cursor = 0
    latency_summary = {}
    latency_journal = directory / "request_latencies.jsonl"
    latency_journal.write_text("", encoding="utf-8")
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
            origins = sampled_origins(
                len(frame), patch, percentiles, prediction_patches=len(probabilities)
            )
            # Missing values are never scored, and never a reason to drop a
            # table. Every model still forecasts an origin whose labels are
            # all missing; it just has nothing to score. The one request no
            # model can be compared on is a target without any observed
            # history: the naive forecast and the MASE scale do not exist.
            values = frame[targets].to_numpy(dtype=float)
            with_history = [
                bool(np.isfinite(values[max(0, origin - cap) : origin]).any(0).all())
                for origin in origins
            ]
            origins_without_history += len(origins) - sum(with_history)
            origins = [origin for origin, keep in zip(origins, with_history) if keep]
            if not origins:
                unscored_tables += 1
                logger.warning(
                    "Holdout table has no origin with observed target history: {}",
                    record["source"],
                )
                continue
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
            for origin_index, (origin, context) in enumerate(zip(origins, contexts)):
                forecast_tokens = len(probabilities)
                horizon = forecast_tokens * patch
                stop = min(origin + horizon, len(frame))
                history = frame.iloc[max(0, origin - cap) : origin]
                request = {
                    "table_index": table_index,
                    "source": record["source"],
                    "origin": origin,
                    "context_points": len(history),
                    "targets": len(targets),
                    "covariates": len(covariates),
                    "horizon": horizon,
                    "forecast_tokens": forecast_tokens,
                    "scored_horizon": stop - origin,
                }
                foundation, classic = {}, {}
                regressors = None
                # Rotate serial execution order to spread cache/clock/order effects.
                shift = (table_index + origin_index) % len(methods)
                order = methods[shift:] + methods[:shift]
                for order_index, name in enumerate(order):
                    request["order_index"] = order_index
                    if name == "rwkv":
                        prediction = timer.measure(
                            name,
                            lambda: list(
                                forecast_contexts(
                                    model,
                                    [context],
                                    1,
                                    **(
                                        {"prediction_patches": forecast_tokens}
                                        if forecast_tokens > 1
                                        else {}
                                    ),
                                )
                            )[0],
                            device=device,
                            request=request,
                        )
                    elif name in ("prophet", "naive"):
                        points, bundle = timer.measure(
                            name,
                            lambda name=name: classical_request(
                                name, history, targets, covariates, horizon, config
                            ),
                            device="cpu",
                            request=request,
                        )
                        classic[name] = points
                        if name == "prophet":
                            regressors = bundle
                    else:
                        foundation[name] = timer.measure(
                            name,
                            lambda name=name: benchmarks.predict(
                                [context], horizon, model_names=[name]
                            )[0][name],
                            device=benchmarks.metadata[name]["device"],
                            request=request,
                        )
                quantiles = prediction[:, :horizon].numpy()
                if quantiles.shape != (
                    len(targets),
                    horizon,
                    len(model.config.quantiles),
                ):
                    raise ValueError(
                        "Holdout model must return every requested forecast point"
                    )
                point = quantiles[..., model.config.quantiles.index(0.5)]
                if not np.isfinite(quantiles).all():
                    raise FloatingPointError("Nonfinite holdout neural forecast")
                for target_index, target in enumerate(targets):
                    actual = np.full(horizon, np.nan)
                    actual[: stop - origin] = (
                        frame[target].iloc[origin:stop].to_numpy(dtype=float)
                    )
                    denominator = mase_scale(history[target].to_numpy(), lag)
                    scale = float(np.nanstd(history[target].to_numpy(dtype=float)))
                    scale = scale if math.isfinite(scale) and scale > 0 else 1.0
                    error = (actual[:, None] - quantiles[target_index]) / scale
                    levels = np.asarray(model.config.quantiles)
                    pinball = np.maximum(levels * error, (levels - 1) * error).mean(-1)
                    current_rows = []
                    available = [
                        np.isfinite(actual[i * patch : (i + 1) * patch]).sum()
                        for i in range(forecast_tokens)
                    ]
                    total_weight = sum(
                        w for w, count in zip(probabilities, available) if count
                    )
                    for offset, value in enumerate(actual):
                        token = offset // patch
                        current_rows.append(
                            {
                                "source": record["source"],
                                "target": target,
                                "origin": origin,
                                "time_index": origin + offset,
                                "context_points": len(history),
                                "forecast_tokens": forecast_tokens,
                                "forecast_horizon": horizon,
                                "forecast_token": token + 1,
                                "metric_weight": probabilities[token]
                                / available[token]
                                / total_weight
                                if np.isfinite(value)
                                else 0.0,
                                "actual": value,
                                "rwkv": float(point[target_index, offset]),
                                **{
                                    name: float(classic[name][target_index, offset])
                                    if name in ("prophet", "naive")
                                    else float(foundation[name][target_index, offset])
                                    for name in benchmarks.names
                                },
                                "context_scale": scale,
                                "mase_scale": denominator,
                                "normalized_pinball": float(pinball[offset])
                                if np.isfinite(value)
                                else np.nan,
                            }
                        )
                    # Forecasts are kept for every point, observed or not.
                    table_rows.extend(current_rows)
                    if total_weight == 0:
                        origins_without_labels += 1
                        continue  # every label of this target is missing here
                    metric = {
                        "source": record["source"],
                        "target": target,
                        "origin": origin,
                        "forecast_tokens": forecast_tokens,
                        "forecast_horizon": horizon,
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
            if not np.isfinite(table_frame.actual).any():
                # Forecast everywhere, observed nowhere: predictions are saved,
                # but the table has no score.
                unscored_tables += 1
                table_frame.to_parquet(
                    directory / f"test_table_{table_index:03d}.parquet", index=False
                )
                continue
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
                    target_index = targets.index(target)
                    write_holdout_plot(
                        frame,
                        table_frame,
                        target,
                        methods,
                        directory
                        / f"test_table_{table_index:03d}_target_{target_index}.html",
                        config,
                        table_index,
                    )
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
                "primary_metric": "table_mase_median",
            }
            # Append only new requests; do not rewrite the entire timing history per table.
            with latency_journal.open("a", encoding="utf-8") as journal:
                for row in timer.rows[latency_cursor:]:
                    journal.write(json.dumps(row) + "\n")
            latency_cursor = len(timer.rows)
            if completed % progress_every == 0 or completed + unscored_tables == len(
                records
            ):
                latency_summary = timer.summary()
                logger.info(
                    "HOLDOUT tables {:,}/{:,} ({:.1f}%) | origins={:,} | elapsed {} | ETA {}",
                    completed,
                    len(records),
                    progress["progress_percent"],
                    completed_origins,
                    duration(elapsed),
                    duration(progress["eta_seconds"]),
                )
            progress["latency"] = latency_summary
            progress["rwkv_inference"] = inference_execution_summary(model)
            progress["latency_summary_updates_every_tables"] = progress_every
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
        pd.DataFrame(timer.rows).to_parquet(
            directory / "request_latencies.parquet", index=False
        )
        result = {
            # The name is kept for the four-token models so earlier results stay comparable.
            "scoring_protocol_version": "four_token_complete_weighted_v2"
            if len(probabilities) == 4
            else "complete_block_weighted_v3",
            # One score per table, then the middle and both ends of the
            # tables. The pooled mean ("mase") stays for continuity; a single
            # request with a near-zero scale can decide it for every model.
            "primary_metric": "table_mase_median",
            "primary_metrics": [
                "table_mase_median",
                "table_mase_p10",
                "table_mase_p90",
            ],
            "mase_definition": {
                "seasonality": lag,
                "scale": "mean absolute lag-m difference of finite historical target pairs within the capped context at each origin",
                "aggregation": "weighted token means per origin and target, then equal origin-target weight; pooled point quantiles use the same weights; table_summary gives equal-table score percentiles",
                "undefined": "zero scale, insufficient history or no finite lag pairs: null MASE and explicit excluded counts; no epsilon substitution",
            },
            "latency_protocol": timer.metadata(),
            "latency": timer.summary(),
            "protocol": f"dataset-disjoint, sampled patch origins, always {len(probabilities)} forecast tokens, training-weighted scoring",
            "horizon_policy": {
                "loss_weights": probabilities,
                "forecast_tokens": len(probabilities),
                "boundary": f"all origins reserve {len(probabilities)} full future tokens; final origin is the latest patch-aligned start with {len(probabilities) * patch} points remaining; missing actuals alone are excluded from scoring",
                "counts_by_tokens": {
                    str(k): len(
                        {
                            (row["source"], row["origin"])
                            for row in origin_metrics
                            if row["forecast_tokens"] == k
                        }
                    )
                    for k in range(1, len(probabilities) + 1)
                },
            },
            "percentiles": percentiles,
            "origin_selection": f"first eligible after two context tokens, nearest row percentiles among starts with {len(probabilities)} full future tokens, final eligible; all eligible origins for short tables",
            "patch_size": patch,
            "context_limit": cap,
            "initial_context_tokens": 2,
            "minimum_table_points": minimum,
            "missing_values": "never scored; forecasts are still made and saved for them",
            "skipped_origins_without_observed_history": origins_without_history,
            "unscored_origin_targets_without_observed_labels": origins_without_labels,
            "unscored_tables": unscored_tables,
            "selected_tables_before_length_filter": requested_tables,
            "skipped_short_tables": requested_tables - len(records),
            "models": methods,
            "graph_profile_plan": graph_plan,
            "benchmark_metadata": benchmarks.metadata,
            "plot_selection": plot_selection,
            "baseline_covariates": "Naive uses historical targets; foundation models use past covariates directly; Prophet uses configured historical regressors with forecast-derived future values, no observed future covariates",
            "table_count": len(table_metrics),
            "origin_target_count": len(origin_metrics),
            "seconds": perf_counter() - started,
            "pooled": summarize(all_predictions),
        }
        write_json(directory / "table_metrics.json", table_metrics, level="DEBUG")
        write_json(directory / "origin_metrics.json", origin_metrics, level="DEBUG")
        result["detail_files"] = {
            "tables": "table_metrics.json",
            "origins": "origin_metrics.json",
            "predictions": "test_predictions.parquet",
            "latencies": "request_latencies.parquet",
        }
        write_error_distribution(result, directory, config, tables=table_metrics)
        write_latency_distribution(result, pd.DataFrame(timer.rows), directory, config)
        result["leaderboard"] = sorted(
            [
                {
                    "model": name,
                    "table_mase_profile": table_mase_profile(
                        *(
                            result["table_summary"]["models"][name]["mase"][key]
                            for key in ("p10", "median", "p90")
                        )
                    ),
                    "mase": result["pooled"][name]["mase"],
                    "median_absolute_scaled_error": result["pooled"][name][
                        "median_absolute_scaled_error"
                    ],
                    "table_mase_mean": result["table_summary"]["models"][name]["mase"][
                        "mean"
                    ],
                    "table_mase_median": result["table_summary"]["models"][name][
                        "mase"
                    ]["median"],
                    "table_mase_p10": result["table_summary"]["models"][name]["mase"][
                        "p10"
                    ],
                    "table_mase_p90": result["table_summary"]["models"][name]["mase"][
                        "p90"
                    ],
                    "median_latency_ms": result["latency"][name]["median_ms"],
                    "p95_latency_ms": result["latency"][name]["p95_ms"],
                }
                for name in methods
            ],
            key=lambda item: (
                item["table_mase_median"] is None,
                item["table_mase_median"] or 0,
            ),
        )
        result["paired_latency"] = timer.paired_comparison()
        result["rwkv_inference"] = inference_execution_summary(model)
        result["by_forecast_token"] = {
            str(token): summarize(group)
            for token, group in all_predictions.groupby("forecast_token")
            if np.isfinite(group.actual).any()
        }
        write_json(directory / "test_metrics.json", result)
        logger.success("MASE / request latency leaderboard: {}", result["leaderboard"])
        logger.success("Final out-of-sample comparison: {}", result["pooled"])
        return result
    finally:
        model.train(was_training)


def holdout_directory(config):
    """Dedicated results beside the training directory, also for older configs."""
    if "holdout_test" in config["paths"]:
        return location(config, "holdout_test")
    return location(config, "training").parent / "holdout_test"

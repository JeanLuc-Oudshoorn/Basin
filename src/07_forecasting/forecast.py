"""Forecast held-out horizons and compare against Prophet and last-observation carry-forward, without training."""
# ruff: noqa: E402

import argparse
import sys
from pathlib import Path
from time import perf_counter

import numpy as np
import pandas as pd
import torch
from loguru import logger

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src/05_model_training"))
sys.path.insert(0, str(ROOT / "src/03_transforming_input_data"))
sys.path.insert(0, str(ROOT / "src/06_holdout_test"))
from benchmarks import sample_holdout
from holdout import evaluate_holdout
from performance import configure_cuda_memory
from train import RWKVForecaster, launch_wsl
from transform import (
    describe,
    file_digest,
    load_config,
    location,
    read_artifact,
    run_context,
    timed,
    write_json,
)
from variate_attention import configure_variate_execution


def forecast(config):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    directory = location(config, "forecasts")
    directory.mkdir(parents=True, exist_ok=True)
    data = read_artifact(location(config, "embedded") / "embeddings.pt", config)
    checkpoint = location(config, "training") / "model.pt"
    metadata = read_artifact(checkpoint, config)
    if metadata["embedding_sha256"] != file_digest(
        location(config, "embedded") / "embeddings.pt"
    ):
        raise ValueError(
            "Embedding artifact changed since training: train again before evaluation"
        )
    with timed("load trained checkpoint", cuda=True):
        torch.set_num_threads(config["training"].get("cpu_threads", 4))
        configure_cuda_memory(config["training"])
        model = RWKVForecaster.load(checkpoint).eval()
        configure_variate_execution(model, config["training"], ROOT)
    if "dataset_split" in metadata:
        test_sources = set(metadata["dataset_split"]["test_sources"])
        records = [
            r
            for r in data["records"]
            if r["split"] == "evaluation" and r["source"] in test_sources
        ]
        if {r["source"] for r in records} != test_sources:
            raise ValueError(
                "Checkpoint holdout sources missing from embedding artifact"
            )
        records, selection = sample_holdout(records, config)
        from holdout import holdout_directory

        write_json(holdout_directory(config) / "sample_selection.json", selection)
        return evaluate_holdout(model, records, config)
    metrics = []
    evaluation_records = [r for r in data["records"] if r["split"] == "evaluation"]
    plot_indices = set(
        np.linspace(
            0,
            len(evaluation_records) - 1,
            min(
                len(evaluation_records),
                config["plot"].get("max_plots", len(evaluation_records)),
            ),
            dtype=int,
        ).tolist()
    )
    evaluation_index = -1
    for record in data["records"]:
        if record["split"] != "evaluation":
            continue
        evaluation_index += 1
        batch = record["batch"]
        inputs = {
            k: v.to("cuda") if isinstance(v, torch.Tensor) else v
            for k, v in batch.items()
            if k != "future_target"
        }
        with timed(f"forecast unseen horizon {record['id']}", cuda=True):
            start = perf_counter()
            output = model.forecast(**inputs)
            torch.cuda.synchronize()
            rwkv_seconds = perf_counter() - start
            describe("forecast quantiles", output.quantiles)
        if not torch.isfinite(output.quantiles).all():
            raise FloatingPointError("Forecast contains nonfinite values")
        predictions = output.point.cpu().numpy()[0]
        actuals = batch["future_target"].numpy()[0]
        for index, target in enumerate(record["targets"]):
            context = batch["context"].numpy()[0, index]
            actual = actuals[index]
            valid = np.isfinite(actual)
            if not valid.any():
                raise ValueError(f"No evaluation labels for {target}")
            baseline_kind = "Last observation carried forward"
            if not np.isfinite(context).all():
                logger.warning(
                    "Imputing missing historical values for Last observation carried forward only"
                )
                context_baseline = (
                    pd.Series(context).interpolate(limit_direction="both").to_numpy()
                )
                if not np.isfinite(context_baseline).all():
                    raise ValueError("Cannot fit baseline to entirely missing context")
            else:
                context_baseline = context
            start = perf_counter()
            baseline = np.full(batch["horizon"], context_baseline[-1], dtype=float)
            baseline_seconds = perf_counter() - start
            if not np.isfinite(baseline).all():
                raise FloatingPointError(
                    "Last observation carried forward produced nonfinite forecasts"
                )
            rmse = float(
                np.sqrt(np.mean((predictions[index, valid] - actual[valid]) ** 2))
            )
            baseline_rmse = float(
                np.sqrt(np.mean((baseline[valid] - actual[valid]) ** 2))
            )
            prophet_prediction, prophet_rmse, prophet_seconds = None, None, None
            prophet_config = config.get("prophet", {})
            if prophet_config.get("enabled", False):
                from prophet import Prophet

                with timed(f"Prophet fit+forecast target={target}"):
                    start = perf_counter()
                    dates = pd.date_range(
                        prophet_config["start_date"],
                        periods=len(context) + len(actual),
                        freq=prophet_config["frequency"],
                    )
                    options = {
                        k: prophet_config[k]
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
                    prophet = Prophet(**options)
                    prophet.add_seasonality(
                        name="configured_period",
                        period=prophet_config["seasonality_period"],
                        fourier_order=prophet_config["fourier_order"],
                    )
                    prophet.fit(
                        pd.DataFrame(
                            {
                                "ds": dates[: len(context)],
                                "y": context_baseline.astype(np.float64),
                            }
                        ),
                        seed=config["seed"],
                    )
                    prophet_prediction = prophet.predict(
                        pd.DataFrame({"ds": dates[len(context) :]})
                    )["yhat"].to_numpy()
                    if not np.isfinite(prophet_prediction).all():
                        raise FloatingPointError("Nonfinite Prophet forecast")
                    prophet_seconds = perf_counter() - start
                    prophet_rmse = float(
                        np.sqrt(
                            np.mean((prophet_prediction[valid] - actual[valid]) ** 2)
                        )
                    )
            context_scale = max(float(np.nanstd(context)), 1e-5)
            row = {
                "window": record["id"],
                "target": target,
                "observed_labels": int(valid.sum()),
                "rwkv_rmse": rmse,
                "naive_rmse": baseline_rmse,
                "rwkv_group_seconds": rwkv_seconds,
                "naive_seconds": baseline_seconds,
                "baseline_kind": baseline_kind,
                "prophet_rmse": prophet_rmse,
                "prophet_seconds": prophet_seconds,
                "context_std": context_scale,
                "rwkv_scaled_rmse": rmse / context_scale,
                "naive_scaled_rmse": baseline_rmse / context_scale,
                "prophet_scaled_rmse": None
                if prophet_rmse is None
                else prophet_rmse / context_scale,
                "source": record["source"],
            }
            metrics.append(row)
            logger.info("EVALUATION {}", row)
            horizon = np.arange(record["cutoff"], record["cutoff"] + batch["horizon"])
            table = pd.DataFrame(
                {
                    "time_index": horizon,
                    "actual": actual,
                    "rwkv": predictions[index],
                    "naive": baseline,
                }
            )
            if prophet_prediction is not None:
                table["prophet"] = prophet_prediction
            for q, level in enumerate(model.config.quantiles):
                table[f"q{level:g}"] = output.quantiles[0, index, :, q].cpu().numpy()
            stem = f"{record['id']}_target_{index}"
            table.to_csv(directory / f"{stem}.csv", index=False)
            write_json(
                directory / "metrics_progress.json",
                {
                    "completed": len(metrics),
                    "expected": len(evaluation_records),
                    "metrics": metrics,
                },
            )
            if evaluation_index not in plot_indices:
                continue
            with timed("plot forecast/actual/baseline"):
                fig, axis = plt.subplots(
                    figsize=(config["plot"]["width"], config["plot"]["height"])
                )
                plot_context = context[
                    -config["plot"].get("context_points", len(context)) :
                ]
                axis.plot(
                    np.arange(record["cutoff"] - len(plot_context), record["cutoff"]),
                    plot_context,
                    color="gray",
                    label="Context",
                )
                axis.plot(horizon, actual, color="black", label="Actual held-out")
                axis.plot(horizon, predictions[index], label=f"RWKV (RMSE {rmse:.3g})")
                axis.plot(
                    horizon,
                    baseline,
                    label=f"{baseline_kind} (RMSE {baseline_rmse:.3g})",
                )
                if prophet_prediction is not None:
                    axis.plot(
                        horizon,
                        prophet_prediction,
                        label=f"Prophet (RMSE {prophet_rmse:.3g})",
                    )
                axis.fill_between(
                    horizon,
                    table[f"q{model.config.quantiles[0]:g}"],
                    table[f"q{model.config.quantiles[-1]:g}"],
                    alpha=0.15,
                    label="RWKV quantile range",
                )
                axis.axvline(record["cutoff"], linestyle="--", color="gray")
                axis.set(
                    title=f"{record['id']} / {target} (smoke training)",
                    xlabel="Row index",
                    ylabel=target,
                )
                axis.legend()
                fig.tight_layout()
                fig.savefig(directory / f"{stem}.png", dpi=config["plot"]["dpi"])
                plt.close(fig)
            logger.info("Saved forecast CSV and PNG: {}", directory / stem)
    if not metrics:
        raise ValueError("No evaluation windows")
    write_json(
        directory / "metrics.json",
        {
            "config": config,
            "metrics": metrics,
            "note": "Same held-out horizons and context; RWKV may use covariates, Last observation carried forward is univariate. Smoke accuracy only.",
        },
    )
    summary = {"evaluated_targets": len(metrics), "models": {}}
    for method in ("rwkv", "naive", "prophet"):
        scores = [
            r[f"{method}_rmse"] for r in metrics if r[f"{method}_rmse"] is not None
        ]
        scaled = [
            r[f"{method}_scaled_rmse"]
            for r in metrics
            if r[f"{method}_scaled_rmse"] is not None
        ]
        if scores:
            summary["models"][method] = {
                "count": len(scores),
                "mean_rmse": float(np.mean(scores)),
                "median_rmse": float(np.median(scores)),
                "mean_scaled_rmse": float(np.mean(scaled)),
            }
    summary["rwkv_wins_vs_naive"] = sum(
        r["rwkv_rmse"] < r["naive_rmse"] for r in metrics
    )
    summary["rwkv_wins_vs_prophet"] = sum(
        r["prophet_rmse"] is not None and r["rwkv_rmse"] < r["prophet_rmse"]
        for r in metrics
    )
    write_json(directory / "summary.json", summary)
    logger.info("SUMMARY {}", summary)
    logger.success("Forecast complete: {} target evaluations", len(metrics))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default=str(ROOT / "configs/config.json"))
    args = parser.parse_args()
    config = load_config(args.config)
    with run_context(config, "forecast"):
        if launch_wsl(__file__, args.config):
            return
        if not torch.cuda.is_available() or not torch.cuda.is_bf16_supported():
            raise RuntimeError("Use the configured WSL CUDA bf16 runtime")
        with timed("FORECAST stage", cuda=True):
            forecast(config)


if __name__ == "__main__":
    main()

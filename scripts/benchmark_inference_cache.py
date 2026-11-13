"""Replay the saved holdout requests without overwriting production predictions."""
# ruff: noqa: E402

import importlib.util
import sys
from pathlib import Path
from time import perf_counter

import numpy as np
import pandas as pd
import torch
from loguru import logger

ROOT = Path(__file__).resolve().parents[1]
for stage in ("03_transforming_input_data", "05_model_training", "06_holdout_test"):
    sys.path.insert(0, str(ROOT / "src" / stage))
from fast_inference import FastInference
from inference import forecast_contexts
from performance import configure_cuda_memory
from train import RWKVForecaster
from transform import load_config, run_context, write_json
from variate_attention import configure_variate_execution

OUTPUT = ROOT / "docs/audits/2026-11-03/inference_cache"


def summary(times):
    return {
        "requests": len(times),
        "median_ms": float(np.median(times)),
        "p95_ms": float(np.percentile(times, 95)),
        "mean_ms": float(np.mean(times)),
        "total_seconds": float(sum(times) / 1000),
    }


@torch.inference_mode()
def main():
    config = load_config()
    torch.set_num_threads(config["training"].get("cpu_threads", 4))
    configure_cuda_memory(config["training"])
    saved_dir = ROOT / "results/model/holdout_test"
    saved = pd.read_parquet(saved_dir / "test_predictions.parquet")
    latencies = pd.read_parquet(saved_dir / "request_latencies.parquet")
    requests = latencies[
        (latencies.model == "rwkv")
        & (latencies.phase == "scored")
        & (latencies.repeat == 0)
    ]
    spec = importlib.util.spec_from_file_location(
        "baseline_fast_inference", OUTPUT / "baseline_fast_inference.py"
    )
    baseline = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(baseline)
    with run_context(config, "inference_cache_benchmark"):
        model = RWKVForecaster.load(ROOT / "results/model/training/model.pt").eval()
        configure_variate_execution(model, config["training"], ROOT)
        prepared_requests = []
        frames = {}
        for row in requests.itertuples():
            if row.source not in frames:
                frames[row.source] = pd.read_parquet(ROOT / row.source)
            frame = frames[row.source]
            # Current trained policy is outcome plus all non-lag covariates.
            targets = config["data"]["targets"]
            covariates = [
                c
                for c in frame
                if c not in targets + [config["data"]["time_column"]]
                and "lag" not in c.lower()
            ]
            history = frame.iloc[
                max(0, row.origin - model.config.context_length) : row.origin
            ]
            context = {
                "context": torch.tensor(
                    history[targets].to_numpy(dtype=np.float32).T.copy()
                )[None],
                "past_covariates": torch.tensor(
                    history[covariates].to_numpy(dtype=np.float32).T.copy()
                )[None]
                if covariates
                else None,
            }
            reference = (
                saved[(saved.source == row.source) & (saved.origin == row.origin)]
                .sort_values("time_index")
                .rwkv.to_numpy()
            )
            prepared_requests.append((row, context, reference))
        rows, results = [], {}
        for mode in ("original_cache_8", "bounded_cache_256"):
            engine = (
                baseline.FastInference(model, max_graphs=8)
                if mode == "original_cache_8"
                else FastInference(model, max_graphs=256, graph_memory_gib=1.5)
            )
            model.fast_inference = engine
            for phase in ("capture_inclusive", "warm"):
                times = []
                maximum_error = 0.0
                for index, (row, context, reference) in enumerate(prepared_requests):
                    torch.cuda.synchronize()
                    started = perf_counter()
                    prediction = list(
                        forecast_contexts(
                            model,
                            [context],
                            1,
                            prediction_patches=row.horizon // model.config.patch_size,
                        )
                    )[0]
                    torch.cuda.synchronize()
                    milliseconds = (perf_counter() - started) * 1000
                    times.append(milliseconds)
                    point = prediction[..., model.config.quantiles.index(0.5)].numpy()[
                        0, : len(reference)
                    ]
                    error = float(np.max(np.abs(point - reference)))
                    maximum_error = max(maximum_error, error)
                    if error != 0:
                        raise RuntimeError(
                            f"Forecast changed: {mode} {row.source} {row.origin} max_error={error}"
                        )
                    rows.append(
                        {
                            "mode": mode,
                            "phase": phase,
                            "source": row.source,
                            "origin": row.origin,
                            "horizon": row.horizon,
                            "context_points": row.context_points,
                            "covariates": row.covariates,
                            "milliseconds": milliseconds,
                        }
                    )
                    if (index + 1) % 250 == 0:
                        logger.info(
                            "{} {} {}/{} median_ms={:.3f}",
                            mode,
                            phase,
                            index + 1,
                            len(prepared_requests),
                            np.median(times),
                        )
                results[f"{mode}_{phase}"] = {
                    **summary(times),
                    "max_abs_prediction_difference": maximum_error,
                    "graph_cache": engine.graphs.summary()
                    if hasattr(engine.graphs, "summary")
                    else {
                        "cached_profiles": len(engine.graphs.entries),
                        "max_profiles": 8,
                    },
                }
                write_json(OUTPUT / "benchmark.json", results)
                pd.DataFrame(rows).to_parquet(OUTPUT / "requests.parquet", index=False)
            engine.clear()
            model.fast_inference = None
            del engine
            torch.cuda.synchronize()
            torch.cuda.empty_cache()
        write_json(OUTPUT / "benchmark.json", results)


if __name__ == "__main__":
    main()

"""Matched GPU latency matrix and instrumented replay of historical RWKV tails."""
# ruff: noqa: E402

import argparse
import copy
import gc
import gzip
import hashlib
import inspect
import itertools
import json
import platform
import sys
from dataclasses import replace
from datetime import datetime, timezone
from pathlib import Path
from time import perf_counter, process_time

import numpy as np
import pandas as pd
import torch
from loguru import logger

ROOT = Path(__file__).resolve().parents[1]
for stage in ("03_transforming_input_data", "05_model_training", "06_holdout_test"):
    sys.path.insert(0, str(ROOT / "src" / stage))
from benchmarks import FoundationBenchmarks
from inference import forecast_contexts
from performance import configure_cuda_memory
from train import RWKVForecaster
from transform import load_config, run_context, write_json
from variate_attention import configure_variate_execution


def compiler_count():
    from torch._dynamo.utils import counters

    return int(counters["stats"]["unique_graphs"])


def depth_variant(model, layers):
    """A parameter-sharing inference view; leave the original model intact."""
    reduced = copy.copy(model)
    reduced._modules = model._modules.copy()
    reduced._buffers = model._buffers.copy()
    reduced.blocks = torch.nn.ModuleList(list(model.blocks)[:layers])
    reduced.variate = torch.nn.ModuleList(list(model.variate)[:layers])
    reduced.config = replace(model.config, layers=layers)
    return reduced


class Meter:
    def __init__(self):
        self.rows = []
        self.gc_start = {}
        self.gc_ms = 0.0
        self.gc_events = 0

    def callback(self, phase, info):
        generation = info["generation"]
        if phase == "start":
            self.gc_start[generation] = perf_counter()
        else:
            started = self.gc_start.pop(generation, None)
            if started is not None:
                self.gc_ms += (perf_counter() - started) * 1000
                self.gc_events += 1

    def call(self, function, **labels):
        torch.cuda.synchronize()
        gc_ms, gc_events = self.gc_ms, self.gc_events
        compiled = compiler_count()
        started_cpu = process_time()
        started = perf_counter()
        value = function()
        torch.cuda.synchronize()
        row = dict(
            labels,
            milliseconds=(perf_counter() - started) * 1000,
            process_cpu_ms=(process_time() - started_cpu) * 1000,
            gc_ms=self.gc_ms - gc_ms,
            gc_events=self.gc_events - gc_events,
            new_compiled_graphs=compiler_count() - compiled,
            allocated_gib=torch.cuda.memory_allocated() / 2**30,
            reserved_gib=torch.cuda.memory_reserved() / 2**30,
        )
        self.rows.append(row)
        if not np.isfinite(value).all():
            raise FloatingPointError(f"Nonfinite prediction: {labels}")
        return value


@torch.inference_mode()
def rwkv_prediction(model, context, horizon):
    """Use the holdout patch adapter, then feed medians back for longer horizons."""
    history = context["context"]
    covariates = context.get("past_covariates")
    parts = []
    patch = model.config.patch_size
    for _ in range((horizon + patch - 1) // patch):
        result = list(
            forecast_contexts(
                model, [{"context": history, "past_covariates": covariates}], 1
            )
        )[0]
        parts.append(result)
        history = torch.cat(
            [history, result[..., model.config.quantiles.index(0.5)][None]], -1
        )[..., -model.config.context_length :]
        if covariates is not None:
            missing = covariates.new_full((*covariates.shape[:-1], patch), float("nan"))
            covariates = torch.cat([covariates, missing], -1)[
                ..., -model.config.context_length :
            ]
    return torch.cat(parts, -2)[:, :horizon].numpy()


def foundation_prediction(suite, name, context, horizon, levels, precision):
    # Every provider returns the same target x horizon x quantiles CPU contract.
    autocast = torch.autocast("cuda", dtype=torch.bfloat16, enabled=precision == "bf16")
    with torch.inference_mode(), autocast:
        provider = "timestral" if name.startswith("timestral") else name
        model = suite.models[provider]
        if provider == "chronos":
            item = {"target": context["context"][0].numpy()}
            if context.get("past_covariates") is not None:
                item["past_covariates"] = {
                    f"covariate_{i}": row.numpy()
                    for i, row in enumerate(context["past_covariates"][0])
                }
            quantiles, _ = model.predict_quantiles(
                [item],
                prediction_length=horizon,
                quantile_levels=levels,
                batch_size=32,
                context_length=2048,
                cross_learning=False,
            )
            return quantiles[0].float().cpu().numpy()
        result = model.forecast(
            context["context"],
            prediction_length=horizon,
            past_covariates=context.get("past_covariates"),
            symmetric_averaging=name != "timestral",
        )
        # Check the actual library distribution contract before accepting it.
        quantiles = result.quantiles
        indices = [result.quantile_levels.index(level) for level in levels]
        return quantiles[0, ..., indices].float().cpu().numpy()


def history(length, covariates, seed):
    generator = np.random.default_rng(seed)
    t = np.arange(length)
    target = 0.01 * t + np.sin(t * 2 * np.pi / 31) + generator.normal(0, 0.1, length)
    past = (
        np.stack(
            [
                np.cos(t * 2 * np.pi / (13 + i * 7)) + generator.normal(0, 0.1, length)
                for i in range(covariates)
            ]
        )
        if covariates
        else None
    )
    return {
        "context": torch.tensor(target, dtype=torch.float32)[None, None],
        "past_covariates": None
        if past is None
        else torch.tensor(past, dtype=torch.float32)[None],
    }


def historical_analysis(output):
    directory = ROOT / "results/447m_model/holdout_test"
    source = directory / "request_latencies.parquet"
    frame = pd.read_parquet(source)
    rwkv = frame[(frame.model == "rwkv") & (frame.phase == "scored")].copy()
    median = float(rwkv.seconds.median())
    worst = rwkv.nlargest(10, "seconds")
    worst.to_csv(output / "historical_worst_requests.csv", index=False)
    report = {
        "source": str(source),
        "sha256": hashlib.sha256(source.read_bytes()).hexdigest(),
        "requests": len(rwkv),
        "median_ms": median * 1000,
        "max_ms": float(rwkv.seconds.max() * 1000),
        "max_over_median": float(rwkv.seconds.max() / median),
        "p99_ms": float(rwkv.seconds.quantile(0.99) * 1000),
        "over_ten_times_median": int((rwkv.seconds > 10 * median).sum()),
        "outlier": worst.iloc[0].to_dict(),
        "historical_runtime_metadata": json.loads(
            (directory / "test_metrics.json").read_text()
        )["benchmark_metadata"],
    }
    write_json(output / "historical_analysis.json", report)
    return report


def summarize(rows, output):
    frame = pd.DataFrame(rows)
    frame.to_parquet(output / "requests.parquet", index=False)
    frame.to_csv(output / "requests.csv", index=False)
    scored = frame[frame.phase == "scored"]
    grouped = scored.groupby(
        ["model", "precision", "context", "horizon", "covariates"], dropna=False
    )
    summary = grouped.milliseconds.agg(
        count="size", mean_ms="mean", median_ms="median", min_ms="min", max_ms="max"
    )
    summary["p95_ms"] = grouped.milliseconds.quantile(0.95)
    summary["p99_ms"] = grouped.milliseconds.quantile(0.99)
    summary.reset_index().to_csv(output / "summary.csv", index=False)


def tail_diagnostics(model, metadata, historical, meter, options, output):
    """Replay the actual slowest request and separate plausible host-side causes."""
    from holdout_test import populate_test_schema

    request = historical["outlier"]
    record = {"source": request["source"]}
    populate_test_schema([record], metadata)
    frame = pd.read_parquet(ROOT / record["source"])
    origin = int(request["origin"])
    frame = frame.iloc[max(0, origin - model.config.context_length) : origin]
    context = {
        "context": torch.tensor(
            frame[record["targets"]].to_numpy(dtype=np.float32).T.copy()
        )[None],
        "past_covariates": torch.tensor(
            frame[record["roles"]["past"]].to_numpy(dtype=np.float32).T.copy()
        )[None],
    }
    labels = dict(
        model="rwkv_447m",
        precision="bf16",
        context=len(frame),
        horizon=16,
        covariates=len(record["roles"]["past"]),
    )
    for repeat in range(5):
        meter.call(
            lambda: rwkv_prediction(model, context, 16),
            phase="diagnostic_warmup",
            repeat=repeat,
            **labels,
        )
    for phase, repeats in (
        ("diagnostic_gc_enabled", options["diagnostic_repeats"]),
        ("diagnostic_gc_disabled", options["diagnostic_repeats"]),
        ("diagnostic_forced_gc", 5),
    ):
        gc.collect()
        if phase == "diagnostic_gc_disabled":
            gc.disable()
        try:
            for repeat in range(repeats):

                def predict():
                    if phase == "diagnostic_forced_gc":
                        gc.collect(2)
                    return rwkv_prediction(model, context, 16)

                meter.call(predict, phase=phase, repeat=repeat, **labels)
        finally:
            gc.enable()
    # Cold graph-cache replay (binaries remain cached on disk).
    torch._dynamo.reset()
    for repeat in range(3):
        meter.call(
            lambda: rwkv_prediction(model, context, 16),
            phase="diagnostic_graph_cache_reset",
            repeat=repeat,
            **labels,
        )
    with torch.profiler.profile(
        activities=[
            torch.profiler.ProfilerActivity.CPU,
            torch.profiler.ProfilerActivity.CUDA,
        ],
        record_shapes=True,
    ) as profile:
        rwkv_prediction(model, context, 16)
        torch.cuda.synchronize()
    trace_path = ROOT / ".cache/time_benchmark_traces/rwkv_warm_trace.json"
    trace_path.parent.mkdir(parents=True, exist_ok=True)
    profile.export_chrome_trace(str(trace_path))
    # Keep bulky profiler JSON out of IDE indexing; publish its compressed form.
    (output / "rwkv_warm_trace.json.gz").write_bytes(
        gzip.compress(trace_path.read_bytes())
    )
    (output / "rwkv_warm_profile.txt").write_text(
        profile.key_averages().table(sort_by="self_device_time_total", row_limit=35),
        encoding="utf-8",
    )
    diagnostics = pd.DataFrame(
        [row for row in meter.rows if row["phase"].startswith("diagnostic")]
    )
    diagnostics.to_csv(output / "tail_replays.csv", index=False)
    logger.info("Exact historical outlier replay and GC/compiler diagnostics complete")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default=str(ROOT / "configs/time_benchmark.json"))
    parser.add_argument("--probe", action="store_true")
    parser.add_argument(
        "--paired",
        action="store_true",
        help="Interleave full/reduced RWKV in the same timed matrix",
    )
    args = parser.parse_args()
    options = json.loads(Path(args.config).read_text())
    config = load_config()
    output = ROOT / options["output"]
    if args.paired:
        output = output / "paired"
    output.mkdir(parents=True, exist_ok=True)
    if not torch.cuda.is_available():
        raise RuntimeError(
            "Run in WSL CUDA environment; the vendored RWKV backend has no CPU path"
        )
    torch.set_num_threads(options["cpu_threads"])
    configure_cuda_memory(config["training"])
    torch.manual_seed(options["seed"])
    config["logging"]["level"] = "INFO"
    with run_context(config, "time_benchmark"):
        historical = historical_analysis(output)
        checkpoint = ROOT / config["paths"]["training"] / "model.pt"
        payload = torch.load(
            checkpoint, map_location="cpu", weights_only=True, mmap=True
        )
        model = RWKVForecaster.load(checkpoint, checkpoint=payload).eval()
        configure_variate_execution(model, config["training"], ROOT)
        config["holdout"]["benchmark_models"] = ["chronos", "timestral"]
        suite = FoundationBenchmarks(config, 2048)
        write_json(
            output / "timestral_api.json",
            {
                "forecast_signature": str(
                    inspect.signature(suite.models["timestral"].forecast)
                ),
                "source_file": inspect.getfile(type(suite.models["timestral"])),
            },
        )
        metadata = {
            "started_utc": datetime.now(timezone.utc).isoformat(),
            "python": sys.version,
            "platform": platform.platform(),
            "torch": torch.__version__,
            "cuda": torch.version.cuda,
            "gpu": torch.cuda.get_device_name(),
            "cpu_threads": torch.get_num_threads(),
            "config": options,
            "models": suite.metadata,
            "checkpoint": str(checkpoint),
            "rwkv_parameters": sum(p.numel() for p in model.parameters()),
            "paired_depth_comparison": args.paired,
            "contract": "batch=1, one target, identical CPU float32 historical payload, same GPU, nine quantiles on CPU, no future covariates",
            "scope": "wall clock including adapter preprocessing, transfers and CPU outputs; GPU synchronized; excludes loading and payload generation",
            "rwkv_rollout": "16-point nonoverlapping patches, median feedback, unknown future covariates filled with NaN; capped at 2048; no recurrent state caching",
            "cpu_benchmark": "not comparable: vendored RWKV kernels require CUDA/bf16; no substitute architecture was benchmarked",
        }
        for name in suite.models:
            suite.models[name].model.to(dtype=torch.bfloat16)
            suite.metadata[name]["native_dtype"] = suite.metadata[name]["dtype"]
            suite.metadata[name]["dtype"] = str(
                next(suite.models[name].model.parameters()).dtype
            )
            suite.metadata[name]["quantile_levels"] = (
                list(suite.models[name].quantile_levels)
                if name == "timestral"
                else options["quantiles"]
            )
        metadata["timestral_modes"] = {
            "timestral": "symmetric_averaging=False; single non-mirrored pass per decode block",
            "timestral_default": "symmetric_averaging=True; original library default with mirrored pass",
        }
        metadata["tf32_allowed"] = torch.backends.cuda.matmul.allow_tf32
        probe = history(32, 3, options["seed"])
        for name in suite.models:
            value = foundation_prediction(
                suite, name, probe, 16, options["quantiles"], "bf16"
            )
            logger.info(
                "Probe {} shape={} parameters={} dtype={}",
                name,
                value.shape,
                sum(p.numel() for p in suite.models[name].model.parameters()),
                next(suite.models[name].model.parameters()).dtype,
            )
        write_json(output / "environment.json", metadata)
        if args.probe:
            return
        reduced = depth_variant(model, options["reduced_layers"])
        metadata["reduced_parameters"] = sum(p.numel() for p in reduced.parameters())
        metadata["reduced_model"] = (
            "first 17 existing layers, same width; parameter-sharing depth view, no retraining"
        )
        metadata["chronos_forecasting_config"] = vars(
            suite.models["chronos"].model.chronos_config
        )
        meter = Meter()
        gc.callbacks.append(meter.callback)
        try:
            cases = list(
                itertools.product(
                    options["contexts"], options["horizons"], options["covariates"]
                )
            )

            def run_matrix(names, precision, depth_name):
                functions = (
                    {depth_name: lambda c, h: rwkv_prediction(model, c, h)}
                    if depth_name
                    else {}
                )
                if depth_name == "paired":
                    functions = {
                        "rwkv_447m": lambda c, h: rwkv_prediction(model, c, h),
                        "rwkv_307m": lambda c, h: rwkv_prediction(reduced, c, h),
                    }
                functions.update(
                    {
                        name: lambda c, h, name=name: foundation_prediction(
                            suite, name, c, h, options["quantiles"], precision
                        )
                        for name in names
                    }
                )
                for case_index, (length, horizon, covariates) in enumerate(cases):
                    context = history(length, covariates, options["seed"] + case_index)
                    original_context = {
                        key: None if value is None else value.clone()
                        for key, value in context.items()
                    }
                    common = dict(
                        precision=precision,
                        context=length,
                        horizon=horizon,
                        covariates=covariates,
                    )
                    for name, fn in functions.items():
                        for repeat in range(options["warmups"]):
                            result = meter.call(
                                lambda: fn(context, horizon),
                                model=name,
                                phase="warmup",
                                repeat=repeat,
                                **common,
                            )
                            if result.shape != (1, horizon, len(options["quantiles"])):
                                raise ValueError(
                                    f"Contract shape failed: {name} {result.shape}"
                                )
                    # Interleave measured models, rotating their order each repeat.
                    ordered = list(functions)
                    for repeat in range(options["repeats"]):
                        shift = (repeat + case_index) % len(ordered)
                        for name in ordered[shift:] + ordered[:shift]:
                            meter.call(
                                lambda: functions[name](context, horizon),
                                model=name,
                                phase="scored",
                                repeat=repeat,
                                **common,
                            )
                    summarize(meter.rows, output)
                    for key, value in original_context.items():
                        if value is not None:
                            torch.testing.assert_close(
                                context[key], value, equal_nan=True, rtol=0, atol=0
                            )
                    logger.info(
                        "TIME matrix={} precision={} case {}/{} context={} horizon={} covariates={}",
                        list(functions),
                        precision,
                        case_index + 1,
                        len(cases),
                        length,
                        horizon,
                        covariates,
                    )

            run_matrix(
                ["chronos", "timestral", "timestral_default"],
                "bf16",
                "paired" if args.paired else "rwkv_447m",
            )
            tail_diagnostics(model, payload, historical, meter, options, output)
            del payload
            # Benchmark a depth ablation, not a claim about a trained 300M model's accuracy.
            if not args.paired:
                model = reduced
                run_matrix([], "bf16", "rwkv_307m")
            for name in suite.models:
                suite.models[name].model.float()
            run_matrix(["chronos", "timestral", "timestral_default"], "fp32", None)
        finally:
            gc.callbacks.remove(meter.callback)
            summarize(meter.rows, output)
            write_json(output / "environment.json", metadata)


if __name__ == "__main__":
    main()

"""Synchronized end-to-end, single-origin request latency with an audit trail."""

import platform
from time import perf_counter

import numpy as np
import pandas as pd
import torch


class RequestTimer:
    """Time actual adapters, including preparation and CPU result transfer."""

    def __init__(self, config):
        options = config.get("holdout", {}).get("latency", {})
        self.warmup_requests = options.get("warmup_requests", 1)
        self.repeats = options.get("repeats", 1)
        for name, minimum in (("warmup_requests", 0), ("repeats", 1)):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, int) or value < minimum:
                raise ValueError(
                    f"holdout.latency.{name} must be an integer >= {minimum}"
                )
        self.rows = []
        self.warmed = set()

    def measure(self, method, function, *, device, request):
        """Warm each adapter once per run; retain every cold and scored duration."""
        if method not in self.warmed:
            for repeat in range(self.warmup_requests):
                self._call(method, function, device, request, "warmup", repeat)
            self.warmed.add(method)
        result = None
        for repeat in range(self.repeats):
            output = self._call(method, function, device, request, "scored", repeat)
            if repeat == 0:
                result = output  # repetitions never change the forecast being scored
        return result

    def _call(self, method, function, device, request, phase, repeat):
        device = torch.device(device)
        if device.type == "cuda":
            torch.cuda.synchronize(device)
        started = perf_counter()
        result = function()
        if device.type == "cuda":
            torch.cuda.synchronize(device)
        elapsed = perf_counter() - started
        self.rows.append(
            dict(
                request,
                model=method,
                phase=phase,
                repeat=repeat,
                seconds=elapsed,
                device=str(device),
            )
        )
        return result

    def summary(self):
        frame = pd.DataFrame(self.rows)
        result = {}
        if frame.empty:
            return result
        for method, group in frame.groupby("model", sort=False):
            scored = group[group.phase == "scored"]
            values = scored.seconds.to_numpy() * 1000
            result[method] = {
                "requests": int(len(scored)),
                "forecast_origins": int(len(scored[scored.repeat == 0])),
                "mean_ms": float(values.mean()),
                "median_ms": float(np.median(values)),
                "p95_ms": float(np.percentile(values, 95)),
                "min_ms": float(values.min()),
                "max_ms": float(values.max()),
                "total_seconds": float(scored.seconds.sum()),
                "warmup_requests": int((group.phase == "warmup").sum()),
                "warmup_seconds": float(
                    group.loc[group.phase == "warmup", "seconds"].sum()
                ),
            }
            if "context_points" in scored:
                buckets = {}
                for lower, upper in (
                    (1, 128),
                    (129, 512),
                    (513, 1024),
                    (1025, 2048),
                    (2049, None),
                ):
                    selected = scored[scored.context_points >= lower]
                    if upper is not None:
                        selected = selected[selected.context_points <= upper]
                    if selected.empty:
                        continue
                    times = selected.seconds.to_numpy() * 1000
                    buckets[f"{lower}-{upper or 'plus'}"] = {
                        "requests": len(selected),
                        "median_ms": float(np.median(times)),
                        "p95_ms": float(np.percentile(times, 95)),
                    }
                result[method]["by_context_points"] = buckets
            if "horizon" in scored:
                result[method]["by_horizon"] = {
                    str(int(horizon)): {
                        "requests": len(group),
                        "median_ms": float(np.median(group.seconds.to_numpy() * 1000)),
                        "p95_ms": float(
                            np.percentile(group.seconds.to_numpy() * 1000, 95)
                        ),
                    }
                    for horizon, group in scored.groupby("horizon")
                }
        return result

    def paired_comparison(self):
        """Compare equal origins, rather than ratios of unpaired percentile summaries."""
        frame = pd.DataFrame(self.rows)
        if frame.empty:
            return {}
        scored = frame[frame.phase == "scored"]
        index = ["source", "origin", "repeat"]
        if "table_index" in scored:
            index.insert(0, "table_index")
        paired = scored.pivot(index=index, columns="model", values="seconds")
        result = {}
        for name in paired.columns:
            if name == "rwkv":
                continue
            pairs = paired[["rwkv", name]].dropna()
            pairs = pairs[pairs[name] > 0]
            result[name] = {
                "paired_requests": len(pairs),
                "median_rwkv_over_baseline_latency": float(
                    (pairs.rwkv / pairs[name]).median()
                )
                if len(pairs)
                else None,
                "interpretation": "ratio > 1 means RWKV was slower on the same request",
            }
        return result

    def metadata(self):
        return {
            "unit": "one table at one origin, all targets, the same requested horizon for every model, CPU output",
            "batch_size": 1,
            "warmup_requests_per_model": self.warmup_requests,
            "repeats_per_origin": self.repeats,
            "scope": "resident CPU history to CPU forecast; includes tokenization, normalization, device transfers and adapter execution; Prophet includes fitting and covariate forecasts",
            "excluded": "model loading/downloads, parquet reads, shared history extraction, scoring, plotting and writing results",
            "synchronization": "CUDA device synchronized before and after each GPU request",
            "execution_order": "serial model order rotates by table and origin; order_index retained per request",
            "warmup_policy": "first origin only per model; retained separately; later shape compilation remains included in scored latency",
            "interpretation": "serial local wall-clock latency, not concurrent serving throughput; precision and model outputs differ as recorded in benchmark_metadata",
            "torch_version": torch.__version__,
            "platform": platform.platform(),
            "cpu_threads": torch.get_num_threads(),
            "cuda_version": torch.version.cuda,
            "gpu": torch.cuda.get_device_name() if torch.cuda.is_available() else None,
        }

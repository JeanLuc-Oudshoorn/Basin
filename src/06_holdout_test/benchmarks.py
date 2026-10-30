"""Configurable, zero-shot foundation model adapters for identical holdout origins."""

import hashlib
import math

import numpy as np
import torch
from loguru import logger
from transform import ROOT

DEFAULT_MODELS = ["prophet", "naive", "chronos"]
MODEL_DEFAULTS = {
    "chronos": {
        "model_id": "amazon/chronos-2",
        "revision": "29ec3766d36d6f73f0696f85560a422f50e8498c",
        "batch_size": 32,
    },
    "timestral": {
        "model_id": "mistralai/Timestral-209M",
        "revision": "9874d252ab0aae16f7f7c999f98f98d6ad7d8365",
    },
}


def benchmark_names(config):
    names = config.get("holdout", {}).get("benchmark_models", DEFAULT_MODELS)
    if (
        not isinstance(names, list)
        or any(
            not isinstance(n, str)
            or n not in {"prophet", "naive", "chronos", "timestral"}
            for n in names
        )
        or len(set(names)) != len(names)
    ):
        raise ValueError(
            "holdout.benchmark_models must be a unique list of prophet, naive, chronos, timestral; RWKV is always included"
        )
    return list(names)


def sample_holdout(records, config):
    """Seeded uniform hash ranking of whole test tables, independent of input order."""
    settings = config.get("holdout", {})
    percent = settings.get("sample_percent", 100.0)
    seed = settings.get("sample_seed", config["seed"])
    if (
        isinstance(percent, bool)
        or not isinstance(percent, (float, int))
        or not math.isfinite(percent)
        or not 0 < percent <= 100
    ):
        raise ValueError(
            "holdout.sample_percent must be greater than 0 and at most 100"
        )
    if isinstance(seed, bool) or not isinstance(seed, int):
        raise ValueError("holdout.sample_seed must be an integer")
    count = min(len(records), max(1, math.ceil(len(records) * percent / 100)))

    def key(record):
        return record.get("source_key", record["source"])

    ranked = sorted(
        records,
        key=lambda r: (hashlib.sha256(f"{seed}:{key(r)}".encode()).hexdigest(), key(r)),
    )
    selected = sorted(ranked[:count], key=key)
    audit = {
        "available_holdout_tables": len(records),
        "evaluated_tables": len(selected),
        "requested_sample_percent": percent,
        "sample_seed": seed,
        "selection": "seeded SHA-256 ranking; ceil percentage; at least one table",
        "selected_sources": [r["source"] for r in selected],
    }
    return selected, audit


def select_plot_tables(records, config):
    """Reproducible random examples from the tables evaluated in this run."""
    options = config.get("plot", {})
    count = options.get("max_plots", 10)
    seed = options.get("sample_seed", config["seed"])
    if isinstance(count, bool) or not isinstance(count, int) or count < 0:
        raise ValueError("plot.max_plots must be a nonnegative integer")
    if isinstance(seed, bool) or not isinstance(seed, int):
        raise ValueError("plot.sample_seed must be an integer")
    ranked = sorted(
        range(len(records)),
        key=lambda i: (
            hashlib.sha256(
                f"plots:{seed}:{records[i].get('source_key', records[i]['source'])}".encode()
            ).hexdigest(),
            records[i]["source"],
        ),
    )
    selected = sorted(ranked[:count])
    return set(selected), {
        "format": "plotly_html",
        "requested_tables": count,
        "selected_tables": len(selected),
        "sample_seed": seed,
        "selection": "seeded SHA-256 ranking within evaluated tables; independent of holdout sampling",
        "tables": [
            {
                "table_index": i,
                "source": records[i]["source"],
                "plots": [
                    f"test_table_{i:03d}_target_{j}.html"
                    for j in range(len(records[i]["targets"]))
                ],
            }
            for i in selected
        ],
    }


class FoundationBenchmarks:
    """Load each requested model once; never send future values or other table labels."""

    def __init__(self, config, context_limit):
        settings = config.get("holdout", {})
        self.names = benchmark_names(config)
        self.context_limit = context_limit
        self.models = {}
        self.metadata = {}
        self.options = {}
        cache = ROOT / settings.get("model_cache_dir", ".cache/huggingface")
        cache.mkdir(parents=True, exist_ok=True)
        device = settings.get(
            "foundation_device", "cuda" if torch.cuda.is_available() else "cpu"
        )
        for name in self.names:
            if name not in MODEL_DEFAULTS:
                self.metadata[name] = {
                    "inputs": "historical target only",
                    "point_forecast": "last finite value"
                    if name == "naive"
                    else "yhat",
                }
                if name == "prophet":
                    enabled = config["prophet"].get("use_covariates", True)
                    self.metadata[name]["use_covariates"] = enabled
                    self.metadata[name]["covariate_forecast"] = config["prophet"].get(
                        "covariate_forecast", "prophet"
                    )
                    if enabled:
                        self.metadata[name]["inputs"] = (
                            "historical target and past covariates; future regressor values forecast from history"
                        )
                continue
            options = dict(MODEL_DEFAULTS[name], **settings.get(name, {}))
            self.options[name] = options
            logger.info(
                "Loading benchmark {} model={} revision={} device={} cache={}",
                name,
                options["model_id"],
                options["revision"],
                device,
                cache,
            )
            if name == "chronos":
                from chronos import Chronos2Pipeline

                model = Chronos2Pipeline.from_pretrained(
                    options["model_id"],
                    revision=options["revision"],
                    device_map=device,
                    torch_dtype=torch.float32,
                    cache_dir=str(cache),
                    token=False,
                )
            else:
                try:
                    from timestral import TimestralForecaster
                except ImportError as error:
                    raise ImportError(
                        "Timestral requires: uv sync --inexact --extra benchmarks --extra validation"
                    ) from error
                model = TimestralForecaster.from_pretrained(
                    options["model_id"],
                    revision=options["revision"],
                    device=device,
                    max_context=context_limit,
                    cache_dir=str(cache),
                    token=False,
                )
            self.models[name] = model
            parameters = sum(p.numel() for p in model.model.parameters())
            self.metadata[name] = {
                "model_id": options["model_id"],
                "revision": options["revision"],
                "parameters": parameters,
                "dtype": str(next(model.model.parameters()).dtype),
                "device": str(next(model.model.parameters()).device),
                "point_forecast": "median",
                "inputs": "historical targets and time-matched past covariates",
                "authentication": "anonymous token=False",
            }
            logger.info("Loaded benchmark {} parameters={:,}", name, parameters)

    @torch.inference_mode()
    def predict(self, contexts, horizon):
        predictions = [{} for _ in contexts]
        for name, model in self.models.items():
            if name == "chronos":
                inputs = []
                for context in contexts:
                    item = {"target": context["context"][0].numpy()}
                    if context.get("past_covariates") is not None:
                        item["past_covariates"] = {
                            f"covariate_{i}": row.numpy()
                            for i, row in enumerate(context["past_covariates"][0])
                        }
                    inputs.append(item)
                _, points = model.predict_quantiles(
                    inputs,
                    prediction_length=horizon,
                    quantile_levels=[0.5],
                    batch_size=self.options[name]["batch_size"],
                    context_length=self.context_limit,
                    cross_learning=False,
                )
            else:
                points = [
                    model.forecast(
                        c["context"],
                        prediction_length=horizon,
                        past_covariates=c.get("past_covariates"),
                    ).median[0]
                    for c in contexts
                ]
            if len(points) != len(contexts):
                raise ValueError(
                    f"{name} returned a different number of forecast contexts"
                )
            for context, point, result in zip(contexts, points, predictions):
                point = torch.as_tensor(point).float().cpu().numpy()
                expected = (context["context"].shape[1], horizon)
                if point.shape != expected or not np.isfinite(point).all():
                    raise ValueError(
                        f"Invalid {name} forecast: shape={point.shape}, expected={expected}"
                    )
                result[name] = point
        return predictions

"""Plan profitable graph shapes using only request metadata, never future values."""

import math
from collections import Counter

from fast_inference import FastInference
from horizon_policy import evaluation_weights
from training_protocol import sampled_origins


def profile_counts(records, model_config, config, device):
    counts = Counter()
    cap = min(model_config.context_length, config["data"]["context_length"])
    patch, width = model_config.patch_size, model_config.width
    settings = config.get("holdout", {})
    for record in records:
        targets = len(record["targets"])
        variables = targets + len(record["roles"]["past"])
        length = record["source_rows"]
        for origin in sampled_origins(
            length,
            patch,
            settings.get("percentiles", [1, 25, 50, 75, 99]),
            prediction_patches=len(evaluation_weights(config)),
        ):
            horizon = len(evaluation_weights(config))
            for start in range(0, horizon, model_config.output_patches):
                patches = min(horizon - start, model_config.output_patches)
                context = min(
                    min(origin, cap) + start * patch, model_config.context_length
                )
                padded = math.ceil(math.ceil(context / patch) / 16) * 16
                key = ((1, variables, padded, width), targets, patches, str(device))
                counts[key] += settings.get("latency", {}).get("repeats", 1)
    return counts


def configure_holdout_graph_profiles(model, records, config):
    engine = getattr(model, "fast_inference", None)
    if not isinstance(engine, FastInference):
        return None
    minimum = config.get("holdout", {}).get("graph_min_profile_requests", 8)
    if isinstance(minimum, bool) or not isinstance(minimum, int) or minimum < 1:
        raise ValueError(
            "holdout.graph_min_profile_requests must be a positive integer"
        )
    counts = profile_counts(
        records, model.config, config, next(model.parameters()).device
    )
    engine.graphs.allowed_keys = {
        key for key, count in counts.items() if count >= minimum
    }
    return {
        "minimum_requests": minimum,
        "unique_profiles": len(counts),
        "admitted_profiles": len(engine.graphs.allowed_keys),
        "planned_requests": sum(counts.values()),
        "planned_replay_requests": sum(
            counts[key] for key in engine.graphs.allowed_keys
        ),
        "policy": "capture only recurring profiles; no hidden warmup or changed latency boundary; rare profiles use forward-only execution",
    }

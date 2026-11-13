"""Four-token evaluation weights and legacy horizon-draw compatibility helpers."""

import hashlib
import math


def evaluation_weights(config):
    """Evaluate every lead, using the training objective's weights, never draws."""
    weights = config.get("model", {}).get(
        "horizon_loss_weights", [0.8, 0.16, 0.03, 0.01]
    )
    if (
        not isinstance(weights, (list, tuple))
        or len(weights) != 4
        or any(
            isinstance(w, bool)
            or not isinstance(w, (int, float))
            or not math.isfinite(w)
            or w <= 0
            for w in weights
        )
        or not math.isclose(sum(weights), 1.0, abs_tol=1e-8)
    ):
        raise ValueError(
            "model.horizon_loss_weights must contain four positive weights summing to one"
        )
    return list(weights)


def horizon_probabilities(config):
    probabilities = config.get("holdout", {}).get("horizon_probabilities", [1.0])
    if (
        not isinstance(probabilities, (list, tuple))
        or not probabilities
        or any(
            isinstance(p, bool)
            or not isinstance(p, (float, int))
            or not math.isfinite(p)
            or p < 0
            for p in probabilities
        )
        or not math.isclose(sum(probabilities), 1.0, abs_tol=1e-8)
    ):
        raise ValueError(
            "holdout.horizon_probabilities must be finite, nonnegative and sum to one"
        )
    return probabilities


def sampled_horizon(record, origin, length, patch_size, config):
    probabilities = horizon_probabilities(config)
    available = max(1, min(len(probabilities), (length - origin) // patch_size))
    weights = probabilities[:available]
    total = sum(weights)
    if total <= 0:
        raise ValueError(
            "No positive holdout horizon probability fits available observations"
        )
    settings = config.get("holdout", {})
    seed = settings.get(
        "horizon_seed", settings.get("sample_seed", config.get("seed", 7))
    )
    key = record.get("source_key", record["source"])
    digest = hashlib.sha256(
        f"{seed}:{key}:{origin}:forecast_horizon_v1".encode()
    ).digest()
    draw = int.from_bytes(digest[:8], "big") / 2**64 * total
    cumulative = 0.0
    for tokens, weight in enumerate(weights, 1):
        cumulative += weight
        if draw < cumulative:
            return tokens
    return available

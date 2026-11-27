"""Describe a generated table and decide which annealing categories it belongs to.

A category names a kind of series the model is known to forecast less well
than it could. Membership depends on the series as a whole, never on what
follows a particular forecast origin, so inside a selected table every
situation still ends the way the generator makes it end.
"""

import numpy as np

# The natural distribution: every table qualifies, so this share of the set is
# drawn exactly as the training corpus was.
STANDARD = "standard"

DEFAULT_THRESHOLDS = {
    # Share of observed steps that repeat the previous value, and the longest
    # such stretch. Stuck sensors, shutdowns, clipping and coarse quantisation.
    "flat": {"min_flat_share": 0.05, "min_longest_run": 16},
    # Mean absolute step relative to the series' standard deviation: slow,
    # clean signals where a forecast has to be precise to beat the last value.
    "smooth": {"max_roughness": 0.03, "max_flat_share": 0.05},
}


def table_statistics(values):
    """Statistics of one series; NaN marks a missing value and is skipped."""
    values = np.asarray(values, dtype=np.float64)
    observed = values[np.isfinite(values)]
    result = {
        "rows": int(values.size),
        "observed": int(observed.size),
        "flat_share": 0.0,
        "longest_flat_run": 0,
        "roughness": float("nan"),
    }
    if observed.size < 2:
        return result
    steps = np.diff(observed)
    same = steps == 0
    result["flat_share"] = float(same.mean())
    if same.any():
        # Lengths of the runs of repeated values, in steps.
        edges = np.flatnonzero(np.diff(np.concatenate(([0], same.view(np.int8), [0]))))
        result["longest_flat_run"] = int((edges[1::2] - edges[0::2]).max())
    spread = float(observed.std())
    if spread > 0:
        result["roughness"] = float(np.abs(steps).mean() / spread)
    return result


def _flat(statistics, options):
    return (
        statistics["flat_share"] >= options["min_flat_share"]
        and statistics["longest_flat_run"] >= options["min_longest_run"]
    )


def _smooth(statistics, options):
    return (
        np.isfinite(statistics["roughness"])
        and statistics["roughness"] <= options["max_roughness"]
        and statistics["flat_share"] < options["max_flat_share"]
    )


RULES = {"flat": _flat, "smooth": _smooth}


def category_thresholds(overrides=None):
    """Defaults with a recipe's overrides; unknown names are an error."""
    thresholds = {name: dict(options) for name, options in DEFAULT_THRESHOLDS.items()}
    for name, options in (overrides or {}).items():
        if name not in thresholds:
            raise ValueError(
                f"Unknown annealing category {name!r}; known: {sorted(thresholds)}"
            )
        unknown = set(options) - set(thresholds[name])
        if unknown:
            raise ValueError(f"Unknown {name} thresholds: {sorted(unknown)}")
        for key, value in options.items():
            if (
                isinstance(value, bool)
                or not isinstance(value, (int, float))
                or not np.isfinite(value)
                or value < 0
            ):
                raise ValueError(f"categories.{name}.{key} must be a number >= 0")
        thresholds[name].update(options)
    return thresholds


def matching_categories(statistics, thresholds):
    """Names of the categories a table with these statistics belongs to."""
    return [name for name, rule in RULES.items() if rule(statistics, thresholds[name])]

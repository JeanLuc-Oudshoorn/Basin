"""What the historian recorded, as opposed to what the process did.

Applied AFTER simulation, so no artifact feeds back into the causal equations:
redundant sensors, removed rows, per-column sensor artifacts, then missing
values (NaN). Sizes are saved as fractions of each clean column's own range,
the way an instrument's span and resolution are chosen for the signal it
measures. Randomness comes from a seed namespace separate from the simulator.
"""

import numpy as np

ARTIFACTS = (
    "noise",
    "quantize",
    "clip",
    "deadband",
    "hold",
    "stuck",
    "spikes",
    "dropout",
    "drift",
    "recalibration",
)
DEFAULT_ARTIFACT_WEIGHTS = {
    "noise": 4.0,
    "quantize": 4.0,
    "clip": 2.0,
    "deadband": 1.5,
    "hold": 1.0,
    "stuck": 1.5,
    "spikes": 2.0,
    "dropout": 1.0,
    "drift": 1.0,
    "recalibration": 1.0,
}
MISSING_PARADIGMS = ("mcar", "mar", "structural")
DEFAULT_MISSINGNESS_WEIGHTS = {"mcar": 0.35, "mar": 0.25, "structural": 0.40}
# Fixed evaluation order: slow bias first, digitisation late, failures last.
_ORDER = (
    "drift",
    "recalibration",
    "noise",
    "clip",
    "spikes",
    "deadband",
    "hold",
    "quantize",
    "stuck",
    "dropout",
)
_STRUCTURAL = ("periodic", "multirate", "late_start", "early_end", "outage", "censor")
PROTECTED_TAIL = 80  # Final outcome rows always observed: evaluation labels.
# Stored resolution: distinct levels over a signal's range, log-uniform. From a
# status-like reading with a handful of levels to one with five digits.
RESOLUTION_LEVELS = (6, 20000)


def _log_uniform(rng, low, high):
    return float(np.exp(rng.uniform(np.log(low), np.log(high))))


def _segments(rng, low, high, maximum):
    return [
        {"start": float(rng.uniform(0, 1)), "length": _log_uniform(rng, low, high)}
        for _ in range(int(rng.integers(1, maximum + 1)))
    ]


def sample_row_gaps(rng, rows):
    """Rows the logger never wrote: outage blocks or scattered single samples."""
    if rng.random() < 0.5:
        lengths = [
            int(min(rows // 4, _log_uniform(rng, 4, 400)))
            for _ in range(int(rng.integers(1, 4)))
        ]
        lengths = [n for n in lengths if n > 0] or [1]
        return {
            "kind": "blocks",
            "lengths": lengths,
            "positions": rng.uniform(0.05, 0.95, len(lengths)).tolist(),
            "total": int(sum(lengths)),
        }
    return {
        "kind": "scattered",
        "total": max(1, int(rows * _log_uniform(rng, 0.005, 0.06))),
    }


def _sample_artifact(rng, kind):
    spec = {"kind": kind}
    if kind == "noise":
        spec["fraction"] = _log_uniform(rng, 0.002, 0.06)
    elif kind == "quantize":
        spec.update(levels=_log_uniform(rng, 6, 3000), phase=float(rng.uniform(0, 1)))
    elif kind == "clip":
        spec.update(
            side=str(rng.choice(["upper", "lower", "both"], p=[0.4, 0.4, 0.2])),
            upper=float(rng.uniform(0.6, 0.985)),
            lower=float(rng.uniform(0.015, 0.4)),
        )
    elif kind == "deadband":
        spec["fraction"] = _log_uniform(rng, 0.003, 0.06)
    elif kind == "hold":
        spec["every"] = int(rng.integers(2, 13))
    elif kind == "stuck":
        spec["segments"] = _segments(rng, 0.005, 0.15, 3)
    elif kind == "spikes":
        spec.update(
            rate=_log_uniform(rng, 0.0005, 0.01),
            magnitude=float(rng.uniform(2, 12)),
            direction=str(rng.choice(["up", "down", "both"])),
            width=int(rng.choice([1, 1, 1, 2, 3])),
        )
    elif kind == "dropout":
        spec.update(
            segments=_segments(rng, 0.003, 0.08, 3),
            value=str(rng.choice(["zero", "floor"])),
        )
    elif kind == "drift":
        spec.update(
            magnitude=float(rng.choice([-1, 1]) * rng.uniform(0.3, 2.0)),
            resets=np.sort(rng.uniform(0.1, 0.9, int(rng.integers(0, 4)))).tolist(),
        )
    elif kind == "recalibration":
        spec.update(
            position=float(rng.uniform(0.15, 0.85)),
            gain=float(rng.uniform(0.8, 1.25)),
            offset=float(rng.choice([-1, 1]) * rng.uniform(0.2, 1.5)),
        )
    else:
        raise ValueError(f"Unknown measurement artifact: {kind}")
    return spec


def _sample_missingness(rng, columns, weights):
    names = list(weights)
    probability = np.array([weights[n] for n in names], dtype=float)
    paradigm = str(rng.choice(names, p=probability / probability.sum()))
    every = [c["name"] for c in columns]
    covariates = [n for n in every if n != "outcome"]
    # The outcome usually takes part; a covariate-only pattern is also real.
    affected = ["outcome"] if not covariates or rng.random() < 0.7 else []
    affected += [n for n in covariates if rng.random() < 0.5]
    affected = affected or [str(rng.choice(every))]
    spec = {"paradigm": paradigm, "columns": affected}
    if paradigm == "mcar":
        # Independent of every value: scattered points and/or random blocks.
        mode = str(rng.choice(["points", "blocks", "both"]))
        spec.update(
            rate=_log_uniform(rng, 0.002, 0.15) if mode != "blocks" else 0.0,
            blocks=_segments(rng, 0.002, 0.08, 6) if mode != "points" else [],
        )
    elif paradigm == "mar":
        # Depends only on something that stays observed: another column, the
        # clock, or this column's own previous reading.
        target = str(rng.choice(affected))
        others = [n for n in every if n != target]
        driver = (
            str(rng.choice(others))
            if others and rng.random() < 0.75
            else str(rng.choice(["clock", "previous"]))
        )
        spec.update(
            columns=[target],
            driver=driver,
            quantile=float(rng.uniform(0.5, 0.95)),
            direction=str(rng.choice(["above", "below"])),
            high=_log_uniform(rng, 0.15, 1.0),
            low=0.0 if rng.random() < 0.6 else _log_uniform(rng, 0.001, 0.03),
            clock_period=int(_log_uniform(rng, 12, 512)),
        )
    else:
        pattern = str(rng.choice(_STRUCTURAL, p=[0.22, 0.18, 0.18, 0.12, 0.15, 0.15]))
        if pattern == "early_end" and not covariates:
            pattern = "late_start"  # A target that stops has nothing to forecast.
        spec["pattern"] = pattern
        if pattern == "periodic":
            period = int(_log_uniform(rng, 12, 512))
            spec.update(
                period=period,
                length=int(max(1, period * rng.uniform(0.03, 0.35))),
                shift=int(rng.integers(period)),
            )
        elif pattern == "multirate":
            spec.update(every=int(rng.integers(2, 9)), shift=int(rng.integers(8)))
        elif pattern == "late_start":
            spec["fraction"] = float(rng.uniform(0.05, 0.5))
        elif pattern == "early_end":
            spec["fraction"] = float(rng.uniform(0.05, 0.3))
            spec["columns"] = [n for n in affected if n != "outcome"] or covariates[:1]
        elif pattern == "outage":
            # One cause, every column: the logger or the network was down.
            spec.update(columns=every, blocks=_segments(rng, 0.003, 0.06, 3))
        elif pattern == "censor":
            # The value decides its own absence (out-of-range not reported).
            spec.update(
                quantile=float(rng.uniform(0.85, 0.99)),
                direction=str(rng.choice(["above", "below"])),
            )
    return spec


def sample_observation(rng, columns, settings):
    """Draw the recording layer for the given columns.

    `columns` is a list of {"name", "discrete"} in export order, outcome first.
    Discrete columns keep exact levels: they can go missing but get no noise.
    """
    spec = {"redundant": [], "artifacts": {}, "missing": []}
    continuous = [c["name"] for c in columns if not c["discrete"]]
    columns = list(columns)
    chance = settings["redundant_sensor_probability"]
    if continuous and chance and rng.random() < chance:
        for _ in range(int(rng.choice([1, 2], p=[0.75, 0.25]))):
            name = f"covariate_{settings['next_index'] + len(spec['redundant']):02d}"
            binary = bool(rng.random() < 0.15)
            spec["redundant"].append(
                {
                    "name": name,
                    "source": "outcome"
                    if rng.random() < 0.6
                    else str(rng.choice(continuous)),
                    "form": "threshold" if binary else "analog",
                    "quantile": float(rng.uniform(0.15, 0.85)),
                    "gain": float(rng.choice([-1, 1]) * _log_uniform(rng, 0.2, 5)),
                    "offset": float(rng.uniform(-2, 2)),
                    "lag": int(rng.choice([0, 0, 0, 1, 2, 3, 5])),
                    "tau": 0.0 if rng.random() < 0.6 else _log_uniform(rng, 1, 12),
                    "power": float(rng.choice([1.0, 1.0, 1.0, 1.0, 0.5, 2.0])),
                    "noise": _log_uniform(rng, 0.003, 0.1),
                }
            )
            columns.append({"name": name, "discrete": binary})
    names = list(settings["artifact_weights"])
    weights = np.array([settings["artifact_weights"][n] for n in names], dtype=float)
    for column in columns:
        if column["discrete"] or not settings["measurement_probability"]:
            continue
        if rng.random() < settings["measurement_probability"]:
            count = int(rng.choice([1, 2, 3], p=[0.6, 0.3, 0.1]))
            kinds = rng.choice(
                names,
                size=min(count, int((weights > 0).sum())),
                replace=False,
                p=weights / weights.sum(),
            )
            spec["artifacts"][column["name"]] = [
                _sample_artifact(rng, str(kind)) for kind in kinds
            ]
        chance = settings.get("resolution_probability", 0)
        if chance and rng.random() < chance:
            # A historian stores a reading with a fixed number of digits. This
            # is the normal state of recorded data, not a fault of the sensor.
            drawn = spec["artifacts"].setdefault(column["name"], [])
            low, high = RESOLUTION_LEVELS
            levels, phase = _log_uniform(rng, low, high), float(rng.uniform(0, 1))
            if not any(a["kind"] == "quantize" for a in drawn):
                drawn.append({"kind": "quantize", "levels": levels, "phase": phase})
    chance = settings["missingness_probability"]
    if chance and rng.random() < chance:
        spec["missing"].append(
            _sample_missingness(rng, columns, settings["missingness_weights"])
        )
    return spec


def _blocks(segments, rows):
    mask = np.zeros(rows, dtype=bool)
    for segment in segments:
        length = max(1, int(segment["length"] * rows))
        start = int(segment["start"] * max(1, rows - length))
        mask[start : start + length] = True
    return mask


def _apply_artifacts(clean, artifacts, rng):
    """One column. `clean` fixes the instrument span; `value` is the reading."""
    rows = len(clean)
    low, high = np.percentile(clean, [1, 99])
    span, spread = float(high - low), float(np.std(clean))
    if span <= 0 or spread <= 0:
        return clean  # A constant signal has no span to size artifacts against.
    value = clean.astype(float).copy()
    position = np.arange(rows) / max(1, rows - 1)
    for kind in _ORDER:
        for spec in (a for a in artifacts if a["kind"] == kind):
            if kind == "drift":
                # Bias grows linearly and returns to zero at each recalibration.
                since = position.copy()
                for reset in spec["resets"]:
                    since = np.where(position >= reset, position - reset, since)
                value += spec["magnitude"] * spread * since
            elif kind == "recalibration":
                after = position >= spec["position"]
                center = (low + high) / 2
                value[after] = (
                    center
                    + spec["gain"] * (value[after] - center)
                    + spec["offset"] * spread
                )
            elif kind == "noise":
                value += rng.normal(0, spec["fraction"] * span, rows)
            elif kind == "clip":
                floor, ceiling = np.quantile(clean, [spec["lower"], spec["upper"]])
                value = np.clip(
                    value,
                    floor if spec["side"] != "upper" else -np.inf,
                    ceiling if spec["side"] != "lower" else np.inf,
                )
            elif kind == "spikes":
                starts = np.flatnonzero(rng.random(rows) < spec["rate"])
                signs = (
                    rng.choice([-1, 1], len(starts))
                    if spec["direction"] == "both"
                    else np.full(len(starts), 1 if spec["direction"] == "up" else -1)
                )
                sizes = spec["magnitude"] * spread * rng.uniform(0.5, 1.0, len(starts))
                for start, sign, size in zip(starts, signs, sizes):
                    value[start : start + spec["width"]] += sign * size
            elif kind == "deadband":
                # Report by exception: a new value only after a large enough move.
                threshold, last = spec["fraction"] * span, value[0]
                for index in range(rows):
                    if abs(value[index] - last) > threshold:
                        last = value[index]
                    value[index] = last
            elif kind == "hold":
                value = value[np.arange(rows) // spec["every"] * spec["every"]]
            elif kind == "quantize":
                step = span / spec["levels"]
                origin = low + spec["phase"] * step
                value = origin + step * np.round((value - origin) / step)
            elif kind == "stuck":
                # The transmitter freezes on its last reading, then recovers.
                for segment in spec["segments"]:
                    length = max(2, int(segment["length"] * rows))
                    start = int(segment["start"] * max(1, rows - length))
                    value[start : start + length] = value[start]
            elif kind == "dropout":
                fill = 0.0 if spec["value"] == "zero" else float(clean.min())
                value[_blocks(spec["segments"], rows)] = fill
    return value


def _missing_mask(spec, name, recorded, rows, rng):
    """Boolean mask of readings to blank in one column, for one mechanism."""
    index = np.arange(rows)
    paradigm = spec["paradigm"]
    if paradigm == "mcar":
        return (rng.random(rows) < spec["rate"]) | _blocks(spec["blocks"], rows)
    if paradigm == "mar":
        driver = spec["driver"]
        if driver == "clock":
            reference = (index % spec["clock_period"]) / spec["clock_period"]
        elif driver == "previous":
            reference = np.concatenate([recorded[name][:1], recorded[name][:-1]])
        else:
            reference = recorded[driver]
        finite = reference[np.isfinite(reference)]
        if not len(finite):
            return np.zeros(rows, dtype=bool)
        above = spec["direction"] == "above"
        if len(np.unique(finite)) <= 2:
            # A binary driver: one of its two states carries the risk.
            risky = reference == (finite.max() if above else finite.min())
        else:
            threshold = np.quantile(finite, spec["quantile"])
            risky = reference > threshold if above else reference < threshold
        return rng.random(rows) < np.where(risky, spec["high"], spec["low"])
    pattern = spec["pattern"]
    if pattern == "periodic":
        return (index + spec["shift"]) % spec["period"] < spec["length"]
    if pattern == "multirate":
        return (index + spec["shift"]) % spec["every"] != 0
    if pattern == "late_start":
        # The target keeps most of its history; a covariate may arrive late.
        fraction = min(spec["fraction"], 0.3 if name == "outcome" else 1.0)
        return index < int(fraction * rows)
    if pattern == "early_end":
        return index >= rows - int(spec["fraction"] * rows)
    if pattern == "outage":
        return _blocks(spec["blocks"], rows)
    if pattern == "censor":
        values, above = recorded[name], spec["direction"] == "above"
        threshold = np.nanquantile(
            values, spec["quantile"] if above else 1 - spec["quantile"]
        )
        return values > threshold if above else values < threshold
    raise ValueError(f"Unknown structural missingness: {pattern}")


def _redundant(spec, source, rng):
    """A second instrument on the same quantity: own lag, filter, scale, noise."""
    value = source.astype(float)
    if spec["lag"]:
        value = np.concatenate([np.full(spec["lag"], value[0]), value[: -spec["lag"]]])
    if spec["tau"]:
        alpha, state = 1 / (1 + spec["tau"]), value[0]
        filtered = np.empty(len(value))
        for index, reading in enumerate(value):
            state += alpha * (reading - state)
            filtered[index] = state
        value = filtered
    low, high = np.percentile(value, [1, 99])
    span = float(high - low)
    if spec["form"] == "threshold":
        return (value > np.quantile(value, spec["quantile"])).astype(float)
    if span > 0 and spec["power"] != 1.0:
        # Square-root/square laws (flow versus differential pressure).
        unit = np.clip((value - low) / span, 0.0, None)
        value = low + span * unit ** spec["power"]
    value = spec["offset"] + spec["gain"] * value
    return value + rng.normal(0, spec["noise"] * abs(spec["gain"]) * span, len(value))


def recorded_window(frame, recorded_rows):
    """The clean rows that would be kept if nothing else were recorded badly."""
    if recorded_rows is None or recorded_rows >= len(frame):
        return frame
    return frame.iloc[len(frame) - recorded_rows :].reset_index(drop=True)


def apply_observation(frame, spec, recorded_rows, row_gaps, seed):
    """Clean full-span frame in, recorded table out. Deterministic given seed."""
    rng = np.random.default_rng(np.random.SeedSequence([seed, 0x0B5E7]))
    frame = frame.copy()
    for sensor in spec.get("redundant", []):
        frame[sensor["name"]] = _redundant(
            sensor, frame[sensor["source"]].to_numpy(), rng
        )
    recorded_rows = len(frame) if recorded_rows is None else recorded_rows
    if row_gaps:
        # Remove rows from the tail segment so the recorded length is unchanged.
        segment = min(len(frame), recorded_rows + row_gaps["total"])
        keep = np.ones(segment, dtype=bool)
        if row_gaps["kind"] == "blocks":
            for position, length in zip(row_gaps["positions"], row_gaps["lengths"]):
                start = int(position * max(1, segment - length))
                keep[start : start + length] = False
        else:
            count = min(row_gaps["total"], segment - 2)
            keep[rng.choice(segment, count, replace=False)] = False
        frame = frame.iloc[len(frame) - segment :][keep]
    frame = frame.iloc[max(0, len(frame) - recorded_rows) :].reset_index(drop=True)
    rows = len(frame)
    frame["time_index"] = np.arange(rows)
    for name, artifacts in spec.get("artifacts", {}).items():
        frame[name] = _apply_artifacts(
            frame[name].to_numpy(dtype=float), artifacts, rng
        )
    recorded = {name: frame[name].to_numpy(dtype=float) for name in frame}
    for missing in spec.get("missing", []):
        for name in missing["columns"]:
            mask = _missing_mask(missing, name, recorded, rows, rng)
            limit = 0.3 if name == "outcome" else 0.6
            if mask.mean() > limit:
                # Keep the pattern, thin it: a column must stay mostly usable.
                mask &= rng.random(rows) < limit / mask.mean()
            if name == "outcome":
                mask[max(0, rows - PROTECTED_TAIL) :] = False
            values = frame[name].to_numpy(dtype=float).copy()
            values[mask] = np.nan
            if np.isfinite(values).sum() >= 2:
                frame[name] = values
    return frame

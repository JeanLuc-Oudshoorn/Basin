"""Causal process envelopes and exogenous discrete operating states."""

import numpy as np

DISCRETE_KINDS = ("bernoulli", "markov")
EXOGENOUS_KINDS = ("event", "switch", "formula") + DISCRETE_KINDS
FORMULA_FAMILIES = (
    "ramp",
    "power",
    "sigmoid",
    "bump",
    "damped_wave",
    "chirp",
    "sawtooth",
    "triangle",
    "state_machine",
)


def ramp_fraction(progress, shape, curvature=1.0):
    """A finite causal transition, with independently chosen rise/release shape."""
    u = np.clip(progress, 0.0, 1.0)
    if shape == "linear":
        return u
    if shape == "smoothstep":
        return u * u * (3 - 2 * u)
    if shape == "cosine":
        return 0.5 * (1 - np.cos(np.pi * u))
    if shape == "power":
        return u**curvature
    if shape == "release":
        # A sharp initial change followed by a long tail, with an exact endpoint.
        return -np.expm1(-curvature * u) / -np.expm1(-curvature)
    raise ValueError(f"Unknown process ramp: {shape}")


def scheduled_path(spec, time):
    """Response to saved setpoint transitions; future commands cannot affect a prefix.

    At a command time, start from the previous response at t-1. Unlike joining
    control points by interpolation, this never begins moving toward a future
    setpoint before the corresponding command actually arrives.
    """
    time = np.asarray(time)
    result = np.full(time.shape, spec.get("initial_level", 0.0), dtype=float)
    previous, start_level, last_start = None, spec.get("initial_level", 0.0), -np.inf
    for transition in spec["transitions"]:
        start = transition["start"]
        if start <= last_start or transition["steps"] <= 0:
            raise ValueError(
                "Process commands must be ordered with positive ramp durations"
            )
        if previous is not None:
            fraction = ramp_fraction(
                (start - previous["start"]) / previous["steps"],
                previous["shape"],
                previous.get("curvature", 1.0),
            )
            start_level += (previous["target"] - start_level) * float(fraction)
        mask = time >= start
        fraction = ramp_fraction(
            (time[mask] - start + 1) / transition["steps"],
            transition["shape"],
            transition.get("curvature", 1.0),
        )
        result[mask] = start_level + (transition["target"] - start_level) * fraction
        previous, last_start = transition, start
    for pulse in spec.get("priming", []):
        phase = (time - pulse["start"]) / pulse["duration"]
        mask = (phase >= 0) & (phase <= 1)
        result[mask] += pulse["amplitude"] * np.sin(np.pi * phase[mask]) ** 2
    return result


def formula_path(spec, time):
    """Clock-driven exogenous equations, evaluated without empirical normalization.

    Noise is added by the simulator using the same saved noise laws as other
    nodes. This function is the deterministic part, also used for SCM screening.
    """
    time = np.asarray(time, dtype=float)
    family = spec["family"]
    if family in ("process_run", "state_machine"):
        wave = scheduled_path(spec, time)
    else:
        p = spec["parameters"]
        u = time / p["duration"]
        phase = 2 * np.pi * time / p["period"] + p["phase"]
        if family == "ramp":
            wave = np.clip(u, 0, 1)
        elif family == "power":
            wave = np.clip(u, 0, 1) ** p["power"]
        elif family == "sigmoid":
            wave = 0.5 * (1 + np.tanh((u - p["center"]) / p["width"]))
        elif family == "bump":
            wave = np.exp(-0.5 * ((u - p["center"]) / p["width"]) ** 2)
        elif family == "damped_wave":
            wave = np.exp(-p["decay"] * np.maximum(u, 0)) * np.sin(phase)
        elif family == "chirp":
            wave = np.sin(
                phase + np.pi * p["chirp"] * time**2 / (p["period"] * p["duration"])
            )
        elif family == "sawtooth":
            wave = 2 * ((phase / (2 * np.pi)) % 1) - 1
        elif family == "triangle":
            wave = 1 - 4 * np.abs((phase / (2 * np.pi)) % 1 - 0.5)
        else:
            raise ValueError(f"Unknown formula family: {family}")
    return spec.get("offset", 0.0) + spec.get("amplitude", 1.0) * wave


def discrete_path(spec, rng, length):
    """Draw exact discrete levels; Markov transitions use only the previous state."""
    if spec["kind"] == "bernoulli":
        return rng.binomial(1, spec["probability"], length)
    values = np.asarray(spec["values"], dtype=float)
    transition = np.asarray(spec["transition"], dtype=float)
    if (
        transition.shape != (len(values), len(values))
        or not np.isfinite(transition).all()
        or (transition < 0).any()
        or not np.allclose(transition.sum(axis=1), 1)
        or not np.isfinite(values).all()
    ):
        raise ValueError("Invalid Markov levels or transition matrix")
    state = spec["initial_state"]
    if not isinstance(state, int) or not 0 <= state < len(values):
        raise ValueError("Invalid initial Markov state")
    cumulative = transition.cumsum(axis=1)
    cumulative[:, -1] = 1.0  # Remove cumulative floating-point roundoff only.
    uniforms = rng.random(max(0, length - 1))
    result = np.empty(length)
    if length:
        result[0] = values[state]
    for index, draw in enumerate(uniforms, 1):
        state = int(np.searchsorted(cumulative[state], draw, side="right"))
        result[index] = values[state]
    return result


def episode_envelope(operation, command):
    """Finite startup/release response; never consult the next command or row.

    A rising command starts a ramp from the current activity to one; a falling
    command starts release from the current activity to zero. Thus even short
    interrupted runs and repeated bursts behave causally and continuously.
    """
    result = np.empty(len(command))
    value, start, elapsed, previous_on = 0.0, 0.0, 0, False
    for index, signal in enumerate(command):
        on = bool(signal)
        if on != previous_on:
            start, elapsed = value, 0
        elapsed += 1
        steps = operation["rise_steps"] if on else operation["fall_steps"]
        progress = min(1.0, elapsed / steps)
        shape = operation.get("ramp_shape", "smoothstep")
        if shape == "smoothstep":
            progress = progress * progress * (3 - 2 * progress)
        elif shape == "cosine":
            progress = 0.5 * (1 - np.cos(np.pi * progress))
        elif shape != "linear":
            raise ValueError(f"Unknown episode ramp: {shape}")
        value = start + (1 - start) * progress if on else start * (1 - progress)
        result[index] = value
        previous_on = on
    return result


def sensor_reading(value, spec):
    """Hard measurement limits, expressed in the variable's recorded units."""
    if spec.get("floor") is not None and value < spec["floor"]:
        return spec["floor"]
    if spec.get("ceiling") is not None and value > spec["ceiling"]:
        return spec["ceiling"]
    return value

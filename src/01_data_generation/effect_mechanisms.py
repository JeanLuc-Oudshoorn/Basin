"""Additional static maps and causal stateful operators for SCM edges.

All parameters belong to a saved instruction; state belongs to one edge in one
simulation. Physical names denote shape priors on standardized inputs, not a
dimensionally calibrated simulator of a particular pump, reactor or controller.
"""

from collections import deque

import numpy as np

STATIC_OPERATORS = (
    "saturation",
    "hill",
    "michaelis_menten",
    "langmuir",
    "arrhenius",
    "antoine",
    "gaussian_optimum",
    "parabolic_optimum",
    "deadband",
    "clip",
    "knee",
    "leaky_relu",
    "decreasing_quadratic",
    "quadratic_friction",
    "stefan_boltzmann",
    "queue",
    "min",
    "max",
    "soft_min",
    "ratio",
    "swiglu",
)
TEMPORAL_OPERATORS = (
    "first_order_lag",
    "delay",
    "integrator",
    "hysteresis",
    "rate_limiter",
    "erlang",
    "diffusion",
    "resonance",
    "regime_switch",
    "pid",
    "volatility_coupling",
)


def clip_scalar(value, low, high):
    """Exact scalar clipping without NumPy's array-dispatch overhead.

    Comparisons preserve in-range values (including signed zero) and NaN, as
    np.clip does. Arrays still use np.clip at vectorized call sites.
    """
    return low if value < low else high if value > high else value


def evaluate_extra(op, z, b, instruction, state):
    """Evaluate one instruction. Temporal updates consume only current/past input."""
    p = instruction
    k, width, threshold = p["k"], p["width"], p["threshold"]
    # Stable softplus supplies a strictly positive temperature/concentration.
    positive = float(np.logaddexp(0.0, z)) + p["floor"]
    if op == "saturation":
        return z / (k + abs(z))
    if op in ("hill", "michaelis_menten", "langmuir"):
        exponent = p["shape"] if op == "hill" else 1.0
        log_ratio = exponent * (np.log(positive) - np.log(k))
        return float(np.exp(-np.logaddexp(0.0, -log_ratio)))
    if op == "arrhenius":
        return np.exp(-k / positive)
    if op == "antoine":
        # Log-pressure form A - B/(C+T), before the shared capacity bound.
        return p["level"] - k / (threshold + positive)
    if op == "gaussian_optimum":
        return np.exp(-0.5 * ((z - p["center"]) / width) ** 2)
    if op == "parabolic_optimum":
        return p["level"] - ((z - p["center"]) / width) ** 2
    if op == "deadband":
        return np.sign(z) * max(0.0, abs(z) - threshold)
    if op == "clip":
        return clip_scalar(z, p["center"] - width, p["center"] + width)
    if op == "knee":
        return z if z <= threshold else threshold + p["leak"] * (z - threshold)
    if op == "leaky_relu":
        return z if z >= 0 else p["leak"] * z
    if op == "decreasing_quadratic":
        return p["level"] - k * positive**2
    if op == "quadratic_friction":
        return k * positive**2
    if op == "stefan_boltzmann":
        return (positive / k) ** 4
    if op == "queue":
        # Utilization below one; finite headroom is sampled per edge. This
        # bounds the singularity without hiding it behind inverse-square input.
        rho = (1 - p["floor"]) * np.exp(-np.logaddexp(0.0, -z))
        return 1 / max(p["floor"], 1 - rho) - 1
    if op == "min":
        return min(z, b)
    if op == "max":
        return max(z, b)
    if op == "soft_min":
        return -width * np.logaddexp(-z / width, -b / width)
    if op == "ratio":
        denominator = np.copysign(max(abs(b), p["floor"]), b)
        return z / denominator
    if op == "swiglu":
        return z * np.exp(-np.logaddexp(0.0, -z)) * b

    # Shared defaults are used only on the first call of an instruction. The
    # simulator owns this dictionary; never mutate the saved SCM itself.
    previous = state.get("output", 0.0)
    alpha = -np.expm1(-1 / p["tau"])
    if op == "first_order_lag":
        result = previous + alpha * (z - previous)
    elif op == "delay":
        queue = state.setdefault("queue", deque([0.0] * p["steps"], maxlen=p["steps"]))
        result = queue.popleft()
        queue.append(z)
    elif op == "integrator":
        # A sampled leak and finite capacity keep feedback stable. Leak zero
        # supports a true bounded accumulator; integration starts in warm-up.
        result = (1 - p["leak"]) * previous + z / p["tau"]
    elif op == "hysteresis":
        on = state.get("on", False)
        if z > p["center"] + threshold:
            on = True
        elif z < p["center"] - threshold:
            on = False
        state["on"] = on
        result = float(on)
    elif op == "rate_limiter":
        result = previous + clip_scalar(z - previous, -p["fall_rate"], p["rise_rate"])
    elif op in ("erlang", "diffusion"):
        stages = state.setdefault("stages", [0.0] * p["stages"])
        # Cascaded first-order filters give Erlang-like impulse responses.
        # Geometrically spaced time constants make the diffusion variant broader.
        result = z
        for index in range(len(stages)):
            tau = p["tau"] * (p["spread"] ** index if op == "diffusion" else 1)
            stages[index] += -np.expm1(-1 / tau) * (result - stages[index])
            result = stages[index]
    elif op == "resonance":
        # Stable complex pole: an input impulse excites a damped oscillator.
        pole = np.exp(-1 / p["tau"] + 2j * np.pi / p["period"])
        oscillator = pole * state.get("oscillator", 0j) + (1 - abs(pole)) * z
        state["oscillator"] = oscillator
        result = oscillator.imag
    elif op == "regime_switch":
        # Hysteretic regime selection avoids rapid chattering near a threshold.
        on = state.get("on", False)
        if z > p["center"] + threshold:
            on = True
        elif z < p["center"] - threshold:
            on = False
        state["on"] = on
        result = (p["high_gain"] if on else p["leak"]) * z
    elif op == "pid":
        # A genuine local feedback controller: error measures a saved setpoint
        # against the causal input, and its output can drive downstream nodes.
        error = p["center"] - z
        integral = clip_scalar(
            state.get("integral", 0.0) + error / p["tau"], -p["limit"], p["limit"]
        )
        derivative = error - state.get("error", error)
        state.update(integral=float(integral), error=error)
        result = k * error + p["ki"] * integral + p["kd"] * derivative
    elif op == "volatility_coupling":
        # Deterministic volatility estimate, not an extra random innovation.
        mean = state.get("mean", 0.0)
        variance = (1 - alpha) * state.get("variance", 0.0) + alpha * (z - mean) ** 2
        state.update(mean=mean + alpha * (z - mean), variance=variance)
        result = z * (1 + k * np.sqrt(variance))
    else:
        raise ValueError(f"Unknown operator: {op}")
    state["output"] = float(clip_scalar(result, -p["limit"], p["limit"]))
    return state["output"]


def combine_parents(effects, spec, prepared=None):
    """Joint maps use different parents, rather than intermediate values of one."""
    if spec.get("version", 1) == 2:
        return _combine_selected_parents(effects, spec, prepared)
    # Freeze original arithmetic for SCMs saved before selected-parent maps.
    values = np.asarray(effects, dtype=float)
    if not len(values):
        return 0.0
    if len(values) == 1 or spec["mode"] == "sum":
        return float(values.sum())
    mode, scale = spec["mode"], spec["scale"]
    # Preserve parent weights from _effect; normalization applies only inside
    # joint nonlinear maps and prevents catastrophic products in feedback loops.
    z = values / scale
    if mode == "product":
        result = scale * np.prod(np.tanh(z))
    elif mode == "ratio":
        result = values[0] / max(spec["floor"], float(np.logaddexp(0, z[1:].mean())))
    elif mode == "min":
        result = values.min()
    elif mode == "max":
        result = values.max()
    elif mode == "soft_min":
        result = -scale * np.logaddexp.reduce(-z)
    elif mode == "gate":
        result = values[:-1].sum() * np.exp(-np.logaddexp(0, -z[-1]))
    elif mode == "regime_switch":
        result = values[:-1].sum() * (
            spec["high_gain"] if z[-1] > spec["threshold"] else spec["low_gain"]
        )
    elif mode == "volatility_coupling":
        result = values[0] * (1 + abs(np.tanh(z[1:].mean())))
    elif mode == "swiglu":
        result = values[0] * np.exp(-np.logaddexp(0, -z[0])) * np.tanh(z[1:].mean())
    else:
        raise ValueError(f"Unknown parent combination: {mode}")
    return float(spec["limit"] * np.tanh(result / spec["limit"]))


def _positive(values, spec):
    if spec.get("positive_map", "softplus") == "exp":
        return np.exp(np.clip(values, -30, 30)) + spec["floor"]
    return np.logaddexp(0.0, values) + spec["floor"]


def prepare_parent_combination(spec, parent_count):
    """Validate and materialize constant arrays once per simulation, not per row."""
    if (
        not spec
        or spec.get("version", 1) != 2
        or spec["mode"] == "sum"
        or not parent_count
    ):
        return None
    indices = np.asarray(spec["indices"], dtype=int)
    if (
        len(indices) < 2
        or len(set(indices.tolist())) != len(indices)
        or (indices < 0).any()
        or (indices >= parent_count).any()
    ):
        raise ValueError(
            "Joint-parent indices must identify at least two distinct available parents"
        )
    unused = np.ones(parent_count, dtype=bool)
    unused[indices] = False
    prepared = {"indices": indices, "unused": unused}
    mode = spec["mode"]
    if mode == "convex_mix":
        weights = np.asarray(spec["weights"], float)
        if (
            weights.shape != indices.shape
            or (weights < 0).any()
            or not np.isclose(weights.sum(), 1)
        ):
            raise ValueError("Convex weights must be nonnegative and sum to one")
        prepared["weights"] = weights
    if mode == "mlp":
        prepared.update((key, np.asarray(spec[key])) for key in ("w", "bias", "v"))
    if mode == "log_sum":
        prepared["powers"] = np.asarray(spec["powers"])
    return prepared


def _combine_selected_parents(effects, spec, prepared=None):
    """Replace a selected group's sum, or add its interaction to the full sum.

    Unselected parents retain their additive contributions. The large numeric
    guard prevents overflow; unlike tanh saturation it does not force products
    to remain smaller than the additive contributions they can replace.
    """
    values = np.asarray(effects, dtype=float)
    if not len(values) or spec["mode"] == "sum":
        return float(values.sum())
    prepared = (
        prepare_parent_combination(spec, len(values)) if prepared is None else prepared
    )
    indices = prepared["indices"]
    selected = values[indices]
    z = selected / spec["scale"]
    positive = _positive(z, spec)
    mode, scale = spec["mode"], spec["scale"]
    if mode == "product":
        joint = scale * np.exp(clip_scalar(np.log(positive).sum(), -30, 30))
    elif mode == "saddle":
        joint = scale * np.prod(np.tanh(z))
    elif mode == "ratio":
        joint = scale * positive[0] / max(spec["floor"], float(positive[1:].mean()))
    elif mode == "min":
        joint = selected.min()
    elif mode == "max":
        joint = selected.max()
    elif mode == "soft_min":
        joint = -scale * np.logaddexp.reduce(-z)
    elif mode == "gate":
        gate = np.exp(
            -np.logaddexp(0.0, -(z[-1] - spec["threshold"]) / spec["temperature"])
        )
        joint = selected[:-1].sum() * gate
    elif mode == "exponent_gate":
        gate = np.exp(
            -np.logaddexp(0.0, -(z[-1] - spec["threshold"]) / spec["temperature"])
        )
        exponent = spec["low_gain"] + (spec["high_gain"] - spec["low_gain"]) * gate
        joint = scale * np.exp(clip_scalar(exponent * np.log(positive[0]), -30, 30))
    elif mode == "threshold":
        joint = selected[:-1].sum() if z[-1] > spec["threshold"] else 0.0
    elif mode == "convex_mix":
        weights = prepared["weights"]
        joint = float(np.dot(weights, selected))
    elif mode == "mlp":
        hidden = np.tanh(prepared["w"] @ z + prepared["bias"])
        joint = scale * float(np.dot(prepared["v"], hidden))
    elif mode == "log_sum":
        # Together with the exponential node link this gives a true positive
        # multi-parent power law: exp(sum(exponent_i * log(parent_i))).
        joint = float(np.dot(prepared["powers"], np.log(positive)))
    elif mode == "regime_switch":
        joint = selected[:-1].sum() * (
            spec["high_gain"] if z[-1] > spec["threshold"] else spec["low_gain"]
        )
    elif mode == "volatility_coupling":
        joint = selected[0] * (1 + abs(np.tanh(z[1:].mean())))
    elif mode == "swiglu":
        joint = selected[0] * np.exp(-np.logaddexp(0, -z[0])) * np.tanh(z[1:].mean())
    else:
        raise ValueError(f"Unknown parent combination: {mode}")
    # Convex mixing and log-sum preserve their exact mathematical meanings.
    strength = 1.0 if mode in ("convex_mix", "log_sum") else spec["strength"]
    if spec["application"] == "additive":
        result = values.sum() + strength * joint
    elif spec["application"] == "replace":
        result = values[prepared["unused"]].sum() + strength * joint
    else:
        raise ValueError("Joint application must be additive or replace")
    return float(clip_scalar(result, -spec["numeric_limit"], spec["numeric_limit"]))


def apply_node_link(forcing, spec):
    """Transform the parent aggregate; stochastic innovations remain outside g."""
    if spec is None or spec["mode"] == "identity":
        return forcing
    z = (forcing - spec["input_offset"]) / spec["scale"]
    mode = spec["mode"]
    if mode == "saturation":
        result = spec["capacity"] * np.tanh(z / spec["capacity"])
    elif mode == "softplus":
        result = np.logaddexp(0.0, z)
    elif mode == "exponential":
        result = np.exp(clip_scalar(z, -30, 30))
    elif mode == "sigmoid":
        result = np.exp(-np.logaddexp(0.0, -z))
    elif mode == "logarithm":
        result = np.sign(z) * np.log1p(abs(z))
    else:
        raise ValueError(f"Unknown node link: {mode}")
    result = spec["gain"] * result + spec["output_offset"]
    return float(clip_scalar(result, -spec["numeric_limit"], spec["numeric_limit"]))

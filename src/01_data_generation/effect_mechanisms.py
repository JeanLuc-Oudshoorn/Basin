"""Additional static maps and causal stateful operators for SCM edges.

All parameters belong to a saved instruction; state belongs to one edge in one
simulation. Physical names denote shape priors on standardized inputs, not a
dimensionally calibrated simulator of a particular pump, reactor or controller.
"""

from collections import deque
from functools import lru_cache

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
    "second_order",
)


def clip_scalar(value, low, high):
    """Exact scalar clipping without NumPy's array-dispatch overhead.

    Comparisons preserve in-range values (including signed zero) and NaN, as
    np.clip does. Arrays still use np.clip at vectorized call sites.
    """
    return low if value < low else high if value > high else value


def second_order_coefficients(period, damping):
    """Exact one-step update of y'' + 2*zeta*w*y' + w*w*y = w*w*u, input held.

    Returns (a11, a12, a21, a22, b1, b2) for the state (y, y'). Unit static
    gain: a step input settles at the input, with overshoot below zeta=1.
    Matrix exponential by scaling and squaring; no SciPy dependency.
    """
    omega = 2 * np.pi / period
    system = np.array([[0.0, 1.0], [-omega * omega, -2 * damping * omega]])
    squarings = max(0, int(np.ceil(np.log2(max(1e-12, np.abs(system).sum())))) + 4)
    scaled = system / 2**squarings
    transition, term = np.eye(2), np.eye(2)
    for order in range(1, 13):
        term = term @ scaled / order
        transition = transition + term
    for _ in range(squarings):
        transition = transition @ transition
    drive = np.linalg.solve(
        system, (transition - np.eye(2)) @ np.array([0.0, omega * omega])
    )
    return tuple(float(v) for v in (*transition.ravel(), *drive))


_POSITIVE_PARENT_MODES = frozenset(("product", "ratio", "exponent_gate", "log_sum"))


@lru_cache(maxsize=1024)
def _lag_alpha(tau):
    # Parameters are fixed for a whole SCM. Reuse the identical scalar NumPy
    # calculation across rows; never approximate expm1 or change its dtype.
    # Key by the parameter, so edited/replayed specifications remain valid.
    # Bound memory in workers generating millions of unrelated SCMs.
    return -np.expm1(-1 / tau)


@lru_cache(maxsize=1024)
def _cascade_alphas(tau, count, spread):
    return tuple(_lag_alpha(tau * spread**index) for index in range(count))


@lru_cache(maxsize=1024)
def _resonance_pole(tau, period):
    return np.exp(-1 / tau + 2j * np.pi / period)


def _base_core(op, p):
    """Maps of the current instruction input z and an earlier result b."""
    if op == "affine":
        return lambda z, b: z
    if op == "sine":
        return lambda z, b: np.sin(z)
    if op == "cosine":
        return lambda z, b: np.cos(z)
    if op in ("damped_sine", "damped_cosine"):
        # For nonnegative input this is exp(-decay*z)*sin/cos(w*z+phase).
        # abs(z) extends it to signed covariates without exponential
        # blow-up. Time-domain ringing is modeled separately.
        frequency, phase, decay = p["frequency"], p["phase"], p["decay"]
        wave = np.sin if op == "damped_sine" else np.cos
        return lambda z, b: np.exp(-decay * abs(z)) * wave(frequency * z + phase)
    if op == "tangent":
        cap = p["angle_limit"]
        return lambda z, b: np.tan(cap * np.tanh(z / cap))
    if op == "sigmoid":
        # 2*sigmoid(z)-1, without exponential overflow.
        return lambda z, b: np.tanh(z / 2)
    if op == "exponential":
        limit = p["limit"]
        return lambda z, b: np.expm1(limit * np.tanh(z / limit)) / limit
    if op == "power":
        power = p["power"]
        return lambda z, b: np.sign(z) * abs(z) ** power
    if op == "inverse_square":
        # Exactly 1/z**2 outside a sampled near-zero floor, followed by the
        # same soft actuator saturation as the other operators. The
        # instruction's gain/offset shift and scale the singularity.
        floor = p["floor"]
        return lambda z, b: (1 / max(abs(z), floor)) ** 2
    if op == "log":
        return lambda z, b: np.sign(z) * np.log1p(abs(z))
    if op == "relu":
        return lambda z, b: max(0, z)
    if op == "mlp":
        w, bias, v = np.asarray(p["w"]), np.asarray(p["bias"]), np.asarray(p["v"])
        return lambda z, b: np.dot(v, np.tanh(w * z + bias))
    if op == "add":
        return lambda z, b: z + b
    if op == "multiply":
        return lambda z, b: z * b
    if op == "gate":
        return lambda z, b: z * (1 + np.tanh(b)) / 2
    return None


def _static_core(op, p):
    """Memoryless shape priors. Some consume a softplus of the input."""
    k, width, threshold = p["k"], p["width"], p["threshold"]
    if op == "saturation":
        return lambda z, b: z / (k + abs(z))
    if op in ("hill", "michaelis_menten", "langmuir"):
        floor = p["floor"]
        exponent = p["shape"] if op == "hill" else 1.0
        log_k = np.log(k)

        def core(z, b):
            positive = float(np.logaddexp(0.0, z)) + floor
            log_ratio = exponent * (np.log(positive) - log_k)
            return float(np.exp(-np.logaddexp(0.0, -log_ratio)))

        return core
    if op == "arrhenius":
        floor = p["floor"]
        return lambda z, b: np.exp(-k / (float(np.logaddexp(0.0, z)) + floor))
    if op == "antoine":
        # Log-pressure form A - B/(C+T), before the shared capacity bound.
        floor, level = p["floor"], p["level"]
        return lambda z, b: (
            level - k / (threshold + (float(np.logaddexp(0.0, z)) + floor))
        )
    if op == "gaussian_optimum":
        center = p["center"]
        return lambda z, b: np.exp(-0.5 * ((z - center) / width) ** 2)
    if op == "parabolic_optimum":
        center, level = p["center"], p["level"]
        return lambda z, b: level - ((z - center) / width) ** 2
    if op == "deadband":
        return lambda z, b: np.sign(z) * max(0.0, abs(z) - threshold)
    if op == "clip":
        low, high = p["center"] - width, p["center"] + width
        return lambda z, b: low if z < low else high if z > high else z
    if op == "knee":
        leak = p["leak"]
        return lambda z, b: z if z <= threshold else threshold + leak * (z - threshold)
    if op == "leaky_relu":
        leak = p["leak"]
        return lambda z, b: z if z >= 0 else leak * z
    if op == "decreasing_quadratic":
        floor, level = p["floor"], p["level"]
        return lambda z, b: level - k * (float(np.logaddexp(0.0, z)) + floor) ** 2
    if op == "quadratic_friction":
        floor = p["floor"]
        return lambda z, b: k * (float(np.logaddexp(0.0, z)) + floor) ** 2
    if op == "stefan_boltzmann":
        floor = p["floor"]
        return lambda z, b: ((float(np.logaddexp(0.0, z)) + floor) / k) ** 4
    if op == "queue":
        # Utilization below one; finite headroom is sampled per edge. This
        # bounds the singularity without hiding it behind inverse-square input.
        floor = p["floor"]
        headroom = 1 - floor

        def core(z, b):
            rho = headroom * np.exp(-np.logaddexp(0.0, -z))
            return 1 / max(floor, 1 - rho) - 1

        return core
    if op == "min":
        return lambda z, b: min(z, b)
    if op == "max":
        return lambda z, b: max(z, b)
    if op == "soft_min":
        return lambda z, b: -width * np.logaddexp(-z / width, -b / width)
    if op == "ratio":
        floor = p["floor"]
        return lambda z, b: z / np.copysign(max(abs(b), floor), b)
    if op == "swiglu":
        return lambda z, b: z * np.exp(-np.logaddexp(0.0, -z)) * b
    raise ValueError(f"Unknown operator: {op}")


def _temporal_update(op, p):
    """update(z, previous): one causal step from the previous bounded output.

    An update owns its memory, which starts empty. It consumes only the current
    and past input, never a later one.
    """
    k, threshold = p["k"], p["threshold"]
    if op == "first_order_lag":
        alpha = _lag_alpha(p["tau"])
        return lambda z, previous: previous + alpha * (z - previous)
    if op == "delay":
        queue = deque([0.0] * p["steps"], maxlen=p["steps"])

        def update(z, previous):
            result = queue.popleft()
            queue.append(z)
            return result

        return update
    if op == "integrator":
        # A sampled leak and finite capacity keep feedback stable. Leak zero
        # supports a true bounded accumulator; integration starts in warm-up.
        keep, tau = 1 - p["leak"], p["tau"]
        return lambda z, previous: keep * previous + z / tau
    if op in ("hysteresis", "regime_switch"):
        # Hysteretic selection avoids rapid chattering near a threshold.
        upper, lower = p["center"] + threshold, p["center"] - threshold
        on = False
        if op == "hysteresis":

            def update(z, previous):
                nonlocal on
                if z > upper:
                    on = True
                elif z < lower:
                    on = False
                return float(on)

            return update
        high_gain, leak = p["high_gain"], p["leak"]

        def update(z, previous):
            nonlocal on
            if z > upper:
                on = True
            elif z < lower:
                on = False
            return (high_gain if on else leak) * z

        return update
    if op == "rate_limiter":
        fall, rise = -p["fall_rate"], p["rise_rate"]

        def update(z, previous):
            change = z - previous
            return previous + (
                fall if change < fall else rise if change > rise else change
            )

        return update
    if op in ("erlang", "diffusion"):
        # Cascaded first-order filters give Erlang-like impulse responses.
        # Geometrically spaced time constants make the diffusion variant broader.
        stages = [0.0] * p["stages"]
        alphas = _cascade_alphas(
            p["tau"], len(stages), p["spread"] if op == "diffusion" else 1
        )

        def update(z, previous):
            result = z
            for index, alpha in enumerate(alphas):
                stages[index] += alpha * (result - stages[index])
                result = stages[index]
            return result

        return update
    if op == "resonance":
        # Stable complex pole: an input impulse excites a damped oscillator.
        pole = _resonance_pole(p["tau"], p["period"])
        drive = 1 - abs(pole)
        oscillator = 0j

        def update(z, previous):
            nonlocal oscillator
            oscillator = pole * oscillator + drive * z
            return oscillator.imag

        return update
    if op == "pid":
        # A genuine local feedback controller: error measures a saved setpoint
        # against the causal input, and its output can drive downstream nodes.
        center, tau, ki, kd = p["center"], p["tau"], p["ki"], p["kd"]
        low, high = -p["limit"], p["limit"]
        integral, last_error = 0.0, None

        def update(z, previous):
            nonlocal integral, last_error
            error = center - z
            total = integral + error / tau
            total = low if total < low else high if total > high else total
            derivative = error - (error if last_error is None else last_error)
            integral, last_error = float(total), error
            return k * error + ki * total + kd * derivative

        return update
    if op == "volatility_coupling":
        # Deterministic volatility estimate, not an extra random innovation.
        alpha = _lag_alpha(p["tau"])
        keep = 1 - alpha
        mean, variance = 0.0, 0.0

        def update(z, previous):
            nonlocal mean, variance
            variance = keep * variance + alpha * (z - mean) ** 2
            mean = mean + alpha * (z - mean)
            return z * (1 + k * np.sqrt(variance))

        return update
    if op == "second_order":
        # Settling with optional overshoot and ringing around the NEW level.
        # (resonance above is band-pass: it rings and returns to zero.)
        a11, a12, a21, a22, b1, b2 = second_order_coefficients(
            p["period"], p.get("damping", 0.5)
        )
        position, velocity = 0.0, 0.0

        def update(z, previous):
            nonlocal position, velocity
            position, velocity = (
                a11 * position + a12 * velocity + b1 * z,
                a21 * position + a22 * velocity + b2 * z,
            )
            return position

        return update
    raise ValueError(f"Unknown operator: {op}")


_TEMPORAL_SET = frozenset(TEMPORAL_OPERATORS)


def _extra_core(op, p):
    """core(z, b) of a static or temporal operator, before saturation."""
    if op not in _TEMPORAL_SET:
        return _static_core(op, p)
    update = _temporal_update(op, p)
    low, high = -p["limit"], p["limit"]
    output = 0.0

    def core(z, b):
        nonlocal output
        result = update(z, output)
        output = float(low if result < low else high if result > high else result)
        return output

    return core


def compile_instruction(instruction):
    """Bind one instruction's saved parameters once; return step(a, b).

    The simulator evaluates every instruction on every row, so the operator and
    its parameters are resolved here instead of per row. A step owns the memory
    of a temporal operator: compile again to start a fresh response. The saved
    instruction is only read, never changed.
    """
    op = instruction["op"]
    gain, offset = instruction["gain"], instruction["offset"]
    limit = instruction["limit"]
    core = _base_core(op, instruction)
    if core is None:
        core = _extra_core(op, instruction)

    def step(a, b):
        # Soft saturation models finite capacity and controls nonlinear
        # feedback. Accumulated trend/walk states are NOT clipped this way.
        return float(limit * np.tanh(core(gain * a + offset, b) / limit))

    return step


def compile_steps(program):
    """[(step, a, b)]: each step scales result `a` and may also use result `b`.

    Index 0 is the edge input; instruction i produces result i + 1.
    """
    return [(compile_instruction(i), i["a"], i["b"]) for i in program]


def run_steps(steps):
    """Expression DAG as run(value), with reusable intermediate results.

    Evaluation in instruction order is dynamic programming: shared results are
    computed once, and the program contains no dead instructions.
    """

    def run(value):
        values = [value]
        for step, a, b in steps:
            values.append(step(values[a], values[b]))
        return values[-1]

    return run


def compile_program(program):
    """run(value) for a saved program, with fresh memory for temporal steps."""
    return run_steps(compile_steps(program))


def evaluate_extra(op, z, b, instruction, state):
    """Evaluate one static or temporal instruction, before its saturation.

    `state` is this instruction's memory in one simulation: pass the same
    dictionary on every row, and an empty one to start a fresh response. It
    keeps the operator bound to the parameters of the first call.
    """
    core = state.get("core")
    if core is None:
        core = state["core"] = _extra_core(op, instruction)
    return core(z, b)


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


def _sums_in_order():
    """Whether ndarray.sum() adds fewer than eight doubles from zero, in order.

    That is how NumPy sums a short array; longer ones are summed pairwise. The
    probes span 24 orders of magnitude, so another order of additions, or
    another starting value (signed zeros), does not reproduce their sums.
    """
    rng = np.random.default_rng(0)
    probes = [[-0.0] * count for count in range(1, 8)]
    for count in range(8):
        for _ in range(48):
            scale = 10.0 ** rng.integers(-12, 13, size=count)
            probes.append((rng.normal(size=count) * scale).tolist())
    for probe in probes:
        total = 0.0
        for value in probe:
            total += value
        if total.hex() != float(np.asarray(probe, dtype=float).sum()).hex():
            return False
    return True


# Summing a few parent effects through an array costs more than the joint map
# itself. Add them directly if the check confirms that this NumPy does exactly
# that, and leave the sum to NumPy otherwise.
_IN_ORDER_BELOW = 8 if _sums_in_order() else 0


def _total(effects):
    """Exactly np.asarray(effects, dtype=float).sum(), without the array."""
    if len(effects) >= _IN_ORDER_BELOW:
        return np.asarray(effects, dtype=float).sum()
    total = 0.0
    for effect in effects:
        total += effect
    return total


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
    prepared = {
        "indices": indices,
        "unused": unused,
        "unused_indices": np.flatnonzero(unused).tolist(),
    }
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
    if not len(effects) or spec["mode"] == "sum":
        return float(_total(effects))
    values = np.asarray(effects, dtype=float)
    prepared = (
        prepare_parent_combination(spec, len(values)) if prepared is None else prepared
    )
    indices = prepared["indices"]
    selected = values[indices]
    z = selected / spec["scale"]
    mode, scale = spec["mode"], spec["scale"]
    if mode in _POSITIVE_PARENT_MODES:
        positive = _positive(z, spec)
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
        result = _total(effects) + strength * joint
    elif spec["application"] == "replace":
        result = (
            _total([effects[index] for index in prepared["unused_indices"]])
            + strength * joint
        )
    else:
        raise ValueError("Joint application must be additive or replace")
    return float(clip_scalar(result, -spec["numeric_limit"], spec["numeric_limit"]))


def bind_node_link(spec):
    """link(forcing) of one node, with its saved parameters read once."""
    if spec is None or spec["mode"] == "identity":
        return lambda forcing: forcing
    mode = spec["mode"]
    if mode == "saturation":
        capacity = spec["capacity"]

        def shape(z):
            return capacity * np.tanh(z / capacity)
    elif mode == "softplus":

        def shape(z):
            return np.logaddexp(0.0, z)
    elif mode == "exponential":

        def shape(z):
            return np.exp(clip_scalar(z, -30, 30))
    elif mode == "sigmoid":

        def shape(z):
            return np.exp(-np.logaddexp(0.0, -z))
    elif mode == "logarithm":

        def shape(z):
            return np.sign(z) * np.log1p(abs(z))
    else:
        raise ValueError(f"Unknown node link: {mode}")
    input_offset, scale = spec["input_offset"], spec["scale"]
    gain, output_offset = spec["gain"], spec["output_offset"]
    low, high = -spec["numeric_limit"], spec["numeric_limit"]

    def link(forcing):
        result = gain * shape((forcing - input_offset) / scale) + output_offset
        return float(low if result < low else high if result > high else result)

    return link


def apply_node_link(forcing, spec):
    """Transform the parent aggregate; stochastic innovations remain outside g."""
    return bind_node_link(spec)(forcing)

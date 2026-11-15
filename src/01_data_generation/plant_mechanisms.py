"""Closed-loop units and lifecycle shapes that one-way SCM edges cannot express.

A plant reads the standardized SCM outcome as its demand and returns recorded
channels. The loop is INSIDE the unit: the controlled value decides the next
actuator state, which moves the controlled value. Every parameter is a saved
plain number; evaluation is causal and never uses whole-series statistics.

Families (band and actuator ranges are 0..1 before the saved output gain):
  relay          on/off control with hysteresis: tank, compressor, thermostat
  pid            setpoint tracking with dead time, saturation and windup
  batch          fill-and-reset cycles: hopper, clogging filter, totalizer
  step_response  first/second-order settling after level changes
  totalizer      running total of a flow: meter, counter, operating hours
"""

import math
from collections import deque

import numpy as np

try:
    from .effect_mechanisms import second_order_coefficients
except ImportError:
    from effect_mechanisms import second_order_coefficients

PLANT_FAMILIES = (
    "relay",
    "pid",
    "batch",
    "step_response",
    "totalizer",
    "stick_slip",
    "interlock",
    "coupled_storage",
    "rotating_machine",
)
DEFAULT_PLANT_WEIGHTS = {
    "relay": 0.4,
    "pid": 0.25,
    "batch": 0.2,
    "step_response": 0.15,
    "totalizer": 0.1,
    "stick_slip": 0.0,  # Opt-in; runner enables these lightweight families.
    "interlock": 0.0,
    "coupled_storage": 0.0,
    "rotating_machine": 0.0,
}
LIFECYCLE_KINDS = (
    "saturating",
    "exponential_wear",
    "onset",
    "logistic",
    "maintenance_sawtooth",
    "level_shifts",
)
# Which channel becomes the recorded outcome, and how often the others are
# exported as covariates. Binary status is a covariate, never the target.
_OUTCOME_CHANNELS = {
    "coupled_storage": (("pv", "inventory_b", "outflow"), (0.5, 0.3, 0.2)),
    "rotating_machine": (("pv",), (1.0,)),
    "relay": (("pv", "actuator", "secondary"), (0.6, 0.2, 0.2)),
    "pid": (("pv", "mv"), (0.65, 0.35)),
    "batch": (("pv",), (1.0,)),
    "step_response": (("pv",), (1.0,)),
    "totalizer": (("pv",), (1.0,)),
    "stick_slip": (("pv",), (1.0,)),
    "interlock": (("pv", "actuator"), (0.7, 0.3)),
}
_EXPORTS = {
    "coupled_storage": {
        "pv": 0.4,
        "inventory_b": 0.5,
        "inflow": 0.5,
        "transfer": 0.4,
        "outflow": 0.5,
        "mode": 0.6,
    },
    "rotating_machine": {"speed": 0.7, "mode": 0.6, "load_torque": 0.4},
    "relay": {"pv": 0.4, "status": 0.45, "actuator": 0.3, "secondary": 0.25},
    "pid": {"pv": 0.5, "setpoint": 0.6, "mv": 0.45},
    "batch": {"status": 0.3},
    "step_response": {"setpoint": 0.5},
    "totalizer": {"rate": 0.4, "status": 0.3},
    "stick_slip": {"position": 0.5, "drive": 0.5, "status": 0.4},
    "interlock": {"pv": 0.5, "actuator": 0.5, "status": 0.5, "reset": 0.5},
}


def _log_uniform(rng, low, high):
    return float(np.exp(rng.uniform(np.log(low), np.log(high))))


def _levels(rng, rows):
    """A piecewise-constant command: saved levels and a dwell-time range.

    Dwell is capped by the table length so that level changes are recorded.
    """
    low = _log_uniform(rng, 15, max(16, min(500, rows / 3)))
    return {
        "levels": rng.uniform(0.15, 0.85, int(rng.integers(2, 6))).tolist(),
        "dwell": [int(low), int(low * rng.uniform(1.2, 3.0)) + 1],
        "ramp": 0 if rng.random() < 0.7 else int(_log_uniform(rng, 5, 150)),
    }


def sample_plant(rng, rows, weights, first_export_index):
    """Draw one unit. Durations are in samples, not fractions of the table."""
    names = list(weights)
    probability = np.array([weights[n] for n in names], dtype=float)
    family = str(rng.choice(names, p=probability / probability.sum()))
    spec = {
        "family": family,
        "seed": int(rng.integers(0, 2**63)),
        "duration": int(rows),
        # Demand is centred by a slow causal running mean/scale (a relative
        # load), so its influence cannot sit saturated for the whole table.
        "load_tau": _log_uniform(rng, 100, 600),
        "load_width": float(rng.uniform(0.7, 2.5)),
        "load_coupling": 0.0 if rng.random() < 0.3 else float(rng.uniform(0.2, 0.9)),
        "load_lag": int(rng.integers(0, 4)),
        "process_noise": 0.0 if rng.random() < 0.4 else _log_uniform(rng, 0.001, 0.03),
        "output_gain": float(rng.choice([-1, 1]) * _log_uniform(rng, 0.5, 8)),
    }
    if family in ("coupled_storage", "rotating_machine"):
        # A single categorical state coordinates all actuator settings. It is
        # semi-Markov (persistent dwell), not several independent binary flags.
        count = int(rng.integers(3, 6))
        command = _levels(rng, rows)
        command.update(levels=list(range(count)), ramp=0)
        spec.update(mode_command=command)
        if family == "coupled_storage":
            spec.update(
                initial_inventory=rng.uniform(0.1, 1.5, 2).tolist(),
                inlet_by_mode=rng.uniform(0.0, 0.04, count).tolist(),
                transfer_by_mode=rng.uniform(0.005, 0.10, count).tolist(),
                outlet_by_mode=rng.uniform(0.005, 0.10, count).tolist(),
                equations={
                    "inflow[t]": "max(0, inlet[mode[t]] * drive[t] * (1 + noise[t]))",
                    "transfer[t]": "transfer_rate[mode[t]] * inventory_a[t-1]",
                    "outflow[t]": "outlet_rate[mode[t]] * inventory_b[t-1]",
                    "inventory_a[t]": "inventory_a[t-1] + inflow[t] - transfer[t]",
                    "inventory_b[t]": "inventory_b[t-1] + transfer[t] - outflow[t]",
                },
            )
        else:
            harmonics = int(rng.integers(1, 5))
            spec.update(
                speed_by_mode=rng.uniform(0.002, 0.1 / harmonics, count).tolist(),
                amplitude_by_mode=rng.uniform(0.2, 1.5, count).tolist(),
                torque_by_mode=rng.uniform(0.1, 1.0, count).tolist(),
                speed_tau=_log_uniform(rng, 3, 80),
                initial_phase=float(rng.uniform(-np.pi, np.pi)),
                harmonics=(
                    rng.uniform(0.1, 1, harmonics) / np.arange(1, harmonics + 1)
                ).tolist(),
                harmonic_phases=rng.uniform(-np.pi, np.pi, harmonics).tolist(),
                equations={
                    "speed[t]": "speed[t-1] + alpha * (command_speed[mode[t]] * drive[t] - speed[t-1])",
                    "phase[t]": "phase[t-1] + 2*pi*speed[t-1]",
                    "pv[t]": "amplitude[mode[t]] * sum(harmonic[k] * sin(k*phase[t]+offset[k])) + noise[t]",
                },
            )
    elif family == "relay":
        spec.update(
            on_time=_log_uniform(rng, 4, 300),
            off_time=_log_uniform(rng, 4, 300),
            # Zero leak gives straight sawtooth edges (tank); a positive leak
            # gives exponential charge/discharge curves (thermal).
            leak=0.0 if rng.random() < 0.45 else float(rng.uniform(0.3, 3.0)),
            ambient=float(-rng.uniform(0.3, 2.0)),
            dead_time=int(rng.choice([0, 0, 0, 1, 2, 3, 5])),
            min_on=int(rng.integers(1, 6)),
            min_off=int(rng.integers(1, 6)),
            band_jitter=float(rng.uniform(0, 0.08)),
            # Capacity fade: the unit needs ever longer to recover the band.
            wear=0.0 if rng.random() < 0.75 else float(rng.uniform(0.2, 0.7)),
            inrush=0.0 if rng.random() < 0.5 else float(rng.uniform(0.3, 3.0)),
            inrush_decay=float(rng.uniform(0.3, 0.85)),
            actuator_ripple=_log_uniform(rng, 0.002, 0.05),
            secondary_tau=_log_uniform(rng, 5, 300),
        )
    elif family == "pid":
        mode = str(rng.choice(["constant", "steps", "ramps"], p=[0.35, 0.45, 0.2]))
        command = _levels(rng, rows)
        if mode == "constant":
            command["levels"] = command["levels"][:1]
        if mode == "ramps":
            command["ramp"] = int(_log_uniform(rng, 10, 200))
        spec.update(
            tau=_log_uniform(rng, 3, 150),
            dead_time=int(_log_uniform(rng, 1, 20)) - 1,
            second_tau=0.0 if rng.random() < 0.6 else _log_uniform(rng, 1, 40),
            # Below one is sluggish; above one overshoots and can hunt.
            aggressiveness=_log_uniform(rng, 0.25, 2.5),
            integral_ratio=_log_uniform(rng, 0.3, 3.0),
            derivative=0.0 if rng.random() < 0.7 else float(rng.uniform(0.05, 0.3)),
            anti_windup=bool(rng.random() < 0.8),
            sensor_noise=0.0 if rng.random() < 0.5 else _log_uniform(rng, 0.001, 0.02),
            disturbance_gain=float(rng.uniform(0.2, 0.8)),
            setpoint_mode=mode,
            command=command,
        )
        if mode == "constant" and not (spec["load_coupling"] or spec["sensor_noise"]):
            # Pure regulation with nothing to reject would be an exact constant.
            spec["process_noise"] = max(spec["process_noise"], 0.002)
    elif family == "batch":
        cycle = _log_uniform(rng, 8, 600)
        spec.update(
            cycle_time=cycle,
            shape=str(rng.choice(["linear", "saturating", "accelerating"])),
            curvature=float(rng.uniform(1.5, 4.0)),
            threshold_jitter=float(rng.uniform(0, 0.25)),
            rate_jitter=float(rng.uniform(0, 0.4)),
            reset_steps=1
            if rng.random() < 0.5
            else int(rng.integers(2, max(3, int(0.15 * cycle)) + 1)),
            reset_level_jitter=float(rng.uniform(0, 0.1)),
            hold=0
            if rng.random() < 0.65
            else int(_log_uniform(rng, 1, 0.5 * cycle + 2)),
            trigger=str(rng.choice(["threshold", "timer"], p=[0.7, 0.3])),
            timer_period=int(max(4, cycle * rng.uniform(0.6, 1.2))),
        )
    elif family == "step_response":
        spec.update(
            first_order=bool(rng.random() < 0.35),
            period=_log_uniform(rng, 8, 300),
            damping=_log_uniform(rng, 0.04, 1.6),
            tau=_log_uniform(rng, 3, 250),
            dead_time=int(rng.choice([0, 0, 1, 2, 4, 8])),
            command=_levels(rng, rows),
        )
    elif family == "totalizer":
        run = _log_uniform(rng, 10, max(11, min(600, rows / 2)))
        spec.update(
            # Pauses give plateaus: the counted flow stops, the total holds.
            pauses=bool(rng.random() < 0.5),
            command={
                "levels": [0.0, 1.0],
                "dwell": [int(run), int(run * rng.uniform(1.2, 3.0)) + 1],
                "ramp": 0,
            },
            rate_floor=float(rng.uniform(0, 0.6)),
        )
    elif family == "stick_slip":
        spec.update(
            static_friction=float(rng.uniform(0.7, 1.4)),
            kinetic_ratio=float(rng.uniform(0.15, 0.6)),
            drive_rate=_log_uniform(rng, 0.003, 0.08),
            release_fraction=float(rng.uniform(0.35, 0.9)),
        )
    elif family == "interlock":
        spec.update(
            trip_level=float(rng.uniform(0.65, 0.9)),
            safe_level=float(rng.uniform(0.1, 0.4)),
            tau=_log_uniform(rng, 5, 60),
            cooling_tau=_log_uniform(rng, 5, 80),
            command={"levels": [0.0, 1.0], "dwell": [15, 80], "ramp": 0},
        )
    else:
        raise ValueError(f"Unknown plant family: {family}")
    channels, probability = _OUTCOME_CHANNELS[family]
    spec["outcome_channel"] = str(rng.choice(channels, p=probability))
    exports = [
        channel
        for channel, chance in _EXPORTS[family].items()
        if channel != spec["outcome_channel"] and rng.random() < chance
    ]
    if spec["load_coupling"] and rng.random() < 0.25:
        exports.append("load")
    spec["exports"] = [
        {"name": f"covariate_{first_export_index + i:02d}", "channel": channel}
        for i, channel in enumerate(exports)
    ]
    return spec


def _command_path(command, rng, length):
    """Semi-Markov level changes, optionally joined by finite ramps."""
    levels = command["levels"]
    result = np.empty(length)
    position, index, value = 0, int(rng.integers(len(levels))), None
    while position < length:
        target = levels[index]
        dwell = int(rng.integers(command["dwell"][0], command["dwell"][1] + 1))
        stop = min(length, position + max(1, dwell))
        ramp = min(command["ramp"], stop - position) if value is not None else 0
        result[position:stop] = target
        if ramp:
            result[position : position + ramp] = value + (target - value) * (
                np.arange(1, ramp + 1) / ramp
            )
        value, position = target, stop
        if len(levels) > 1:
            index = (index + 1 + int(rng.integers(len(levels) - 1))) % len(levels)
    return result


def relative_load(spec, demand):
    """Demand as a 0..1 load relative to its own slow causal mean and spread."""
    alpha = 1 / spec["load_tau"]
    mean, variance = float(demand[0]), 1.0
    result = np.empty(len(demand))
    for index, value in enumerate(demand):
        deviation = value - mean
        result[index] = 0.5 * (
            1 + math.tanh(deviation / (spec["load_width"] * math.sqrt(variance) + 1e-9))
        )
        mean += alpha * deviation
        variance += alpha * (deviation * deviation - variance)
    lag = spec["load_lag"]
    if lag:
        result = np.concatenate([np.full(lag, result[0]), result[:-lag]])
    return result


def simulate_plant(spec, demand, time):
    """Return every channel over the full simulated clock (warm-up included)."""
    rng = np.random.default_rng(spec["seed"])
    length = len(demand)
    load = relative_load(spec, np.asarray(demand, dtype=float))
    # Relative demand around one: 1 is nominal, coupling zero means autonomous.
    drive = 1 + spec["load_coupling"] * (2 * load - 1)
    noise = rng.normal(0, 1, length) * spec["process_noise"]
    family = spec["family"]
    channels = {"load": load}
    if family == "coupled_storage":
        channels.update(_coupled_storage(spec, drive, noise, rng))
    elif family == "rotating_machine":
        channels.update(_rotating_machine(spec, drive, noise, rng))
    elif family == "relay":
        channels.update(_relay(spec, drive, noise, time, rng))
    elif family == "pid":
        channels.update(_pid(spec, load, noise, rng))
    elif family == "batch":
        channels.update(_batch(spec, drive, noise, time, rng))
    elif family == "step_response":
        channels.update(_step_response(spec, load, noise, rng))
    elif family == "totalizer":
        running = (
            _command_path(spec["command"], rng, length)
            if spec["pauses"]
            else np.ones(length)
        )
        floor = spec["rate_floor"]
        rate = running * np.maximum(0.0, floor + (1 - floor) * drive + noise)
        # A full-rate run over the whole table counts to about one.
        channels.update(
            pv=np.cumsum(rate) / spec["duration"], rate=rate, status=running
        )
    elif family == "stick_slip":
        channels.update(_stick_slip(spec, drive, noise))
    elif family == "interlock":
        channels.update(_interlock(spec, drive, noise, rng))
    else:
        raise ValueError(f"Unknown plant family: {family}")
    return channels


def _stick_slip(spec, drive, noise):
    """Overdamped spring/slider: static breakaway exceeds sliding friction.

    A small causal state update per row avoids a numerical ODE solver. During
    sticking, the driven spring loads up; slip releases stored displacement.
    """
    length = len(drive)
    pv, position, command, status = (np.empty(length) for _ in range(4))
    static = spec["static_friction"]
    kinetic = static * spec["kinetic_ratio"]
    rate, release = spec["drive_rate"], spec["release_fraction"]
    target = slider = 0.0
    slipping = False
    for i in range(length):
        step = rate * max(0.05, drive[i])
        target += step
        force = target - slider
        if not slipping and force >= static:
            slipping = True
        if slipping:
            slider += release * max(0.0, force - kinetic)
            force = target - slider
            if force <= kinetic + 2 * step:
                slipping = False
        pv[i] = force + noise[i]
        position[i], command[i], status[i] = slider, target, float(slipping)
    return {"pv": pv, "position": position, "drive": command, "status": status}


def _interlock(spec, drive, noise, rng):
    """Latched thermal/pressure trip, with safe-state AND explicit reset edge.

    Decisions read the previous physical state. A reset while unsafe is ignored;
    becoming safe alone never clears a trip. No future commands are consulted.
    """
    reset = _command_path(spec["command"], rng, len(drive))
    pv, actuator, status = (np.empty(len(drive)) for _ in range(3))
    value, tripped, previous_reset = 0.0, False, False
    high, low = spec["trip_level"], spec["safe_level"]
    tau, cooling = spec["tau"], spec["cooling_tau"]
    for i in range(len(drive)):
        reset_on = bool(reset[i])
        if value >= high:
            tripped = True
        elif tripped and value <= low and reset_on and not previous_reset:
            tripped = False
        enabled = not tripped
        value += ((drive[i] - value) / tau if enabled else -value / cooling) + noise[i]
        pv[i], actuator[i], status[i] = value, float(enabled), float(tripped)
        previous_reset = reset_on
    return {"pv": pv, "actuator": actuator, "status": status, "reset": reset}


def _relay(spec, drive, noise, time, rng):
    length = len(drive)
    leak_rate = spec["leak"] / spec["off_time"]
    ambient = spec["ambient"]
    # Nominal mid-band fall takes off_time; nominal rise at the top of the band
    # still proceeds at 1/on_time. Demand above nominal can exceed capacity.
    drain = max(0.0, 1 / spec["off_time"] - leak_rate * (0.5 - ambient))
    gain = 1 / spec["on_time"] + drain + leak_rate * (1 - ambient)
    jitter = rng.normal(0, 1, length) * spec["band_jitter"]
    ripple = rng.normal(0, 1, length) * spec["actuator_ripple"]
    pending = deque([0.0] * spec["dead_time"])
    value, on, since, surge, warm = 0.5, False, 10**6, 0.0, 0.0
    low, high = 0.0, 1.0
    pv, status, actuator, secondary = (np.empty(length) for _ in range(4))
    for i in range(length):
        if on and value >= high and since >= spec["min_on"]:
            on, since, low = False, 0, jitter[i]
        elif not on and value <= low and since >= spec["min_off"]:
            on, since, high, surge = True, 0, 1 + jitter[i], spec["inrush"]
        since += 1
        pending.append(float(on))
        effective = pending.popleft()
        # Wear eats the net recovery margin, so on-phases stretch but still end.
        fade = spec["wear"] * min(1.0, max(0.0, time[i] / spec["duration"]))
        value += (
            effective * (gain - fade / spec["on_time"])
            - drain * max(0.05, drive[i])
            - leak_rate * (value - ambient)
            + noise[i]
        )
        value = min(2.5, max(-1.5, value))  # Physical empty/full limits.
        pv[i], status[i] = value, float(on)
        actuator[i] = (1 + ripple[i] + surge) if on else 0.0
        surge *= spec["inrush_decay"]
        warm += (float(on) - warm) / spec["secondary_tau"]
        secondary[i] = warm
    return {"pv": pv, "status": status, "actuator": actuator, "secondary": secondary}


def _pid(spec, load, noise, rng):
    length = len(load)
    setpoint = _command_path(spec["command"], rng, length)
    sensor = rng.normal(0, 1, length) * spec["sensor_noise"]
    tau, second = spec["tau"], spec["second_tau"]
    lag = spec["dead_time"] + second + 1
    gain = min(20.0, spec["aggressiveness"] * tau / lag)
    integral_gain = gain / (min(tau, 4 * lag) * spec["integral_ratio"])
    disturbance = spec["load_coupling"] * spec["disturbance_gain"] * (2 * load - 1)
    pending = deque([setpoint[0]] * spec["dead_time"])
    first = output = measured = previous = float(setpoint[0])
    integral = float(setpoint[0])
    pv, mv = np.empty(length), np.empty(length)
    for i in range(length):
        error = setpoint[i] - measured
        raw = (
            gain * error + integral - spec["derivative"] * gain * (measured - previous)
        )
        command = min(1.0, max(0.0, raw))
        if not spec["anti_windup"] or command == raw or (raw - command) * error < 0:
            integral = min(3.0, max(-3.0, integral + integral_gain * error))
        pending.append(command)
        first += (pending.popleft() - disturbance[i] - first) / tau + noise[i]
        output = output + (first - output) / second if second else first
        previous, measured = measured, output + sensor[i]
        pv[i], mv[i] = measured, command
    return {"pv": pv, "setpoint": setpoint, "mv": mv}


def _batch(spec, drive, noise, time, rng):
    length = len(drive)
    curvature, shape = spec["curvature"], spec["shape"]

    def level(progress):
        progress = min(1.15, max(0.0, progress))
        if shape == "saturating":
            return 1 - (1 - min(1.0, progress)) ** curvature
        if shape == "accelerating":
            return progress**curvature
        return progress

    def cycle():
        return (
            max(0.3, 1 + spec["threshold_jitter"] * rng.normal()),
            float(np.exp(spec["rate_jitter"] * rng.normal())),
            abs(spec["reset_level_jitter"] * rng.normal()),
        )

    def progress_at(value):
        value = min(1.0, max(0.0, value))
        if shape == "saturating":
            return 1 - (1 - value) ** (1 / curvature)
        if shape == "accelerating":
            return value ** (1 / curvature)
        return value

    threshold, rate, floor = cycle()
    progress, phase, counter, start = float(rng.uniform(0, 1)), "fill", 0, 0.0
    pv, status = np.empty(length), np.zeros(length)
    for i in range(length):
        if phase == "fill":
            progress += rate * max(0.05, drive[i]) / spec["cycle_time"] + noise[i]
            value = level(progress)
            timed = spec["trigger"] == "timer" and time[i] % spec["timer_period"] == 0
            if timed or (spec["trigger"] == "threshold" and progress >= threshold):
                phase, counter, start = "hold", spec["hold"], value
                if not spec["hold"]:
                    phase, counter = "dump", 0
        elif phase == "hold":
            value = start
            counter -= 1
            if counter <= 0:
                phase, counter = "dump", 0
        else:
            # The reset is recorded: the peak, then a finite or instant dump.
            counter += 1
            status[i] = 1.0
            value = start + (floor - start) * counter / spec["reset_steps"]
            if counter >= spec["reset_steps"]:
                progress, phase = progress_at(floor), "fill"
                threshold, rate, floor = cycle()
        pv[i] = value
    return {"pv": pv, "status": status}


def _step_response(spec, load, noise, rng):
    length = len(load)
    command = _command_path(spec["command"], rng, length) + spec["load_coupling"] * (
        load - 0.5
    )
    pending = deque([command[0]] * spec["dead_time"])
    position, velocity = float(command[0]), 0.0
    if spec["first_order"]:
        a11, a12, a21, a22 = float(np.exp(-1 / spec["tau"])), 0.0, 0.0, 0.0
        b1, b2 = 1 - a11, 0.0
    else:
        a11, a12, a21, a22, b1, b2 = second_order_coefficients(
            spec["period"], spec["damping"]
        )
    pv = np.empty(length)
    for i in range(length):
        pending.append(command[i])
        target = pending.popleft()
        position, velocity = (
            a11 * position + a12 * velocity + b1 * target + noise[i],
            a21 * position + a22 * velocity + b2 * target,
        )
        pv[i] = position
    return {"pv": pv, "setpoint": command}


def sample_lifecycle(rng, rows, amplitude):
    """A saved deterministic wear/commissioning curve reaching about `amplitude`."""
    kind = str(rng.choice(LIFECYCLE_KINDS))
    trend = {
        "kind": kind,
        "duration": int(rows),
        "amplitude": float(amplitude),
        # Keys shared with the classic trend kinds; unused by lifecycle shapes.
        "initial_slope": float(amplitude / rows),
        "event_parent": None,
        "event_time": None,
        "slope_multiplier": 1.0,
    }
    if kind == "saturating":
        trend["time_constant"] = float(rng.uniform(0.08, 0.4))
    elif kind == "exponential_wear":
        trend["rate"] = float(rng.uniform(2.5, 7.0))
    elif kind == "onset":
        trend.update(
            start=float(rng.uniform(0.2, 0.7)), power=float(rng.uniform(1.0, 3.0))
        )
    elif kind == "logistic":
        trend.update(
            center=float(rng.uniform(0.25, 0.75)), width=float(rng.uniform(0.03, 0.15))
        )
    elif kind == "maintenance_sawtooth":
        # Wear accumulates, service restores part of it. Irregular intervals.
        count = int(rng.integers(2, 9))
        gaps = rng.dirichlet(np.full(count + 1, 4.0))
        trend.update(
            resets=np.cumsum(gaps)[:-1].tolist(),
            restore=rng.uniform(0.6, 1.0, count).tolist(),
            rates=np.exp(rng.normal(0, 0.3, count + 1)).tolist(),
            power=float(rng.uniform(0.7, 2.5)),
        )
    elif kind == "level_shifts":
        count = int(rng.integers(1, 5))
        trend.update(
            times=np.sort(rng.uniform(0.08, 0.92, count)).tolist(),
            steps=(rng.choice([-1, 1], count) * rng.uniform(0.3, 1.0, count)).tolist(),
            widths=np.where(
                rng.random(count) < 0.6, 0.0, rng.uniform(0.005, 0.08, count)
            ).tolist(),
        )
    return trend


def lifecycle_path(trend, time):
    """Trend contribution at each clock time; zero during warm-up (time < 0)."""
    time = np.asarray(time, dtype=float)
    u = np.clip(time / trend["duration"], 0.0, 1.0)
    kind = trend["kind"]
    if kind == "saturating":
        c = trend["time_constant"]
        shape = -np.expm1(-u / c) / -np.expm1(-1 / c)
    elif kind == "exponential_wear":
        shape = np.expm1(trend["rate"] * u) / np.expm1(trend["rate"])
    elif kind == "onset":
        shape = (
            np.maximum(0.0, (u - trend["start"]) / (1 - trend["start"]))
            ** trend["power"]
        )
    elif kind == "logistic":
        c, w = trend["center"], trend["width"]
        curve = np.tanh((u - c) / (2 * w))
        low, high = np.tanh(-c / (2 * w)), np.tanh((1 - c) / (2 * w))
        shape = (curve - low) / (high - low)
    elif kind == "maintenance_sawtooth":
        edges = [0.0] + list(trend["resets"]) + [1.0]
        mean_gap = 1 / (len(edges) - 1)
        shape, base = np.zeros(len(u)), 0.0
        for index in range(len(edges) - 1):
            start, stop = edges[index], edges[index + 1]
            mask = (u >= start) & ((u < stop) | (index == len(edges) - 2))
            growth = (
                trend["rates"][index] * ((u[mask] - start) / mean_gap) ** trend["power"]
            )
            shape[mask] = base + growth
            if index < len(trend["restore"]):
                peak = (
                    base
                    + trend["rates"][index]
                    * ((stop - start) / mean_gap) ** (trend["power"])
                )
                base = peak * (1 - trend["restore"][index])
    elif kind == "level_shifts":
        shape = np.zeros(len(u))
        for moment, step, width in zip(trend["times"], trend["steps"], trend["widths"]):
            shape += step * (
                np.clip((u - moment) / width, 0.0, 1.0) if width else (u >= moment)
            )
    else:
        raise ValueError(f"Unknown lifecycle trend: {kind}")
    return trend["amplitude"] * np.where(time < 0, 0.0, shape)


def _coupled_storage(spec, drive, noise, rng):
    """Two inventories with an explicit shared transfer, in units per sample.

    Transfer appears with opposite signs in the balances, so it cannot create
    material. Rates below one withdraw only available previous inventory; no
    clipping correction or independent inventory noise can break conservation.
    Random disturbances act on inlet flow, where they enter the balance.
    """
    mode = _command_path(spec["mode_command"], rng, len(drive)).astype(int)
    inlet = np.asarray(spec["inlet_by_mode"])[mode]
    transfer_rate = np.asarray(spec["transfer_by_mode"])[mode]
    outlet_rate = np.asarray(spec["outlet_by_mode"])[mode]
    inflow = np.maximum(0, inlet * drive * (1 + noise))
    a, b, transfer, outflow = (np.empty(len(drive)) for _ in range(4))
    left, right = spec["initial_inventory"]
    for i in range(len(drive)):
        transfer[i] = transfer_rate[i] * left
        outflow[i] = outlet_rate[i] * right
        left += inflow[i] - transfer[i]
        right += transfer[i] - outflow[i]
        a[i], b[i] = left, right
    return dict(
        pv=a,
        inventory_b=b,
        inflow=inflow,
        transfer=transfer,
        outflow=outflow,
        mode=mode,
    )


def _rotating_machine(spec, drive, noise, rng):
    """Integrate actual speed: changing mode never resets oscillator phase.

    Speed is in revolutions/sample. Its bounded commands keep all sampled
    harmonics below Nyquist, including the largest permitted demand gain.
    Phase uses previous speed, so future speed cannot change earlier cycles.
    """
    mode = _command_path(spec["mode_command"], rng, len(drive)).astype(int)
    command = np.asarray(spec["speed_by_mode"])[mode] * drive
    speed = np.empty(len(drive))
    alpha = -math.expm1(-1 / spec["speed_tau"])
    previous = spec["speed_by_mode"][int(mode[0])] if len(mode) else 0.0
    for i, target in enumerate(command):
        previous += alpha * (target - previous)
        speed[i] = previous
    phase = np.empty(len(drive))
    if len(phase):
        phase[0] = spec["initial_phase"]
        phase[1:] = phase[0] + 2 * np.pi * np.cumsum(speed[:-1])
    wave = np.zeros(len(drive))
    for k, (gain, offset) in enumerate(
        zip(spec["harmonics"], spec["harmonic_phases"]), 1
    ):
        wave += gain * np.sin(k * phase + offset)
    amplitude = np.asarray(spec["amplitude_by_mode"])[mode]
    torque = np.asarray(spec["torque_by_mode"])[mode] * drive
    return dict(
        pv=amplitude * wave + noise,
        speed=speed,
        phase=phase,
        mode=mode,
        load_torque=torque,
    )


def physical_dag_view(spec):
    """Unroll one physical update for the two shared-state SCM templates.

    This runs only when rendering. The generic source SCM supplies demand;
    inventory/phase recurrences below, rather than that latent demand, define
    the recorded target. Export names are retained for visible channels.
    """
    import networkx as nx

    if spec["family"] == "coupled_storage":
        relations = [
            ("mode", 1, "mode"),
            ("mode", 0, "inflow"),
            ("mode", 0, "transfer"),
            ("mode", 0, "outflow"),
            ("load", 0, "inflow"),
            ("pv", 1, "transfer"),
            ("inventory_b", 1, "outflow"),
            ("pv", 1, "pv"),
            ("inventory_b", 1, "inventory_b"),
            ("inflow", 0, "pv"),
            ("transfer", 0, "pv"),
            ("transfer", 0, "inventory_b"),
            ("outflow", 0, "inventory_b"),
        ]
    else:
        relations = [
            ("mode", 1, "mode"),
            ("mode", 0, "speed"),
            ("load", 0, "speed"),
            ("speed", 1, "speed"),
            ("phase", 1, "phase"),
            ("speed", 1, "phase"),
            ("phase", 0, "pv"),
            ("mode", 0, "pv"),
            ("mode", 0, "load_torque"),
            ("load", 0, "load_torque"),
        ]
    exported = {e["channel"]: e["name"] for e in spec["exports"]}
    graph = nx.DiGraph()

    def vertex(channel, lag):
        target = channel == spec["outcome_channel"]
        variable = (
            "outcome" if target else exported.get(channel, f"covariate_{channel}")
        )
        name = variable if lag == 0 else f"{variable}[t-{lag}]"
        graph.add_node(
            name,
            variable=variable,
            time_offset=-lag,
            visible=target or channel in exported,
            kind="history" if lag else "outcome" if target else "process",
        )
        return name

    for parent, lag, child in relations:
        graph.add_edge(
            vertex(parent, lag), vertex(child, 0), lag=lag, labels=["balance/state"]
        )
    positions = {"outcome": np.array([0.0, 0.0])}
    others = [name for name in graph if name != "outcome"]
    for index, name in enumerate(others):
        angle = 2 * np.pi * index / len(others)
        positions[name] = 3 * np.array([np.cos(angle), np.sin(angle)])
    return graph, positions, set()

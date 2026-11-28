"""Patterns of recorded operational data that an SCM with one-way edges rarely makes.

Each pattern is a small generative model of a kind of signal that is common
in the records of physical and digital systems. It is used like a unit of
plant_mechanisms.py: the standardized SCM outcome arrives as a relative
demand (`drive`, around one), so that the covariates of the table can carry
information about the target, and the pattern's signal becomes the recorded
outcome. All parameters are saved plain numbers with wide ranges; nothing is
fitted to, or copied from, a recorded series.

  held_value     a controlled variable: noise around a setpoint that sometimes moves
  daily_load     a daily profile with busy and quiet hours, rest days, and days of
                 different size: the load of a building, a grid, a service
  daylight       a source that works only part of the day and rests at zero:
                 solar generation, daytime-only activity
  duty_cycle     equipment that switches on and off: plateaus or charge/discharge
  random_walk    sum of independent changes: an inventory, a queue, a price
  smooth_drift   a slow physical variable: temperatures, levels, set by ramps and
                 holds, sometimes logged rarely and joined by straight lines
  counts         events per interval: requests, alarms, items, with idle intervals
  staircase      a value that moves in steps and holds: a counter, a recipe level
  oscillation    a repeating waveform whose length and size vary from cycle to
                 cycle: rotation, breathing, pulsation
  resource_use   how busy a shared resource is: a noisy level that cannot go below
                 zero, with bursts, sometimes idle, sometimes at its limit
"""

import math

import numpy as np
from scipy.signal import lfilter

PATTERNS = (
    "held_value",
    "daily_load",
    "daylight",
    "duty_cycle",
    "random_walk",
    "smooth_drift",
    "counts",
    "staircase",
    "oscillation",
    "resource_use",
)
# Patterns that can ride on a trend (a growing load, a drifting setpoint),
# with the usual spread of their signal: a trend is sized against it.
TREND_SPREAD = {
    "held_value": 1.3,
    "daily_load": 0.4,
    "duty_cycle": 0.3,
    "smooth_drift": 0.8,
    "staircase": 2.0,
    "oscillation": 1.0,
    "resource_use": 0.5,
}
# Patterns whose direction has a meaning: a count or a generation is not negative.
UPRIGHT = ("daylight", "counts", "resource_use")
# Laws of the independent changes of a random walk, all centred on zero.
WALK_LAWS = (
    "normal",
    "student_t",
    "laplace",
    "uniform",
    "two_point",
    "gamma",
    "poisson",
    "jumps",
    "lazy",
    "two_sizes",
)
# Points per cycle of recorded data: a day at the common sampling rates, a
# week of hours or days, a year of months.
CALENDAR_CYCLES = {
    24: 4.0,
    48: 2.0,
    96: 3.0,
    144: 2.0,
    288: 1.0,
    168: 0.7,
    12: 0.5,
    7: 0.5,
}
DAY_CYCLES = {24: 4.0, 48: 2.0, 96: 3.0, 144: 2.0, 288: 1.0}


def _log_uniform(rng, low, high):
    return float(np.exp(rng.uniform(np.log(low), np.log(high))))


def _cycle(rng, rows, cycles=CALENDAR_CYCLES, free=0.15):
    """A cycle length: mostly from the calendar, sometimes any length.

    A table may hold as little as one and a half repetitions: a day of
    5-minute readings is 288 points, and a history is often shorter than
    three days. A model has to learn a cycle from that little.
    """
    longest = max(8.0, rows / 1.5)
    if rng.random() < free:
        return _log_uniform(rng, 6, min(400.0, longest))
    lengths = [p for p in cycles if p <= longest] or [min(cycles)]
    weights = np.array([cycles[p] for p in lengths])
    return float(rng.choice(lengths, p=weights / weights.sum()))


def _profile(rng):
    """A day's shape as saved knots: a quiet stretch and some busy ones."""
    count = int(rng.integers(3, 9))
    positions = np.sort(rng.uniform(0, 1, count))
    values = rng.uniform(0.15, 1.0, count)
    # At least one stretch is clearly the quiet one (the night).
    values[int(rng.integers(count))] = rng.uniform(0.0, 0.2)
    return {
        "positions": positions.tolist(),
        "values": values.tolist(),
        "hold": bool(rng.random() < 0.55),  # Plateaus instead of slopes.
        "smoothing": float(rng.uniform(0.01, 0.2)),
    }


def _profile_values(profile, grid=512):
    """The saved shape on a regular grid over one cycle, between 0 and 1."""
    positions = np.array(profile["positions"])
    values = np.array(profile["values"])
    u = np.arange(grid) / grid
    if profile["hold"]:
        index = np.searchsorted(positions, u, side="right") - 1
        shape = values[index]  # Index -1 is the last knot: the cycle wraps.
    else:
        shape = np.interp(
            u,
            np.concatenate([[positions[-1] - 1], positions, [positions[0] + 1]]),
            np.concatenate([[values[-1]], values, [values[0]]]),
        )
    width = max(1, int(profile["smoothing"] * grid))
    kernel = np.ones(width) / width
    shape = np.convolve(np.concatenate([shape, shape, shape]), kernel, mode="same")
    shape = shape[grid : 2 * grid]
    low, high = shape.min(), shape.max()
    return (shape - low) / (high - low) if high > low else np.zeros(grid)


def _wander(rng, length, tau, scale):
    """A stationary first-order path: slow, bounded, returning to zero."""
    persistence = math.exp(-1 / tau)
    shocks = rng.normal(0, scale * math.sqrt(1 - persistence**2), length)
    shocks[0] = rng.normal(0, scale)
    return lfilter([1.0], [1.0, -persistence], shocks)


def _per_cycle(rng, cycle, spread):
    """A size for every cycle; `cycle` holds the cycle number of every point."""
    cycle = cycle - cycle.min()
    return rng.lognormal(0.0, spread, int(cycle.max()) + 1)[cycle]


def sample_pattern(rng, rows, family):
    """The saved parameters of one pattern. Durations are in samples."""
    spec = _sample_family(rng, rows, family)
    # Calm and busy stretches: the size of the noise wanders, as it does
    # when the conditions around a sensor change.
    spec["busy"] = 0.0 if rng.random() < 0.5 else float(rng.uniform(0.4, 1.3))
    spec["busy_tau"] = _log_uniform(rng, 5, 300)
    return spec


def _noise(spec, rng, length):
    """Unit noise, larger in busy stretches and smaller in calm ones."""
    noise = rng.normal(size=length)
    stretch = _wander(rng, length, spec["busy_tau"], 1.0)
    if spec["busy"]:
        noise = noise * np.exp(spec["busy"] * stretch - spec["busy"] ** 2 / 2)
    return noise


def _sample_family(rng, rows, family):
    if family == "held_value":
        colour = rng.random()
        return {
            # White, sluggish, or alternating around the setpoint.
            "memory": 0.0
            if colour < 0.3
            else float(1 - _log_uniform(rng, 0.004, 0.8))
            if colour < 0.92
            else float(-rng.uniform(0.2, 0.7)),
            "tail": 0.0 if rng.random() < 0.6 else float(rng.uniform(2.5, 8.0)),
            # Setpoint changes per 500 points, and their size in noise spreads.
            "moves": 0.0 if rng.random() < 0.45 else _log_uniform(rng, 0.5, 12),
            "move_size": _log_uniform(rng, 0.5, 10),
            "recipe": int(rng.integers(2, 6)) if rng.random() < 0.5 else 0,
            "settle": 0.0 if rng.random() < 0.5 else _log_uniform(rng, 2, 40),
            "drift": 0.0 if rng.random() < 0.5 else _log_uniform(rng, 0.2, 2.5),
            "drift_tau": _log_uniform(rng, 50, 1000),
            "spikes": 0.0 if rng.random() < 0.7 else _log_uniform(rng, 0.001, 0.02),
            "spike_size": float(rng.choice([-1, 1]) * rng.uniform(4, 15)),
            "load_gain": float(rng.uniform(0, 4)),
            # Levels the reading can take within six noise spreads.
            "levels": 0 if rng.random() < 0.3 else int(_log_uniform(rng, 3, 200)),
        }
    if family == "daily_load":
        return {
            "period": _cycle(rng, rows),
            "offset": float(rng.uniform(0, 1)),
            "profile": _profile(rng),
            "base": float(rng.uniform(0, 1.5)),
            "day_spread": _log_uniform(rng, 0.02, 0.4),
            "rest_days": int(rng.integers(1, 3)) if rng.random() < 0.5 else 0,
            "rest_factor": float(rng.uniform(0.2, 1.0)),
            "week": int(rng.choice([7, 7, 7, 5, 10, 14])),
            "week_shift": int(rng.integers(14)),
            "level_wander": 0.0 if rng.random() < 0.4 else _log_uniform(rng, 0.03, 0.5),
            "level_tau": _log_uniform(rng, 100, 2000),
            "noise": _log_uniform(rng, 0.003, 0.4),
            # One user is jagged from point to point; the sum of many is smooth.
            "noise_memory": 0.0
            if rng.random() < 0.3
            else float(1 - _log_uniform(rng, 0.01, 0.6)),
            # Noise that grows with the load, as in the sum of many users.
            "noise_with_load": bool(rng.random() < 0.5),
        }
    if family == "daylight":
        return {
            "period": _cycle(rng, rows, DAY_CYCLES, free=0.05),
            "offset": float(rng.uniform(0, 1)),
            "day_fraction": float(rng.uniform(0.25, 0.7)),
            "power": float(rng.uniform(0.6, 2.5)),
            "skew": float(rng.uniform(-0.25, 0.25)),
            # Size of a day: a beta law, from mostly clear to mostly overcast.
            "clear": [float(rng.uniform(1, 12)), float(rng.uniform(0.3, 2.5))],
            "cloud_depth": 0.0 if rng.random() < 0.3 else float(rng.uniform(0, 0.95)),
            "cloud_tau": _log_uniform(rng, 2, 80),
            "season": float(rng.uniform(-0.6, 0.6)),
            "noise": 0.0 if rng.random() < 0.75 else _log_uniform(rng, 0.002, 0.05),
        }
    if family == "duty_cycle":
        on = _log_uniform(rng, 3, 200)
        off = _log_uniform(rng, 3, 300)
        return {
            "on_time": on,
            "off_time": off,
            "jitter": _log_uniform(rng, 0.01, 0.5),
            "form": str(rng.choice(["plateau", "charge"])),
            "rise": int(rng.choice([0, 0, 0, 1, 2, 4])),
            "tau_on": on * _log_uniform(rng, 0.1, 2.0),
            "tau_off": off * _log_uniform(rng, 0.1, 2.0),
            "inrush": 0.0 if rng.random() < 0.6 else float(rng.uniform(0.1, 1.5)),
            "inrush_decay": float(rng.uniform(0.2, 0.9)),
            "droop": float(rng.uniform(-0.4, 0.4)) if rng.random() < 0.5 else 0.0,
            "ripple": _log_uniform(rng, 0.001, 0.1),
            "level_spread": 0.0
            if rng.random() < 0.4
            else float(rng.uniform(0.01, 0.25)),
            "idle_noise": 0.0 if rng.random() < 0.6 else _log_uniform(rng, 0.001, 0.03),
        }
    if family == "random_walk":
        return {
            # Nothing drives it and nothing pulls it back: the best forecast
            # of such a series is its last value, whatever its recent direction.
            "law": str(rng.choice(WALK_LAWS)),
            "shape": float(rng.uniform(0, 1)),
            "sign": float(rng.choice([-1, 1])),
            # The size of the changes is constant, changes once, or wanders.
            "volatility": str(
                rng.choice(["constant", "regime", "wander"], p=[0.6, 0.2, 0.2])
            ),
            "volatility_change": float(rng.choice([-1, 1]) * rng.uniform(0.5, 1.6)),
            "volatility_position": float(rng.uniform(0.2, 0.8)),
            "volatility_tau": _log_uniform(rng, 20, 400),
        }
    if family == "smooth_drift":
        count = int(rng.integers(1, 4))
        return {
            "taus": [_log_uniform(rng, 20, 1500) for _ in range(count)],
            "scales": rng.uniform(0.3, 1.0, count).tolist(),
            "smooth": bool(rng.random() < 0.85),
            # A program of ramps and holds on top of, or instead of, the wander.
            "program": bool(rng.random() < 0.4),
            "segment": _log_uniform(rng, 20, 500),
            "hold_share": float(rng.uniform(0.2, 0.8)),
            "program_scale": _log_uniform(rng, 0.5, 4),
            # Logged every so many points and joined by straight lines.
            "sparse": 0 if rng.random() < 0.55 else int(_log_uniform(rng, 4, 150)),
            "noise": 0.0 if rng.random() < 0.65 else _log_uniform(rng, 0.0005, 0.04),
        }
    if family == "counts":
        return {
            "rate": _log_uniform(rng, 0.03, 300),
            "period": _cycle(rng, rows),
            "offset": float(rng.uniform(0, 1)),
            "profile": _profile(rng) if rng.random() < 0.6 else None,
            "depth": float(rng.uniform(0.3, 1.0)),
            "wander": 0.0 if rng.random() < 0.3 else _log_uniform(rng, 0.1, 1.2),
            "wander_tau": _log_uniform(rng, 10, 800),
            # Clumping of events: none (Poisson) or negative binomial.
            "clumping": 0.0 if rng.random() < 0.5 else _log_uniform(rng, 0.3, 20),
            "idle": 0.0 if rng.random() < 0.7 else float(rng.uniform(0.05, 0.6)),
            "bursts": 0.0 if rng.random() < 0.7 else _log_uniform(rng, 0.002, 0.02),
            "burst_size": float(rng.uniform(3, 20)),
            "burst_length": int(_log_uniform(rng, 1, 12)),
            "load_power": float(rng.uniform(0, 3)),
        }
    if family == "staircase":
        return {
            "counter": bool(rng.random() < 0.4),
            "dwell": _log_uniform(rng, 3, 200),
            "dwell_spread": _log_uniform(rng, 0.05, 1.0),
            "step_spread": _log_uniform(rng, 0.05, 1.0),
            "recipe": int(rng.integers(2, 7)) if rng.random() < 0.5 else 0,
            "ramp": 0.0 if rng.random() < 0.5 else float(rng.uniform(0.05, 0.9)),
            "noise": 0.0 if rng.random() < 0.3 else _log_uniform(rng, 0.003, 0.4),
            "noise_memory": 0.0
            if rng.random() < 0.4
            else float(rng.uniform(0.3, 0.99)),
        }
    if family == "oscillation":
        kind = str(rng.choice(["harmonics", "pulse", "skewed"]))
        count = int(rng.integers(1, 7))
        bumps = int(rng.integers(1, 4))
        return {
            "period": _log_uniform(rng, 6, 160),
            "wobble": _log_uniform(rng, 0.002, 0.08),
            "wobble_tau": float(rng.uniform(2, 12)),
            "kind": kind,
            "harmonics": (
                rng.normal(size=count) / np.arange(1, count + 1) ** rng.uniform(0.5, 2)
            ).tolist(),
            "phases": rng.uniform(-np.pi, np.pi, count).tolist(),
            "bump_at": rng.uniform(0.05, 0.95, bumps).tolist(),
            "bump_width": [_log_uniform(rng, 0.015, 0.15) for _ in range(bumps)],
            "bump_size": (
                rng.choice([-1, 1], bumps) * rng.uniform(0.3, 1.5, bumps)
            ).tolist(),
            "skew": float(rng.uniform(0.05, 0.45)),
            "size_spread": 0.0 if rng.random() < 0.3 else _log_uniform(rng, 0.02, 0.4),
            "size_wander": 0.0 if rng.random() < 0.5 else float(rng.uniform(0.1, 0.6)),
            "baseline": 0.0 if rng.random() < 0.5 else _log_uniform(rng, 0.05, 1.0),
            "noise": _log_uniform(rng, 0.002, 0.15),
        }
    if family == "resource_use":
        return {
            "noise": _log_uniform(rng, 0.03, 0.7),
            "noise_memory": 0.0
            if rng.random() < 0.4
            else float(1 - _log_uniform(rng, 0.02, 0.7)),
            # Bursts per point, their size in units of the usual level (with a
            # heavy tail), and how many points one takes to die away.
            "bursts": 0.0 if rng.random() < 0.25 else _log_uniform(rng, 0.002, 0.08),
            "burst_size": _log_uniform(rng, 0.5, 20),
            "burst_tail": float(rng.uniform(1.3, 4.0)),
            "burst_decay": 0.0 if rng.random() < 0.4 else _log_uniform(rng, 1, 40),
            "period": _cycle(rng, rows),
            "offset": float(rng.uniform(0, 1)),
            "profile": _profile(rng) if rng.random() < 0.5 else None,
            "depth": float(rng.uniform(0.2, 0.95)),
            # Changes of the usual level that stay: a new release, a new user.
            "shifts": 0.0 if rng.random() < 0.65 else _log_uniform(rng, 0.5, 6),
            "shift_size": float(rng.uniform(0.2, 1.0)),
            "idle_below": 0.0 if rng.random() < 0.7 else float(rng.uniform(0.3, 1.2)),
            "limit": 0.0 if rng.random() < 0.75 else float(rng.uniform(1.5, 6)),
        }
    raise ValueError(f"Unknown operational pattern: {family}")


def simulate_pattern(spec, drive, rng):
    """The pattern's signal over the full simulated clock."""
    family = spec["family"]
    simulate = {
        "held_value": _held_value,
        "daily_load": _daily_load,
        "daylight": _daylight,
        "duty_cycle": _duty_cycle,
        "random_walk": _random_walk,
        "smooth_drift": _smooth_drift,
        "counts": _counts,
        "staircase": _staircase,
        "oscillation": _oscillation,
        "resource_use": _resource_use,
    }
    if family not in simulate:
        raise ValueError(f"Unknown operational pattern: {family}")
    return simulate[family](spec, np.asarray(drive, dtype=float), rng)


def _dwell_path(rng, length, mean, spread, levels):
    """A value that holds and then moves; `levels(count)` gives the values."""
    count = int(length / max(1.0, mean) * 3) + 8
    dwells = np.maximum(1, rng.lognormal(np.log(mean), spread, count)).astype(int)
    starts = np.concatenate([[0], np.cumsum(dwells)])
    while starts[-1] < length:  # Short dwells drawn by chance: extend.
        more = np.maximum(1, rng.lognormal(np.log(mean), spread, count)).astype(int)
        starts = np.concatenate([starts, starts[-1] + np.cumsum(more)])
    starts = starts[starts < length]
    index = np.searchsorted(starts, np.arange(length), side="right") - 1
    return levels(len(starts))[index], starts


def _held_value(spec, drive, rng):
    length = len(drive)
    noise = (
        rng.standard_t(spec["tail"], length)
        / math.sqrt(spec["tail"] / (spec["tail"] - 2))
        if spec["tail"]
        else _noise(spec, rng, length)
    )
    if spec["memory"]:
        memory = spec["memory"]
        noise = lfilter([math.sqrt(1 - memory**2)], [1.0, -memory], noise)
    level = np.zeros(length)
    if spec["moves"]:
        size = spec["move_size"]
        if spec["recipe"]:
            options = rng.uniform(-size, size, spec["recipe"])

            def levels(count):
                return options[rng.integers(0, len(options), count)]
        else:

            def levels(count):
                return np.cumsum(rng.normal(0, size / 2, count))

        level, _ = _dwell_path(rng, length, 500 / spec["moves"], 0.8, levels)
        if spec["settle"]:
            alpha = 1 - math.exp(-1 / spec["settle"])
            level = lfilter([alpha], [1.0, alpha - 1], level - level[0]) + level[0]
    if spec["drift"]:
        level = level + _wander(rng, length, spec["drift_tau"], spec["drift"])
    value = level + noise + spec["load_gain"] * (drive - 1)
    if spec["spikes"]:
        hits = rng.random(length) < spec["spikes"]
        value = value + hits * spec["spike_size"] * rng.uniform(0.5, 1.0, length)
    if spec["levels"]:
        step = 6.0 / spec["levels"]
        value = step * np.round(value / step)
    return value


def _cycle_position(spec, length):
    position = np.arange(length) / spec["period"] + spec["offset"]
    return position % 1.0, np.floor(position).astype(int)


def _daily_load(spec, drive, rng):
    length = len(drive)
    within, day = _cycle_position(spec, length)
    shape = _profile_values(spec["profile"])
    load = shape[(within * len(shape)).astype(int)]
    size = _per_cycle(rng, day, spec["day_spread"])
    if spec["rest_days"]:
        resting = (day + spec["week_shift"]) % spec["week"] < spec["rest_days"]
        size = np.where(resting, size * spec["rest_factor"], size)
    value = spec["base"] + load * size * drive
    if spec["level_wander"]:
        value = value * np.exp(
            _wander(rng, length, spec["level_tau"], spec["level_wander"])
        )
    noise = _noise(spec, rng, length)
    if spec["noise_memory"]:
        memory = spec["noise_memory"]
        noise = lfilter([math.sqrt(1 - memory**2)], [1.0, -memory], noise)
    scale = spec["noise"] * (0.15 + load if spec["noise_with_load"] else 1.0)
    return value + scale * noise


def _daylight(spec, drive, rng):
    length = len(drive)
    within, day = _cycle_position(spec, length)
    light = within < spec["day_fraction"]
    progress = np.clip(within / spec["day_fraction"], 0, 1) ** (1 + spec["skew"])
    arc = np.sin(np.pi * progress) ** spec["power"]
    day = day - day.min()
    clear = rng.beta(spec["clear"][0], spec["clear"][1], int(day.max()) + 1)[day]
    cloud = 1 / (1 + np.exp(-2 * _wander(rng, length, spec["cloud_tau"], 1.0)))
    season = 1 + spec["season"] * np.arange(length) / max(1, length - 1)
    value = arc * clear * (1 - spec["cloud_depth"] * cloud) * season * drive
    if spec["noise"]:
        value = value + spec["noise"] * rng.normal(size=length)
    # Night, and whatever the noise pushed below it, is exactly zero.
    return np.where(light, np.maximum(value, 0.0), 0.0)


def _duty_cycle(spec, drive, rng):
    length = len(drive)
    value = np.zeros(length)
    # Drawn before the cycles: their number depends on the demand, and a later
    # demand must not change the noise of an earlier point.
    ripple = spec["ripple"] * _noise(spec, rng, length)
    idle = spec["idle_noise"] * rng.normal(size=length)
    position, on, current = 0, bool(rng.random() < 0.5), 0.0
    while position < length:
        # More demand, shorter rests; read when the rest begins.
        demand = float(np.clip(drive[position], 0.3, 3.0))
        mean = spec["on_time"] if on else spec["off_time"] / demand
        duration = max(1, int(rng.lognormal(np.log(mean), spec["jitter"])))
        stop = min(length, position + duration)
        steps = np.arange(1, stop - position + 1)
        target = (1 + rng.normal(0, spec["level_spread"])) if on else 0.0
        if spec["form"] == "charge":
            tau = spec["tau_on"] if on else spec["tau_off"]
            part = target + (current - target) * np.exp(-steps / tau)
        else:
            part = np.full(len(steps), target)
            if on:
                part = part * (1 + spec["droop"] * steps / len(steps))
                part = part + spec["inrush"] * spec["inrush_decay"] ** (steps - 1)
            if spec["rise"]:
                edge = np.minimum(1.0, steps / (spec["rise"] + 1))
                part = current + (part - current) * edge
        value[position:stop] = part
        current, position, on = float(part[-1]), stop, not on
    return value + (value > 0.05) * ripple + idle


def walk_steps(law, shape, rng, length):
    """Independent changes with mean zero; `shape` in 0..1 sets the law's form."""
    if law == "normal":
        return rng.normal(size=length)
    if law == "student_t":
        return rng.standard_t(2.5 + 10 * shape, size=length)
    if law == "laplace":
        return rng.laplace(size=length)
    if law == "uniform":
        return rng.uniform(-1, 1, size=length)
    if law == "two_point":
        # Up with one chance and down with the other, sized to cancel out.
        chance = 0.15 + 0.7 * shape
        return np.where(rng.random(length) < chance, 1 - chance, -chance)
    if law == "gamma":
        form = 0.3 + 4 * shape
        return rng.gamma(form, size=length) - form
    if law == "poisson":
        rate = 0.2 + 8 * shape
        return rng.poisson(rate, size=length) - rate
    if law == "jumps":
        # Small changes, and now and then a large one.
        large = rng.random(length) < 0.01 + 0.07 * shape
        return rng.normal(size=length) * np.where(large, 8.0, 1.0)
    if law == "lazy":
        # Most of the time nothing changes.
        moving = rng.random(length) < 0.1 + 0.6 * shape
        return moving * rng.choice([-1.0, 1.0], size=length)
    if law == "two_sizes":
        return rng.normal(size=length) + rng.choice([-1, 1], size=length) * (
            1 + 3 * shape
        )
    raise ValueError(f"Unknown random-walk law: {law}")


def _random_walk(spec, drive, rng):
    length = len(drive)
    steps = spec["sign"] * walk_steps(spec["law"], spec["shape"], rng, length)
    if spec["volatility"] == "regime":
        late = np.arange(length) >= spec["volatility_position"] * length
        steps = steps * np.exp(spec["volatility_change"] * late)
    elif spec["volatility"] == "wander":
        steps = steps * np.exp(_wander(rng, length, spec["volatility_tau"], 0.7))
    # About one unit of spread over the table for changes of unit size.
    return np.cumsum(steps) / math.sqrt(length)


def _smooth_drift(spec, drive, rng):
    length = len(drive)
    value = np.zeros(length)
    for tau, scale in zip(spec["taus"], spec["scales"]):
        part = _wander(rng, length, tau, scale)
        if spec["smooth"]:
            alpha = 1 - math.exp(-4 / tau)
            part = lfilter([alpha], [1.0, alpha - 1], part - part[0]) + part[0]
        value += part
    if spec["program"]:
        # Knots of a ramp-and-hold program: a hold repeats the previous level.
        count = int(length / spec["segment"] * 3) + 8
        gaps = np.maximum(2, rng.lognormal(np.log(spec["segment"]), 0.6, count))
        knots = np.concatenate([[0], np.cumsum(gaps)])
        levels = rng.normal(0, spec["program_scale"], len(knots))
        hold = rng.random(len(knots)) < spec["hold_share"]
        for index in range(1, len(knots)):
            if hold[index]:
                levels[index] = levels[index - 1]
        value = value * 0.2 + np.interp(np.arange(length), knots, levels)
    load = 0.5 * (drive - 1)
    if spec["sparse"]:
        kept = np.arange(0, length, spec["sparse"])
        clock = np.arange(length)
        value = np.interp(clock, kept, value[kept])
        # The demand's part is joined one logging interval late, so that no
        # point depends on a demand that comes after it.
        load = np.interp(clock - spec["sparse"], kept, load[kept])
    value = value + load
    if spec["noise"]:
        value = value + spec["noise"] * _noise(spec, rng, length)
    return value


def _counts(spec, drive, rng):
    length = len(drive)
    rate = np.full(length, spec["rate"])
    if spec["profile"]:
        within, _ = _cycle_position(spec, length)
        shape = _profile_values(spec["profile"])
        rate = rate * (
            1 - spec["depth"] + spec["depth"] * shape[(within * len(shape)).astype(int)]
        )
    if spec["wander"]:
        rate = rate * np.exp(_wander(rng, length, spec["wander_tau"], spec["wander"]))
    rate = rate * np.clip(drive, 0.05, None) ** spec["load_power"]
    if spec["bursts"]:
        starts = np.flatnonzero(rng.random(length) < spec["bursts"])
        for start in starts:
            rate[start : start + spec["burst_length"]] *= spec["burst_size"]
    if spec["idle"]:
        rate = rate * (rng.random(length) >= spec["idle"])
    rate = np.minimum(rate, 1e6)
    if spec["clumping"]:
        clumping = spec["clumping"]
        return rng.negative_binomial(clumping, clumping / (clumping + rate)).astype(
            float
        )
    return rng.poisson(rate).astype(float)


def _staircase(spec, drive, rng):
    length = len(drive)
    spread = spec["step_spread"]
    if spec["counter"]:

        def levels(count):
            return np.cumsum(rng.lognormal(0.0, spread, count))
    elif spec["recipe"]:
        options = rng.uniform(-2, 2, spec["recipe"])

        def levels(count):
            return options[rng.integers(0, len(options), count)]
    else:

        def levels(count):
            return np.cumsum(
                rng.normal(0, 1, count) * rng.lognormal(0.0, spread, count)
            )

    value, starts = _dwell_path(
        rng, length, spec["dwell"], spec["dwell_spread"], levels
    )
    if spec["ramp"] and len(starts) > 1:
        # Part of every dwell is spent moving to the new level.
        ends = np.concatenate([starts[1:], [length]])
        moving = ends - (spec["ramp"] * (ends - starts)).astype(int)
        knots = np.sort(np.concatenate([starts, np.maximum(moving, starts + 1)]))
        value = np.interp(
            np.arange(length), knots, value[np.minimum(knots, length - 1)]
        )
    if spec["noise"]:
        noise = _noise(spec, rng, length)
        if spec["noise_memory"]:
            memory = spec["noise_memory"]
            noise = lfilter([math.sqrt(1 - memory**2)], [1.0, -memory], noise)
        value = value + spec["noise"] * noise
    return value + 0.3 * (drive - 1)


def _oscillation(spec, drive, rng):
    length = len(drive)
    speed = np.exp(
        _wander(rng, length, spec["wobble_tau"] * spec["period"], spec["wobble"])
    )
    position = np.cumsum(speed * np.clip(drive, 0.3, 3.0) ** 0.3) / spec["period"]
    within, cycle = position % 1.0, np.floor(position).astype(int)
    grid = np.arange(1024) / 1024
    if spec["kind"] == "harmonics":
        wave = sum(
            weight * np.sin(2 * np.pi * (order + 1) * grid + phase)
            for order, (weight, phase) in enumerate(
                zip(spec["harmonics"], spec["phases"])
            )
        )
    elif spec["kind"] == "pulse":
        wave = np.zeros(len(grid))
        for at, width, size in zip(
            spec["bump_at"], spec["bump_width"], spec["bump_size"]
        ):
            distance = (grid - at + 0.5) % 1 - 0.5
            wave += size * np.exp(-0.5 * (distance / width) ** 2)
    else:
        # A fast rise and a slow fall, or the reverse.
        rise = spec["skew"]
        wave = np.where(grid < rise, grid / rise, (1 - grid) / (1 - rise))
    spread = wave.std()
    wave = (wave - wave.mean()) / (spread if spread > 0 else 1.0)
    value = wave[(within * len(wave)).astype(int)]
    # Every random number is drawn for the whole clock, whatever the demand
    # makes of the number of cycles: a later demand cannot change the past.
    sizes = rng.lognormal(0.0, spec["size_spread"] or 1.0, length)
    size_wander = _wander(rng, length, 6 * spec["period"], spec["size_wander"] or 1.0)
    baseline = _wander(rng, length, 5 * spec["period"], spec["baseline"] or 1.0)
    noise = spec["noise"] * _noise(spec, rng, length)
    if spec["size_spread"]:
        value = value * sizes[np.clip(cycle - cycle[0], 0, length - 1)]
    if spec["size_wander"]:
        value = value * np.exp(size_wander)
    if spec["baseline"]:
        value = value + baseline
    return value + noise


def _resource_use(spec, drive, rng):
    length = len(drive)
    noise = _noise(spec, rng, length)
    if spec["noise_memory"]:
        memory = spec["noise_memory"]
        noise = lfilter([math.sqrt(1 - memory**2)], [1.0, -memory], noise)
    level = np.clip(drive, 0.05, None)
    if spec["profile"]:
        within, _ = _cycle_position(spec, length)
        shape = _profile_values(spec["profile"])
        level = level * (
            1 - spec["depth"] + spec["depth"] * shape[(within * len(shape)).astype(int)]
        )
    if spec["shifts"]:

        def levels(count):
            return np.exp(np.cumsum(rng.normal(0, spec["shift_size"], count)) * 0.5)

        level = level * _dwell_path(rng, length, 500 / spec["shifts"], 0.8, levels)[0]
    value = level * np.exp(spec["noise"] * noise - spec["noise"] ** 2 / 2)
    hits = rng.random(length) < spec["bursts"]
    sizes = spec["burst_size"] * (rng.pareto(spec["burst_tail"], length) + 0.3)
    burst = hits * sizes
    if spec["burst_decay"]:
        burst = lfilter([1.0], [1.0, -math.exp(-1 / spec["burst_decay"])], burst)
    value = value + burst
    if spec["idle_below"]:
        value = np.maximum(value - spec["idle_below"], 0.0)
    if spec["limit"]:
        value = np.minimum(value, spec["limit"])
    return value

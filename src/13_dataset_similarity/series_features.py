"""Describe a series by numbers that do not depend on its level or scale.

Every descriptor is computed on a window of fixed length after filling
missing values by linear interpolation and standardising the window to mean 0
and spread 1, so the mean and the spread themselves carry no information.
The descriptors are grouped in families; in a distance every family counts
equally, so that five ways of measuring seasonality do not count five times.

The definitions follow the feature sets that are in common use for comparing
collections of series: trend and seasonal strength, spectral entropy, the
long-memory (Hurst) exponent, stability and lumpiness as in Hyndman's
`tsfeatures` (the six that GIFT-Eval reports per dataset), and the families
of `catch22` (distribution shape, autocorrelation, entropy, scaling of
fluctuations). Added here are descriptors of sensor behaviour that those
sets lack and that operational data shows: flat runs, few distinct values,
resting at a limit, sudden jumps.
"""

import numpy as np
from scipy.ndimage import median_filter

# Descriptor -> family. The order is the order of every table and file.
FAMILIES = {
    "acf_1": "memory",
    "acf_4": "memory",
    "acf_16": "memory",
    "acf_64": "memory",
    "diff_acf_1": "memory",
    "long_memory": "memory",
    "decorrelation_time": "memory",
    "acf_first_minimum": "memory",
    "partial_acf": "memory",
    "diff2_acf_1": "memory",
    "spectral_entropy": "spectrum",
    "spectral_slope": "spectrum",
    "power_slow": "spectrum",
    "power_mid": "spectrum",
    "power_fast": "spectrum",
    "power_jitter": "spectrum",
    "spectral_centroid": "spectrum",
    "peak_share": "spectrum",
    "linear_trend": "trend",
    "trend_strength": "trend",
    "curvature": "trend",
    "seasonal_strength": "seasonality",
    "season_acf": "seasonality",
    "season_harmonics": "seasonality",
    "season_amplitude_change": "seasonality",
    "season_rest_share": "seasonality",
    "stability": "changes over time",
    "lumpiness": "changes over time",
    "level_shift": "changes over time",
    "variance_shift": "changes over time",
    "volatility_clustering": "changes over time",
    "longest_above_mean": "changes over time",
    "skewness": "distribution",
    "kurtosis": "distribution",
    "bimodality": "distribution",
    "diff_skewness": "distribution",
    "diff_kurtosis": "distribution",
    "jump_share": "distribution",
    "histogram_mode": "distribution",
    "crest": "distribution",
    "rise_share": "distribution",
    "floor_rest_share": "distribution",
    "ceiling_rest_share": "distribution",
    "flat_share": "sensor behaviour",
    "distinct_share": "sensor behaviour",
    "mode_share": "sensor behaviour",
    "at_extreme_share": "sensor behaviour",
    "missing_share": "sensor behaviour",
    "longest_flat_run": "sensor behaviour",
    "quantum_fit": "sensor behaviour",
    "noise_share": "noise and complexity",
    "permutation_entropy": "noise and complexity",
    "sample_entropy": "noise and complexity",
    "crossing_rate": "noise and complexity",
    "forecast_error": "noise and complexity",
    "turning_rate": "noise and complexity",
}
FEATURES = tuple(FAMILIES)
# Reported, but not part of a distance: undefined for a series without a cycle.
EXTRA = ("season_period",)
DESCRIPTIONS = {
    "acf_1": "Correlation with the previous point",
    "acf_4": "Correlation with the point 4 steps back",
    "acf_16": "Correlation with the point 16 steps back (one patch)",
    "acf_64": "Correlation with the point 64 steps back",
    "diff_acf_1": "Correlation between successive changes (negative: noise, positive: smooth)",
    "long_memory": "Long-memory exponent (0.5 white noise, 1.5 random walk)",
    "spectral_entropy": "Spectral entropy (0 one pure cycle, 1 white noise)",
    "spectral_slope": "How fast power falls with frequency (0 white noise, 2 random walk)",
    "power_slow": "Share of variance in movements slower than 128 points",
    "power_mid": "Share of variance in cycles of 16 to 128 points",
    "power_fast": "Share of variance in cycles of 4 to 16 points",
    "power_jitter": "Share of variance in cycles shorter than 4 points",
    "linear_trend": "Share of variance explained by a straight line",
    "trend_strength": "Trend strength (slow component against the remainder)",
    "seasonal_strength": "Seasonal strength (repeating cycle against the remainder)",
    "season_acf": "Correlation with the point one cycle back",
    "season_harmonics": "Share of the cycle's shape that is not a plain sine wave",
    "season_amplitude_change": "How much the size of the cycle varies from cycle to cycle",
    "stability": "Variation of the level between 16 stretches of the window",
    "lumpiness": "Variation of the spread between 16 stretches of the window",
    "level_shift": "Largest shift of the level between neighbouring stretches",
    "variance_shift": "Largest change of the spread between neighbouring stretches",
    "volatility_clustering": "Whether large changes follow large changes",
    "skewness": "Skewness of the values",
    "kurtosis": "Heaviness of the tails of the values",
    "bimodality": "Tendency to sit at two levels (on/off behaviour)",
    "diff_skewness": "Asymmetry between rises and falls",
    "diff_kurtosis": "Heaviness of the tails of the changes",
    "jump_share": "Share of all movement in the largest 1% of the changes",
    "flat_share": "Share of points equal to the previous point",
    "distinct_share": "Distinct values per point (low: coarse readings)",
    "mode_share": "Share of points at the most common value",
    "at_extreme_share": "Share of points at the lowest or highest value",
    "missing_share": "Share of missing points",
    "noise_share": "Spread of point-to-point noise against the spread of the series",
    "permutation_entropy": "Disorder of the up/down patterns of three points",
    "season_period": "Length of the cycle in points",
    "decorrelation_time": "How long the series remembers (lag where the correlation falls below 1/e, log scale)",
    "acf_first_minimum": "Lag of the first dip of the correlation (log scale; half a cycle where there is one)",
    "partial_acf": "Sum of the squared partial correlations at the first 5 lags",
    "diff2_acf_1": "Correlation between successive changes of the changes",
    "spectral_centroid": "Mean frequency of the spectrum (0 slow, 1 fastest)",
    "peak_share": "Share of variance in the single strongest cycle",
    "curvature": "Share of variance a bend explains beyond a straight line",
    "season_rest_share": "Share of the average cycle spent at its floor",
    "longest_above_mean": "Longest stretch above the mean, as a share of the window",
    "histogram_mode": "Where the series sits most often, in spreads from its mean",
    "crest": "Largest distance from the mean, in spreads",
    "rise_share": "Share of the changes that are rises (low: fast rise, slow fall)",
    "floor_rest_share": "Share of points within 2% of the range from the lowest value",
    "ceiling_rest_share": "Share of points within 2% of the range from the highest value",
    "longest_flat_run": "Longest run of unchanged points, as a share of the window",
    "quantum_fit": "Whether the values lie on a regular grid (gaps between values as multiples of the smallest gap)",
    "sample_entropy": "Sample entropy (low: repeats itself, high: unpredictable)",
    "crossing_rate": "How often the series crosses its median",
    "forecast_error": "Error of forecasting a point by the mean of the three before it",
    "turning_rate": "Share of points that are a local peak or dip",
}
# Bands by cycle length in points, relative to what Basin works with: a patch
# is 16 points and a forecast 112.
BANDS = (("power_slow", 128, np.inf), ("power_mid", 16, 128), ("power_fast", 4, 16))


def prepare(values):
    """(raw values with gaps filled, standardised values, share missing) or None.

    None for a window with fewer than half of its points observed or without
    any variation: neither has a shape to describe.
    """
    raw = np.asarray(values, dtype=np.float64)
    observed = np.isfinite(raw)
    if observed.sum() < max(16, len(raw) // 2):
        return None
    missing = 1.0 - observed.mean()
    if missing:
        position = np.arange(len(raw))
        raw = np.interp(position, position[observed], raw[observed])
    spread = raw.std()
    if not spread > 1e-9 * max(1.0, abs(raw.mean())):
        return None
    return raw, (raw - raw.mean()) / spread, float(missing)


def correlation(a, b):
    a, b = a - a.mean(), b - b.mean()
    scale = np.sqrt((a * a).sum() * (b * b).sum())
    return float((a * b).sum() / scale) if scale > 0 else 0.0


def autocorrelation(x):
    """The autocorrelation at every lag below the length of the series."""
    n = len(x)
    x = x - x.mean()
    spectrum = np.abs(np.fft.rfft(x, 2 * n)) ** 2
    acf = np.fft.irfft(spectrum)[:n]
    return acf / acf[0] if acf[0] > 0 else np.zeros(n)


def long_memory(x, scales=(8, 16, 32, 64, 128)):
    """Detrended fluctuation analysis: how fluctuations grow with the scale."""
    profile = np.cumsum(x - x.mean())
    sizes, fluctuations = [], []
    for scale in scales:
        boxes = len(profile) // scale
        if boxes < 4:
            break
        parts = profile[: boxes * scale].reshape(boxes, scale)
        time = np.arange(scale) - (scale - 1) / 2
        slope = (parts * time).sum(axis=1) / (time * time).sum()
        rest = parts - parts.mean(axis=1, keepdims=True) - slope[:, None] * time
        fluctuation = np.sqrt((rest * rest).mean())
        if fluctuation > 0:
            sizes.append(scale)
            fluctuations.append(fluctuation)
    if len(sizes) < 3:
        return 0.0
    # A series of a few flat steps has almost no fluctuation inside a short
    # stretch, which sends the slope far beyond the range of real processes.
    slope = np.polyfit(np.log(sizes), np.log(fluctuations), 1)[0]
    return float(np.clip(slope, 0.0, 2.5))


def first_below(acf, level):
    below = np.flatnonzero(acf < level)
    return int(below[0]) if len(below) else len(acf)


def memory_features(x, acf, change):
    """Descriptors of memory as in catch22 and tsfeatures."""
    n = len(x)
    result = {
        "decorrelation_time": float(np.log(first_below(acf, np.exp(-1))) / np.log(n))
    }
    inner = acf[1:-1]
    dips = np.flatnonzero((inner < acf[:-2]) & (inner <= acf[2:]))
    lag = int(dips[0]) + 1 if len(dips) else n
    result["acf_first_minimum"] = float(np.log(lag) / np.log(n))
    # Partial autocorrelations by the Levinson-Durbin recursion.
    total, weights, error = 0.0, np.zeros(0), 1.0
    for order in range(1, 6):
        if error <= 1e-12:
            break
        partial = (acf[order] - (weights * acf[order - 1 : 0 : -1]).sum()) / error
        partial = float(np.clip(partial, -1.0, 1.0))
        weights = np.concatenate([weights - partial * weights[::-1], [partial]])
        error *= 1.0 - partial * partial
        total += partial * partial
    result["partial_acf"] = total
    second = np.diff(change)
    result["diff2_acf_1"] = correlation(second[:-1], second[1:])
    return result


def sample_entropy(x, length=2, tolerance=0.2, points=512):
    """Sample entropy of the latest `points` points (capped at 5)."""
    x = x[-points:]
    count = len(x) - length

    def matches(size):
        parts = np.lib.stride_tricks.sliding_window_view(x, size)[:count]
        distance = np.abs(parts[:, None, :] - parts[None, :, :]).max(axis=2)
        return float((distance <= tolerance).sum() - count)

    shorter, longer = matches(length), matches(length + 1)
    if shorter <= 0 or longer <= 0:
        return 5.0
    return float(min(5.0, -np.log(longer / shorter)))


def run_length(flags):
    """The longest run of True."""
    if not flags.any():
        return 0
    edges = np.diff(np.concatenate([[0], flags.astype(np.int8), [0]]))
    return int((np.flatnonzero(edges == -1) - np.flatnonzero(edges == 1)).max())


def spectrum_features(x):
    n = len(x)
    power = np.abs(np.fft.rfft(x)[1:]) ** 2
    frequency = np.arange(1, len(power) + 1) / n
    total = power.sum()
    share = power / total
    positive = share[share > 0]
    result = {
        "spectral_entropy": float(
            -(positive * np.log(positive)).sum() / np.log(len(share))
        )
    }
    # The slope over bins of equal width on the logarithmic frequency axis,
    # so that the many high frequencies do not decide it alone.
    edges = np.geomspace(frequency[0], frequency[-1], 13)
    centres, levels = [], []
    for low, high in zip(edges[:-1], edges[1:]):
        chosen = (frequency >= low) & (frequency <= high)
        if chosen.any() and power[chosen].mean() > 0:
            centres.append(np.sqrt(low * high))
            levels.append(power[chosen].mean())
    slope = np.polyfit(np.log(centres), np.log(levels), 1)[0] if len(centres) > 2 else 0
    result["spectral_slope"] = float(-slope)
    period = 1.0 / frequency
    for name, shortest, longest in BANDS:
        result[name] = float(share[(period > shortest) & (period <= longest)].sum())
    result["power_jitter"] = float(share[period <= 4].sum())
    result["spectral_centroid"] = float((frequency * share).sum() / 0.5)
    # The strongest cycle that repeats at least three times, with the bins
    # beside it: a cycle whose length is not a whole part of the window leaks.
    peak = 2 + int(np.argmax(power[2:])) if len(power) > 3 else 0
    result["peak_share"] = float(share[max(0, peak - 1) : peak + 2].sum())
    return result


def moving_average(x, width):
    """Centred moving average; the ends are continued by reflection."""
    width = int(max(1, min(width, len(x) - 1)))
    padded = np.pad(x, (width // 2, width - 1 - width // 2), mode="reflect")
    total = np.cumsum(np.concatenate([[0.0], padded]))
    return (total[width:] - total[:-width]) / width


def find_period(x):
    """The length of the repeating cycle in points, or 0 when there is none.

    The strongest peak of the spectrum of the series without its straight
    line gives a first estimate; the autocorrelation refines it. A cycle
    needs three repetitions in the window and a correlation of 0.3 with the
    point one cycle back.
    """
    n = len(x)
    time = np.arange(n)
    rest = x - np.polyval(np.polyfit(time, x, 1), time)
    if rest.std() < 1e-9:
        return 0, 0.0
    acf = autocorrelation(rest)
    power = np.abs(np.fft.rfft(rest)) ** 2
    power[:3] = 0
    peak = int(np.argmax(power))
    if peak < 3 or power[peak] <= 0:
        return 0, 0.0
    low, high = int(np.floor(n / (peak + 0.5))), int(np.ceil(n / (peak - 0.5)))
    low, high = max(2, low), min(n // 3, high)
    if high < low:
        return 0, 0.0
    period = low + int(np.argmax(acf[low : high + 1]))
    # A strong second harmonic (two humps a day) can win the spectrum: prefer
    # the cycle twice as long when the series repeats clearly better there.
    double = 2 * period
    if double + 1 <= n // 3:
        around = acf[double - 1 : double + 2]
        if around.max() > acf[period] + 0.1:
            period = double - 1 + int(np.argmax(around))
    strength = float(acf[period])
    if period < 3 or strength < 0.3:
        return 0, 0.0
    return period, strength


def decomposition_features(x):
    """Trend, cycle and remainder, and what they say about the series."""
    n = len(x)
    time = np.arange(n)
    line = np.polyval(np.polyfit(time, x, 1), time)
    result = {"linear_trend": float(line.var() / x.var())}
    bend = np.polyval(np.polyfit(time / n, x, 2), time / n)
    result["curvature"] = float(max(0.0, (bend.var() - line.var()) / x.var()))
    period, repeat = find_period(x)
    # The trend is what moves slower than an eighth of the window, over whole
    # cycles where there is one.
    width = n // 8 if not period else period * int(np.ceil(n / 8 / period))
    trend = moving_average(x, width)
    detrended = x - trend
    season = np.zeros(n)
    if period:
        phase = time % period
        counts = np.bincount(phase, minlength=period)
        profile = np.bincount(phase, weights=detrended, minlength=period) / counts
        profile -= profile.mean()
        season = profile[phase]
    rest = detrended - season

    def strength(component):
        whole = (component + rest).var()
        return float(max(0.0, 1.0 - rest.var() / whole)) if whole > 0 else 0.0

    result["trend_strength"] = strength(trend)
    result["seasonal_strength"] = strength(season) if period else 0.0
    result["season_acf"] = repeat
    result["season_period"] = float(period)
    harmonics = change = rest_share = 0.0
    if period:
        height = np.ptp(profile)
        if height > 0:
            rest_share = float((profile <= profile.min() + 0.1 * height).mean())
        shape = np.abs(np.fft.rfft(profile)[1:]) ** 2
        if shape.sum() > 0:
            harmonics = float(1.0 - shape[0] / shape.sum())
        cycles = n // period
        sizes = detrended[n - cycles * period :].reshape(cycles, period).std(axis=1)
        if sizes.mean() > 0:
            change = float(sizes.std() / sizes.mean())
    result["season_harmonics"] = harmonics
    result["season_amplitude_change"] = change
    result["season_rest_share"] = rest_share
    return result


def change_features(x, stretches=16):
    n = len(x)
    width = n // stretches
    tiles = x[n - stretches * width :].reshape(stretches, width)
    result = {
        "stability": float(tiles.mean(axis=1).var()),
        "lumpiness": float(tiles.var(axis=1).var()),
    }
    total = np.cumsum(np.concatenate([[0.0], x]))
    squares = np.cumsum(np.concatenate([[0.0], x * x]))
    mean = (total[width:] - total[:-width]) / width
    variance = np.maximum((squares[width:] - squares[:-width]) / width - mean**2, 0)
    result["level_shift"] = float(np.abs(mean[width:] - mean[:-width]).max())
    spread = np.log(variance + 1e-4)
    result["variance_shift"] = float(np.abs(spread[width:] - spread[:-width]).max() / 2)
    return result


def moments(values):
    spread = values.std()
    if spread <= 0:
        return 0.0, 0.0
    z = (values - values.mean()) / spread
    return float((z**3).mean()), float((z**4).mean() - 3.0)


def describe(values):
    """Every descriptor of one window, or None when it has no shape."""
    prepared = prepare(values)
    if prepared is None:
        return None
    raw, x, missing = prepared
    n = len(x)
    change = np.diff(x)
    acf = autocorrelation(x)
    result = {f"acf_{lag}": float(acf[lag]) for lag in (1, 4, 16, 64)}
    result["diff_acf_1"] = correlation(change[:-1], change[1:])
    result["long_memory"] = long_memory(x)
    result.update(memory_features(x, acf, change))
    result.update(spectrum_features(x))
    result.update(decomposition_features(x))
    result.update(change_features(x))
    size = np.abs(change)
    result["volatility_clustering"] = correlation(size[:-1], size[1:])
    skewness, kurtosis = moments(x)
    result["skewness"], result["kurtosis"] = skewness, kurtosis
    result["bimodality"] = float((skewness**2 + 1.0) / (kurtosis + 3.0))
    result["diff_skewness"], result["diff_kurtosis"] = moments(change)
    squared = np.sort(change * change)
    top = max(1, int(round(0.01 * len(squared))))
    result["jump_share"] = (
        float(squared[-top:].sum() / squared.sum()) if squared[-1] > 0 else 0.0
    )
    steps = np.diff(raw)
    levels, counts = np.unique(raw, return_counts=True)
    result["flat_share"] = float((steps == 0).mean())
    result["distinct_share"] = float(len(counts) / n)
    result["mode_share"] = float(counts.max() / n)
    result["at_extreme_share"] = float(((raw == raw.min()) | (raw == raw.max())).mean())
    result["missing_share"] = missing
    result["noise_share"] = float((x - median_filter(x, size=5, mode="nearest")).std())
    order = (
        (x[:-2] < x[1:-1]).astype(int)
        + 2 * (x[1:-1] < x[2:]).astype(int)
        + 4 * (x[:-2] < x[2:]).astype(int)
    )
    shares = np.bincount(order, minlength=8) / len(order)
    shares = shares[shares > 0]
    result["permutation_entropy"] = float(-(shares * np.log(shares)).sum() / np.log(6))
    result["longest_above_mean"] = run_length(x > 0) / n
    counts, edges = np.histogram(x, bins=10)
    fullest = int(np.argmax(counts))
    result["histogram_mode"] = float((edges[fullest] + edges[fullest + 1]) / 2)
    result["crest"] = float(np.abs(x).max())
    moving = steps[steps != 0]
    result["rise_share"] = float((moving > 0).mean()) if len(moving) else 0.5
    low, height = raw.min(), np.ptp(raw)
    result["floor_rest_share"] = float((raw <= low + 0.02 * height).mean())
    result["ceiling_rest_share"] = float((raw >= low + 0.98 * height).mean())
    result["longest_flat_run"] = run_length(steps == 0) / n
    # Whether the values lie on a regular grid: the gaps between neighbouring
    # distinct values as multiples of the smallest gap. The smallest change
    # from one point to the next would not do, because it can span several
    # grid steps; and stored as 32-bit numbers a gap is not an exact multiple.
    fit = 0.0
    if len(levels) > 2:
        gaps = np.diff(levels)
        ratio = gaps / gaps.min()
        fit = float((np.abs(ratio - np.round(ratio)) < 0.05).mean())
    result["quantum_fit"] = fit
    result["sample_entropy"] = sample_entropy(x)
    above = x > np.median(x)
    result["crossing_rate"] = float((above[1:] != above[:-1]).mean())
    ahead = x[3:] - (x[:-3] + x[1:-2] + x[2:-1]) / 3.0
    result["forecast_error"] = float(ahead.std())
    result["turning_rate"] = float(((change[:-1] * change[1:]) < 0).mean())
    return result


def describe_many(windows):
    """`describe` for every row of a block; rows without a shape give NaN."""
    names = FEATURES + EXTRA
    table = np.full((len(windows), len(names)), np.nan)
    for position, values in enumerate(windows):
        found = describe(values)
        if found is not None:
            table[position] = [found[name] for name in names]
    return table

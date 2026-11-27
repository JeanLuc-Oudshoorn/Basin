"""Holt-Winters (triple exponential smoothing), fitted anew for every forecast.

A classical reference next to the pretrained models. Each request fits the
model to the history of one target and forecasts from it; nothing is carried
over from one request to the next.

The tables do not say how long their seasonal cycle is, so it is taken from
the history: the lag at which the detrended history correlates best with
itself. A model with that season and one without a season (damped-trend
double exponential smoothing) are both fitted, and the one with the lower
AICc forecasts. Trend and season are additive, because the series can be
zero or negative.
"""

import warnings

import numpy as np

# The longest seasonal cycle looked for, in points.
MAX_PERIOD = 512
# Below this autocorrelation at the best lag no seasonal model is tried.
MIN_AUTOCORRELATION = 0.3
# Fewer points than this are forecast with the last value.
MIN_POINTS = 10


def seasonal_period(values, max_period=MAX_PERIOD):
    """The cycle length the history suggests, or None.

    The highest local peak of the autocorrelation of the linearly detrended
    history, among lags the history covers at least twice.
    """
    count = len(values)
    upper = min(max_period, count // 2)
    if upper < 2:
        return None
    position = np.arange(count) - (count - 1) / 2
    centred = values - values.mean()
    residual = centred - position * (position @ centred) / (position @ position)
    spectrum = np.fft.rfft(residual, 2 * count)
    correlation = np.fft.irfft(spectrum * np.conj(spectrum))[: upper + 2]
    if correlation[0] <= 0:
        return None
    correlation = correlation / correlation[0]
    lags = np.arange(2, upper + 1)
    peaks = lags[
        (correlation[lags] > correlation[lags - 1])
        & (correlation[lags] >= correlation[lags + 1])
    ]
    if not len(peaks):
        return None
    best = int(peaks[np.argmax(correlation[peaks])])
    return best if correlation[best] >= MIN_AUTOCORRELATION else None


def _fit(values, period):
    from statsmodels.tsa.holtwinters import ExponentialSmoothing

    options = {"trend": "add", "damped_trend": True}
    if period:
        options.update(seasonal="add", seasonal_periods=period)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return ExponentialSmoothing(
            values, initialization_method="heuristic", **options
        ).fit()


def holt_winters_forecast(history, horizon, *, max_period=MAX_PERIOD, details=None):
    """Fit to `history` (no missing values) and forecast `horizon` points.

    `details`, if given, receives the season used (0: none) under "period".
    """
    history = np.asarray(history, dtype=float)
    last = np.full(horizon, history[-1])
    spread = history.std()
    if details is not None:
        details["period"] = 0
    if len(history) < MIN_POINTS or not spread > 0:
        return last
    # Fitted on standardised values: the model is additive, so this changes
    # nothing but keeps the optimiser away from very large and small numbers.
    values = (history - history.mean()) / spread
    period = seasonal_period(values, max_period)
    best, used = None, 0
    for candidate in (0, period) if period else (0,):
        try:
            fitted = _fit(values, candidate)
        except (ValueError, FloatingPointError, np.linalg.LinAlgError):
            continue
        if np.isfinite(fitted.aicc) and (best is None or fitted.aicc < best.aicc):
            best, used = fitted, candidate
    if best is None:
        return last
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        forecast = np.asarray(best.forecast(horizon)) * spread + history.mean()
    if forecast.shape != (horizon,) or not np.isfinite(forecast).all():
        return last
    if details is not None:
        details["period"] = used
    return forecast

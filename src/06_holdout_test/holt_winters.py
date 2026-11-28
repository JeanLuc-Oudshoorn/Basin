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

The model also says how uncertain its forecast is: `holt_winters_interval`
returns the standard deviation of the forecast error at every step, from the
size of the one-step errors within the history and the fitted smoothing
parameters (the standard formula for additive exponential smoothing, Hyndman
and Athanasopoulos, Forecasting: Principles and Practice, section 8.7).
"""

import multiprocessing
import os
import warnings
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

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


def walk_spread(history, horizon):
    """The forecast error of the last value if the history is a random walk."""
    step = np.sqrt(np.mean(np.diff(history) ** 2)) if len(history) > 1 else 0.0
    return step * np.sqrt(np.arange(1, horizon + 1))


def error_spread(fitted, period, horizon):
    """Standard deviation of the fitted model's forecast error at steps 1 to horizon.

    An error of one step moves the forecast `weights[j]` at j steps later,
    through the level, the damped trend and (every `period` steps) the season.
    """
    parameters = fitted.params
    level = float(parameters["smoothing_level"])
    trend = level * float(np.nan_to_num(parameters["smoothing_trend"]))
    damping = float(np.nan_to_num(parameters["damping_trend"], nan=1.0))
    steps = np.arange(1, horizon)
    weights = level + trend * np.cumsum(damping**steps)
    if period:
        season = float(np.nan_to_num(parameters["smoothing_seasonal"])) * (1 - level)
        weights = weights + season * (steps % period == 0)
    one_step = np.sqrt(fitted.sse / len(fitted.resid))
    return one_step * np.sqrt(1 + np.concatenate([[0.0], np.cumsum(weights**2)]))


def holt_winters_forecast(history, horizon, *, max_period=MAX_PERIOD, details=None):
    """Fit to `history` (no missing values) and forecast `horizon` points.

    `details`, if given, receives the season used (0: none) under "period" and
    the standard deviation of the forecast error at every step under "spread".
    """
    history = np.asarray(history, dtype=float)
    last = np.full(horizon, history[-1])
    spread = history.std()
    if details is not None:
        details["period"] = 0
        details["spread"] = walk_spread(history, horizon)
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
        errors = error_spread(best, used, horizon) * spread
        details["period"] = used
        if np.isfinite(errors).all():
            details["spread"] = errors
    return forecast


def holt_winters_interval(history, horizon, *, max_period=MAX_PERIOD):
    """(forecast, standard deviation of its error), each of `horizon` points.

    Where the last value is forecast instead of a fitted model, the spread is
    that of a random walk with the history's steps.
    """
    details = {}
    forecast = holt_winters_forecast(
        history, horizon, max_period=max_period, details=details
    )
    return forecast, details["spread"]


def _interval(request):
    return holt_winters_interval(*request)


class HoltWintersPool:
    """Holt-Winters fitted to many histories, by several processes at once."""

    def __init__(self, workers):
        self.workers, self.pool = max(1, int(workers)), None

    def __call__(self, histories, horizon):
        """(forecasts, spreads), each [histories, horizon]."""
        requests = [(np.asarray(h, dtype=float), horizon) for h in histories]
        if self.workers == 1:
            results = [_interval(r) for r in requests]
        else:
            if self.pool is None:
                # The processes find this module through this path, and use
                # one thread each: they are the parallelism.
                paths = [str(Path(__file__).resolve().parent)]
                paths.append(os.environ.get("PYTHONPATH", ""))
                os.environ["PYTHONPATH"] = os.pathsep.join(p for p in paths if p)
                for name in (
                    "OMP_NUM_THREADS",
                    "MKL_NUM_THREADS",
                    "OPENBLAS_NUM_THREADS",
                ):
                    os.environ[name] = "1"
                self.pool = ProcessPoolExecutor(
                    self.workers, mp_context=multiprocessing.get_context("spawn")
                )
            results = list(self.pool.map(_interval, requests, chunksize=2))
        return tuple(np.stack(part) for part in zip(*results))

    def close(self):
        if self.pool is not None:
            self.pool.shutdown()
            self.pool = None

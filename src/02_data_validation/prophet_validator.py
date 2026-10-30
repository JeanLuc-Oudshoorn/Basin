"""Chronological Prophet evaluation, without using SCM metadata or future y."""

import errno
import hashlib
import json
import logging
import os
import re
import tempfile
from concurrent.futures import FIRST_COMPLETED, ProcessPoolExecutor, wait
from contextlib import contextmanager
from datetime import datetime
from multiprocessing import get_context
from pathlib import Path
from uuid import uuid4

import numpy as np
import pandas as pd


@contextmanager
def _temporary_storage(directory):
    """Set temp storage before CmdStanPy's first import, including spawned fits."""
    directory = Path(directory).resolve()
    directory.mkdir(parents=True, exist_ok=True)
    previous = {name: os.environ.get(name) for name in ("TMP", "TEMP", "TMPDIR")}
    previous_default = tempfile.tempdir
    try:
        for name in previous:
            os.environ[name] = str(directory)
        tempfile.tempdir = str(directory)
        yield directory
    finally:
        tempfile.tempdir = previous_default
        for name, value in previous.items():
            if value is None:
                os.environ.pop(name, None)
            else:
                os.environ[name] = value


@contextmanager
def _worker_environment():
    """Spawn workers with one native thread per fit, avoiding oversubscription."""
    names = (
        "OMP_NUM_THREADS",
        "OPENBLAS_NUM_THREADS",
        "MKL_NUM_THREADS",
        "BLIS_NUM_THREADS",
        "VECLIB_MAXIMUM_THREADS",
        "NUMEXPR_NUM_THREADS",
        "STAN_NUM_THREADS",
    )
    previous = {name: os.environ.get(name) for name in names}
    try:
        for name in names:
            os.environ[name] = "1"
        yield
    finally:
        for name, value in previous.items():
            if value is None:
                os.environ.pop(name, None)
            else:
                os.environ[name] = value


def _initialize_worker():
    # A root handler also prevents CmdStanPy installing its own INFO handler.
    logging.basicConfig(level=logging.WARNING)
    logging.getLogger("cmdstanpy").setLevel(logging.WARNING)


def multivariate_admitted(results):
    """A successful, finite multivariate MAE may tie the nonnegative baseline."""
    baseline, score = results["baseline_mae"], results["multivariate_mae"]
    admitted = (
        results["status"].eq("ok")
        & np.isfinite(baseline)
        & (baseline >= 0)
        & np.isfinite(score)
        & (score >= 0)
        & (score <= baseline)
    )
    if "multivariate_error" in results:
        admitted &= results["multivariate_error"].fillna("").str.strip().eq("")
    return admitted


class ProphetValidator:
    def __init__(
        self,
        input_dir,
        output_dir,
        *,
        train_fraction=0.8,
        covariate_mode="known_future",
        step_duration="1D",
        seed=42,
        prophet_options=None,
        seasonal_periods=(),
        reserved_tail_rows=0,
        workers=None,
        temp_dir=None,
    ):
        if not 0 < train_fraction < 1:
            raise ValueError("train_fraction must lie strictly between 0 and 1")
        if covariate_mode not in ("known_future", "forecast"):
            raise ValueError("covariate_mode must be known_future or forecast")
        self.input_dir, self.output_dir = Path(input_dir), Path(output_dir)
        self.temp_dir = (
            Path(temp_dir)
            if temp_dir is not None
            else Path(__file__).resolve().parents[2] / ".cache/prophet"
        )
        self.train_fraction, self.covariate_mode = train_fraction, covariate_mode
        self.step_duration = pd.Timedelta(step_duration)
        if self.step_duration <= pd.Timedelta(0):
            raise ValueError("step_duration must be positive")
        self.seed = seed
        if workers is not None and (
            isinstance(workers, bool) or not isinstance(workers, int) or workers < 1
        ):
            raise ValueError("workers must be a positive integer or None")
        self.workers = workers
        if not isinstance(reserved_tail_rows, int) or reserved_tail_rows < 0:
            raise ValueError("reserved_tail_rows must be a nonnegative integer")
        self.reserved_tail_rows = reserved_tail_rows
        # Point forecasts only: no expensive posterior uncertainty simulation.
        self.prophet_options = {"uncertainty_samples": 0, **(prophet_options or {})}
        self.seasonal_periods = tuple(float(p) for p in seasonal_periods)
        if any(not np.isfinite(p) or p <= 2 for p in self.seasonal_periods):
            raise ValueError("Seasonal periods must be finite and exceed two timesteps")

    @staticmethod
    def mae(actual, predicted):
        actual, predicted = np.asarray(actual, float), np.asarray(predicted, float)
        if actual.shape != predicted.shape or not np.isfinite(predicted).all():
            raise ValueError(
                "Predictions must be finite and match the test target shape"
            )
        return float(np.mean(np.abs(actual - predicted)))

    @staticmethod
    def rmse(actual, predicted):
        actual, predicted = np.asarray(actual, float), np.asarray(predicted, float)
        if actual.shape != predicted.shape or not np.isfinite(predicted).all():
            raise ValueError(
                "Predictions must be finite and match the test target shape"
            )
        return float(np.sqrt(np.mean(np.square(actual - predicted))))

    @staticmethod
    def improvement(reference, candidate):
        # A perfect baseline has no percentage denominator. A tie still passes.
        return 100 * (reference - candidate) / reference if reference > 0 else np.nan

    def _forecast(self, train, future, covariates=()):
        with _temporary_storage(self.temp_dir):
            return self._forecast_in_temporary_storage(train, future, covariates)

    def _forecast_in_temporary_storage(self, train, future, covariates):
        from prophet import Prophet

        model = Prophet(**self.prophet_options)
        for period in self.seasonal_periods:
            # Periods are user-specified, not extracted from held-out observations
            # or the privileged SCM. Require two complete training cycles.
            if len(train) >= 2 * period:
                model.add_seasonality(
                    name=f"cycle_{period:g}",
                    period=period * self.step_duration / pd.Timedelta("1D"),
                    fourier_order=min(5, max(1, int(period // 2) - 1)),
                )
        for column in covariates:
            model.add_regressor(column)
        # CmdStan normally retains CSV/stdout files until the worker exits. Across
        # hundreds of thousands of fits that exhausts the system temp partition.
        with tempfile.TemporaryDirectory(prefix="fit_", dir=self.temp_dir) as directory:
            model.fit(train, seed=self.seed, output_dir=directory)
            return model.predict(future)["yhat"].to_numpy().copy()

    def evaluate_table(self, path):
        path = Path(path)
        record = {
            "table": path.parent.name,
            "source": str(path.resolve()),
            "covariate_mode": self.covariate_mode,
            "admission_metric": "mae",
            "status": "ok",
            "error": "",
            "multivariate_error": "",
            "source_sha256": "",
            "source_rows": np.nan,
            "reserved_tail_rows": self.reserved_tail_rows,
            "n_rows": np.nan,
            "n_train": np.nan,
            "n_test": np.nan,
            "n_covariates": 0,
            "covariates": "[]",
        }
        for key in (
            "baseline_mae",
            "multivariate_mae",
            "baseline_rmse",
            "multivariate_rmse",
            "multivariate_improvement_pct",
            "multivariate_beats_baseline",
            "multivariate_not_worse_than_baseline",
        ):
            record[key] = np.nan
        try:
            record["source_sha256"] = hashlib.sha256(path.read_bytes()).hexdigest()
            frame = (
                pd.read_parquet(path)
                .sort_values("time_index", kind="stable")
                .reset_index(drop=True)
            )
            times = frame["time_index"].to_numpy(dtype=float)
            if not np.isfinite(times).all() or (np.diff(times) <= 0).any():
                raise ValueError(
                    "time_index must contain finite, unique increasing values"
                )
            y = frame["outcome"].to_numpy(dtype=float)
            # The reserved final test outcomes must not influence screening.
            record.update(
                source_rows=len(frame), reserved_tail_rows=self.reserved_tail_rows
            )
            if self.reserved_tail_rows:
                frame = frame.iloc[: -self.reserved_tail_rows].copy()
                times = times[: -self.reserved_tail_rows]
                y = y[: -self.reserved_tail_rows]
            if not np.isfinite(y).all():
                raise ValueError(
                    "Screening outcome contains missing or infinite values"
                )
            split = int(len(frame) * self.train_fraction)
            if split < 2 or split >= len(frame):
                raise ValueError("Need at least two training rows and one test row")
            dates = pd.Timestamp("2000-01-01") + pd.to_timedelta(
                (times - times[0]) * self.step_duration.total_seconds(), unit="s"
            )
            train = pd.DataFrame({"ds": dates[:split], "y": y[:split]})
            future = pd.DataFrame({"ds": dates[split:]})
            actual = y[split:]
            baseline_predictions = np.full(len(actual), y[split - 1])
            baseline = self.mae(actual, baseline_predictions)
            record.update(
                n_rows=len(frame),
                n_train=split,
                n_test=len(actual),
                baseline_mae=baseline,
                baseline_rmse=self.rmse(actual, baseline_predictions),
                n_covariates=0,
                covariates="[]",
            )
        except Exception as exc:
            record.update(status="error", error=f"{type(exc).__name__}: {exc}")
            return record

        try:
            training, prediction = train.copy(), future.copy()
            columns = []
            # Lag columns expose held-out outcomes in a fixed-origin forecast.
            for column in sorted(
                c
                for c in frame
                if c.startswith("covariate_")
                and not re.search(r"(?i)(?:^|_)lag(?:ged)?(?:_?\d+)?(?:_|$)", c)
            ):
                past = frame[column].iloc[:split].to_numpy(dtype=float)
                if not np.isfinite(past).all():
                    raise ValueError(f"Nonfinite training covariate: {column}")
                # Select regressors using training observations only.
                if np.unique(past).size < 2:
                    continue
                training[column] = past
                if self.covariate_mode == "known_future":
                    values = frame[column].iloc[split:].to_numpy(dtype=float)
                else:
                    values = self._forecast(
                        pd.DataFrame({"ds": train["ds"], "y": past}), future
                    )
                if not np.isfinite(values).all():
                    raise ValueError(f"Nonfinite future covariate: {column}")
                prediction[column] = values
                columns.append(column)
            record.update(n_covariates=len(columns), covariates=json.dumps(columns))
            predictions = self._forecast(training, prediction, columns)
            score = self.mae(actual, predictions)
            record.update(
                multivariate_mae=score,
                multivariate_rmse=self.rmse(actual, predictions),
                multivariate_improvement_pct=self.improvement(baseline, score),
                multivariate_beats_baseline=int(score < baseline),
                multivariate_not_worse_than_baseline=int(score <= baseline),
            )
        except Exception as exc:
            if (
                "no space left" in str(exc).lower()
                or "not enough space" in str(exc).lower()
                or (isinstance(exc, OSError) and exc.errno == errno.ENOSPC)
            ):
                raise OSError(
                    errno.ENOSPC,
                    "Prophet temporary storage is full; stopping screening. Free space on the configured temp drive and retry failed tables.",
                ) from exc
            record.update(
                status="error", multivariate_error=f"{type(exc).__name__}: {exc}"
            )
        return record

    def worker_count(self, n_tables):
        # Conservative automatic cap; explicit settings may use more cores.
        requested = self.workers or min(4, max(1, (os.cpu_count() or 2) // 2))
        return min(requested, n_tables)

    def evaluate_tables(self, paths):
        """Yield completed (path, record) pairs; only the parent writes reports."""
        paths = [Path(path) for path in paths]
        workers = self.worker_count(len(paths))
        if workers <= 1:
            for path in paths:
                yield path, self.evaluate_table(path)
            return
        with (
            _worker_environment(),
            ProcessPoolExecutor(
                max_workers=workers,
                mp_context=get_context("spawn"),
                initializer=_initialize_worker,
            ) as pool,
        ):
            remaining = iter(paths)
            # Bound queued jobs for large generated cohorts.
            pending = {
                pool.submit(self.evaluate_table, path): path
                for path in [next(remaining, None) for _ in range(2 * workers)]
                if path is not None
            }
            while pending:
                completed, _ = wait(pending, return_when=FIRST_COMPLETED)
                for job in completed:
                    path = pending.pop(job)
                    yield path, job.result()
                    following = next(remaining, None)
                    if following is not None:
                        pending[pool.submit(self.evaluate_table, following)] = following

    def run(self):
        if not self.input_dir.is_dir():
            raise FileNotFoundError(f"Dataset day folder not found: {self.input_dir}")
        # Include a row for missing parquet files too, rather than silently
        # dropping a broken table folder from the evaluation denominator.
        paths = [
            folder / "data.parquet"
            for folder in sorted(self.input_dir.iterdir())
            if folder.is_dir()
        ]
        if not paths:
            raise ValueError(f"No table subfolders found in {self.input_dir}")
        from importlib.metadata import version

        run_dir = self.output_dir / (
            self.input_dir.name
            + "_"
            + datetime.now().strftime("%Y%m%d_%H%M%S")
            + "_"
            + uuid4().hex[:8]
        )
        run_dir.mkdir(parents=True, exist_ok=False)
        (run_dir / "settings.json").write_text(
            json.dumps(
                {
                    "input_dir": str(self.input_dir.resolve()),
                    "train_fraction": self.train_fraction,
                    "covariate_mode": self.covariate_mode,
                    "step_duration": str(self.step_duration),
                    "seed": self.seed,
                    "prophet_options": self.prophet_options,
                    "seasonal_periods": self.seasonal_periods,
                    "reserved_tail_rows": self.reserved_tail_rows,
                    "workers": self.workers,
                    "effective_workers": self.worker_count(len(paths)),
                    "temp_dir": str(self.temp_dir.resolve()),
                    "admission_rule": "multivariate_mae <= baseline_mae",
                    "prophet_version": version("prophet"),
                },
                indent=2,
            ),
            encoding="utf-8",
        )
        records = {}
        for index, (path, record) in enumerate(self.evaluate_tables(paths), 1):
            records[path] = record
            # Append each completion: rewriting all previous rows is quadratic.
            pd.DataFrame([record]).to_csv(
                run_dir / "metrics.csv", mode="a", header=index == 1, index=False
            )
            print(
                f"[{index}/{len(paths)}] {path.parent.name}: {record['status']}",
                flush=True,
            )
        results = pd.DataFrame([records[path] for path in paths])
        for column in (
            "multivariate_beats_baseline",
            "multivariate_not_worse_than_baseline",
        ):
            results[column] = results[column].astype("Int64")
        results.to_csv(run_dir / "metrics.csv", index=False)
        results.to_parquet(run_dir / "metrics.parquet", index=False)
        return results, run_dir

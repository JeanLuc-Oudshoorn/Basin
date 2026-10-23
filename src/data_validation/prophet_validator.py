"""Chronological Prophet evaluation, without using SCM metadata or future y."""

import json
from datetime import datetime
from pathlib import Path
from uuid import uuid4

import numpy as np
import pandas as pd


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
    ):
        if not 0 < train_fraction < 1:
            raise ValueError("train_fraction must lie strictly between 0 and 1")
        if covariate_mode not in ("known_future", "forecast"):
            raise ValueError("covariate_mode must be known_future or forecast")
        self.input_dir, self.output_dir = Path(input_dir), Path(output_dir)
        self.train_fraction, self.covariate_mode = train_fraction, covariate_mode
        self.step_duration = pd.Timedelta(step_duration)
        if self.step_duration <= pd.Timedelta(0):
            raise ValueError("step_duration must be positive")
        self.seed = seed
        # Point forecasts only: no expensive posterior uncertainty simulation.
        self.prophet_options = {"uncertainty_samples": 0, **(prophet_options or {})}
        self.seasonal_periods = tuple(float(p) for p in seasonal_periods)
        if any(not np.isfinite(p) or p <= 2 for p in self.seasonal_periods):
            raise ValueError("Seasonal periods must be finite and exceed two timesteps")

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
        # A perfect baseline has no percentage denominator. Leave this undefined
        # (blank/NaN), while strict win flags and absolute differences remain valid.
        return 100 * (reference - candidate) / reference if reference > 0 else np.nan

    def _forecast(self, train, future, covariates=()):
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
        model.fit(train, seed=self.seed)
        return model.predict(future)["yhat"].to_numpy()

    def evaluate_table(self, path):
        path = Path(path)
        record = {
            "table": path.parent.name,
            "source": str(path.resolve()),
            "covariate_mode": self.covariate_mode,
            "status": "ok",
            "error": "",
            "univariate_error": "",
            "multivariate_error": "",
        }
        for key in (
            "baseline_rmse",
            "univariate_rmse",
            "multivariate_rmse",
            "univariate_improvement_pct",
            "multivariate_improvement_pct",
            "rmse_reduction_multivariate_vs_univariate",
            "multivariate_vs_univariate_improvement_pct",
            "improvement_difference_pp",
            "univariate_beats_baseline",
            "multivariate_beats_baseline",
        ):
            record[key] = np.nan
        try:
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
            if not np.isfinite(y).all():
                raise ValueError("Outcome contains missing or infinite values")
            split = int(len(frame) * self.train_fraction)
            if split < 2 or split >= len(frame):
                raise ValueError("Need at least two training rows and one test row")
            dates = pd.Timestamp("2000-01-01") + pd.to_timedelta(
                (times - times[0]) * self.step_duration.total_seconds(), unit="s"
            )
            train = pd.DataFrame({"ds": dates[:split], "y": y[:split]})
            future = pd.DataFrame({"ds": dates[split:]})
            actual = y[split:]
            baseline = self.rmse(actual, np.full(len(actual), y[split - 1]))
            record.update(
                n_rows=len(frame),
                n_train=split,
                n_test=len(actual),
                baseline_rmse=baseline,
                n_covariates=0,
                covariates="[]",
            )
        except Exception as exc:
            record.update(status="error", error=f"{type(exc).__name__}: {exc}")
            return record

        # Fit the two target models independently. Failure of one must not hide
        # the other's result, and a failed fit must never look like a baseline win.
        for variant in ("univariate", "multivariate"):
            try:
                training, prediction = train.copy(), future.copy()
                columns = []
                if variant == "multivariate":
                    # Outcome lag columns contain test y after the split. They
                    # are NOT usable as fixed-origin future regressors.
                    for column in sorted(
                        c for c in frame if c.startswith("covariate_")
                    ):
                        past = frame[column].iloc[:split].to_numpy(dtype=float)
                        if not np.isfinite(past).all():
                            raise ValueError(f"Nonfinite training covariate: {column}")
                        # Constant training columns cannot identify a regression
                        # coefficient. Selection uses only training observations.
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
                    record.update(
                        n_covariates=len(columns), covariates=json.dumps(columns)
                    )
                predictions = self._forecast(training, prediction, columns)
                score = self.rmse(actual, predictions)
                record.update(
                    {
                        f"{variant}_rmse": score,
                        f"{variant}_improvement_pct": self.improvement(baseline, score),
                        f"{variant}_beats_baseline": int(score < baseline),
                    }
                )
            except Exception as exc:
                record["status"] = "partial_error"
                record[f"{variant}_error"] = f"{type(exc).__name__}: {exc}"
        uni, multi = record["univariate_rmse"], record["multivariate_rmse"]
        record["rmse_reduction_multivariate_vs_univariate"] = uni - multi
        record["multivariate_vs_univariate_improvement_pct"] = self.improvement(
            uni, multi
        )
        record["improvement_difference_pp"] = (
            record["multivariate_improvement_pct"]
            - record["univariate_improvement_pct"]
        )
        if record["univariate_error"] and record["multivariate_error"]:
            record["status"] = "error"
        return record

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
                    "prophet_version": version("prophet"),
                },
                indent=2,
            ),
            encoding="utf-8",
        )
        records = []
        for index, path in enumerate(paths, 1):
            records.append(self.evaluate_table(path))
            results = pd.DataFrame(records)
            for column in ("univariate_beats_baseline", "multivariate_beats_baseline"):
                results[column] = results[column].astype("Int64")
            # Checkpoint after each table so long runs retain completed results.
            results.to_csv(run_dir / "metrics.csv", index=False)
            print(
                f"[{index}/{len(paths)}] {path.parent.name}: {records[-1]['status']}",
                flush=True,
            )
        results.to_parquet(run_dir / "metrics.parquet", index=False)
        return results, run_dir

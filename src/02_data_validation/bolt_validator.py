"""Batched Chronos-Bolt Small admission screening, without future leakage."""

import hashlib
from pathlib import Path
from time import perf_counter

import numpy as np
import pandas as pd
import torch
from loguru import logger


def bolt_admitted(results, tolerance=0.10):
    """Discard valid tables only for MASE strictly above (1+tolerance)*naive."""
    baseline, score = results.baseline_mase, results.chronos_bolt_mase
    valid = results.status.eq("ok")
    finite = np.isfinite(baseline) & np.isfinite(score) & (baseline >= 0) & (score >= 0)
    # Undefined MASE is not evidence that the model is worse: retain and flag it.
    undefined = (
        results.mase_scale.eq(0)
        & np.isfinite(results.baseline_mae)
        & np.isfinite(results.chronos_bolt_mae)
        & results.baseline_mae.ge(0)
        & results.chronos_bolt_mae.ge(0)
    )
    return valid & ((finite & (score <= (1 + tolerance) * baseline)) | undefined)


class BoltValidator:
    engine = "chronos_bolt"
    label = "Chronos-Bolt"
    admission_metric = "mase"
    admitted = staticmethod(bolt_admitted)

    def __init__(self, config):
        self.config = config
        policy = config["validation"]
        self.options = policy[self.engine]
        self.tolerance = policy.get(
            "max_relative_degradation", 0.15 if self.engine == "chronos2" else 0.10
        )
        if (
            isinstance(self.tolerance, bool)
            or not np.isfinite(self.tolerance)
            or self.tolerance < 0
        ):
            raise ValueError(
                "validation.max_relative_degradation must be finite and nonnegative"
            )
        self.train_fraction = policy["train_fraction"]
        if not 0 < self.train_fraction < 1:
            raise ValueError("validation.train_fraction must be between 0 and 1")
        self.batch_size = self.options.get("batch_size", 64)
        self.horizon = policy.get("forecast_horizon", config["data"]["horizon"])
        self.lag = policy.get("mase_seasonality", 1)
        self.context_length = self.options.get("context_length", 2048)
        self.cpu_threads = self.options.get("cpu_threads", 4)
        for name, value in (
            ("batch_size", self.batch_size),
            ("forecast_horizon", self.horizon),
            ("mase_seasonality", self.lag),
            ("context_length", self.context_length),
            ("cpu_threads", self.cpu_threads),
        ):
            if isinstance(value, bool) or not isinstance(value, int) or value < 1:
                raise ValueError(f"validation {name} must be a positive integer")
        self.pipeline = None
        self.temp_dir = Path(__file__).resolve().parents[2] / self.options.get(
            "cache_dir", ".cache/huggingface"
        )
        self.temp_dir.mkdir(parents=True, exist_ok=True)
        self.metadata = {}

    def worker_count(self, count):
        return min(
            1, count
        )  # one resident model; batch tables rather than spawn models

    def _load(self):
        if self.pipeline is not None:
            return
        torch.set_num_threads(self.cpu_threads)

        device = self.options.get("device", "auto")
        if device == "auto":
            device = "cuda" if torch.cuda.is_available() else "cpu"
        started = perf_counter()
        logger.info(
            "Loading screening model={} device={} cache={}",
            self.options["model_id"],
            device,
            self.temp_dir,
        )
        from chronos import BaseChronosPipeline

        self.pipeline = BaseChronosPipeline.from_pretrained(
            self.options["model_id"],
            revision=self.options["revision"],
            device_map=device,
            torch_dtype=torch.float32,
            cache_dir=str(self.temp_dir),
            token=False,
        )
        self.metadata = {
            "model_id": self.options["model_id"],
            "revision": self.options["revision"],
            "device": device,
            "load_seconds": perf_counter() - started,
            "parameters": sum(p.numel() for p in self.pipeline.model.parameters()),
            "inputs": "historical target and covariates"
            if self.engine == "chronos2"
            else "historical target only; Chronos-Bolt does not support covariates",
        }
        logger.info("Screening model loaded {}", self.metadata)

    def _prepare(self, path):
        record = {
            "table": path.parent.name,
            "source": str(path.resolve()),
            "status": "ok",
            "error": "",
            "admission_metric": self.admission_metric,
            "screening_model": self.options["model_id"],
            "source_sha256": "",
            "source_rows": np.nan,
            "reserved_tail_rows": self.config["data"]["horizon"],
            "baseline_mase": np.nan,
            f"{self.engine}_mase": np.nan,
            "baseline_mae": np.nan,
            f"{self.engine}_mae": np.nan,
            "mase_scale": np.nan,
            "mase_defined": False,
        }
        try:
            with path.open("rb") as stream:
                record["source_sha256"] = hashlib.file_digest(
                    stream, "sha256"
                ).hexdigest()
            data = self.config["data"]
            if len(data["targets"]) != 1:
                raise ValueError(f"{self.label} screening requires exactly one target")
            frame = self._read_frame(path)
            times = frame[data["time_column"]].to_numpy(dtype=float)
            if not np.isfinite(times).all() or (np.diff(times) <= 0).any():
                raise ValueError("Time values must be finite and strictly increasing")
            reserved = record["reserved_tail_rows"]
            y = frame[data["targets"][0]].to_numpy(dtype=float)
            record["source_rows"] = len(y)
            y = y[:-reserved] if reserved else y
            if not np.isfinite(y).all():
                raise ValueError("Screening target contains missing/infinite values")
            split = int(len(y) * self.train_fraction)
            if split <= self.lag or split >= len(y):
                raise ValueError("Insufficient historical rows for screening")
            history = y[max(0, split - self.context_length) : split]
            actual = y[split : split + self.horizon]
            if len(history) <= self.lag:
                raise ValueError("Insufficient context for MASE scale")
            scale = float(np.abs(history[self.lag :] - history[: -self.lag]).mean())
            baseline = float(np.abs(actual - history[-1]).mean())
            record.update(
                n_rows=len(y),
                n_train=split,
                context_rows=len(history),
                n_test=len(actual),
                forecast_start=split,
                forecast_stop=split + len(actual),
                mase_scale=scale,
                baseline_mae=baseline,
                baseline_mase=baseline / scale if scale > 0 else np.nan,
                mase_defined=scale > 0,
            )
            inputs = self._model_input(frame, history, split, len(actual), record)
            return record, inputs, actual
        except Exception as error:
            record.update(status="error", error=f"{type(error).__name__}: {error}")
            return record, None, None

    @torch.inference_mode()
    def evaluate_tables(self, paths):
        """Bound memory and batch independent histories with matching horizons."""
        paths = list(map(Path, paths))
        for start in range(0, len(paths), self.batch_size):
            if start == 0:
                logger.info(
                    "Preparing first {} batch: {} tables",
                    self.label,
                    min(self.batch_size, len(paths)),
                )
            prepared = [
                (path, *self._prepare(path))
                for path in paths[start : start + self.batch_size]
            ]
            groups = {}
            for item in prepared:
                if item[2] is not None:
                    groups.setdefault(self._group_key(item), []).append(item)
            for group in groups.values():
                horizon = len(group[0][3])
                self._load()
                started = perf_counter()
                points = self._predict([item[2] for item in group], horizon)
                if (
                    points.shape != (len(group), horizon, 1)
                    or not np.isfinite(points).all()
                ):
                    raise ValueError(
                        f"Invalid {self.label} predictions: {points.shape}"
                    )
                for (_, record, _, actual), point in zip(group, points):
                    error = float(np.abs(actual - point[:, 0]).mean())
                    record[f"{self.engine}_mae"] = error
                    scale = record["mase_scale"]
                    record[f"{self.engine}_mase"] = (
                        error / scale if scale > 0 else np.nan
                    )
                logger.debug(
                    "Screened batch tables={} horizon={} seconds={:.3f}",
                    len(group),
                    horizon,
                    perf_counter() - started,
                )
            for path, record, _, _ in prepared:
                if record["status"] == "ok":
                    record["admitted"] = bool(
                        self.admitted(pd.DataFrame([record]), self.tolerance).iloc[0]
                    )
                    record["decision"] = (
                        "undefined_mase_keep"
                        if not record["mase_defined"] and self.engine == "chronos_bolt"
                        else "within_tolerance"
                        if record["admitted"]
                        else "more_than_tolerance_worse"
                    )
                yield path, record

    def _group_key(self, item):
        return len(item[3])

    def _read_frame(self, path):
        data = self.config["data"]
        return pd.read_parquet(path, columns=[data["time_column"], data["targets"][0]])

    def _model_input(self, frame, history, split, horizon, record):
        return torch.tensor(history, dtype=torch.float32)

    def _predict(self, inputs, horizon):
        quantiles, _ = self.pipeline.predict_quantiles(
            inputs, prediction_length=horizon, quantile_levels=[0.5]
        )
        return torch.as_tensor(quantiles).float().cpu().numpy()

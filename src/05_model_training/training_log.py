"""Automatic JSONL journals for training; imported by train.py, not a CLI script.

The trainer's decorator covers direct Python calls as well as CLI execution.
The outer run context and WSL worker share that journal rather than opening
duplicates. Output is written under the configured paths.logs directory.
"""

import json
import math
import os
import platform
import sys
from contextlib import contextmanager
from contextvars import ContextVar
from datetime import datetime, timezone
from functools import wraps
from pathlib import Path
from time import perf_counter, time
from uuid import uuid4

import torch

ROOT = Path(__file__).resolve().parents[2]
ENV_KEY = "BASIN_TRAINING_JSONL"
_ACTIVE = ContextVar("basin_training_jsonl", default=None)


def clean(value):
    if isinstance(value, float) and not math.isfinite(value):
        return None
    if isinstance(value, dict):
        return {str(k): clean(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [clean(v) for v in value]
    if isinstance(value, Path):
        return str(value)
    return value


def runtime():
    return {
        "platform": platform.platform(),
        "python_executable": sys.executable,
        "python_version": platform.python_version(),
        "torch_version": torch.__version__,
        "cuda_available": torch.cuda.is_available(),
        "pid": os.getpid(),
    }


def memory_metrics():
    result = {
        "cuda_memory": None,
        "cuda_memory_bytes": None,
        "memory": None,
        "process_rss_bytes": None,
        "system_used_bytes": None,
    }
    try:
        if torch.cuda.is_available():
            stats = {
                "allocated": torch.cuda.memory_allocated(),
                "reserved": torch.cuda.memory_reserved(),
                "peak": torch.cuda.max_memory_allocated(),
            }
            result["cuda_memory_bytes"] = stats
            result["cuda_memory"] = ", ".join(
                f"{k}={v / 1024**2:.1f} MiB" for k, v in stats.items()
            )
    except Exception as error:
        result["cuda_memory_error"] = str(error)
    try:
        import psutil

        rss = psutil.Process().memory_info().rss
        used = psutil.virtual_memory().used
        result.update(
            process_rss_bytes=rss,
            system_used_bytes=used,
            memory=f"process RSS={rss / 1024**3:.3f} GiB, system used={used / 1024**3:.3f} GiB",
        )
    except Exception as error:
        result["memory_error"] = str(error)
    return result


class TrainingLog:
    def __init__(self, config, stage):
        self.config, self.stage = config, stage
        directory = ROOT / config["paths"]["logs"]
        directory.mkdir(parents=True, exist_ok=True)
        inherited = os.environ.get(ENV_KEY)
        identity = json.loads(inherited) if inherited else None
        self.attached = bool(identity and identity["stage"] == stage)
        if self.attached:
            filename = identity["filename"]
            if Path(filename).name != filename or not filename.endswith(".jsonl"):
                raise ValueError("Invalid shared training journal filename")
            self.run_id = identity["run_id"]
            if filename != self.run_id + ".jsonl":
                raise ValueError("Shared journal identity mismatch")
            self.started_unix = identity["started_unix"]
            self.path = directory / filename
            if not self.path.is_file():
                raise FileNotFoundError(
                    f"WSL shared training journal missing: {self.path}"
                )
        else:
            stamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S_%fZ")
            self.run_id = f"rwkv7_{stage}_{stamp}_pid{os.getpid()}_{uuid4().hex[:8]}"
            self.path = directory / (self.run_id + ".jsonl")
            self.started_unix = time()
            # Exclusive creation prevents collisions; append mode prevents launcher
            # writes from overwriting events appended by the WSL worker.
            with self.path.open("x", encoding="utf-8"):
                pass
        self.stream = self.path.open("a", encoding="utf-8", buffering=1)
        self.started = perf_counter() - max(0, time() - self.started_unix)
        self.step = 0
        self.best_step = None
        self.best_validation_loss = None
        self.final_validation_loss = None
        self.summary = {}
        self.emit(
            "worker_start" if self.attached else "run_start",
            config=config,
            runtime=runtime(),
            loss_name="normalized_pinball",
            global_step_scope="current_run",
            unit_name="observations",
            unsupported_metrics={
                "perplexity": "Not defined for continuous pinball loss",
                "energy": "Power and energy sampling is not enabled",
            },
        )

    def shared_identity(self):
        return json.dumps(
            {
                "stage": self.stage,
                "filename": self.path.name,
                "run_id": self.run_id,
                "started_unix": self.started_unix,
            }
        )

    def emit(self, event, **data):
        if "global_step" in data:
            self.step = data["global_step"]
        validation = data.get("validation_loss")
        if validation is not None and math.isfinite(validation):
            self.final_validation_loss = validation
            if event != "validation_baseline" and (
                self.best_validation_loss is None
                or validation < self.best_validation_loss
            ):
                self.best_validation_loss, self.best_step = validation, self.step
        if event == "training_complete":
            self.summary.update(data)
        if event == "out_of_sample_evaluation":
            result = data["out_of_sample_metrics"]
            self.summary["out_of_sample_metrics"] = {
                "table_count": result["table_count"],
                "pooled": result["pooled"],
            }
        now = time()
        record = {
            "event": event,
            "experiment": "rwkv7_timeseries",
            "training_mode": self.stage,
            "run_id": self.run_id,
            "timestamp_utc": datetime.now(timezone.utc).isoformat(),
            "unix_seconds": now,
            "elapsed_seconds": max(0, perf_counter() - self.started),
            "global_step": self.step,
            "run_step": self.step,
            "train_loss": None,
            "validation_loss": None,
            "perplexity": None,
            "best_step": self.best_step,
            "best_validation_loss": self.best_validation_loss,
            "training_average_power_watts": None,
            "training_energy_backend": "not_monitored",
            "training_energy_error": None,
            "training_energy_joules": None,
            "training_energy_kwh": None,
            **memory_metrics(),
            **data,
        }
        self.stream.write(
            json.dumps(
                clean(record), allow_nan=False, ensure_ascii=False, sort_keys=True
            )
            + "\n"
        )
        self.stream.flush()
        # Python flushing makes every update visible to readers. At epoch and
        # checkpoint boundaries also commit the journal through the OS cache.
        if event in {
            "epoch_end",
            "validation_baseline",
            "epoch_checkpoint_saved",
            "training_complete",
            "run_complete",
            "run_error",
        }:
            os.fsync(self.stream.fileno())

    def terminal_event(self):
        # Completion records are small, even when the start config is large.
        with self.path.open("rb") as stream:
            stream.seek(0, 2)
            end = stream.tell()
            stream.seek(max(0, end - 65536))
            lines = stream.read().splitlines()
        if not lines:
            return None
        try:
            return json.loads(lines[-1]).get("event")
        except (ValueError, UnicodeDecodeError):
            return None

    def finish(self, error=None):
        terminal = self.terminal_event()
        # A WSL worker has already closed the same journal successfully, or
        # recorded the original failure. The launcher must not duplicate it.
        if terminal == "run_error" or (terminal == "run_complete" and error is None):
            self.stream.close()
            return
        try:
            if error is not None:
                self.emit(
                    "run_error",
                    status="failed",
                    error_type=type(error).__name__,
                    error=str(error),
                )
            else:
                self.emit(
                    "run_complete",
                    **{
                        **self.summary,
                        "status": "complete",
                        "completed_global_step": self.step,
                        "completed_run_step": self.step,
                        "final_validation_loss": self.final_validation_loss,
                    },
                )
        finally:
            self.stream.close()


@contextmanager
def training_run(config, stage):
    active = _ACTIVE.get()
    if active is not None:
        if active.stage != stage:
            raise ValueError("Nested training stages cannot share a run identity")
        yield active
        return
    journal = TrainingLog(config, stage)
    token = _ACTIVE.set(journal)
    try:
        yield journal
    except BaseException as error:
        journal.finish(error)
        raise
    else:
        journal.finish()
    finally:
        _ACTIVE.reset(token)


def logged_training(stage):
    def decorate(function):
        @wraps(function)
        def wrapped(config, *args, **kwargs):
            with training_run(config, stage):
                return function(config, *args, **kwargs)

        return wrapped

    return decorate


def emit_training_event(event, **data):
    journal = _ACTIVE.get()
    if journal is not None:
        journal.emit(event, **data)


def worker_environment():
    environment = os.environ.copy()
    journal = _ACTIVE.get()
    if journal is not None:
        environment[ENV_KEY] = journal.shared_identity()
        # WSL only forwards Windows variables explicitly listed in WSLENV.
        forwarded = [
            entry
            for entry in environment.get("WSLENV", "").split(":")
            if entry and entry.split("/")[0] != ENV_KEY
        ]
        environment["WSLENV"] = ":".join(forwarded + [ENV_KEY])
        journal.emit("worker_launch")
    return environment


def verify_worker_completed():
    journal = _ACTIVE.get()
    if journal is not None and journal.terminal_event() != "run_complete":
        raise RuntimeError(
            "WSL worker did not complete the shared training JSONL journal; check WSLENV forwarding"
        )

"""Stage 1: raw parquet -> explicit, leakage-free train/evaluation windows."""

import argparse
import hashlib
import json
import re
import sys
import uuid
from contextlib import contextmanager, nullcontext
from datetime import date, datetime, timezone
from pathlib import Path
from time import perf_counter

import numpy as np
import pandas as pd
import torch
from loguru import logger
from torch.utils.serialization import config as serialization_config

ROOT = Path(__file__).resolve().parents[2]


def load_config(path=None):
    path = Path(path) if path else ROOT / "configs/config.json"
    if not path.is_absolute():
        path = ROOT / path
    config = json.loads(path.read_text(encoding="utf-8"))
    generated_dates(config)
    d = config["data"]
    for key in (
        "context_length",
        "horizon",
        "train_windows_per_table",
        "stride",
    ):
        if not isinstance(d[key], int) or isinstance(d[key], bool) or d[key] < 1:
            raise ValueError(f"data.{key} must be a positive integer")
    if not d["targets"] or len(set(d["targets"] + d["known_covariates"])) != len(
        d["targets"] + d["known_covariates"]
    ):
        raise ValueError(
            "targets and known_covariates must be nonempty/disjoint as applicable"
        )
    if d["max_tables"] is not None and d["max_tables"] < 1:
        raise ValueError("max_tables must be null (all tables) or positive")
    steps = config["training"]["steps"]
    if (steps is not None and steps < 1) or config["training"]["learning_rate"] <= 0:
        raise ValueError("training steps and learning_rate must be positive")
    return config


def location(config, name):
    return ROOT / config["paths"][name]


def generated_dates(config):
    """Explicit dates are reproducible; absence retains legacy raw-path behavior."""
    if "generated_dates" not in config["data"]:
        return None
    dates = config["data"]["generated_dates"]
    if not isinstance(dates, list) or not dates:
        raise ValueError(
            "data.generated_dates must be a nonempty list of YYYY-MM-DD folder names"
        )
    for value in dates:
        if not isinstance(value, str) or not re.fullmatch(r"\d{4}-\d{2}-\d{2}", value):
            raise ValueError(
                f"Invalid generated date folder: {value!r}; expected YYYY-MM-DD"
            )
        try:
            date.fromisoformat(value)
        except ValueError as error:
            raise ValueError(f"Invalid generated date folder: {value!r}") from error
    if len(set(dates)) != len(dates):
        raise ValueError("data.generated_dates must not contain duplicates")
    return sorted(dates)


def source_selection(config):
    return {
        "raw": str(config["paths"]["raw"]).replace("\\", "/").rstrip("/"),
        "generated_dates": generated_dates(config),
    }


def discover_source_tables(config):
    """Shared source selection for Prophet screening and both transform modes."""
    root = location(config, "raw")
    if not root.is_dir():
        raise FileNotFoundError(f"Generated-data root not found: {root}")
    dates = generated_dates(config)
    directories = [root / day for day in dates] if dates is not None else [root]
    paths = []
    for directory in directories:
        if not directory.is_dir():
            raise FileNotFoundError(
                f"Selected generated date folder not found: {directory}"
            )
        tables = sorted(directory.rglob("data.parquet"))
        if not tables:
            raise ValueError(f"No data.parquet tables in selected folder: {directory}")
        logger.info("Selected generated folder={} tables={}", directory, len(tables))
        paths.extend(tables)
    logger.info(
        "Source selection={} total_tables={}", source_selection(config), len(paths)
    )
    return paths


def signature(config):
    content = {k: config[k] for k in ("seed", "data", "model")}
    if "validation" in config:
        content["validation"] = config["validation"]
    if "generated_dates" in config["data"]:
        # Canonical ordering makes date-list order irrelevant. Bind the root too:
        # the same date names under another source root are a different cohort.
        content["data"] = dict(config["data"], generated_dates=generated_dates(config))
        content["source_selection"] = source_selection(config)
    return hashlib.sha256(json.dumps(content, sort_keys=True).encode()).hexdigest()


def file_digest(path):
    with Path(path).open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def setup_logging(config, stage):
    run_id = (
        datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S_%fZ")
        + f"_{stage}_{uuid.uuid4().hex[:8]}"
    )
    directory = location(config, "logs") / run_id
    directory.mkdir(parents=True, exist_ok=False)
    logger.remove()
    logger.add(sys.stderr, level=config["logging"]["level"])
    logger.add(
        str(directory / "events.log"), level="DEBUG", rotation="10 MB", diagnose=False
    )
    logger.info(
        "Stage={} interpreter={} torch={} CUDA={} root={}",
        stage,
        sys.executable,
        torch.__version__,
        torch.cuda.is_available(),
        ROOT,
    )
    logger.info("Input configuration:\n{}", json.dumps(config, indent=2))
    write_json(directory / "config.json", config)
    sources = [
        ROOT / "rwkv7_backend.py",
        *ROOT.glob("src/02_data_validation/*.py"),
        *ROOT.glob("src/03_transforming_input_data/*.py"),
        *ROOT.glob("src/04_creating_embeddings/*.py"),
        *ROOT.glob("src/05_model_training/*.py"),
        *ROOT.glob("src/06_holdout_test/*.py"),
        *ROOT.glob("src/07_forecasting/*.py"),
    ]
    write_json(
        directory / "provenance.json",
        {
            "run_id": run_id,
            "stage": stage,
            "interpreter": sys.executable,
            "torch": torch.__version__,
            "source_sha256": {
                p.relative_to(ROOT).as_posix(): file_digest(p) for p in sources
            },
        },
    )
    return directory


@contextmanager
def run_context(config, stage):
    if stage in ("train", "capacity_test", "holdout_test"):
        # Training owns the JSONL journal; preprocessing needs only ordinary logs.
        sys.path.insert(0, str(ROOT / "src/05_model_training"))
        from training_log import training_run

        context = training_run(config, stage)
    else:
        context = nullcontext()
    with context as journal:
        directory = setup_logging(config, stage)
        if journal is not None:
            logger.info("Training JSONL: {}", journal.path)
            write_json(
                directory / "training_jsonl.json",
                {
                    "path": str(journal.path),
                    "run_id": journal.run_id,
                    "description": "Pointer to the live JSONL journal; open path for training events. This file is not the journal.",
                    "journal_filename": journal.path.name,
                    "journal_relative_to_this_file": "../" + journal.path.name,
                    "metrics_visibility": "Every event is flushed; epoch and checkpoint events are also synced to disk.",
                    "epoch_metrics_event": "epoch_end",
                },
            )
        started = perf_counter()
        write_json(directory / "status.json", {"status": "running"})
        try:
            yield directory
        except BaseException as error:
            logger.exception("Stage failed: {}", stage)
            write_json(
                directory / "status.json",
                {
                    "status": "failed",
                    "error": str(error),
                    "seconds": perf_counter() - started,
                },
            )
            raise
        else:
            write_json(
                directory / "status.json",
                {"status": "complete", "seconds": perf_counter() - started},
            )


@contextmanager
def timed(label, cuda=False, level="INFO"):
    if cuda:
        torch.cuda.synchronize()
    start = perf_counter()
    logger.log(level, "START {}", label)
    try:
        yield
    except Exception:
        logger.exception("FAILED {} after {:.4f}s", label, perf_counter() - start)
        raise
    finally:
        if cuda:
            torch.cuda.synchronize()
        logger.log(level, "END {} elapsed={:.4f}s", label, perf_counter() - start)


def describe(name, tensor, preview=8):
    x = tensor.detach().cpu()
    finite = torch.isfinite(x)
    logger.debug(
        "{} shape={} dtype={} finite={}/{} preview={}",
        name,
        list(x.shape),
        x.dtype,
        int(finite.sum()),
        x.numel(),
        x.flatten()[:preview].tolist(),
    )
    if not finite.all():
        logger.warning(
            "{} contains {} missing/nonfinite values", name, int((~finite).sum())
        )
    if finite.any():
        values = x[finite].float()
        logger.debug(
            "{} finite statistics min={} max={} mean={}",
            name,
            float(values.min()),
            float(values.max()),
            float(values.mean()),
        )
    return {
        "shape": list(x.shape),
        "dtype": str(x.dtype),
        "finite": int(finite.sum()),
        "elements": x.numel(),
    }


def save_artifact(path, payload):
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(payload, path)
    logger.info("Saved artifact={} bytes={}", path, path.stat().st_size)


def read_artifact(path, config):
    logger.info("Loading artifact={} bytes={}", path, path.stat().st_size)
    # Map tensor storage from the artifact instead of copying the whole corpus
    # into anonymous RAM. Only accessed pages need to become resident.
    # Calculate offsets from the modern checkpoint layout instead of seeking to
    # every tensor header. Large corpora otherwise cause hundreds of thousands
    # of small reads on the project drive even with memory mapping enabled.
    previous_offsets = serialization_config.load.calculate_storage_offsets
    try:
        serialization_config.load.calculate_storage_offsets = True
        artifact = torch.load(path, map_location="cpu", weights_only=True, mmap=True)
    finally:
        serialization_config.load.calculate_storage_offsets = previous_offsets
    if artifact["signature"] != signature(config):
        raise ValueError(
            f"Stale artifact {path}: seed/data/model configuration changed; rerun upstream stages"
        )
    if artifact.get("format") == "lazy_patch_index_v1":
        sys.path.insert(0, str(ROOT / "src/04_creating_embeddings"))
        from embed import open_patch_index

        return open_patch_index(artifact, config)
    return artifact


def write_json(path, value, *, level="INFO"):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, allow_nan=False), encoding="utf-8")
    logger.log(level, "Saved JSON={}", path)


def frame_window(
    frame,
    *,
    cutoff,
    context_length,
    horizon,
    targets=("outcome",),
    past_covariates=(),
    known_covariates=(),
    device="cpu",
):
    columns = list(targets) + list(past_covariates) + list(known_covariates)
    if not targets or len(set(columns)) != len(columns):
        raise ValueError("Column roles must be disjoint and targets nonempty")
    if (
        context_length < 1
        or horizon < 1
        or cutoff < context_length
        or cutoff + horizon > len(frame)
    ):
        raise ValueError("Window lies outside table")

    def tensor(names, start, stop):
        return torch.as_tensor(
            frame[list(names)].iloc[start:stop].to_numpy(dtype=np.float32).T.copy(),
            device=device,
        ).unsqueeze(0)

    batch = {
        "context": tensor(targets, cutoff - context_length, cutoff),
        "horizon": horizon,
        "future_target": tensor(targets, cutoff, cutoff + horizon),
    }
    if past_covariates:
        batch["past_covariates"] = tensor(
            past_covariates, cutoff - context_length, cutoff
        )
    if known_covariates:
        batch["known_covariates"] = tensor(
            known_covariates, cutoff - context_length, cutoff
        )
        batch["future_covariates"] = tensor(known_covariates, cutoff, cutoff + horizon)
    return batch


def validation_run_root(config):
    """Separate cohorts by canonical dates and source root; each run is immutable."""
    selection = source_selection(config)
    dates = generated_dates(config)
    label = "_".join(dates) if dates else "selected_tables"
    if len(label) > 70:
        label = f"{dates[0]}_to_{dates[-1]}"
    identity = hashlib.sha256(
        json.dumps(selection, sort_keys=True).encode()
    ).hexdigest()[:12]
    return (ROOT / config["validation"]["metrics_path"]).parent / f"{label}_{identity}"


def validation_report_path(config):
    """Locate this cohort's most recent fully written report."""
    policy = config["validation"]
    if not policy.get("run_folders", True):
        return ROOT / policy["metrics_path"]
    directory = validation_run_root(config).resolve()
    pointer = directory / "latest.json"
    if not pointer.is_file():
        raise FileNotFoundError(
            f"No validation report for {source_selection(config)}. Run src/02_data_validation/validate_generated_data.py first."
        )
    metadata = json.loads(pointer.read_text(encoding="utf-8"))
    path = (directory / metadata["report"]).resolve()
    if not path.is_relative_to(directory):
        raise ValueError(
            "Validation report pointer must remain inside its cohort folder"
        )
    return path


def validated_sources(config, paths):
    """Admit current successful multivariate forecasts no worse than persistence."""
    policy = config.get("validation", {})
    if not policy.get("enabled", False):
        logger.warning("Prophet learnability gate disabled for this configuration")
        return paths, {
            "enabled": False,
            "excluded": [],
            "source_selection": source_selection(config),
        }
    if (
        config["data"]["targets"] != ["outcome"]
        or config["data"]["time_column"] != "time_index"
    ):
        raise ValueError(
            "Prophet screening currently requires target=outcome and time_column=time_index"
        )
    report_path = validation_report_path(config)
    if not report_path.is_file():
        raise FileNotFoundError(
            f"Missing validation report {report_path}. Run src/02_data_validation/validate_generated_data.py first."
        )
    scores = pd.read_parquet(report_path)
    required = {
        "source_key",
        "source_sha256",
        "status",
        "baseline_mae",
        "multivariate_mae",
        "reserved_tail_rows",
        "covariate_mode",
        "screening_signature",
    }
    if required - set(scores):
        raise ValueError(
            f"Validation report lacks columns: {sorted(required - set(scores))}"
        )
    if scores.source_key.duplicated().any():
        raise ValueError("Duplicate source keys in validation report")
    expected = screening_signature(config)
    indexed = scores.set_index("source_key")
    admitted, excluded = [], []
    for path in paths:
        key = path.relative_to(location(config, "raw")).as_posix()
        reason = "missing validation row"
        if key in indexed.index:
            row = indexed.loc[key]
            reason = "Multivariate Prophet did not produce a valid MAE <= last-observation MAE"
            if row.screening_signature != expected:
                reason = "screening configuration mismatch; rerun validation"
            elif row.source_sha256 != file_digest(path):
                reason = "source content changed; rerun validation"
            elif row.reserved_tail_rows != config["data"]["horizon"]:
                reason = "final test horizon was not reserved"
            elif row.covariate_mode != policy["covariate_mode"]:
                reason = "covariate availability mismatch"
            elif row.status == "ok":
                baseline = row.baseline_mae
                score = row.multivariate_mae
                fit_error = row.get("multivariate_error", "")
                if (
                    np.isfinite(baseline)
                    and baseline >= 0
                    and np.isfinite(score)
                    and 0 <= score <= baseline
                    and (pd.isna(fit_error) or not str(fit_error).strip())
                ):
                    admitted.append(path)
                    logger.info(
                        "ADMIT {} baseline_mae={} multivariate_mae={}",
                        key,
                        baseline,
                        score,
                    )
                    continue
            else:
                reason = "validation failed"
        excluded.append({"source": key, "reason": reason})
        logger.warning("REJECT {}: {}", key, reason)
    audit = {
        "enabled": True,
        "report": str(report_path),
        "report_sha256": file_digest(report_path),
        "screening_signature": expected,
        "source_selection": source_selection(config),
        "source_count": len(paths),
        "admitted_count": len(admitted),
        "excluded": excluded,
    }
    if "smoke_test_only" in scores and scores.smoke_test_only.fillna(False).any():
        audit.update(
            smoke_test_only=True,
            actual_scoring_metric="rmse",
            metric_alias="RMSE values exposed as MAE for the authorized smoke test",
        )
        logger.info(
            "Smoke-test screening: legacy RMSE scores are being used via MAE aliases"
        )
    write_json(location(config, "processed") / "validation_selection.json", audit)
    logger.info(
        "Learnability selection: {} admitted / {} candidates", len(admitted), len(paths)
    )
    if not admitted:
        raise ValueError(
            "No datasets passed Prophet-vs-naive validation; inspect validation_selection.json"
        )
    return admitted, audit


def screening_signature(config):
    """Separate screening settings from training/embedding hyperparameters."""
    policy = {
        k: v
        for k, v in config["validation"].items()
        if k not in ("enabled", "metrics_path", "workers", "run_folders")
    }
    content = {
        "validation": policy,
        "seed": config["seed"],
        "reserved_tail_rows": config["data"]["horizon"],
        "schema": 3,
        "admission_metric": "mae",
    }
    return hashlib.sha256(json.dumps(content, sort_keys=True).encode()).hexdigest()


def transform(config):
    if config["model"].get("objective") == "next_patch":
        return transform_next_patch(config)
    d = config["data"]
    directory = location(config, "processed")
    directory.mkdir(parents=True, exist_ok=True)
    paths = discover_source_tables(config)
    paths, selection = validated_sources(config, paths)
    records, manifest = [], []
    selected = 0
    for path in paths:
        if d["max_tables"] is not None and selected >= d["max_tables"]:
            break
        with timed(f"read/validate {path}"):
            frame = pd.read_parquet(path)
            logger.info("Raw shape={} columns={}", frame.shape, frame.columns.tolist())
            if len(frame) < d["context_length"] + 2 * d["horizon"]:
                logger.warning("Skipping short table={} rows={}", path, len(frame))
                continue
            if d["time_column"] in frame:
                time = frame[d["time_column"]]
                if (
                    time.isna().any()
                    or not time.is_monotonic_increasing
                    or time.duplicated().any()
                ):
                    raise ValueError(f"Non-unique/non-increasing time in {path}")
            else:
                logger.warning("No time column: using existing row order for {}", path)
            excluded = [
                c
                for c in frame
                if any(
                    re.search(pattern, c) for pattern in d["excluded_column_patterns"]
                )
            ]
            past = d["past_covariates"]
            if past == "auto":
                past = [
                    c
                    for c in frame
                    if c
                    not in d["targets"]
                    + d["known_covariates"]
                    + [d["time_column"]]
                    + excluded
                ]
            columns = d["targets"] + list(past) + d["known_covariates"]
            if set(columns) & set(excluded):
                raise ValueError(
                    f"Explicit input roles include excluded lag columns: {set(columns) & set(excluded)}"
                )
            logger.info("Excluded lag-derived columns={}", excluded)
            values = frame[columns].to_numpy(dtype=np.float32)
            if np.isinf(values).any():
                raise ValueError(
                    f"Infinity in {path}; use NaN to mark missing observations"
                )
            logger.info(
                "Roles targets={} past={} known_future={}; missing={}",
                d["targets"],
                past,
                d["known_covariates"],
                int(np.isnan(values).sum()),
            )
        split = len(frame) - d["horizon"]
        train_cutoffs = list(
            range(d["context_length"], split - d["horizon"] + 1, d["stride"])
        )
        train_cutoffs = train_cutoffs[-d["train_windows_per_table"] :]
        source_path = (
            path.relative_to(ROOT).as_posix()
            if path.is_relative_to(ROOT)
            else str(path)
        )
        table_metadata = {
            "table_index": selected,
            "source": source_path,
            "source_sha256": file_digest(path),
            "source_rows": len(frame),
            "evaluation_cutoff": split,
            "excluded_columns": excluded,
            "variable_names": columns,
            "roles": {
                "targets": d["targets"],
                "past": list(past),
                "known": d["known_covariates"],
            },
        }
        for role, cutoff in [("train", c) for c in train_cutoffs] + [
            ("evaluation", split)
        ]:
            with timed(f"window table={selected} split={role} cutoff={cutoff}"):
                batch = frame_window(
                    frame,
                    cutoff=cutoff,
                    context_length=d["context_length"],
                    horizon=d["horizon"],
                    targets=d["targets"],
                    past_covariates=past,
                    known_covariates=d["known_covariates"],
                )
                if not torch.isfinite(batch["future_target"]).any():
                    raise ValueError(f"No observed target labels in {path} at {cutoff}")
                key = f"table_{selected:03d}_{role}_{cutoff}"
                summary = {
                    k: describe(k, v, config["logging"]["preview_values"])
                    for k, v in batch.items()
                    if isinstance(v, torch.Tensor)
                }
                records.append(
                    {
                        **table_metadata,
                        "id": key,
                        "split": role,
                        "batch": batch,
                        "source": (
                            str(path.relative_to(ROOT))
                            if path.is_relative_to(ROOT)
                            else str(path)
                        ),
                        "cutoff": cutoff,
                        "context_rows": [cutoff - d["context_length"], cutoff],
                        "label_rows": [cutoff, cutoff + d["horizon"]],
                        "targets": d["targets"],
                    }
                )
                manifest.append(
                    {
                        **table_metadata,
                        "id": key,
                        "source": (
                            str(path.relative_to(ROOT))
                            if path.is_relative_to(ROOT)
                            else str(path)
                        ),
                        "cutoff": cutoff,
                        "context_rows": [cutoff - d["context_length"], cutoff],
                        "label_rows": [cutoff, cutoff + d["horizon"]],
                        "split": role,
                        "roles": {
                            "targets": d["targets"],
                            "past": past,
                            "known": d["known_covariates"],
                        },
                        "tensors": summary,
                    }
                )
                if d["export_processed_tables"]:
                    export_columns = (
                        [d["time_column"]] if d["time_column"] in frame else []
                    ) + columns
                    frame.loc[:, export_columns].iloc[
                        cutoff - d["context_length"] : cutoff + d["horizon"]
                    ].to_parquet(
                        directory / f"{key}.parquet", index=False, compression="zstd"
                    )
                    logger.info(
                        "Saved inspection Parquet={}", directory / f"{key}.parquet"
                    )
        selected += 1
    if not records:
        raise ValueError(
            "No eligible raw tables. Check paths.raw and context/horizon lengths"
        )
    payload = {
        "signature": signature(config),
        "records": records,
        "source_selection": source_selection(config),
    }
    save_artifact(directory / "windows.pt", payload)
    write_json(
        directory / "manifest.json",
        {"config": config, "windows": manifest, "validation": selection},
    )
    logger.success("Transform complete: {} tables, {} windows", selected, len(records))
    return payload


def transform_next_patch(config):
    """Reserve the final full patch; train every earlier next-patch transition."""
    d, p = config["data"], config["model"]["patch_size"]
    if d["horizon"] != p or d["context_length"] != config["model"]["context_length"]:
        raise ValueError(
            "Next-patch evaluation requires horizon=patch_size and matching context limits"
        )
    if d["known_covariates"]:
        raise ValueError(
            "Next-patch objective currently uses observed, time-aligned covariates only"
        )
    max_tokens = d["context_length"] // p
    paths = discover_source_tables(config)
    paths, selection = validated_sources(config, paths)
    if d["max_tables"] is not None:
        paths = paths[: d["max_tables"]]
    if not paths:
        raise ValueError("No source tables")
    records, manifest, excluded_tables = [], [], selection["excluded"]
    for index, path in enumerate(paths):
        with timed(f"load/split full series {index + 1}/{len(paths)} {path}"):
            frame = pd.read_parquet(path)
            if len(frame) < max(48, 3 * p):
                reason = (
                    f"need at least {max(48, 3 * p)} timesteps (three complete tokens)"
                )
                excluded_tables.append({"source": str(path), "reason": reason})
                logger.warning(
                    "Skipping short table {} rows={}: {}", path, len(frame), reason
                )
                continue
            time = frame[d["time_column"]]
            if (
                time.isna().any()
                or not time.is_monotonic_increasing
                or time.duplicated().any()
            ):
                raise ValueError(f"Invalid time ordering: {path}")
            excluded = [
                c
                for c in frame
                if any(
                    re.search(pattern, c) for pattern in d["excluded_column_patterns"]
                )
            ]
            past = d["past_covariates"]
            if past == "auto":
                past = [
                    c
                    for c in frame
                    if c not in d["targets"] + [d["time_column"]] + excluded
                ]
            columns = d["targets"] + list(past)
            if set(columns) & set(excluded):
                raise ValueError("Explicit roles contain excluded lag columns")
            if np.isinf(frame[columns].to_numpy(dtype=np.float32)).any():
                raise ValueError(f"Infinity in {path}")
            cutoff = len(frame) - p
            left_pad = (-len(frame)) % p
            train_tokens = (cutoff + left_pad) // p
            source = (
                path.relative_to(ROOT).as_posix()
                if path.is_relative_to(ROOT)
                else str(path)
            )
            metadata = {
                "table_index": index,
                "screening_smoke_test_only": selection.get("smoke_test_only", False),
                "source": source,
                "source_key": path.relative_to(location(config, "raw")).as_posix(),
                "source_sha256": file_digest(path),
                "source_rows": len(frame),
                "evaluation_cutoff": cutoff,
                "targets": d["targets"],
                "excluded_columns": excluded,
                "variable_names": columns,
                "roles": {"targets": d["targets"], "past": list(past), "known": []},
                "left_padding_points": left_pad,
                "training_transitions": train_tokens - 1,
            }
            logger.info(
                "Table={} rows={} variables={} train_transitions={} held_out=[{}, {}) excluded={}",
                index,
                len(frame),
                len(columns),
                train_tokens - 1,
                cutoff,
                len(frame),
                excluded,
            )
            # A chunk has at most C input points plus one target patch. Adjacent chunks
            # share one patch as context, never duplicate supervised transitions.
            for token_start in range(0, train_tokens - 1, max_tokens):
                start = max(0, token_start * p - left_pad)
                stop = min(cutoff, (token_start + max_tokens + 1) * p - left_pad)
                batch = {
                    "context": torch.tensor(
                        frame[d["targets"]]
                        .iloc[start:stop]
                        .to_numpy(dtype=np.float32)
                        .T.copy()
                    )[None],
                    "horizon": p,
                }
                if past:
                    batch["past_covariates"] = torch.tensor(
                        frame[list(past)]
                        .iloc[start:stop]
                        .to_numpy(dtype=np.float32)
                        .T.copy()
                    )[None]
                record = {
                    **metadata,
                    "id": f"table_{index:03d}_train_tokens_{token_start}",
                    "split": "train",
                    "batch": batch,
                    "context_rows": [start, stop],
                    "cutoff": stop,
                    "token_start": token_start,
                }
                records.append(record)
            batch = frame_window(
                frame,
                cutoff=cutoff,
                context_length=min(cutoff, d["context_length"]),
                horizon=p,
                targets=d["targets"],
                past_covariates=past,
            )
            records.append(
                {
                    **metadata,
                    "id": f"table_{index:03d}_evaluation_{cutoff}",
                    "split": "evaluation",
                    "batch": batch,
                    "cutoff": cutoff,
                    "context_rows": [max(0, cutoff - d["context_length"]), cutoff],
                    "label_rows": [cutoff, len(frame)],
                }
            )
            manifest.append(metadata)
            if d["export_processed_tables"]:
                directory = location(config, "processed")
                directory.mkdir(parents=True, exist_ok=True)
                frame[[d["time_column"]] + columns].to_parquet(
                    directory / f"table_{index:03d}.parquet",
                    index=False,
                    compression="zstd",
                )
    if not manifest:
        raise ValueError(
            "No admitted datasets have at least three full tokens (48 timesteps minimum)"
        )
    payload = {
        "signature": signature(config),
        "records": records,
        "objective": "next_patch",
        "source_selection": source_selection(config),
    }
    save_artifact(location(config, "processed") / "windows.pt", payload)
    write_json(
        location(config, "processed") / "manifest.json",
        {
            "config": config,
            "source_selection": source_selection(config),
            "tables": manifest,
            "table_count": len(manifest),
            "validation": selection,
            "excluded_tables": excluded_tables,
            "training_transitions": sum(t["training_transitions"] for t in manifest),
            "split_rule": "right-aligned patches; final full patch held out; all earlier next-patch pairs trained",
        },
    )
    return payload


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default=str(ROOT / "configs/config.json"))
    args = parser.parse_args()
    config = load_config(args.config)
    with run_context(config, "transform"):
        with timed("TRANSFORM stage"):
            transform(config)


if __name__ == "__main__":
    main()

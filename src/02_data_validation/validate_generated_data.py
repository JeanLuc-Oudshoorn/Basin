"""Stage 02: screen pre-test history with Prophet versus last observation carried forward."""

import argparse
import logging
import shutil
import sys
from datetime import datetime, timezone
from pathlib import Path
from uuid import uuid4

import pandas as pd
from loguru import logger

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT / "src/03_transforming_input_data"))
from prophet_validator import ProphetValidator, multivariate_admitted


def validate(config, *, retry_errors=False):
    # Keep training/Torch imports out of spawned Prophet workers.
    from transform import (
        discover_source_tables,
        generated_dates,
        location,
        source_selection,
        validation_report_path,
        validation_run_root,
        write_json,
    )

    policy = config["validation"]
    if not policy["enabled"]:
        raise ValueError("Enable validation in config.json to run screening")
    logging.basicConfig(level=logging.WARNING)
    logging.getLogger("cmdstanpy").setLevel(logging.WARNING)
    validator = ProphetValidator(
        location(config, "raw"),
        PROJECT_ROOT / Path(policy["metrics_path"]).parent,
        train_fraction=policy["train_fraction"],
        covariate_mode=policy["covariate_mode"],
        step_duration=policy["step_duration"],
        seed=config["seed"],
        prophet_options=policy["prophet_options"],
        seasonal_periods=policy["seasonal_periods"],
        reserved_tail_rows=config["data"]["horizon"],
        workers=policy.get("workers"),
    )
    paths = discover_source_tables(config)
    if not paths:
        raise ValueError("No source data.parquet files found")
    previous_report = validation_report_path(config) if retry_errors else None
    report = PROJECT_ROOT / policy["metrics_path"]
    started = datetime.now(timezone.utc)
    if policy.get("run_folders", True):
        run_id = started.strftime("%Y%m%dT%H%M%S_%fZ_") + uuid4().hex[:8]
        directory = validation_run_root(config) / run_id
        directory.mkdir(parents=True, exist_ok=False)
        report = directory / report.name
    report.parent.mkdir(parents=True, exist_ok=True)
    metadata = {
        "started_at_utc": started.isoformat(),
        "run_date_utc": started.date().isoformat(),
        "dataset_dates": generated_dates(config),
        "source_selection": source_selection(config),
        "admission_metric": "mae",
        "admission_rule": "multivariate_mae <= baseline_mae",
        "report": str(report.resolve()),
        "retry_source": str(previous_report.resolve()) if previous_report else None,
    }
    write_json(report.with_name("settings.json"), {**metadata, "config": config})
    write_json(
        report.with_name("status.json"),
        {**metadata, "status": "running", "tables": len(paths), "processed": 0},
    )
    logger.info("Validation run folder: {}", report.parent)
    try:
        return _evaluate_run(
            config, validator, paths, report, previous_report, metadata
        )
    except BaseException as exc:
        write_json(
            report.with_name("status.json"),
            {**metadata, "status": "failed", "error": f"{type(exc).__name__}: {exc}"},
        )
        raise


def _evaluate_run(config, validator, paths, report, previous_report, metadata):
    from transform import (
        file_digest,
        location,
        screening_signature,
        source_selection,
        validation_run_root,
        write_json,
    )

    policy = config["validation"]
    signature = screening_signature(config)
    records = {}
    if previous_report is not None:
        if not previous_report.is_file():
            raise FileNotFoundError(
                f"Cannot retry without an existing report: {previous_report}"
            )
        previous = pd.read_parquet(previous_report)
        if previous.source_key.duplicated().any():
            raise ValueError("Duplicate source keys in the existing validation report")
        indexed = previous.set_index("source_key", drop=False)
        for path in paths:
            key = path.relative_to(location(config, "raw")).as_posix()
            if key not in indexed.index:
                continue
            row = indexed.loc[key]
            if (
                row.status == "ok"
                and not bool(row.get("smoke_test_only", False))
                and row.screening_signature == signature
                and row.source_sha256 == file_digest(path)
                and pd.notna(row.get("multivariate_mae"))
            ):
                records[path] = row.to_dict()
    if report.is_file():
        archive = (
            report.parent / "archive" / datetime.now().strftime("%Y%m%d_%H%M%S_%f")
        )
        archive.mkdir(parents=True, exist_ok=False)
        for previous_file in (
            report,
            report.with_name("summary.json"),
            report.with_name("metrics_progress.csv"),
        ):
            if previous_file.is_file():
                shutil.copy2(previous_file, archive / previous_file.name)
        logger.info("Archived previous validation report: {}", archive)
    reused = len(records)
    pending = [path for path in paths if path not in records]
    progress = report.with_name("metrics_progress.csv")
    if records:
        pd.DataFrame(list(records.values())).to_csv(progress, index=False)
    logger.info(
        "Screening {} tables with {} workers; reusing {} successful results",
        len(pending),
        validator.worker_count(len(pending)),
        reused,
    )
    for index, (path, record) in enumerate(
        validator.evaluate_tables(pending), reused + 1
    ):
        key = path.relative_to(location(config, "raw")).as_posix()
        record.update(source_key=key, screening_signature=signature)
        records[path] = record
        logger.info("VALIDATION {}/{} {}", index, len(paths), record)
        # Durable per-table progress without rewriting a growing Parquet report.
        pd.DataFrame([record]).to_csv(
            progress,
            mode="w" if index == 1 else "a",
            header=index == 1,
            index=False,
        )
        if index % 100 == 0:
            write_json(
                report.with_name("status.json"),
                {
                    **metadata,
                    "status": "running",
                    "tables": len(paths),
                    "processed": index,
                    "reused": reused,
                    "reevaluated": index - reused,
                },
            )
    results = pd.DataFrame([records[path] for path in paths])
    results.attrs["report_path"] = str(report.resolve())
    temporary = report.with_suffix(".tmp.parquet")
    results.to_parquet(temporary, index=False)
    temporary.replace(report)
    wins = multivariate_admitted(results)
    summary = {
        **metadata,
        "tables": len(results),
        "successful": int(results.status.eq("ok").sum()),
        "failed": int(results.status.ne("ok").sum()),
        "reused": reused,
        "reevaluated": len(pending),
        "status": "completed_with_errors"
        if results.status.ne("ok").any()
        else "complete",
        "source_selection": source_selection(config),
        "admitted": int(wins.sum()),
        "rejected": int((~wins).sum()),
        "reserved_tail_rows": config["data"]["horizon"],
        "screening_signature": signature,
        "report_sha256": file_digest(report),
        "policy": policy,
        "effective_workers": validator.worker_count(len(paths)),
        "temp_dir": str(validator.temp_dir.resolve()),
    }
    summary["completed_at_utc"] = datetime.now(timezone.utc).isoformat()
    write_json(report.with_name("summary.json"), summary)
    write_json(report.with_name("status.json"), {**summary, "processed": len(results)})
    if policy.get("run_folders", True):
        root = validation_run_root(config)
        pointer = root / ("latest_" + uuid4().hex + ".tmp.json")
        write_json(
            pointer,
            {
                "report": report.relative_to(root).as_posix(),
                "screening_signature": signature,
                "source_selection": source_selection(config),
                "admission_metric": "mae",
            },
        )
        pointer.replace(root / "latest.json")
    if summary["failed"]:
        logger.warning(
            "Validation completed with {} failed tables: {}", summary["failed"], summary
        )
    else:
        logger.success("Validation complete: {}", summary)
    return results


def main():
    from transform import load_config, run_context

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default=str(PROJECT_ROOT / "configs/config.json"))
    parser.add_argument(
        "--retry-errors",
        action="store_true",
        help="Reuse unchanged successful scores and reevaluate failed, changed or missing tables",
    )
    args = parser.parse_args()
    config = load_config(args.config)
    with run_context(config, "validate"):
        validate(config, retry_errors=args.retry_errors)


if __name__ == "__main__":
    main()

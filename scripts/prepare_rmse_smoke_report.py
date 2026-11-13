"""Explicit one-off RMSE-to-MAE aliases for the authorized large-scale smoke test."""

import argparse
import hashlib
import json
import sys
from datetime import datetime, timezone
from pathlib import Path
from uuid import uuid4

import pandas as pd


def main():
    root = Path(__file__).resolve().parents[1]
    sys.path.insert(0, str(root / "src/03_transforming_input_data"))
    sys.path.insert(0, str(root / "src/02_data_validation"))
    from prophet_validator import multivariate_admitted
    from transform import (
        file_digest,
        generated_dates,
        load_config,
        screening_signature,
        source_selection,
        validation_run_root,
        write_json,
    )

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default=str(root / "configs/config.json"))
    parser.add_argument(
        "--source",
        default=str(root / "results/02_data_validation/selected_dates/metrics.parquet"),
    )
    args = parser.parse_args()
    config = load_config(args.config)
    source = Path(args.source).resolve()
    original = json.loads(source.with_name("summary.json").read_text(encoding="utf-8"))
    original_digest = file_digest(source)
    if original_digest != original["report_sha256"]:
        raise ValueError("Original report checksum differs from its summary")
    if original["source_selection"] != source_selection(config):
        raise ValueError(
            "Original report and current configuration select different datasets"
        )
    legacy_policy = {
        k: v
        for k, v in config["validation"].items()
        if k not in ("enabled", "metrics_path", "workers", "run_folders")
    }
    legacy_signature = hashlib.sha256(
        json.dumps(
            {
                "validation": legacy_policy,
                "seed": config["seed"],
                "reserved_tail_rows": config["data"]["horizon"],
                "schema": 2,
            },
            sort_keys=True,
        ).encode()
    ).hexdigest()
    if legacy_signature != original["screening_signature"]:
        raise ValueError(
            "Original RMSE screening settings differ from current settings"
        )
    scores = pd.read_parquet(source)
    if len(scores) != original["tables"] or scores.source_key.duplicated().any():
        raise ValueError("Invalid original report row count or duplicate sources")
    if not scores.screening_signature.eq(legacy_signature).all():
        raise ValueError("Original report contains inconsistent screening signatures")
    scores["baseline_mae"] = scores["baseline_rmse"]
    scores["multivariate_mae"] = scores["multivariate_rmse"]
    scores["original_screening_signature"] = scores["screening_signature"]
    scores["screening_signature"] = screening_signature(config)
    scores["admission_metric"] = "mae"
    scores["actual_scoring_metric"] = "rmse"
    scores["smoke_test_only"] = True
    scores["metric_alias"] = "rmse_as_mae"
    admitted = int(multivariate_admitted(scores).sum())
    if admitted != original["admitted"]:
        raise ValueError("Smoke conversion unexpectedly changed the admission count")
    started = datetime.now(timezone.utc)
    cohort = validation_run_root(config)
    directory = cohort / (
        started.strftime("%Y%m%dT%H%M%S_%fZ_smoke_rmse_alias_") + uuid4().hex[:8]
    )
    directory.mkdir(parents=True, exist_ok=False)
    report = directory / Path(config["validation"]["metrics_path"]).name
    metadata = {
        "started_at_utc": started.isoformat(),
        "run_date_utc": started.date().isoformat(),
        "dataset_dates": generated_dates(config),
        "source_selection": source_selection(config),
        "admission_metric": "mae",
        "actual_scoring_metric": "rmse",
        "smoke_test_only": True,
        "metric_alias": "rmse_as_mae",
        "note": "User-authorized smoke test: MAE columns are copies of RMSE, not measured MAE. No models were refitted. Future validators must not reuse these aliases as MAE measurements.",
        "source_report": str(source),
        "source_report_sha256": original_digest,
        "original_screening_signature": legacy_signature,
        "screening_signature": screening_signature(config),
        "report": str(report),
    }
    scores.attrs = metadata
    scores.to_parquet(report, index=False)
    scores.to_csv(directory / "metrics_progress.csv", index=False)
    write_json(directory / "original_rmse_summary.json", original)
    write_json(directory / "settings.json", {**metadata, "config": config})
    summary = {
        **metadata,
        "tables": len(scores),
        "admitted": admitted,
        "rejected": len(scores) - admitted,
        "successful": int(scores.status.eq("ok").sum()),
        "failed": int(scores.status.ne("ok").sum()),
        "status": "complete",
        "reused": len(scores),
        "reevaluated": 0,
        "reserved_tail_rows": config["data"]["horizon"],
        "policy": config["validation"],
        "admission_rule": "legacy RMSE <= baseline RMSE, exposed via MAE aliases for smoke testing",
        "completed_at_utc": datetime.now(timezone.utc).isoformat(),
        "report_sha256": file_digest(report),
    }
    write_json(directory / "summary.json", summary)
    write_json(directory / "status.json", {**summary, "processed": len(scores)})
    if file_digest(source) != original_digest:
        raise ValueError(
            "Original report changed during conversion; latest pointer not updated"
        )
    pointer = cohort / ("latest_" + uuid4().hex + ".tmp.json")
    write_json(pointer, {**metadata, "report": report.relative_to(cohort).as_posix()})
    pointer.replace(cohort / "latest.json")
    print(
        f"Prepared {report}\n{admitted:,} of {len(scores):,} tables admitted; original report preserved."
    )


if __name__ == "__main__":
    main()

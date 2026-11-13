"""Remove generated holdout artifacts before publishing a new run's identity."""

from pathlib import Path

from loguru import logger

FILES = (
    "test_metrics.json",
    "test_metrics_progress.json",
    "table_metrics.json",
    "origin_metrics.json",
    "test_predictions.parquet",
    "request_latencies.parquet",
    "request_latencies.jsonl",
    "model_used.json",
    "sample_selection.json",
    "plot_selection.json",
)
PATTERNS = (
    "test_table_*.parquet",
    "test_table_*.html",
    "test_table_*.png",
    "test_table_*.svg",
    "error_distribution*.html",
    "error_distribution*.png",
    "error_distribution*.svg",
    "latency_distribution*.html",
    "latency_distribution*.png",
    "latency_distribution*.svg",
)


def clear_holdout_outputs(directory):
    """Delete only known generated files at the output root, never directories."""
    directory = Path(directory).resolve()
    directory.mkdir(parents=True, exist_ok=True)
    candidates = {directory / name for name in FILES}
    for pattern in PATTERNS:
        candidates.update(directory.glob(pattern))
    removed = 0
    for path in sorted(candidates):
        if path.is_file() or path.is_symlink():
            path.unlink()
            removed += 1
    logger.info("Cleared {} stale generated holdout files from {}", removed, directory)
    return removed

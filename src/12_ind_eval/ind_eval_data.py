"""IND-Eval: download the evaluation windows and read them as forecasting series.

IND-Eval (cognitedata/IND-Eval on Hugging Face, arXiv 2610.16652) is a
forecasting benchmark on industrial time series. Every source has a folder of
parquet files under `data/<source>/`; one file is one task (a source at one
sampling rate or product code and one horizon). A row is one channel of one
evaluation window: `context` is the history a model is given, `target` the
values that follow it directly and are to be forecast.
"""

import argparse
import json
import sys
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pyarrow.parquet as pq

ROOT = Path(__file__).resolve().parents[2]
SECTION = "ind_eval"
REPOSITORY = "cognitedata/IND-Eval"
DEFAULTS = {
    "path": "benchmarks/IND-Eval",
    # A tagged release, so that later changes to the dataset do not change
    # the results without anyone noticing.
    "revision": "v0.1.3",
    "models": [
        "basin",
        "naive",
        "holtwinters",
        "chronos2",
        "timestral",
        "timeh2o",
    ],
    # None: every source that has been downloaded.
    "sources": None,
    # Contexts per model call.
    "batch_size": 16,
    # Every model gets the latest points of the history up to this many,
    # and forecasts the first points of the target up to `horizon`. Both are
    # what Basin handles in one call, and the same for every model; the
    # benchmark itself allows 4,096 points of history and horizons to 1,024.
    "context_points": 2048,
    "horizon": 112,
    # None: every series of a task. A number: an even subset of that size,
    # the same for every model (for a quicker, less precise run).
    "max_series_per_task": None,
    # The cycle length of the seasonal-naive reference where a task does not
    # state one (electricity and solar), by sampling rate.
    "seasonality_by_freq": {"10T": 144, "15T": 96, "H": 24, "D": 7, "W": 1},
    # Processes that fit Holt-Winters at the same time.
    "holt_winters_workers": 10,
}
RECORD = "download.json"
# Columns read from a task file, where it has them.
COLUMNS = (
    "instance_id",
    "channel",
    "window",
    "role",
    "context",
    "target",
    "freq",
    "prediction_length",
    "seasonality",
)


def settings(config):
    """The `ind_eval` section of the configuration over the defaults."""
    given = config.get(SECTION, {})
    unknown = sorted(set(given) - set(DEFAULTS))
    if unknown:
        raise ValueError(f"Unknown {SECTION} settings: {unknown}")
    options = {**DEFAULTS, **given}
    options["seasonality_by_freq"] = {
        **DEFAULTS["seasonality_by_freq"],
        **given.get("seasonality_by_freq", {}),
    }
    for name in ("context_points", "horizon"):
        value = options[name]
        if isinstance(value, bool) or not isinstance(value, int) or value < 1:
            raise ValueError(f"{SECTION}.{name} must be a positive whole number")
    return options


def data_directory(options):
    return ROOT / options["path"]


def download(options, log=print):
    """Fetch the evaluation windows (about 11 GB) from Hugging Face."""
    from huggingface_hub import snapshot_download

    directory = data_directory(options)
    patterns = ["README.md", "LICENSE"]
    sources = options["sources"]
    patterns += ["data/**"] if sources is None else [f"data/{s}/**" for s in sources]
    log(f"Downloading {REPOSITORY} at {options['revision']} to {directory}")
    snapshot_download(
        REPOSITORY,
        repo_type="dataset",
        revision=options["revision"],
        allow_patterns=patterns,
        local_dir=directory,
    )
    tasks = list_tasks(options)
    record = {
        "repository": REPOSITORY,
        "revision": options["revision"],
        "tasks": len(tasks),
        "sources": sorted({task.source for task in tasks}),
    }
    (directory / RECORD).write_text(json.dumps(record, indent=2), encoding="utf-8")
    log(f"{len(tasks)} tasks from {len(record['sources'])} sources in {directory}")
    return record


@dataclass(frozen=True)
class Task:
    """One parquet file: a source at one sampling rate or code and one horizon."""

    source: str
    name: str
    path: Path

    @property
    def key(self):
        return f"{self.source}__{self.name}"


def list_tasks(options):
    """Every downloaded task of the chosen sources, in a fixed order."""
    directory = data_directory(options) / "data"
    if not directory.is_dir():
        raise FileNotFoundError(
            f"No IND-Eval data in {directory}. Download it with: "
            "python src/12_ind_eval/ind_eval_data.py"
        )
    wanted = options["sources"]
    found = sorted(p.name for p in directory.iterdir() if p.is_dir())
    missing = sorted(set(wanted or []) - set(found))
    if missing:
        raise FileNotFoundError(f"IND-Eval sources not downloaded: {missing}")
    return [
        Task(source, path.stem, path)
        for source in (found if wanted is None else sorted(wanted))
        for path in sorted((directory / source).rglob("*.parquet"))
    ]


@dataclass
class Series:
    """One channel of one evaluation window."""

    key: str
    # The latest points of the history, up to `context_points`; may hold
    # missing values.
    context: np.ndarray
    target: np.ndarray
    seasonality: int

    @property
    def horizon(self):
        return len(self.target)


def model_context(context, points):
    """The history a model is given: the latest `points`, missing values filled.

    Missing values are filled by linear interpolation, and at the ends with
    the nearest observed value, the same way for every model. Returns None
    for a history without any observed value.
    """
    context = np.asarray(context[-points:], dtype=np.float32)
    observed = np.isfinite(context)
    if observed.all():
        return context
    if not observed.any():
        return None
    position = np.arange(len(context))
    return np.interp(position, position[observed], context[observed]).astype(np.float32)


def _lists(column):
    """A list<float> column as one float32 array per row."""
    column = column.combine_chunks() if hasattr(column, "combine_chunks") else column
    values = column.values.to_numpy(zero_copy_only=False)
    offsets = column.offsets.to_numpy()
    return [
        values[offsets[i] : offsets[i + 1]].astype(np.float32)
        for i in range(len(column))
    ]


def chosen_rows(rows, limit):
    """The rows of an even subset of `limit` out of `rows` (all without a limit)."""
    if limit is None or limit >= rows:
        return None
    return set(np.linspace(0, rows - 1, int(limit)).round().astype(int).tolist())


def read_task(task, options, rows_per_read=512):
    """Yield the task's series in lists of up to `rows_per_read` file rows.

    Rows that hold a covariate (PRONTO's known control inputs) are not
    forecasting targets and are left out. A context is cut to its latest
    `context_points` points and a target to its first `horizon` points.
    """
    file = pq.ParquetFile(task.path)
    names = [name for name in COLUMNS if name in file.schema_arrow.names]
    for needed in ("instance_id", "channel", "context", "target"):
        if needed not in names:
            raise ValueError(f"{task.path} has no column {needed!r}")
    keep = chosen_rows(file.metadata.num_rows, options["max_series_per_task"])
    cap, horizon = options["context_points"], options["horizon"]
    first = 0
    for batch in file.iter_batches(batch_size=rows_per_read, columns=names):
        table = {name: batch.column(name) for name in names}
        plain = {
            name: table[name].to_pylist()
            for name in names
            if name not in ("context", "target")
        }
        contexts, targets = _lists(table["context"]), _lists(table["target"])
        series = []
        for i in range(batch.num_rows):
            if keep is not None and first + i not in keep:
                continue
            if plain.get("role", ["target"] * batch.num_rows)[i] == "covariate":
                continue
            target = targets[i]
            stated = plain.get("prediction_length", [len(target)] * batch.num_rows)[i]
            if len(target) != stated or not len(target):
                raise ValueError(
                    f"{task.path}: a target of {len(target)} points for a "
                    f"prediction length of {stated}"
                )
            seasonality = plain.get("seasonality", [None] * batch.num_rows)[i]
            if seasonality is None:
                freq = plain.get("freq", [None] * batch.num_rows)[i]
                seasonality = options["seasonality_by_freq"].get(freq, 1)
            window = plain.get("window", [0] * batch.num_rows)[i]
            series.append(
                Series(
                    key=f"{plain['instance_id'][i]}|{window}|{plain['channel'][i]}",
                    context=contexts[i][-cap:],
                    target=target[:horizon],
                    seasonality=max(1, int(seasonality)),
                )
            )
        first += batch.num_rows
        if series:
            yield series


def main():
    sys.path.insert(0, str(ROOT / "src" / "03_transforming_input_data"))
    from transform import load_config

    parser = argparse.ArgumentParser(description=download.__doc__)
    parser.add_argument("--config", default=str(ROOT / "configs/config.json"))
    args = parser.parse_args()
    download(settings(load_config(args.config)))


if __name__ == "__main__":
    main()

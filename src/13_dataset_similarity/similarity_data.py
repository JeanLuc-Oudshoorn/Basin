"""Collect the series that the similarity analysis compares.

One row is one target series as a model sees it: the latest points before a
forecast, at most `max_points`. A table with several columns contributes its
target column only, because that is what a model is asked to forecast; this
makes tables with and without covariates comparable.

Four groups are collected:

- `synthetic`: the target of the tables Basin's training reserves for
  validation (or of any folder of generated tables, to test a new generator),
- `fakebench`: the random walks of FakeBench-TS,
- `operational`: the forecast cases of the operational benchmarks (stage 11),
- `ind_eval`: a sample of the IND-Eval series (stage 12), the same number
  from every source.

The values are stored as they are; standardising is left to the analysis.
"""

import hashlib
import heapq
import json
import sys
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow.parquet as pq

ROOT = Path(__file__).resolve().parents[2]
SECTION = "dataset_similarity"
GROUPS = ("synthetic", "fakebench", "operational", "ind_eval")
GROUP_NAMES = {
    "synthetic": "Synthetic (Basin validation)",
    "fakebench": "FakeBench-TS",
    "operational": "Operational benchmarks",
    "ind_eval": "IND-Eval",
}
DEFAULTS = {
    "output": "results/model/dataset_similarity",
    # The longest history kept per series: what Basin reads in one call.
    "max_points": 2048,
    # Every series is described on its latest N points, once per N; shorter
    # series are left out of that view. A fixed length keeps the descriptors
    # comparable: a short series cannot show a slow cycle.
    "window_points": [512, 2048],
    "seed": 7,
    "workers": 4,
    "synthetic": {
        # "validation": the tables of the latest validation check of the
        # training run. Or a folder of generated tables (`*/data.parquet`).
        "source": "validation",
        "target": "outcome",
        "tables": None,
    },
    "fakebench": {"path": "benchmarks/fakebench-ts/data/fakebench_v3.parquet"},
    "operational": {
        "cases": "data/04_benchmarks/operational",
        "results": "results/model/operational_benchmarks",
    },
    # Series per IND-Eval source, drawn at random over all its tasks.
    "ind_eval": {"series_per_source": 3000},
    "neighbours": 5,
    "covered_quantile": 0.95,
    "map": {"synthetic": 7000, "per_group": 5000, "perplexity": 40},
    # Chronos-2 embeddings of a fixed random part of every group (a CPU does
    # about ten series a second). `model_path`: a folder with the model, for
    # a machine that cannot read the project's model cache.
    "embeddings": {
        "enabled": True,
        "synthetic": 8000,
        "per_group": 5000,
        "batch_size": 32,
        "threads": 4,
        "model_path": None,
    },
}
WINDOWS, INDEX = "windows.npy", "windows.parquet"


def settings(config):
    """The `dataset_similarity` section of the configuration over the defaults."""
    given = config.get(SECTION, {})
    unknown = sorted(set(given) - set(DEFAULTS))
    if unknown:
        raise ValueError(f"Unknown {SECTION} settings: {unknown}")
    options = {}
    for name, default in DEFAULTS.items():
        value = given.get(name, default)
        if isinstance(default, dict):
            extra = sorted(set(value) - set(default))
            if extra:
                raise ValueError(f"Unknown {SECTION}.{name} settings: {extra}")
            value = {**default, **value}
        options[name] = value
    points = options["window_points"]
    if not points or any(
        isinstance(n, bool) or not isinstance(n, int) or n < 128 for n in points
    ):
        raise ValueError(f"{SECTION}.window_points must be whole numbers from 128")
    if max(points) > options["max_points"]:
        raise ValueError(f"{SECTION}.window_points cannot exceed max_points")
    return options


def output_directory(options):
    return ROOT / options["output"]


def row(group, dataset, series, values, *, view=0, case=-1, score=np.nan, rows=None):
    """One collected series; `view` limits it to one window length (0: any)."""
    values = np.asarray(values, dtype=np.float32)
    return {
        "group": group,
        "dataset": dataset,
        "series": str(series),
        "values": values,
        "source_points": int(len(values) if rows is None else rows),
        "view": int(view),
        "case": int(case),
        "basin_mase": float(score),
    }


def latest_validation_check(config):
    folder = ROOT / config["paths"]["training"]
    checks = sorted(folder.glob("validation_check_*.parquet"))
    if not checks:
        raise FileNotFoundError(f"No validation check with table scores in {folder}")
    return checks[-1]


def synthetic_tables(config, options):
    """(path, Basin's median validation MASE or NaN) of every synthetic table."""
    chosen = options["synthetic"]
    if chosen["source"] == "validation":
        check = pd.read_parquet(
            latest_validation_check(config), columns=["source", "mase"]
        )
        scores = check.groupby("source")["mase"].median()
        tables = [(ROOT / source, float(score)) for source, score in scores.items()]
    else:
        folder = Path(chosen["source"])
        folder = folder if folder.is_absolute() else ROOT / folder
        tables = [(path, np.nan) for path in sorted(folder.rglob("data.parquet"))]
        if not tables:
            raise FileNotFoundError(f"No generated tables (*/data.parquet) in {folder}")
    limit = chosen["tables"]
    if limit is not None and limit < len(tables):
        # A fixed subset that does not depend on the order of the files.
        tables = heapq.nsmallest(
            limit,
            tables,
            key=lambda t: hashlib.sha256(t[0].as_posix().encode()).digest(),
        )
    return sorted(tables)


def collect_synthetic(config, options, log=print):
    tables = synthetic_tables(config, options)
    target, cap = options["synthetic"]["target"], options["max_points"]

    def read(item):
        path, score = item
        values = pq.read_table(path, columns=[target]).column(0).to_numpy()
        return row(
            "synthetic",
            "synthetic",
            path.parent.name,
            values[-cap:],
            score=score,
            rows=len(values),
        )

    rows = []
    with ThreadPoolExecutor(options["workers"]) as pool:
        for done, item in enumerate(pool.map(read, tables), 1):
            rows.append(item)
            if done % 5000 == 0:
                log(f"synthetic: {done:,} of {len(tables):,} tables read")
    return rows


def collect_fakebench(options, shortest):
    path = ROOT / options["fakebench"]["path"]
    table = pq.read_table(path, columns=["series_id", "family", "context"])
    rows = []
    for series, family, context in zip(
        *(table.column(i).to_pylist() for i in range(3))
    ):
        if len(context) >= shortest:
            rows.append(
                row("fakebench", family, series, context[-options["max_points"] :])
            )
    return rows


def latest_operational_scores(options):
    """Per-case scores of the latest finished run of stage 11, or None."""
    folder = ROOT / options["operational"]["results"]
    runs = sorted(folder.glob("*/per_case.parquet")) if folder.is_dir() else []
    return pd.read_parquet(runs[-1]) if runs else None


def collect_operational(options):
    """Forecast cases; a case belongs to the view of its own context length."""
    path = ROOT / options["operational"]["cases"] / "cases.parquet"
    columns = ["dataset", "series", "task", "context_length", "values", "case"]
    cases = pq.read_table(path, columns=columns).to_pandas()
    cases = cases[
        (cases["task"] == "forecast")
        & cases["context_length"].isin(options["window_points"])
    ]
    return [
        row(
            "operational",
            case.dataset,
            case.series,
            case.values[: case.context_length],
            view=case.context_length,
            case=case.case,
        )
        for case in cases.itertuples()
    ]


def collect_ind_eval(config, options, log=print):
    """The same number of series from every source, drawn by a seeded hash."""
    sys.path.insert(0, str(ROOT / "src" / "12_ind_eval"))
    import ind_eval_data

    reader = ind_eval_data.settings(config)
    reader = {
        **reader,
        "context_points": options["max_points"],
        "max_series_per_task": None,
    }
    tasks = ind_eval_data.list_tasks(reader)
    count, seed = options["ind_eval"]["series_per_source"], options["seed"]
    chosen, seen = {}, {}
    for task in tasks:
        heap = chosen.setdefault(task.source, [])
        keys = seen.setdefault(task.source, set())
        for group in ind_eval_data.read_task(task, reader):
            for series in group:
                # A source's tasks for different horizons repeat histories.
                digest = hashlib.sha256(series.context.tobytes()).digest()
                if digest in keys:
                    continue
                keys.add(digest)
                rank = hashlib.sha256(f"{seed}:".encode() + digest).digest()
                item = (rank, f"{task.name}|{series.key}", series.context)
                if len(heap) < count:
                    heapq.heappush(heap, _Reversed(item))
                elif rank < heap[0].item[0]:
                    heapq.heapreplace(heap, _Reversed(item))
        log(
            f"ind_eval: {task.key} read, {len(keys):,} distinct histories in {task.source}"
        )
    rows = []
    for source, heap in chosen.items():
        for entry in sorted(heap, key=lambda e: e.item[0]):
            _, key, context = entry.item
            rows.append(row("ind_eval", source, key, context, rows=len(context)))
    return rows


class _Reversed:
    """Turns Python's smallest-first heap into a largest-first one."""

    def __init__(self, item):
        self.item = item

    def __lt__(self, other):
        return self.item[0] > other.item[0]


def save(rows, directory, max_points):
    """Right-aligned values (NaN before the series starts) and one row of facts each."""
    directory.mkdir(parents=True, exist_ok=True)
    values = np.full((len(rows), max_points), np.nan, dtype=np.float32)
    for position, item in enumerate(rows):
        series = item["values"][-max_points:]
        values[position, max_points - len(series) :] = series
    np.save(directory / WINDOWS, values)
    index = pd.DataFrame(
        [{k: v for k, v in item.items() if k != "values"} for item in rows]
    )
    index["points"] = [min(len(item["values"]), max_points) for item in rows]
    index.to_parquet(directory / INDEX, index=False)
    return index


def load(directory):
    index = pd.read_parquet(directory / INDEX)
    return index, np.load(directory / WINDOWS, mmap_mode="r")


def view_rows(index, points):
    """Positions of the series that belong to the view of `points` points."""
    own = index["view"].to_numpy()
    enough = index["points"].to_numpy() >= points
    return np.flatnonzero(enough & ((own == points) | (own == 0)))


def features_path(directory, points):
    return directory / f"features_{points}.parquet"


def embeddings_path(directory, points):
    return directory / f"embeddings_{points}.npz"


def ranked(names, seed):
    """A fixed random order of series by name: the same in every step."""
    return np.argsort(
        [hashlib.sha256(f"{seed}:{name}".encode()).hexdigest() for name in names],
        kind="stable",
    )


def collect(config, options, log=print):
    directory = output_directory(options)
    shortest = min(options["window_points"])
    rows = collect_fakebench(options, shortest)
    log(f"fakebench: {len(rows):,} series")
    operational = collect_operational(options)
    log(f"operational: {len(operational):,} forecast cases")
    rows += operational
    ind_eval = collect_ind_eval(config, options, log)
    log(f"ind_eval: {len(ind_eval):,} series")
    rows += ind_eval
    rows += collect_synthetic(config, options, log)
    index = save(rows, directory, options["max_points"])
    summary = {
        group: {
            "series": int((index["group"] == group).sum()),
            "points_quartiles": [
                int(v)
                for v in index.loc[index["group"] == group, "points"].quantile(
                    [0.25, 0.5, 0.75]
                )
            ],
        }
        for group in GROUPS
    }
    (directory / "collected.json").write_text(
        json.dumps(summary, indent=2), encoding="utf-8"
    )
    log(json.dumps(summary))
    return index

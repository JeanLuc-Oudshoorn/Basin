"""Cut CATS and the operational UCR series into the cases the models are run on.

A case is one request to a model: a context of 128, 512 or 2,048 points and
the 112 points that follow it. There are two tasks for each dataset:

* forecast: context and future lie in normal data, away from every labelled
  anomaly. The point forecast is scored.
* anomaly: the context ends shortly before a labelled anomaly begins, so the
  model has seen none of it and the anomaly begins inside the forecast. The
  anomaly counts as detected when a point of it falls outside the model's
  interval.

Usage, from the project root (reads 1.4 GB of CATS; about two minutes):

    python src/11_operational_benchmarks/prepare_cases.py

It writes `cases.parquet` and `prepared.json` (what was included and why) to
`paths.operational_benchmark_data`. The models are run by
`operational_benchmarks.py`, which needs nothing but these two files.
"""
# ruff: noqa: E402

import argparse
import hashlib
import json
import sys
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(Path(__file__).resolve().parent))
from operational_datasets import (
    caster_listing,
    cloud_listing,
    load_caster,
    load_cats,
    load_cloud,
    load_ucr,
    ucr_listing,
)

SECTION = "operational_benchmarks"
DEFAULTS = {
    # The names of FakeBench-TS (stage 9), whose model adapters this stage uses.
    "models": ["basin", "naive", "holtwinters", "chronos2", "timestral", "timeh2o"],
    "horizon": 112,
    "context_lengths": [128, 512, 2048],
    "forecast": {
        # Forecast cases per series and context length at most, spread evenly
        # over the normal data and at least one horizon apart.
        "origins": {"cats": 40, "ucr": 24, "cloud": 24, "caster": 12},
        # A forecast case stays this many points away from labelled anomalies:
        # the labels of real data are not exact to the point. The cloud series
        # are short and have many anomalies, so their margin is smaller.
        "anomaly_margin": {"cats": 100, "ucr": 100, "cloud": 16, "caster": 240},
    },
    "anomaly": {
        # Points between the end of the context and the start of the anomaly.
        "lead_points": [16, 64],
        # This many points of the context, the ones right before the forecast,
        # must be normal. An anomaly that ended earlier may lie further back
        # in a long context; such cases are marked and reported separately.
        "normal_points_before": 128,
        # A point is outside the interval when it lies below the lower or
        # above the upper of these two quantiles of the forecast ("quantiles").
        # With "normal" the interval is the median plus and minus
        # interval_sigmas standard deviations of the normal distribution that
        # has these two quantiles.
        "interval": "quantiles",
        "quantiles": [0.1, 0.9],
        "interval_sigmas": 2.0,
        # A window is flagged when more than this share of its judged points
        # lie outside the interval; 0 flags it for a single point.
        "min_share_outside": 0.0,
    },
    # Where a series has companions measured at the same moments (the other
    # channels of CATS, the other sensors of the caster), every case gets this
    # many of them, drawn at random, as past covariates. The models take five
    # at most.
    "covariates": {"count": 5, "seed": 7},
    # CATS has one row per second. A point is the mean of seconds_per_point
    # seconds, so that 112 steps reach far enough into an anomaly.
    "cats": {"path": "benchmarks/CATS Dataset", "seconds_per_point": 10},
    "ucr": {
        "path": "benchmarks/UCR_TimeSeriesAnomalyDatasets2021",
        "categories": ["electricity", "weather", "spacecraft_telemetry"],
    },
    "cloud": {
        "path": "benchmarks/cloud-monitoring-dataset",
        "categories": ["cloud_service_telemetry", "software_crash_rate"],
    },
    # The caster is recorded twice a second and has no anomaly labels: it
    # gives forecast cases only. A point is the mean of samples_per_point samples.
    "caster": {"path": "benchmarks/micromill-caster", "samples_per_point": 1},
    "batch_size": 16,
    # Processes that fit Holt-Winters at the same time.
    "holt_winters_workers": 10,
}
CASES = "cases.parquet"
RECORD = "prepared.json"


def settings(config):
    """`operational_benchmarks` of the configuration over the defaults."""
    given = config.get(SECTION, {})
    unknown = sorted(set(given) - set(DEFAULTS))
    if unknown:
        raise ValueError(f"Unknown settings in {SECTION}: {', '.join(unknown)}")
    merged = {}
    for key, default in DEFAULTS.items():
        value = given.get(key, default)
        if isinstance(default, dict):
            extra = sorted(set(value) - set(default))
            if extra:
                raise ValueError(
                    f"Unknown settings in {SECTION}.{key}: {', '.join(extra)}"
                )
            value = {**default, **value}
        merged[key] = value
    lengths, horizon = merged["context_lengths"], merged["horizon"]
    if not lengths or any(not isinstance(n, int) or n < 2 for n in [*lengths, horizon]):
        raise ValueError(
            f"{SECTION}.context_lengths and horizon must be whole numbers above 1"
        )
    leads = merged["anomaly"]["lead_points"]
    if not leads or any(not isinstance(n, int) or not 0 <= n < horizon for n in leads):
        raise ValueError(
            f"{SECTION}.anomaly.lead_points must be whole numbers from 0 to "
            f"{horizon - 1}: the anomaly has to begin inside the forecast"
        )
    count = merged["covariates"]["count"]
    if isinstance(count, bool) or not isinstance(count, int) or not 0 <= count <= 5:
        raise ValueError(f"{SECTION}.covariates.count must be a whole number, 0 to 5")
    low, high = merged["anomaly"]["quantiles"]
    if not 0 < low < 0.5 < high < 1:
        raise ValueError(f"{SECTION}.anomaly.quantiles must lie around 0.5")
    if merged["anomaly"]["interval"] not in ("quantiles", "normal"):
        raise ValueError(f'{SECTION}.anomaly.interval must be "quantiles" or "normal"')
    if not 0 <= merged["anomaly"]["min_share_outside"] < 1:
        raise ValueError(f"{SECTION}.anomaly.min_share_outside must be 0 or a share")
    return merged


# Settings that change how forecasts are run or scored, not which cases exist.
SCORING_ONLY = ("interval", "quantiles", "interval_sigmas", "min_share_outside")


# Settings that change a run of the models, not the prepared cases.
RUN_ONLY = ("models", "batch_size", "holt_winters_workers")


def case_settings(options):
    """The settings the prepared cases depend on."""
    kept = {k: v for k, v in options.items() if k not in RUN_ONLY}
    kept["anomaly"] = {
        k: v for k, v in options["anomaly"].items() if k not in SCORING_ONLY
    }
    return json.loads(json.dumps(kept))


def data_directory(config):
    return ROOT / config.get("paths", {}).get(
        "operational_benchmark_data", "data/04_benchmarks/operational"
    )


def for_dataset(value, dataset):
    """A setting given once for all datasets, or per dataset."""
    return value[dataset] if isinstance(value, dict) else value


def abnormal_counts(length, ranges, margin=0):
    """Running count of abnormal points, so any window is checked in one step."""
    marked = np.zeros(length, dtype=bool)
    for start, end in ranges:
        marked[max(0, start - margin) : end + margin] = True
    return np.concatenate([[0], np.cumsum(marked)])


def forecast_origins(series, context, horizon, count, margin):
    """Evenly spread positions whose context and future are both normal.

    At most `count`, at least one horizon apart, with no value missing.
    """
    counts = abnormal_counts(len(series.values), series.abnormal, margin)
    counts = counts + np.concatenate([[0], np.cumsum(~np.isfinite(series.values))])
    origins = np.arange(context, len(series.values) - horizon + 1)
    if not len(origins):
        return []
    normal = counts[origins + horizon] == counts[origins - context]
    eligible = origins[normal]
    if not len(eligible):
        return []
    picks = np.linspace(0, len(eligible) - 1, min(count, len(eligible)))
    origins = []
    for origin in sorted({int(eligible[int(round(p))]) for p in picks}):
        if not origins or origin - origins[-1] >= horizon:
            origins.append(origin)
    return origins


def anomaly_origin(series, anomaly, context, horizon, lead, counts, normal):
    """Where the forecast starts for this anomaly, or why there is no case.

    The context ends `lead` points before the anomaly, so it holds no part of
    it. Its last `normal` points and the lead must be free of other anomalies.
    """
    origin = anomaly.start - lead
    if origin - context < 0:
        return None, "the series starts too late for this context"
    if origin + horizon > len(series.values):
        return None, "the series ends inside the forecast"
    if counts[anomaly.start] != counts[origin - min(context, normal)]:
        return None, "another anomaly lies right before this one"
    return origin, None


def choose_covariates(series, count, seed, key):
    """Names of `count` companions of the series, drawn without replacement."""
    names = sorted(series.companions)
    if not names or count < 1:
        return []
    digest = hashlib.sha256(f"{seed}:{series.dataset}:{series.name}:{key}".encode())
    rng = np.random.default_rng(int.from_bytes(digest.digest()[:8], "little"))
    return [str(n) for n in rng.choice(names, min(count, len(names)), False)]


def build_cases(series_list, options):
    """(case records, counts of anomaly cases that could not be made)."""
    horizon = options["horizon"]
    cases, skipped = [], Counter()
    for series in series_list:
        values = np.asarray(series.values, dtype=np.float32)
        counts = abnormal_counts(len(values), series.abnormal)
        shared = {
            "dataset": series.dataset,
            "series": series.name,
            "category": series.category,
            "variant": series.variant,
        }

        def add(task, context, origin, **more):
            window = values[origin - context : origin + horizon]
            names = choose_covariates(
                series,
                options["covariates"]["count"],
                options["covariates"]["seed"],
                f"{task}:{context}:{origin}",
            )
            # Covariates are given for the context only: nothing of the future.
            past = np.stack(
                [series.companions[n][origin - context : origin] for n in names]
                or [np.empty(0)]
            ).astype(np.float32)
            if not (np.isfinite(window).all() and np.isfinite(past).all()):
                skipped[f"{series.dataset}: values missing in the window"] += 1
                return
            cases.append(
                {
                    **shared,
                    "task": task,
                    "context_length": context,
                    "origin": origin,
                    "anomaly": "",
                    "role": "",
                    "kind": "",
                    "lead": -1,
                    "span_start": -1,
                    "span_length": 0,
                    "anomaly_length": 0,
                    "earlier_anomaly": "",
                    **more,
                    "covariates": names,
                    "values": window,
                    "covariate_values": past.ravel(),
                }
            )

        for context in options["context_lengths"]:
            for origin in forecast_origins(
                series,
                context,
                horizon,
                for_dataset(options["forecast"]["origins"], series.dataset),
                for_dataset(options["forecast"]["anomaly_margin"], series.dataset),
            ):
                add("forecast", context, origin)
            for anomaly in series.anomalies:
                for lead in options["anomaly"]["lead_points"]:
                    origin, reason = anomaly_origin(
                        series,
                        anomaly,
                        context,
                        horizon,
                        lead,
                        counts,
                        options["anomaly"]["normal_points_before"],
                    )
                    if origin is None:
                        skipped[f"{series.dataset}: {reason}"] += 1
                        continue
                    add(
                        "anomaly",
                        context,
                        origin,
                        anomaly=anomaly.label,
                        role=anomaly.role,
                        kind=anomaly.kind,
                        lead=lead,
                        span_start=lead,
                        span_length=min(anomaly.end, origin + horizon) - anomaly.start,
                        anomaly_length=anomaly.end - anomaly.start,
                        earlier_anomaly="no"
                        if counts[origin] == counts[origin - context]
                        else "yes",
                    )
    for number, case in enumerate(cases):
        case["case"] = number
    return cases, dict(sorted(skipped.items()))


def write_cases(cases, path):
    lists = {
        "covariates": pa.list_(pa.string()),
        "values": pa.list_(pa.float32()),
        # One covariate after the other, each as long as the context.
        "covariate_values": pa.list_(pa.float32()),
    }
    table = pa.table(
        {
            name: pa.array([case[name] for case in cases], type=lists.get(name))
            for name in cases[0]
        }
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(".tmp")
    pq.write_table(table, temporary, compression="zstd")
    temporary.replace(path)
    return hashlib.sha256(path.read_bytes()).hexdigest()


def load_cases(directory):
    """(cases as a pandas table, the preparation record); checks they belong together."""
    record = json.loads((directory / RECORD).read_text(encoding="utf-8"))
    path = directory / CASES
    if hashlib.sha256(path.read_bytes()).hexdigest() != record["cases_sha256"]:
        raise ValueError(f"{path} is not the file {RECORD} describes; prepare again")
    return pq.read_table(path).to_pandas(), record


def prepare(config, *, datasets=("cats", "ucr", "cloud", "caster"), log=print):
    options = settings(config)
    series_list, audit = [], {}
    if "caster" in datasets:
        audit["caster"] = []
        for record in caster_listing(ROOT / options["caster"]["path"]):
            sensors = load_caster(record, options["caster"]["samples_per_point"])
            first = sensors[0]
            audit["caster"].append(
                {
                    "recording": record["name"],
                    "category": first.category,
                    "measures": first.description,
                    "included": True,
                    "sensors": [s.name.split("/", 1)[1] for s in sensors],
                    "points": len(first.values),
                    "not_normal_operation": first.abnormal,
                }
            )
            series_list += sensors
        log(f"Caster: {len(audit['caster'])} recordings")
    if "cloud" in datasets:
        listing = cloud_listing(ROOT / options["cloud"]["path"])
        wanted = set(options["cloud"]["categories"])
        chosen = [load_cloud(r) for r in listing if r["category"] in wanted]
        anomalies = {s.name: len(s.anomalies) for s in chosen}
        audit["cloud"] = [
            {
                "series": r["name"],
                "category": r["category"],
                "measures": r["description"],
                "included": r["category"] in wanted,
                **(
                    {"anomalies": anomalies[r["name"]]}
                    if r["name"] in anomalies
                    else {}
                ),
            }
            for r in listing
        ]
        log(f"Cloud monitoring: {len(chosen)} of {len(listing)} series included")
        series_list += chosen
    if "ucr" in datasets:
        listing = ucr_listing(ROOT / options["ucr"]["path"])
        wanted = set(options["ucr"]["categories"])
        audit["ucr"] = [
            {
                "series": r["name"],
                "category": r["category"],
                "measures": r["description"],
                "variant": r["variant"],
                "included": r["category"] in wanted,
                "anomaly": list(r["anomaly"]),
            }
            for r in listing
        ]
        chosen = [r for r in listing if r["category"] in wanted]
        log(f"UCR: {len(chosen)} of {len(listing)} series are operational data")
        series_list += [load_ucr(r) for r in chosen]
    if "cats" in datasets:
        log("CATS: reading data.csv")
        cats = load_cats(
            ROOT / options["cats"]["path"],
            seconds_per_point=options["cats"]["seconds_per_point"],
        )
        audit["cats"] = [
            {
                "series": s.name,
                "category": s.category,
                "measures": s.description,
                "included": True,
                "anomalies": dict(Counter(a.role for a in s.anomalies)),
            }
            for s in cats
        ]
        series_list += cats
    cases, skipped = build_cases(series_list, options)
    directory = data_directory(config)
    digest = write_cases(cases, directory / CASES)
    record = {
        "created_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "settings": options,
        "cases": len(cases),
        "cases_by_dataset_task_context": {
            " ".join(map(str, key)): count
            for key, count in sorted(
                Counter(
                    (c["dataset"], c["task"], c["context_length"]) for c in cases
                ).items()
            )
        },
        "anomaly_cases_not_made": skipped,
        "cases_sha256": digest,
        "series": audit,
    }
    (directory / RECORD).write_text(json.dumps(record, indent=2), encoding="utf-8")
    log(f"{len(cases)} cases written to {directory}")
    return record


def main():
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--config", default=str(ROOT / "configs/config.json"))
    parser.add_argument(
        "--datasets",
        nargs="+",
        default=["cats", "ucr", "cloud", "caster"],
        choices=["cats", "ucr", "cloud", "caster"],
    )
    args = parser.parse_args()
    config = json.loads(Path(args.config).read_text(encoding="utf-8"))
    record = prepare(config, datasets=args.datasets)
    for key, count in record["cases_by_dataset_task_context"].items():
        print(f"  {key}: {count}")
    for reason, count in record["anomaly_cases_not_made"].items():
        print(f"  not made, {reason}: {count}")


if __name__ == "__main__":
    main()

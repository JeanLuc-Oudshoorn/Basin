"""Draw an annealing set: fresh tables from the training generator, in a chosen mix.

The generator and its settings are the ones the training corpus came from
(`generate_operational_data.generator_settings`). Tables are drawn one after
another with independent seeds and then either kept or dropped:

1. The `standard` share keeps every table, so it has the natural distribution.
2. For the other categories the generator keeps running, and only tables that
   belong to a category with room left are kept (rejection sampling). A kept
   table is therefore an ordinary draw of the generator, of the wanted kind.

No table is ever copied or repeated, and none comes from the training corpus,
its validation tables or the holdout.
"""

import copy
import hashlib
import json
import math
import shutil
import subprocess  # nosec B404
import sys
from concurrent.futures import ProcessPoolExecutor
from datetime import datetime, timezone
from multiprocessing import get_context
from pathlib import Path
from time import perf_counter

import numpy as np
import pandas as pd
from annealing_experiment import (
    ROOT,
    SET_FILE,
    existing_set,
    quotas,
    set_definition,
    set_directory,
)
from table_categories import STANDARD, matching_categories, table_statistics

sys.path.insert(0, str(ROOT / "src/01_data_generation"))

# Extra copies of a table (coarser, aggregated, sign-inverted) form a family
# with it. A set holds independent tables only, so each one can be validated on.
INDEPENDENT_TABLES = {
    "coarse_copy_probability": 0.0,
    "aggregate_copy_probability": 0.0,
    "sign_inversion_probability": 0.0,
    "save_scm": False,
    "save_plot": False,
    "save_dag": False,
}
_worker = None


def make_generator(output, seed):
    from generate_operational_data import generator_settings
    from scm_generator import OperationalSCMGenerator

    settings = dict(generator_settings(), **INDEPENDENT_TABLES, seed=seed)
    digest = hashlib.sha256(
        json.dumps(settings, sort_keys=True, default=str).encode()
    ).hexdigest()
    return OperationalSCMGenerator(output, **settings), digest


def initialize_worker(generator, target, thresholds):
    import pyarrow

    pyarrow.set_cpu_count(1)
    pyarrow.set_io_thread_count(1)
    global _worker
    _worker = (generator, target, thresholds)


def draw_tables(jobs, worker=None):
    """Draw one table per job; write it if it is wanted. Returns small records.

    A job is (index, seed, trend_free, directory, wanted): `wanted` is None to
    keep the table whatever it is, or the categories that still have room.
    """
    generator, target, thresholds = worker or _worker
    generator = copy.copy(generator)
    records = []
    for index, seed, trend_free, directory, wanted in jobs:
        generator.rng = np.random.default_rng(seed)
        model, frame = generator._sample_accepted_model(trend_free)
        statistics = table_statistics(frame[target])
        matched = matching_categories(statistics, thresholds)
        written = wanted is None or any(category in wanted for category in matched)
        digest = None
        if written:
            generator._write_table(model, frame, directory)
            with (directory / "data.parquet").open("rb") as stream:
                digest = hashlib.file_digest(stream, "sha256").hexdigest()
        records.append(
            {
                "candidate": index,
                "seed": seed,
                "trend_free": trend_free,
                "directory": str(directory) if written else None,
                "sha256": digest,
                "matched": matched,
                **statistics,
            }
        )
    return records


def holdout_hashes(base):
    """Checksums of the main run's holdout tables, or None if there is no split yet."""
    split = ROOT / base["paths"]["training"] / "dataset_split.json"
    if not split.is_file():
        return None
    sys.path.insert(0, str(ROOT / "src/06_holdout_test"))
    from startup_cache import load_split

    return set(load_split(split)["source_sha256"].values())


def build_set(
    recipe,
    base,
    *,
    workers=6,
    batch_size=16,
    max_candidates_per_table=400,
    check_holdout=True,
    progress=print,
):
    """Create the set of a recipe, or return the finished one of the same definition."""
    saved = existing_set(recipe, base)
    if saved is not None:
        progress(f"Annealing set {recipe['set']['name']!r} exists and matches: reused")
        return saved
    chosen = recipe["set"]
    targets = base["data"]["targets"]
    if len(targets) != 1:
        raise ValueError("Annealing categories are defined for a single target column")
    directory = set_directory(chosen["name"])
    date = datetime.now(timezone.utc).date().isoformat()
    tables_directory = directory / "generated" / date
    tables_directory.mkdir(parents=True)  # fails if the set folder already exists
    generator, settings_digest = make_generator(directory / "generated", chosen["seed"])
    run_id = hashlib.sha256(f"{chosen['name']}:{chosen['seed']}".encode()).hexdigest()[
        :12
    ]
    wanted_counts = quotas(chosen["tables"], chosen["mix"])
    standard = wanted_counts.get(STANDARD, 0)
    order = [category for category in chosen["mix"] if category != STANDARD]
    filled = {category: 0 for category in order}
    limit = standard + max_candidates_per_table * (chosen["tables"] - standard)
    sealed = None
    started = reported = perf_counter()

    def open_categories():
        return [c for c in order if filled[c] < wanted_counts[c]]

    def job(number, wanted):
        # Each candidate has its own random stream, fixed by the set seed and
        # its number: neither batch size nor worker count changes a table.
        rng = np.random.default_rng([chosen["seed"], number])
        return (
            number,
            int(rng.integers(0, 2**63)),
            bool(rng.random() < generator.trend_free_fraction),
            tables_directory / f"table_{run_id}_{number:08d}",
            wanted,
        )

    def jobs():
        index = 0
        while index < limit and (index < standard or open_categories()):
            stop = min(limit, index + batch_size)
            if index < standard:
                stop = min(stop, standard)
            wanted = None if index < standard else tuple(open_categories())
            yield [job(number, wanted) for number in range(index, stop)]
            index = stop

    accepted, seen, matches = [], 0, {category: 0 for category in order}

    def take(records):
        """Apply results in candidate order, so the set depends on the seed alone."""
        nonlocal seen
        for record in records:
            seen += 1
            for category in record["matched"]:
                if category in matches:
                    matches[category] += 1
            if record["candidate"] < standard:
                category = STANDARD
            else:
                category = next(
                    (c for c in open_categories() if c in record["matched"]), None
                )
            if category is None:
                if record["directory"]:
                    # Written while its category still had room, filled since.
                    shutil.rmtree(record["directory"])
                continue
            if category != STANDARD:
                filled[category] += 1
            path = Path(record["directory"]) / "data.parquet"
            accepted.append(
                dict(
                    record,
                    category=category,
                    matched=",".join(record["matched"]),
                    source=path.relative_to(ROOT).as_posix(),
                )
            )

    def report(force=False):
        nonlocal reported
        now = perf_counter()
        if force or now - reported >= 10:
            reported = now
            progress(
                f"[{len(accepted):,}/{chosen['tables']:,} tables kept] drawn {seen:,} | "
                + " | ".join(f"{c} {filled[c]:,}/{wanted_counts[c]:,}" for c in order)
                + f" | {seen / max(now - started, 1e-9):.1f} tables/s"
            )

    source = jobs()
    worker = (generator, targets[0], chosen["categories"])
    main_file = getattr(sys.modules.get("__main__"), "__file__", None)
    if workers == 1 or not main_file or not Path(main_file).is_file():
        for batch in source:
            take(draw_tables(batch, worker))
            report()
    else:
        with ProcessPoolExecutor(
            max_workers=workers,
            mp_context=get_context("spawn"),
            initializer=initialize_worker,
            initargs=worker,
        ) as executor:
            pending = []
            while True:
                # A batch is planned only when the results before it leave room,
                # so at most 2 x workers batches are ever drawn in vain.
                while len(pending) < 2 * workers:
                    batch = next(source, None)
                    if batch is None:
                        break
                    pending.append(executor.submit(draw_tables, batch))
                if not pending:
                    break
                take(pending.pop(0).result())
                report()
    report(force=True)
    missing = {c: wanted_counts[c] - filled[c] for c in open_categories()}
    if missing:
        raise RuntimeError(
            f"Stopped after {seen:,} drawn tables with categories still short: {missing}. "
            f"Share of drawn tables per category: "
            f"{ {c: round(matches[c] / seen, 5) for c in order} }. Lower the share, "
            f"relax the thresholds, or raise max_candidates_per_table. The unfinished "
            f"set is in {directory}."
        )
    frame = pd.DataFrame(accepted).drop(columns=["directory"])
    if check_holdout:
        sealed = holdout_hashes(base)
        if sealed is None:
            progress("No main-run split found: holdout overlap not checked")
        elif sealed & set(frame.sha256):
            raise RuntimeError(
                f"{len(sealed & set(frame.sha256))} tables of this set are identical to "
                "holdout tables of the main run. Use another set.seed. The unfinished "
                f"set is in {directory}."
            )
    frame.to_parquet(directory / "tables.parquet", index=False)
    natural = {category: matches[category] / seen for category in order}
    standard_share = wanted_counts.get(STANDARD, 0) / chosen["tables"]
    summary = {
        "schema": 1,
        "definition": set_definition(recipe, base),
        "date": date,
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "seconds": perf_counter() - started,
        "tables": len(frame),
        "tables_by_category": frame.category.value_counts().to_dict(),
        "drawn_tables": seen,
        # Among all drawn tables, before any selection: how common each kind is.
        "natural_share": natural,
        # Share of the finished set that belongs to each kind, counting the
        # tables of that kind that the standard part contains anyway.
        "effective_share": {
            category: float(
                frame.matched.str.split(",").map(lambda m: category in m).mean()
            )
            for category in order
        },
        "expected_share": {
            category: chosen["mix"][category] + standard_share * natural[category]
            for category in order
        },
        "generator_settings_sha256": settings_digest,
        "generator_commit": _commit(),
        "holdout_overlap_checked": bool(check_holdout and sealed is not None),
        "table_list": "tables.parquet",
    }
    (directory / SET_FILE).write_text(json.dumps(summary, indent=2), encoding="utf-8")
    return summary


def _commit():
    try:
        return subprocess.run(  # nosec B603 B607
            ["git", "-C", str(ROOT), "rev-parse", "HEAD"],
            capture_output=True,
            text=True,
            check=True,
        ).stdout.strip()
    except (OSError, subprocess.CalledProcessError):
        return None


def expected_draws(recipe, natural):
    """Rough number of tables to draw, given each category's natural share."""
    wanted = quotas(recipe["set"]["tables"], recipe["set"]["mix"])
    extra = max(
        (
            wanted[category] / natural[category]
            for category in wanted
            if category != STANDARD and natural.get(category)
        ),
        default=0,
    )
    return math.ceil(wanted.get(STANDARD, 0) + extra)

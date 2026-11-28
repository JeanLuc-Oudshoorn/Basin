"""How easily can generated series be told from real ones?

A trial of the generator: sample tables with the settings of
`src/01_data_generation/generate_operational_data.py` as they are now, take
the latest 512 points of every target, and train a classifier to tell them
from each group of real series. The score of a group is the area under the
curve of that classifier on series it was not trained on: 1.0 means it tells
every generated series from the real ones, 0.5 that it cannot tell them
apart. Every group is scored twice, once on the descriptors of
series_features.py and once on Chronos-2 embeddings, and the two are
averaged. The score of a trial is the mean over the groups; lower is better.

    python src/13_dataset_similarity/generator_realism.py reference
    python src/13_dataset_similarity/generator_realism.py sample --name trial
    python src/13_dataset_similarity/series_embeddings.py --windows ... --output ...
    python src/13_dataset_similarity/generator_realism.py score --name trial

`reference` collects the real groups once: FakeBench-TS, the operational
benchmarks and IND-Eval as stage 13 collected them, and windows of the public
datasets in `data/00_public`. Nothing is written outside the output folder.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import argparse
import hashlib
import json
import re
import tempfile
from concurrent.futures import ProcessPoolExecutor
from time import perf_counter

import numpy as np
import pandas as pd
import series_features
import similarity_data

ROOT = Path(__file__).resolve().parents[2]
GENERATION = ROOT / "src" / "01_data_generation"
OUTPUT = ROOT / "results" / "model" / "generator_realism"
POINTS = 512
REAL_GROUPS = ("fakebench", "operational", "ind_eval", "public")
GROUP_NAMES = {
    "fakebench": "FakeBench-TS",
    "operational": "Operational benchmarks",
    "ind_eval": "IND-Eval",
    "public": "Public datasets",
}
PER_GROUP = 5000
PUBLIC = ROOT / "data" / "00_public"
PER_FILE, PER_COLUMN = 300, 24
# Columns that are not measurements: row counters, calendar columns, flags,
# copies of the target at a lag, documented random columns, installed capacity.
NOT_A_SIGNAL = re.compile(
    r"^(unnamed|instant|year$|yr$|month$|mnth$|hour$|hr$|season$|weekday$|holiday$"
    r"|workingday$|weathersit$|is_|rv1$|rv2$|att_flag)|_lag_|capacity",
    re.IGNORECASE,
)
START = perf_counter()


def log(message):
    print(f"[{perf_counter() - START:7.1f}s] {message}", flush=True)


def rank(text, seed=7):
    return hashlib.sha256(f"{seed}:{text}".encode()).hexdigest()


def read_public(path):
    frame = pd.read_csv(path, low_memory=False)
    if frame.shape[1] == 1:
        frame = pd.read_csv(path, sep=";", low_memory=False)
    return frame.select_dtypes("number")


def collect_public(points=POINTS):
    """Windows of the measured columns of every public dataset.

    Windows do not overlap. A column gives at most PER_COLUMN windows and a
    file at most PER_FILE, chosen by a hash, so that one long recording does
    not decide the group.
    """
    rows = []
    for path in sorted(PUBLIC.glob("*.csv")):
        frame = read_public(path)
        found = []
        for column in frame.columns:
            if NOT_A_SIGNAL.search(str(column)):
                continue
            values = frame[column].to_numpy(dtype=np.float64)
            parts = []
            for begin in range(0, len(values) - points + 1, points):
                part = values[begin : begin + points]
                if len(np.unique(part[np.isfinite(part)])) <= 2:
                    continue  # Flags and constant stretches.
                if series_features.prepare(part) is None:
                    continue
                parts.append((rank(f"{path.name}|{column}|{begin}"), begin, part))
            for order, begin, part in sorted(parts)[:PER_COLUMN]:
                found.append((order, str(column), begin, part))
        for order, column, begin, part in sorted(found)[:PER_FILE]:
            rows.append(
                {
                    "group": "public",
                    "dataset": path.stem,
                    "series": f"{column}|{begin}",
                    "key": f"public|{path.stem}|{column}",
                    "values": part.astype(np.float32),
                }
            )
        log(f"public: {path.name}: {min(len(found), PER_FILE)} windows")
    return rows


def collect_benchmarks(points=POINTS):
    """The series of the three benchmark groups that stage 13 collected."""
    import similarity_analysis

    directory = ROOT / similarity_data.DEFAULTS["output"]
    index, windows = similarity_data.load(directory)
    in_view = similarity_data.view_rows(index, points)
    part = index.iloc[in_view]
    keys = similarity_analysis.underlying(part)
    rows = []
    for group in REAL_GROUPS[:3]:
        chosen = np.flatnonzero((part["group"] == group).to_numpy())
        order = similarity_data.ranked(part["series"].to_numpy()[chosen], 7)
        for position in chosen[order[:PER_GROUP]]:
            rows.append(
                {
                    "group": group,
                    "dataset": part["dataset"].iloc[position],
                    "series": part["series"].iloc[position],
                    "key": keys[position],
                    "values": np.asarray(windows[in_view[position], -points:]),
                }
            )
        log(f"{group}: {min(len(chosen), PER_GROUP):,} series")
    return rows


def describe(windows, workers):
    """The descriptors of every window, as a table."""
    names = series_features.FEATURES + series_features.EXTRA
    blocks = [windows[start : start + 200] for start in range(0, len(windows), 200)]
    with ProcessPoolExecutor(workers) as pool:
        table = np.concatenate(list(pool.map(series_features.describe_many, blocks)))
    return pd.DataFrame(table, columns=names)


def reference(args):
    folder = OUTPUT / "reference"
    folder.mkdir(parents=True, exist_ok=True)
    rows = collect_benchmarks() + collect_public()
    windows = np.stack([item.pop("values") for item in rows]).astype(np.float32)
    np.save(folder / "windows.npy", windows)
    index = pd.DataFrame(rows)
    index.to_parquet(folder / "index.parquet", index=False)
    describe(windows, args.workers).to_parquet(folder / "features.parquet", index=False)
    log(f"reference: {index.groupby('group').size().to_dict()} in {folder}")


def table_facts(model, frame):
    """What the generator drew for a table, for reading a trial afterwards."""
    outcome = model.get("outcome", {})
    observation = model.get("observation", {})
    facts = {
        "rows": int(len(frame)),
        "trend_free": bool(model.get("trend_free", False)),
        "univariate": bool(model.get("univariate", False)),
        "kind": str(model.get("kind", "scm")),
        "plant": str(model.get("plant", {}).get("family", "none")),
        "noise_profile": str(model.get("noise_profile", {}).get("name", "none")),
        "trend": str(outcome.get("trend", {}).get("kind", "none")),
        "operation": str(outcome.get("operation", {}).get("mode", "none")),
        "seasons": int(len(outcome.get("seasons", []))),
        "quantized": bool(outcome.get("quantization_step", 0) > 0),
        "sensor_bounds": bool("sensor" in outcome),
        "artifacts": ",".join(
            sorted(
                a["kind"] for a in observation.get("artifacts", {}).get("outcome", [])
            )
        ),
    }
    periods = [s.get("period") for s in outcome.get("seasons", []) if s.get("period")]
    facts["period"] = float(periods[0]) if periods else 0.0
    return facts


_GENERATOR = None


def worker_generator():
    global _GENERATOR
    if _GENERATOR is None:
        sys.path.insert(0, str(GENERATION))
        import generate_operational_data as runner
        from scm_generator import OperationalSCMGenerator

        settings = runner.generator_settings()
        unused = Path(tempfile.gettempdir()) / "basin_generator_realism_unused"
        _GENERATOR = OperationalSCMGenerator(unused, **settings)
    return _GENERATOR


def sample_block(jobs):
    """(latest points or None, facts) of the tables of some seeds."""
    generator = worker_generator()
    results = []
    for seed, trend_free in jobs:
        generator.rng = np.random.default_rng(seed)
        model, frame = generator._sample_accepted_model(trend_free)
        values = frame["outcome"].to_numpy(dtype=np.float32)
        facts = table_facts(model, frame)
        facts["seed"] = str(seed)
        results.append((values[-POINTS:] if len(values) >= POINTS else None, facts))
    return results


def sample(args):
    """Draw tables with the generator's current settings; keep their targets."""
    sys.path.insert(0, str(GENERATION))
    import generate_operational_data as runner

    folder = OUTPUT / "trials" / args.name
    folder.mkdir(parents=True, exist_ok=True)
    count = args.tables
    rng = np.random.default_rng(args.seed)
    seeds = rng.integers(0, 2**63, size=count)
    profiles = np.zeros(count, dtype=bool)
    profiles[: int(np.ceil(count * runner.TREND_FREE_FRACTION))] = True
    rng.shuffle(profiles)
    jobs = [(int(seed), bool(flag)) for seed, flag in zip(seeds, profiles)]
    blocks = [jobs[start : start + 16] for start in range(0, count, 16)]
    windows, facts = [], []
    with ProcessPoolExecutor(args.workers) as pool:
        for done, block in enumerate(pool.map(sample_block, blocks), 1):
            for values, fact in block:
                fact["long_enough"] = values is not None
                facts.append(fact)
                if values is not None:
                    windows.append(values)
            if done % 100 == 0:
                log(f"sample: {len(facts):,} of {count:,} tables")
    windows = np.stack(windows)
    np.save(folder / "windows.npy", windows)
    pd.DataFrame(facts).to_parquet(folder / "tables.parquet", index=False)
    log(f"sample: {len(windows):,} of {count:,} tables have {POINTS} points")
    describe(windows, args.workers).to_parquet(folder / "features.parquet", index=False)
    log(f"sample: described, in {folder}")


def fit_scores(data, positive, keys, seed=7, workers=8, wide=False):
    """Out-of-fold chance of being a positive row, by LightGBM.

    Both sides get the same total weight, and the windows of one recording
    stay in one fold. Also returns the mean gain of every column.
    """
    from lightgbm import LGBMClassifier
    from sklearn.model_selection import StratifiedGroupKFold

    folds = StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=seed)
    scores = np.zeros(len(positive))
    gain = np.zeros(data.shape[1])
    for train, test in folds.split(data, positive, keys):
        model = LGBMClassifier(
            n_estimators=200,
            learning_rate=0.08,
            num_leaves=31,
            min_child_samples=20,
            subsample=0.8,
            subsample_freq=1,
            colsample_bytree=0.3 if wide else 0.8,
            max_bin=63 if wide else 255,
            class_weight="balanced",
            importance_type="gain",
            n_jobs=workers,
            random_state=seed,
            verbose=-1,
        )
        model.fit(data[train], positive[train])
        scores[test] = model.predict_proba(data[test])[:, 1]
        gain += model.feature_importances_ / 5
    return scores, gain


def load_reference():
    folder = OUTPUT / "reference"
    index = pd.read_parquet(folder / "index.parquet")
    features = pd.read_parquet(folder / "features.parquet")
    embeddings = np.load(folder / "embeddings.npy")
    return index, features, embeddings


def score(args):
    from sklearn.metrics import roc_auc_score

    index, real_features, real_vectors = load_reference()
    folder = OUTPUT / "trials" / args.name
    features = pd.read_parquet(folder / "features.parquet")
    vectors = np.load(folder / "embeddings.npy")
    names = list(series_features.FEATURES)
    shaped = features[names[0]].notna().to_numpy()
    synthetic = features.loc[shaped, names].to_numpy()
    synthetic_vectors = vectors[shaped]
    synthetic_keys = np.array([f"synthetic|{i}" for i in range(len(synthetic))])
    result = {
        "name": args.name,
        "synthetic_series": int(len(synthetic)),
        "groups": {},
    }
    likeness = np.zeros(len(synthetic))
    for group in REAL_GROUPS:
        chosen = (index["group"] == group).to_numpy()
        chosen = chosen & real_features[names[0]].notna().to_numpy()
        keys = np.concatenate([index["key"].to_numpy()[chosen], synthetic_keys])
        positive = np.arange(len(keys)) < chosen.sum()
        entry = {"series": int(chosen.sum())}
        for kind, real, own, wide in (
            (
                "descriptors",
                real_features.loc[chosen, names].to_numpy(),
                synthetic,
                False,
            ),
            ("embeddings", real_vectors[chosen], synthetic_vectors, True),
        ):
            data = np.concatenate([real, own])
            scores, gain = fit_scores(
                data, positive, keys, workers=args.workers, wide=wide
            )
            entry[kind] = float(roc_auc_score(positive, scores))
            entry[f"{kind}_synthetic_resembling"] = float(
                (scores[~positive] >= 0.5).mean()
            )
            likeness += scores[~positive] / (2 * len(REAL_GROUPS))
            if kind == "descriptors":
                order = np.argsort(gain)[::-1][:8]
                entry["top_descriptors"] = {
                    names[i]: round(float(gain[i] / gain.sum()), 3) for i in order
                }
                real_scores = pd.Series(scores[positive])
                datasets = index["dataset"].to_numpy()[chosen]
                # Per dataset: how surely its series are told from generated ones.
                entry["datasets"] = {
                    str(dataset): round(
                        float(
                            roc_auc_score(
                                np.concatenate(
                                    [np.ones(len(part)), np.zeros((~positive).sum())]
                                ),
                                np.concatenate([part.to_numpy(), scores[~positive]]),
                            )
                        ),
                        4,
                    )
                    for dataset, part in real_scores.groupby(datasets)
                }
            log(f"score: {group} {kind} AUC {entry[kind]:.4f}")
        entry["mean"] = (entry["descriptors"] + entry["embeddings"]) / 2
        result["groups"][group] = entry
    result["score"] = float(np.mean([g["mean"] for g in result["groups"].values()]))
    (folder / "score.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
    kept = pd.read_parquet(folder / "tables.parquet")
    kept = kept[kept["long_enough"]].reset_index(drop=True)[shaped]
    kept.assign(real_likeness=likeness).to_parquet(
        folder / "likeness.parquet", index=False
    )
    with open(OUTPUT / "trials.jsonl", "a", encoding="utf-8") as handle:
        handle.write(json.dumps(result) + "\n")
    print(summary(result))
    return result


def summary(result):
    lines = [
        f"trial {result['name']}: score {result['score']:.4f} "
        f"({result['synthetic_series']:,} generated series)",
        "| Group | Descriptors | Embeddings | Mean |",
        "|---|---|---|---|",
    ]
    for group, entry in result["groups"].items():
        lines.append(
            f"| {GROUP_NAMES[group]} | {entry['descriptors']:.4f} "
            f"| {entry['embeddings']:.4f} | {entry['mean']:.4f} |"
        )
    return "\n".join(lines)


def main():
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("step", choices=("reference", "sample", "score"))
    parser.add_argument("--name", default="trial")
    parser.add_argument("--tables", type=int, default=12000)
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--workers", type=int, default=10)
    args = parser.parse_args()
    {"reference": reference, "sample": sample, "score": score}[args.step](args)


if __name__ == "__main__":
    main()

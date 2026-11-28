"""Does Basin, as trained, forecast the generator's new kinds of series well?

Draws tables with the generator of the tree this file is run from, keeps for
each a history of 512 points and the 112 points that follow, and scores
forecasts of them by the kind of table. Nothing is written to the data
folders, and no model is trained.

    python pattern_forecast_check.py sample <output folder> [tables] [seed]   (CPU)
    python pattern_forecast_check.py forecast <folder> <model>                (GPU, in WSL)
    python pattern_forecast_check.py report <folder> [<folder of another generator>]

`model` is basin, chronos2 or timestral; the naive forecast needs no step.
"""

import json
import sys
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
CONTEXT, HORIZON = 512, 112


def draw(jobs):
    sys.path.insert(0, str(ROOT / "src" / "13_dataset_similarity"))
    import generator_realism

    generator = generator_realism.worker_generator()
    rows = []
    for seed, trend_free in jobs:
        generator.rng = np.random.default_rng(seed)
        model, frame = generator._sample_accepted_model(trend_free)
        values = frame["outcome"].to_numpy(dtype=np.float64)[-(CONTEXT + HORIZON) :]
        if len(values) < CONTEXT + HORIZON or not np.isfinite(values[CONTEXT:]).all():
            continue
        history = values[:CONTEXT]
        seen = np.isfinite(history)
        if seen.sum() < CONTEXT // 2:
            continue
        position = np.arange(CONTEXT)
        history = np.interp(position, position[seen], history[seen])
        if np.abs(np.diff(history)).mean() <= 0:
            continue  # A flat history has no scale to measure an error by.
        plant = model.get("plant", {})
        kind = plant.get("family", "plain causal model")
        if kind != "plain causal model" and plant.get("trend_gain"):
            kind += " on a trend"
        drawn = {
            key: value
            for key, value in plant.items()
            if isinstance(value, (int, float, str, bool))
        }
        drawn.update(profile_hold=(plant.get("profile") or {}).get("hold"))
        rows.append((kind, history, values[CONTEXT:], json.dumps(drawn)))
    return rows


def sample(folder, tables, seed):
    sys.path.insert(0, str(ROOT / "src" / "01_data_generation"))
    import generate_operational_data as runner

    rng = np.random.default_rng(seed)
    seeds = rng.integers(0, 2**63, size=tables)
    profiles = np.zeros(tables, dtype=bool)
    profiles[: int(np.ceil(tables * runner.TREND_FREE_FRACTION))] = True
    rng.shuffle(profiles)
    jobs = [(int(s), bool(f)) for s, f in zip(seeds, profiles)]
    blocks = [jobs[start : start + 16] for start in range(0, tables, 16)]
    with ProcessPoolExecutor(10) as pool:
        rows = [row for block in pool.map(draw, blocks) for row in block]
    folder.mkdir(parents=True, exist_ok=True)
    np.save(folder / "contexts.npy", np.stack([r[1] for r in rows]).astype(np.float32))
    np.save(folder / "futures.npy", np.stack([r[2] for r in rows]).astype(np.float32))
    pd.DataFrame(
        {"kind": [r[0] for r in rows], "drawn": [r[3] for r in rows]}
    ).to_parquet(folder / "kinds.parquet")
    print(f"{len(rows):,} of {tables:,} tables kept in {folder}")


def forecast(folder, name):
    sys.path.insert(0, str(ROOT / "src" / "09_fakebench"))
    import fakebench
    from transform import load_config

    config = load_config(ROOT / "configs" / "config.json")
    model = fakebench.build(name, config, batch_size=64)
    contexts = np.load(folder / "contexts.npy")
    parts = [
        np.asarray(model(list(contexts[start : start + 512]), HORIZON))
        for start in range(0, len(contexts), 512)
    ]
    np.save(folder / f"forecast_{name}.npy", np.concatenate(parts).astype(np.float32))
    print(f"{name}: {len(contexts):,} forecasts written")


def scores(folder):
    contexts = np.load(folder / "contexts.npy").astype(np.float64)
    futures = np.load(folder / "futures.npy").astype(np.float64)
    scale = np.abs(np.diff(contexts, axis=1)).mean(axis=1)
    table = pd.read_parquet(folder / "kinds.parquet")
    np.seterr(divide="ignore", invalid="ignore")
    for path in sorted(folder.glob("forecast_*.npy")):
        name = path.stem.split("_", 1)[1]
        values = np.load(path)
        if values.ndim == 3:
            # Nine quantile levels from 10% to 90%: the middle one is the median.
            values = values[:, values.shape[1] // 2]
        table[name] = np.abs(futures - values).mean(axis=1) / scale
    # Stored as 32-bit numbers, a few almost flat histories have no scale left.
    with np.errstate(divide="ignore", invalid="ignore"):
        table["naive"] = np.abs(futures - contexts[:, -1:]).mean(axis=1) / scale
    return table[scale > 0].reset_index(drop=True)


def summary(table, label):
    others = [c for c in ("chronos2", "timestral") if c in table]
    rows = []
    for kind, part in list(table.groupby("kind")) + [(f"ALL ({label})", table)]:
        if len(part) < 30:
            continue
        entry = {"kind": kind, "series": len(part)}
        for name in ["basin", *others, "naive"]:
            entry[name] = part[name].median()
        # A forecast can be exact (a flat future): keep the ratio defined.
        floor = 1e-6
        reference = np.exp(
            np.mean([np.log(part[o].clip(lower=floor)) for o in others], axis=0)
        )
        entry["basin / others"] = float(
            np.median(part["basin"].clip(lower=floor) / reference)
        )
        entry["basin better than both"] = float(
            (part["basin"] < part[others].min(axis=1)).mean()
        )
        rows.append(entry)
    return pd.DataFrame(rows)


def main():
    step, folder = sys.argv[1], Path(sys.argv[2])
    if step == "sample":
        tables = int(sys.argv[3]) if len(sys.argv) > 3 else 6000
        sample(folder, tables, int(sys.argv[4]) if len(sys.argv) > 4 else 1)
    elif step == "forecast":
        forecast(folder, sys.argv[3])
    else:
        pd.set_option("display.width", 200)
        frames = [summary(scores(folder), "this generator")]
        if len(sys.argv) > 3:
            frames.append(summary(scores(Path(sys.argv[3])), "other generator"))
        result = pd.concat(frames).sort_values("basin / others")
        print(result.round(3).to_string(index=False))
        (folder / "report.json").write_text(
            json.dumps(result.to_dict(orient="records"), indent=2), encoding="utf-8"
        )


if __name__ == "__main__":
    main()

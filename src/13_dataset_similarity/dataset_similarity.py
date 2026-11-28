"""How much do Basin's synthetic series resemble the series of the benchmarks?

Steps, each of which reads what the earlier ones saved:

    collect   the series of the four groups (similarity_data.py)
    describe  the descriptors of every series (series_features.py)
    embed     Chronos-2 encoder embeddings of the same windows (optional)
    analyse   distances, coverage, what separates the groups, the maps
    report    figures, tables and the interactive map

    python src/13_dataset_similarity/dataset_similarity.py
    python src/13_dataset_similarity/dataset_similarity.py --steps analyse report

Everything runs on the CPU.
"""

import argparse
import sys
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
from time import perf_counter

import numpy as np
import pandas as pd

sys.path.insert(
    0, str(Path(__file__).resolve().parents[1] / "03_transforming_input_data")
)
import series_features
import similarity_data
from transform import load_config

ROOT = Path(__file__).resolve().parents[2]
STEPS = ("collect", "describe", "embed", "analyse", "report")
START = perf_counter()


def log(message):
    print(f"[{perf_counter() - START:7.0f}s] {message}", flush=True)


def describe(options):
    directory = similarity_data.output_directory(options)
    index, windows = similarity_data.load(directory)
    names = series_features.FEATURES + series_features.EXTRA
    for points in options["window_points"]:
        rows = similarity_data.view_rows(index, points)
        blocks = np.array_split(rows, max(1, len(rows) // 250))
        parts = (np.asarray(windows[block][:, -points:]) for block in blocks)
        table = []
        with ProcessPoolExecutor(options["workers"]) as pool:
            for done, part in enumerate(
                pool.map(series_features.describe_many, parts), 1
            ):
                table.append(part)
                if done % 40 == 0:
                    log(f"describe {points}: {done} of {len(blocks)} blocks")
        frame = pd.DataFrame(np.concatenate(table), columns=list(names))
        frame.insert(0, "row", rows)
        frame.to_parquet(similarity_data.features_path(directory, points), index=False)
        shapeless = int(frame[names[0]].isna().sum())
        log(f"describe {points}: {len(frame):,} series, {shapeless:,} without a shape")


def main():
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--config", default=str(ROOT / "configs/config.json"))
    parser.add_argument("--steps", nargs="+", choices=STEPS, default=list(STEPS))
    parser.add_argument(
        "--embedding-model",
        help="folder with the Chronos-2 model (default: the project's model cache)",
    )
    args = parser.parse_args()
    config = load_config(args.config)
    options = similarity_data.settings(config)
    if args.embedding_model:
        options["embeddings"]["model_path"] = args.embedding_model
    for step in args.steps:
        log(f"step: {step}")
        if step == "collect":
            similarity_data.collect(config, options, log)
        elif step == "describe":
            describe(options)
        elif step == "embed":
            import series_embeddings

            series_embeddings.embed(config, options, log)
        elif step == "analyse":
            import similarity_analysis

            similarity_analysis.analyse(options, log)
        elif step == "report":
            import similarity_report

            similarity_report.report(options, log)
    log("done")


if __name__ == "__main__":
    main()

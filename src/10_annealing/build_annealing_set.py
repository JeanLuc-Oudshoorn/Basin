"""Create an annealing experiment from a recipe: its set of tables and its folder.

CPU only.

    python src/10_annealing/build_annealing_set.py --recipe configs/annealing/flat25.json

Draws the recipe's annealing set with the training generator (or reuses the
finished set of the same name and definition), then creates the experiment
folder with its configuration and prints the commands that follow. Existing
sets and experiments are never changed; see docs/annealing/ANNEALING.md.
"""

import argparse
import json
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from annealing_experiment import (
    ROOT,
    existing_set,
    experiment_directory,
    load_base_config,
    load_recipe,
    prepare_experiment,
    set_directory,
)
from annealing_set import build_set


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--recipe", required=True, help="recipe JSON file")
    parser.add_argument(
        "--workers",
        type=int,
        default=max(1, min(8, (os.cpu_count() or 2) // 2)),
        help="generator processes (default: half the logical processors, at most 8)",
    )
    parser.add_argument(
        "--skip-holdout-check",
        action="store_true",
        help="do not compare the set's tables with the main run's holdout tables",
    )
    args = parser.parse_args()
    recipe = load_recipe(args.recipe)
    base = load_base_config(recipe)
    # Refuse a clashing experiment name before any table is drawn.
    if experiment_directory(recipe["experiment"]).exists():
        saved = existing_set(recipe, base)
        if saved is None:
            raise FileNotFoundError(
                f"Experiment {recipe['experiment']!r} exists but its set "
                f"{recipe['set']['name']!r} does not; choose a new experiment name"
            )
        prepare_experiment(recipe, base, saved["date"])
    summary = build_set(
        recipe, base, workers=args.workers, check_holdout=not args.skip_holdout_check
    )
    directory, config, created = prepare_experiment(recipe, base, summary["date"])
    config_argument = config.relative_to(ROOT).as_posix()
    print(
        json.dumps(
            {
                key: summary[key]
                for key in (
                    "tables",
                    "tables_by_category",
                    "drawn_tables",
                    "natural_share",
                    "effective_share",
                )
            },
            indent=2,
        )
    )
    print(f"Annealing set: {set_directory(recipe['set']['name'])}")
    print(f"Experiment {'created' if created else 'exists, unchanged'}: {directory}")
    print("Next:")
    if not (
        set_directory(recipe["set"]["name"]) / "patch_index/patch_index.pt"
    ).is_file():
        print(
            f"  python src/03_transforming_input_data/transform.py --config {config_argument}"
        )
        print(
            f"  python src/04_patch_indexing/index_patches.py --config {config_argument}"
        )
    print(f"  python src/10_annealing/anneal.py --experiment {recipe['experiment']}")


if __name__ == "__main__":
    main()

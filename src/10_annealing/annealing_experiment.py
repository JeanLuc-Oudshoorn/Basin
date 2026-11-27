"""Recipes, folders and derived configurations of annealing experiments.

A recipe (one JSON file) describes an experiment completely: the annealing set
(how many tables, which mix of categories), the checkpoint it starts from and
the training settings that differ from the main run.

    data/04_annealing/<set name>/            tables, processed data, patch index
    results/model/annealing/<experiment>/    configuration, checkpoints, results

Nothing here writes into the main run's folders, and nothing overwrites an
existing set or experiment: a name can be used once per definition. Sets are
shared, so experiments that differ only in training settings reuse one set.
"""

import copy
import hashlib
import json
import re
from datetime import datetime, timezone
from pathlib import Path

from table_categories import STANDARD, category_thresholds

ROOT = Path(__file__).resolve().parents[2]
SET_ROOT = "data/04_annealing"
EXPERIMENT_ROOT = "results/model/annealing"
SET_FILE = "annealing_set.json"  # written last: its presence marks a complete set
NAME = re.compile(r"[A-Za-z0-9][A-Za-z0-9_.-]{0,63}")

# Settings of the annealing phase that differ from the main run unless the
# recipe says otherwise. Everything else is inherited from the base config.
TRAINING_DEFAULTS = {
    # The learning rate the main run ends at; annealing brings it to zero.
    "learning_rate": 1e-5,
    "scheduler": {
        "name": "warmup_cosine",
        "warmup_fraction": 0.03,
        "min_learning_rate": 0.0,
    },
    "epochs": 1,
    "steps": None,
    "max_run_hours": None,
    # A short run needs frequent checks; they use a small panel of the set.
    "validation_interval_seconds": 1800,
    "checkpoint_interval_seconds": 1800,
    "validation_sampling": {"tables": 2000},
    "holdout_tables": 500,
    "validate_before_training": True,
    # Run the whole schedule; the best check is still the one that is exported.
    "early_stopping": {"patience_checks": 1000},
}
# Replaced as a whole instead of merged key by key: a schedule is one choice.
REPLACED = {"scheduler"}
# Fixed by the annealing code; a recipe must not set them.
RESERVED = {"resume"}


def _name(value, what):
    if not isinstance(value, str) or not NAME.fullmatch(value):
        raise ValueError(
            f"{what} must be 1-64 letters, digits, '_', '-' or '.', starting with a "
            f"letter or digit; got {value!r}"
        )
    return value


def _relative(path):
    """A path under the repository as the configs write it: relative, forward slashes."""
    path = Path(path)
    path = path if path.is_absolute() else ROOT / path
    return path.resolve().relative_to(ROOT.resolve()).as_posix()


def merge(base, overrides):
    """`base` with `overrides` applied; nested objects merge, other values replace."""
    result = copy.deepcopy(base)
    for key, value in overrides.items():
        if (
            key not in REPLACED
            and isinstance(value, dict)
            and isinstance(result.get(key), dict)
        ):
            result[key] = merge(result[key], value)
        else:
            result[key] = copy.deepcopy(value)
    return result


def load_recipe(path):
    """Read and check a recipe; returns it with every default filled in."""
    path = Path(path)
    path = path if path.is_absolute() else ROOT / path
    recipe = json.loads(path.read_text(encoding="utf-8"))
    return validate_recipe(recipe)


def validate_recipe(recipe):
    unknown = set(recipe) - {
        "experiment",
        "description",
        "base_config",
        "set",
        "start",
        "training",
    }
    if unknown:
        raise ValueError(f"Unknown recipe fields: {sorted(unknown)}")
    chosen = recipe.get("set")
    if not isinstance(chosen, dict):
        raise ValueError("The recipe needs a 'set' object")
    unknown = set(chosen) - {"name", "tables", "seed", "mix", "categories"}
    if unknown:
        raise ValueError(f"Unknown set fields: {sorted(unknown)}")
    tables, seed, mix = chosen.get("tables"), chosen.get("seed"), chosen.get("mix")
    if isinstance(tables, bool) or not isinstance(tables, int) or tables < 10:
        raise ValueError("set.tables must be an integer of at least 10")
    if isinstance(seed, bool) or not isinstance(seed, int) or seed < 0:
        raise ValueError("set.seed must be a non-negative integer")
    thresholds = category_thresholds(chosen.get("categories"))
    if not isinstance(mix, dict) or not mix:
        raise ValueError("set.mix must map category names to shares")
    for category, share in mix.items():
        if category != STANDARD and category not in thresholds:
            raise ValueError(
                f"Unknown category {category!r} in set.mix; known: "
                f"{[STANDARD, *sorted(thresholds)]}"
            )
        if (
            isinstance(share, bool)
            or not isinstance(share, (int, float))
            or not 0 < share <= 1
        ):
            raise ValueError(f"set.mix.{category} must be a share in (0, 1]")
    if abs(sum(mix.values()) - 1) > 1e-6:
        raise ValueError(f"set.mix shares must add up to 1, not {sum(mix.values())}")
    start = dict(recipe.get("start") or {})
    unknown = set(start) - {"checkpoint", "weights", "allow_unfinished"}
    if unknown:
        raise ValueError(f"Unknown start fields: {sorted(unknown)}")
    if start.get("weights", "best") not in ("best", "last"):
        raise ValueError("start.weights must be 'best' or 'last'")
    if not isinstance(start.get("allow_unfinished", False), bool):
        raise ValueError("start.allow_unfinished must be true or false")
    training = recipe.get("training") or {}
    if not isinstance(training, dict) or set(training) & RESERVED:
        raise ValueError(
            f"training must be an object without {sorted(RESERVED)}: an experiment "
            "always continues its own checkpoint and never another one"
        )
    return {
        "experiment": _name(recipe.get("experiment"), "experiment"),
        "description": str(recipe.get("description", "")),
        "base_config": _relative(recipe.get("base_config", "configs/config.json")),
        "set": {
            "name": _name(chosen.get("name"), "set.name"),
            "tables": tables,
            "seed": seed,
            "mix": {category: float(share) for category, share in mix.items()},
            "categories": thresholds,
        },
        "start": {
            "checkpoint": start.get("checkpoint"),
            "weights": start.get("weights", "best"),
            "allow_unfinished": start.get("allow_unfinished", False),
        },
        "training": training,
    }


def load_base_config(recipe):
    return json.loads((ROOT / recipe["base_config"]).read_text(encoding="utf-8"))


def set_directory(name):
    return ROOT / SET_ROOT / name


def experiment_directory(name):
    return ROOT / EXPERIMENT_ROOT / name


def pipeline_digest(base):
    """What the processed data and the patch index of a set depend on."""
    content = {
        "seed": base["seed"],
        "data": {k: v for k, v in base["data"].items() if k != "generated_dates"},
        "model": base["model"],
    }
    return hashlib.sha256(json.dumps(content, sort_keys=True).encode()).hexdigest()


def set_definition(recipe, base):
    """Everything that decides which tables a set holds and how they are indexed."""
    return {"set": recipe["set"], "pipeline_sha256": pipeline_digest(base)}


def quotas(tables, mix):
    """Whole tables per category; rounding goes to the largest remainders."""
    exact = {category: tables * share for category, share in mix.items()}
    counts = {category: int(value) for category, value in exact.items()}
    by_remainder = sorted(mix, key=lambda c: (counts[c] - exact[c], c))
    for category in by_remainder[: tables - sum(counts.values())]:
        counts[category] += 1
    return counts


def existing_set(recipe, base):
    """The finished set of this name, or None if there is no folder yet.

    A folder with another definition, or one left unfinished, is an error:
    it is never reused and never removed here.
    """
    directory = set_directory(recipe["set"]["name"])
    if not directory.exists():
        return None
    manifest = directory / SET_FILE
    if not manifest.is_file():
        raise FileExistsError(
            f"{directory} exists but holds no finished annealing set (no {SET_FILE}). "
            "An earlier build was interrupted. Remove the folder yourself after "
            "checking it, or choose a new set.name."
        )
    saved = json.loads(manifest.read_text(encoding="utf-8"))
    if saved["definition"] != set_definition(recipe, base):
        raise FileExistsError(
            f"Annealing set {recipe['set']['name']!r} already exists with a different "
            "definition (tables, seed, mix, thresholds or data/model settings). "
            "Choose a new set.name; existing sets are never overwritten."
        )
    return saved


def start_checkpoint(recipe, base):
    return _relative(
        recipe["start"]["checkpoint"]
        or Path(base["paths"]["training"]) / "latest_checkpoint.pt"
    )


def experiment_config(recipe, base, date):
    """The pipeline configuration of one experiment, derived from the main one.

    `date` is the dated folder of the set's tables. Data paths point at the
    set, every output path at the experiment's own folder.
    """
    data = f"{SET_ROOT}/{recipe['set']['name']}"
    output = f"{EXPERIMENT_ROOT}/{recipe['experiment']}"
    config = copy.deepcopy(base)
    config["paths"] = dict(
        base["paths"],
        raw=f"{data}/generated",
        processed=f"{data}/processed",
        patch_index=f"{data}/patch_index",
        training=f"{output}/training",
        holdout_test=f"{output}/holdout_test",
        forecasts=f"{output}/forecasts",
        fakebench=f"{output}/fakebench",
        logs=f"{output}/logs",
    )
    config["data"] = dict(base["data"], generated_dates=[date])
    training = merge(merge(base["training"], TRAINING_DEFAULTS), recipe["training"])
    # Only ever its own checkpoint, in its own folder.
    training["resume"] = "auto"
    config["training"] = training
    start = start_checkpoint(recipe, base)
    # The holdout test evaluates the annealed model on the main run's sealed
    # holdout tables, so results compare directly with the main run's.
    config["holdout"] = dict(
        base.get("holdout", {}),
        split_checkpoint=start,
        split_config=recipe["base_config"],
    )
    config["holdout"].pop("checkpoint", None)
    config["annealing"] = {
        "experiment": recipe["experiment"],
        "description": recipe["description"],
        "set": recipe["set"]["name"],
        "base_config": recipe["base_config"],
        "start": dict(recipe["start"], checkpoint=start),
    }
    return config


def prepare_experiment(recipe, base, date):
    """Create the experiment folder, or return the existing one if it is the same.

    Returns (directory, config path, created). The same name with any other
    setting is an error, so an experiment's results always belong to exactly
    the configuration saved next to them.
    """
    directory = experiment_directory(recipe["experiment"])
    config = experiment_config(recipe, base, date)
    path = directory / "config.json"
    if directory.exists():
        if not path.is_file():
            raise FileExistsError(
                f"{directory} exists but holds no config.json; choose a new "
                "experiment name or remove the folder yourself after checking it"
            )
        if json.loads(path.read_text(encoding="utf-8")) != config:
            raise FileExistsError(
                f"Annealing experiment {recipe['experiment']!r} already exists with "
                "different settings. Choose a new experiment name; existing "
                "experiments are never overwritten."
            )
        return directory, path, False
    directory.mkdir(parents=True)
    (directory / "recipe.json").write_text(
        json.dumps(
            dict(recipe, created_utc=datetime.now(timezone.utc).isoformat()), indent=2
        ),
        encoding="utf-8",
    )
    path.write_text(json.dumps(config, indent=2), encoding="utf-8")
    return directory, path, True


def experiment_config_path(name):
    path = experiment_directory(_name(name, "experiment")) / "config.json"
    if not path.is_file():
        raise FileNotFoundError(
            f"No annealing experiment {name!r}: {path} does not exist. Run "
            "build_annealing_set.py with its recipe first."
        )
    return path

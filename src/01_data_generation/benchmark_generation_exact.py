"""Exact before/after audit using a copy of the generator taken before editing.

Set REFERENCE_SOURCE to that directory (including sibling modules), then run
this file. The production entry point is only imported for its settings; its
main() and multi-million-table generate() call are never executed.
"""

import ast
import hashlib
import importlib
import json
import platform
import statistics
import sys
import tempfile
import time
import types
from pathlib import Path

import generate_operational_data as settings
import numpy as np
import pandas as pd
from scm_generator import OperationalSCMGenerator

REFERENCE_SOURCE = None
TABLES = 128
REPEATS = 3
PLOT_TABLES = 3
OUTPUT = (
    Path(__file__).resolve().parents[2]
    / "results/data_generation/generation_exact_benchmark.json"
)


def runner_options():
    """Read keyword-to-setting mappings without running the production job."""
    tree = ast.parse(Path(settings.__file__).read_text(encoding="utf-8"))
    main = next(
        n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == "main"
    )
    call = main.body[0].value
    if not isinstance(call, ast.Call) or call.func.id != "OperationalSCMGenerator":
        raise ValueError("Update this audit for the runner's new constructor layout")
    options = {}
    for keyword in call.keywords:
        if not isinstance(keyword.value, ast.Name):
            raise ValueError("Runner options must refer to named top-level settings")
        options[keyword.arg] = getattr(settings, keyword.value.id)
    # Keep the distribution, burn-in, history, rejection and recording settings.
    # Time one worker's work, independently of OS process scheduling/startup.
    options.update(generation_workers=1, save_plot=False, save_dag=False, save_scm=True)
    return options


def reference_class(source):
    package = types.ModuleType("_exact_generation_reference")
    package.__path__ = [str(source)]
    sys.modules[package.__name__] = package
    return importlib.import_module(
        package.__name__ + ".scm_generator"
    ).OperationalSCMGenerator


def assert_exact(left, right):
    """Check schema, values, null positions AND IEEE floating-point bit patterns."""
    pd.testing.assert_frame_equal(left, right, check_exact=True)
    for column in left:
        if left[column].to_numpy().tobytes() != right[column].to_numpy().tobytes():
            raise AssertionError(f"Different bit patterns in {column}")


def compare_files(before, after):
    if {p.name for p in before.iterdir()} != {p.name for p in after.iterdir()}:
        raise AssertionError("Different generated artifacts")
    for path in before.iterdir():
        candidate = after / path.name
        if path.read_bytes() != candidate.read_bytes():
            raise AssertionError(f"Different artifact bytes: {path.name}")
        if path.suffix == ".parquet":
            assert_exact(pd.read_parquet(path), pd.read_parquet(candidate))


def verify_effects(reference):
    """Force rare operators too; a random batch may never draw one of them."""
    inputs = [-1000.0, -3.0, -0.0, 0.0, 0.01, 1.0, 4.0, 1000.0] * 16
    programs = []
    for op in reference.OPERATORS:
        generator = reference(
            "unused",
            seed=17,
            composite_effect_probability=0,
            effect_weights={name: float(name == op) for name in reference.OPERATORS},
        )
        programs.append(generator._expression())
    generator = reference("unused", seed=19, composite_effect_probability=1)
    programs.extend(generator._expression() for _ in range(32))
    for program in programs:
        before, after = {}, {}
        saved = json.dumps(program)
        for value in inputs:
            old = reference._evaluate(program, value, before)
            new = OperationalSCMGenerator._evaluate(program, value, after)
            if float(old).hex() != float(new).hex():
                raise AssertionError("An effect's exact response changed")
        if json.dumps(program) != saved:
            raise AssertionError("Evaluating an effect mutated its saved specification")
    return len(programs)


def verify_scenarios(reference, options):
    """Exercise uncommon process/recording paths without changing their laws."""
    scenarios = [
        {"process_episode_probability": 1.0, "process_priming_probability": 1.0},
        {
            "switching_probability": 1.0,
            "process_episode_probability": 0.0,
            "startup_transient_probability": 1.0,
        },
        {"formula_covariate_probability": 1.0},
        {"discrete_covariate_probability": 1.0, "formula_covariate_probability": 0.0},
        {
            "sensor_bounds_probability": 1.0,
            "measurement_probability": 1.0,
            "missingness_probability": 1.0,
            "redundant_sensor_probability": 1.0,
        },
        {
            "full_length_probability": 1.0,
            "prehistory_probability": 1.0,
            "row_gap_probability": 1.0,
        },
        {"multiplicative_trend_probability": 1.0, "lifecycle_trend_probability": 0.0},
        {"variance_dynamics_probability": 1.0, "slow_dynamics_probability": 1.0},
    ]
    scenarios.extend(
        {
            "plant_probability": 1.0,
            "plant_weights": {
                key: float(key == family) for key in options["plant_weights"]
            },
        }
        for family in options["plant_weights"]
    )
    count = 0
    for index, overrides in enumerate(scenarios):
        config = {**options, "seed": 700 + index, **overrides}
        before = reference("unused", **config)
        after = OperationalSCMGenerator("unused", **config)
        for draw in range(4):
            model = before.sample_model(trend_free=bool(draw % 2))
            saved = json.dumps(model)
            assert_exact(before.simulate(model), after.simulate(model))
            if json.dumps(model) != saved:
                raise AssertionError("Simulation mutated its saved SCM")
            count += 1
    return count


def main():
    if REFERENCE_SOURCE is None:
        raise ValueError(
            "Set REFERENCE_SOURCE to a pre-edit copy of the generator directory"
        )
    source = Path(REFERENCE_SOURCE).resolve()
    if source == Path(__file__).resolve().parent:
        raise ValueError("Reference and candidate source directories must differ")
    reference = reference_class(source)
    options = runner_options()
    # A fixed independent audit seed also makes runs repeatable if SEED=None.
    options["seed"] = 1
    timings = {"reference": [], "optimized": []}
    effect_programs = verify_effects(reference)
    scenario_models = verify_scenarios(reference, options)
    pairs_by_variant = {}
    output_count = None
    with tempfile.TemporaryDirectory(prefix="basin_exact_audit_") as temporary:
        root = Path(temporary)
        for repeat in range(REPEATS):
            variants = [
                ("reference", reference),
                ("optimized", OperationalSCMGenerator),
            ]
            for name, cls in variants[:: 1 if repeat % 2 == 0 else -1]:
                generator = cls(root, **options)
                directory = root / f"{name}_{repeat}"
                directory.mkdir()
                pairs = []
                started = time.perf_counter()
                for index in range(TABLES):
                    model, frame = generator._sample_accepted_model(bool(index % 2))
                    path = directory / f"table_{index:06d}"
                    generator._write_table(model, frame, path)
                    generator._write_extra_copies(model, frame, path)
                    pairs.append((model, frame))
                timings[name].append(time.perf_counter() - started)
                pairs_by_variant[name] = pairs
                print(
                    f"{repeat + 1}/{REPEATS} {name}: {timings[name][-1]:.3f}s",
                    flush=True,
                )
            for index, ((old_model, old_frame), (model, frame)) in enumerate(
                zip(
                    pairs_by_variant["reference"],
                    pairs_by_variant["optimized"],
                    strict=True,
                )
            ):
                if old_model != model:
                    raise AssertionError(f"SCM/screening changed for table {index}")
                assert_exact(old_frame, frame)
            before, after = root / f"reference_{repeat}", root / f"optimized_{repeat}"
            if {p.name for p in before.iterdir()} != {p.name for p in after.iterdir()}:
                raise AssertionError("Augmentation choices changed")
            for path in before.iterdir():
                compare_files(path, after / path.name)
            output_count = len(list(before.iterdir()))

        # Plot bytes also have to agree; keep rendering outside the timed batch
        # because the production runner currently disables both kinds of plots.
        for index, (model, frame) in enumerate(
            pairs_by_variant["reference"][:PLOT_TABLES]
        ):
            directories = []
            for name, cls in (
                ("reference", reference),
                ("optimized", OperationalSCMGenerator),
            ):
                generator = cls(root, **options)
                directory = root / f"plot_{name}_{index}"
                directory.mkdir()
                generator._plot(frame, directory / "series.png")
                generator._plot_dag(model, directory / "dag.png")
                directories.append(directory)
            compare_files(*directories)

    medians = {name: statistics.median(values) for name, values in timings.items()}
    hashes = {}
    for name, directory in (
        ("reference", source),
        ("optimized", Path(__file__).parent),
    ):
        hashes[name] = {
            path.name: hashlib.sha256(path.read_bytes()).hexdigest()
            for path in sorted(directory.glob("*.py"))
        }
    report = {
        "python": sys.version,
        "platform": platform.platform(),
        "numpy": np.__version__,
        "pandas": pd.__version__,
        "reference_source": str(source),
        "source_sha256": hashes,
        "options": options,
        "original_tables": TABLES,
        "saved_tables_including_copies": output_count,
        "repeats": REPEATS,
        "seconds": timings,
        "median_seconds": medians,
        "time_reduction_percent": 100
        * (1 - medians["optimized"] / medians["reference"]),
        "throughput_speedup": medians["reference"] / medians["optimized"],
        "plot_tables": PLOT_TABLES,
        "effect_programs_checked": effect_programs,
        "additional_scenario_models_checked": scenario_models,
        "checks": [
            "identical SCM parameters, quality scores and rejection counts",
            "identical frame schema, IEEE value bits and missing-value positions",
            "identical original/augmented Parquet, SCM and provenance bytes",
            "identical series and DAG PNG bytes",
            "exact responses for every effect operator and 32 composite programs",
            "exact replay of forced process, plant, history and recording scenarios",
        ],
    }
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    OUTPUT.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(f"Time reduction: {report['time_reduction_percent']:.1f}%; report: {OUTPUT}")


if __name__ == "__main__":
    main()

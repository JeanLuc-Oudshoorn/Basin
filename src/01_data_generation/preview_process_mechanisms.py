"""Render actual SCM examples of process episodes, sensor limits and categories."""

import json
from pathlib import Path

from matplotlib.backends.backend_agg import FigureCanvasAgg
from matplotlib.figure import Figure
from scm_generator import OperationalSCMGenerator

OUTPUT_DIR = Path(__file__).resolve().parents[2] / "results" / "data_generation"
ROWS = 256
SEED = 31


def discrete_example(family, minimum_levels):
    generator = OperationalSCMGenerator(
        "unused",
        seed=SEED,
        min_rows=ROWS,
        max_rows=ROWS,
        min_covariates=8,
        max_covariates=8,
        process_episode_probability=0,
        switching_probability=0,
        discrete_covariate_probability=1,
        discrete_weights={
            name: int(name == family)
            for name in OperationalSCMGenerator.DEFAULT_DISCRETE_WEIGHTS
        },
    )
    for _ in range(30):
        model = generator.sample_model(trend_free=True, visible_covariate_count=8)
        for node in model["nodes"]:
            node["visible"] = True
        model["univariate"] = False
        frame = generator.simulate(model)
        for node in model["nodes"]:
            if (
                node.get("family") == family
                and frame[node["name"]].nunique() >= minimum_levels
            ):
                return model, frame, node["name"]
    raise RuntimeError(f"Could not draw an illustrative {family} example")


def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    generator = OperationalSCMGenerator(
        "unused",
        seed=SEED,
        min_rows=ROWS,
        max_rows=ROWS,
        process_episode_probability=1,
        sensor_bounds_probability=0,
    )
    episode = generator.sample_model(trend_free=True)
    # An illustrative saved seasonal equation demonstrates that the run
    # envelope gates an oscillating process, rather than replacing it by a hump.
    episode["outcome"]["seasons"] = [
        {
            "period": 18.0,
            "phase": 0.0,
            "amplitude": 0.65,
            "modulation": 0.0,
            "modulation_period": float(ROWS),
        }
    ]
    episode["outcome"]["operation"]["seasonal_fraction"] = 0.25
    episode_frame = generator.simulate(episode)
    # Bounds are fixed from saved scale/level, never inferred from observations.
    generator = OperationalSCMGenerator(
        "unused",
        seed=SEED,
        min_rows=ROWS,
        max_rows=ROWS,
        process_episode_probability=0,
        switching_probability=0,
        sensor_bounds_probability=0,
    )
    for _ in range(40):
        capped = generator.sample_model(trend_free=True)
        spec = capped["outcome"]
        spec["sensor"] = {
            "floor": spec["baseline"] - spec["scale"],
            "ceiling": spec["baseline"] + spec["scale"],
        }
        cap_frame = generator.simulate(capped)
        levels = cap_frame.outcome.to_numpy()
        inside = (levels > spec["sensor"]["floor"]) & (
            levels < spec["sensor"]["ceiling"]
        )
        if inside.any() and not inside.all():
            break
    else:
        raise RuntimeError("Could not draw an illustrative clipped sensor")
    bernoulli, bernoulli_frame, bname = discrete_example("bernoulli", 2)
    binary, binary_frame, biname = discrete_example("binary_markov", 2)
    multi, multi_frame, mname = discrete_example("categorical_markov", 3)
    figure = Figure(figsize=(13, 8), layout="constrained")
    FigureCanvasAgg(figure)
    axes = figure.subplots(2, 2).ravel()
    axes[0].plot(episode_frame.time_index, episode_frame.outcome, color="#214e80")
    axes[0].axhline(
        episode["outcome"]["operation"]["idle_value"],
        color="#7b8794",
        linestyle="--",
        linewidth=1,
    )
    axes[0].set_title("Single process run: idle → startup → work → shutdown → idle")
    axes[1].plot(cap_frame.time_index, cap_frame.outcome, color="#214e80")
    for label, value in capped["outcome"]["sensor"].items():
        axes[1].axhline(
            value, linestyle="--", color="#c45b42", linewidth=1, label=label
        )
    axes[1].legend()
    axes[1].set_title("Hard sensor floor and ceiling")
    axes[2].step(
        bernoulli_frame.time_index,
        bernoulli_frame[bname] + 1.5,
        where="post",
        label="Bernoulli (+1.5)",
        alpha=0.75,
    )
    axes[2].step(
        binary_frame.time_index,
        binary_frame[biname],
        where="post",
        label="Persistent binary Markov",
    )
    axes[2].legend()
    axes[2].set_title("Independent binary draws versus persistent states")
    axes[3].step(
        multi_frame.time_index, multi_frame[mname], where="post", color="#318b72"
    )
    axes[3].set_title("Random multi-state Markov covariate")
    for axis in axes:
        axis.set_xlabel("Time step")
        axis.set_ylabel("Recorded value")
        axis.grid(alpha=0.15)
    figure.suptitle(
        "Operational process mechanisms · all examples generated from saved SCMs"
    )
    figure.savefig(OUTPUT_DIR / "process_mechanisms.png", dpi=150)
    figure.clear()
    models = {
        "episode": episode,
        "sensor": capped,
        "bernoulli": bernoulli,
        "binary_markov": binary,
        "categorical_markov": multi,
    }
    (OUTPUT_DIR / "process_mechanisms.json").write_text(
        json.dumps(models, indent=2), encoding="utf-8"
    )
    print(OUTPUT_DIR / "process_mechanisms.png")


if __name__ == "__main__":
    main()

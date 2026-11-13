"""Inspect actual SCM outcomes driven by asymmetric exogenous machine programs."""

import json
from pathlib import Path

from matplotlib.backends.backend_agg import FigureCanvasAgg
from matplotlib.figure import Figure
from process_mechanisms import FORMULA_FAMILIES, formula_path
from scm_generator import OperationalSCMGenerator

OUTPUT_DIR = Path(__file__).resolve().parents[2] / "results" / "data_generation"
ROWS = 512
SEED = 93
MAX_PROPOSALS = 400


def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    criteria = {
        "Fast startup, slow release": lambda d: (
            d["transitions"][-1]["steps"] > 4 * d["transitions"][0]["steps"]
        ),
        "Slow startup, fast release": lambda d: (
            d["transitions"][0]["steps"] > 4 * d["transitions"][-1]["steps"]
        ),
        "Long idle before operation": lambda d: (
            d["start"] > 4 * max(1, d["idle_after"])
        ),
        "Long idle after operation": lambda d: d["idle_after"] > 4 * max(1, d["start"]),
        "Multiple operating settings": lambda d: len(d["transitions"]) >= 4,
        "Startup priming / mild jitters": lambda d: bool(d["priming"]),
    }
    saved = {}
    figure = Figure(figsize=(15, 10), layout="constrained")
    FigureCanvasAgg(figure)
    for index, (axis, (title, matches)) in enumerate(
        zip(figure.subplots(3, 2).ravel(), criteria.items())
    ):
        generator = OperationalSCMGenerator(
            "unused",
            seed=SEED + index,
            min_rows=ROWS,
            max_rows=ROWS,
            min_covariates=3,
            max_covariates=5,
            process_episode_probability=1,
            sensor_bounds_probability=0,
        )
        for _ in range(MAX_PROPOSALS):
            model = generator.sample_model(
                trend_free=False, visible_covariate_count=100
            )
            driver = model["nodes"][1]
            if not matches(driver):
                continue
            frame = generator.simulate(model)
            if generator.trend_strength(frame, model) >= 0.4:
                break
        else:
            raise RuntimeError(f"Could not sample a screened example: {title}")
        op = model["outcome"]["operation"]
        activity = formula_path(driver, frame.time_index.to_numpy() - op["lag"])
        axis.plot(frame.time_index, frame.outcome, linewidth=1, label="Outcome")
        axis.plot(
            frame.time_index,
            op["idle_value"] + op["amplitude"] * activity,
            color="#d1743b",
            alpha=0.65,
            linewidth=1,
            linestyle="--",
            label="Scaled SCM driver",
        )
        axis.set(title=title, xlabel="Sample", ylabel="Value")
        axis.grid(alpha=0.15)
        axis.legend(fontsize=8)
        saved[title] = model
    figure.suptitle(
        "Actual SCM runs: exogenous setpoints → causal process response; no post-hoc reshaping"
    )
    figure.savefig(OUTPUT_DIR / "process_drivers.png", dpi=130)
    figure.clear()

    generator = OperationalSCMGenerator(
        "unused",
        seed=SEED,
        min_rows=ROWS,
        max_rows=ROWS,
        min_covariates=6,
        max_covariates=6,
        process_episode_probability=0,
        switching_probability=0,
        formula_covariate_probability=1,
    )
    figure = Figure(figsize=(15, 10), layout="constrained")
    FigureCanvasAgg(figure)
    for axis, family in zip(figure.subplots(3, 3).ravel(), FORMULA_FAMILIES):
        for _ in range(MAX_PROPOSALS):
            model = generator.sample_model(trend_free=True, visible_covariate_count=100)
            node = next((n for n in model["nodes"] if n.get("family") == family), None)
            if node is not None:
                break
        else:
            raise RuntimeError(f"Could not sample formula family: {family}")
        frame = generator.simulate(model)
        axis.plot(
            frame.time_index,
            frame[node["name"]],
            linewidth=0.9,
            label="Recorded source",
        )
        axis.plot(
            frame.time_index,
            formula_path(node, frame.time_index),
            linewidth=1,
            alpha=0.65,
            linestyle="--",
            color="#d1743b",
            label="Formula",
        )
        axis.set(title=family.replace("_", " "), xlabel="Sample", ylabel="Source value")
        axis.grid(alpha=0.15)
        saved[f"formula_{family}"] = model
    figure.suptitle("Formulaic exogenous covariates, with optional saved noise")
    figure.savefig(OUTPUT_DIR / "formula_sources.png", dpi=130)
    figure.clear()
    (OUTPUT_DIR / "process_drivers.json").write_text(
        json.dumps(saved, indent=2), encoding="utf-8"
    )
    print(OUTPUT_DIR / "process_drivers.png")
    print(OUTPUT_DIR / "formula_sources.png")


if __name__ == "__main__":
    main()

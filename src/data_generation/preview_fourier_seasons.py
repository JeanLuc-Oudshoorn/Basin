"""Render sampled seasonal equations in isolation to inspect the shape prior."""

import json
from pathlib import Path

import numpy as np
from matplotlib.backends.backend_agg import FigureCanvasAgg
from matplotlib.figure import Figure
from scm_generator import OperationalSCMGenerator

OUTPUT_DIR = Path(__file__).resolve().parents[2] / "results" / "data_generation"
SEED = 23
PERIOD = 96
CYCLES = 4
VARIANTS = 3


def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    families = OperationalSCMGenerator.DEFAULT_FOURIER_WEIGHTS
    figure = Figure(figsize=(14, 13), layout="constrained")
    FigureCanvasAgg(figure)
    axes = figure.subplots(3, 3).ravel()
    time = np.arange(PERIOD * CYCLES)
    saved = []
    for axis, family in zip(axes, families):
        generator = OperationalSCMGenerator(
            "unused",
            seed=SEED,
            fourier_weights={key: int(key == family) for key in families},
            seasonal_shape_drift_probability=0,
        )
        for variant in range(VARIANTS):
            season = {
                "period": PERIOD,
                "phase": 0.0,
                "amplitude": 1.0,
                "modulation": 0.0,
                "modulation_period": len(time),
            }
            season["fourier"] = generator._sample_fourier(season, len(time))
            axis.plot(
                time / PERIOD,
                generator._seasonal([season], time),
                linewidth=1,
                alpha=0.8,
                label=f"Sample {variant + 1}",
            )
            saved.append(season)
        axis.set(
            title=family.replace("_", " "),
            xlabel="Cycles",
            ylabel="Seasonal contribution",
        )
        axis.grid(alpha=0.2)
    figure.suptitle(
        "Sampled Fourier seasonal shapes\nIsolated equations; full SCMs also add changing amplitude, timing, noise and causal influences"
    )
    figure.savefig(OUTPUT_DIR / "fourier_seasons.png", dpi=140)
    (OUTPUT_DIR / "fourier_seasons.json").write_text(
        json.dumps(saved, indent=2), encoding="utf-8"
    )
    print(OUTPUT_DIR / "fourier_seasons.png")


if __name__ == "__main__":
    main()

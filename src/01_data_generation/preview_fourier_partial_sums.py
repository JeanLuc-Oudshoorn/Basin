"""Illustrate exact square-wave partial sums using the generator's evaluator."""

import json
from pathlib import Path

import numpy as np
from matplotlib.backends.backend_agg import FigureCanvasAgg
from matplotlib.figure import Figure
from scm_generator import OperationalSCMGenerator

# These are illustrative equations, rather than a change to the sampling prior.
OUTPUT_DIR = Path(__file__).resolve().parents[2] / "results" / "01_data_generation"
PERIOD = 96
CYCLES = 2
PARTIAL_SUM_TERMS = (1, 2, 3, 6)


def square_season(terms):
    """Six nonzero terms use harmonic orders 1, 3, 5, 7, 9 and 11.

    Keep textbook 4/(pi*k) coefficients here. The random generator normally
    normalizes spectral energy to vary complexity independently of amplitude.
    Both conventions are supported by the same saved-equation evaluator.
    """
    orders = np.arange(1, 2 * terms, 2)
    if orders[-1] / PERIOD >= 0.5:
        raise ValueError("Increase PERIOD to resolve the highest harmonic")
    return {
        "period": PERIOD,
        "phase": 0.0,
        "amplitude": 1.0,
        "modulation": 0.0,
        "modulation_period": PERIOD * CYCLES,
        "fourier": {
            "family": "pulse",
            "orders": orders.tolist(),
            "start": {
                "cos": np.zeros(terms).tolist(),
                "sin": (4 / (np.pi * orders)).tolist(),
            },
        },
    }


def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    time = np.arange(PERIOD * CYCLES)
    phase = 2 * np.pi * time / PERIOD
    seasons = [square_season(terms) for terms in PARTIAL_SUM_TERMS]
    figure = Figure(figsize=(12, 10), layout="constrained")
    FigureCanvasAgg(figure)
    axes = figure.subplots(3, 1)
    target = np.where(np.sin(phase) >= 0, 1.0, -1.0)
    axes[0].step(
        time / PERIOD,
        target,
        where="post",
        color="black",
        alpha=0.3,
        linestyle="--",
        label="Ideal square wave",
    )
    for terms, season in zip(PARTIAL_SUM_TERMS, seasons):
        axes[0].plot(
            time / PERIOD,
            OperationalSCMGenerator._seasonal([season], time),
            label=f"{terms} nonzero term(s)",
        )
    axes[0].set_title(
        "Partial sums approach a square wave; overshoot near jumps remains"
    )
    axes[0].legend(ncol=3)
    six = square_season(6)
    spectrum = six["fourier"]
    for order, coefficient in zip(spectrum["orders"], spectrum["start"]["sin"]):
        axes[1].plot(
            time / PERIOD,
            coefficient * np.sin(order * phase),
            linewidth=0.8,
            alpha=0.6,
            label=f"k={order}",
        )
    axes[1].plot(
        time / PERIOD,
        OperationalSCMGenerator._seasonal([six], time),
        color="black",
        linewidth=2,
        label="Sum",
    )
    axes[1].set_title("Six sine components and their sum, evaluated by the generator")
    axes[1].legend(ncol=7)
    amplitudes = np.zeros(12)
    amplitudes[np.asarray(spectrum["orders"])] = spectrum["start"]["sin"]
    axes[2].stem(np.arange(12), amplitudes)
    axes[2].set(
        title="Amplitude spectrum: only odd harmonics, proportional to 1/k",
        xlabel="Harmonic order k",
        ylabel="Sine amplitude",
    )
    for axis in axes[:2]:
        axis.set(xlabel="Cycles", ylabel="Value")
    for axis in axes:
        axis.grid(alpha=0.2)
    figure.savefig(OUTPUT_DIR / "fourier_partial_sums.png", dpi=150)
    figure.clear()
    (OUTPUT_DIR / "fourier_partial_sums.json").write_text(
        json.dumps(seasons, indent=2), encoding="utf-8"
    )
    print(OUTPUT_DIR / "fourier_partial_sums.png")


if __name__ == "__main__":
    main()

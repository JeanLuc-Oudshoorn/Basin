"""Render source-controlled architecture illustrations as PNG and editable SVG."""

import hashlib
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch, Rectangle

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "docs/model_training/figures"
CONFIG = json.loads((ROOT / "configs/config.json").read_text())
MODEL = CONFIG["model"]
P, D, L, H = (MODEL[k] for k in ("patch_size", "width", "layers", "variate_heads"))
BG, INK, MUTED, LINE = "#f5f7fb", "#15263d", "#52657e", "#d9e2ed"
BLUE, TEAL, PURPLE, AMBER = "#2364c4", "#087f83", "#7654b5", "#ac6412"
PALE = {BLUE: "#eaf1fd", TEAL: "#e5f5f2", PURPLE: "#f1ecfa", AMBER: "#fff2da"}
plt.rcParams.update({"font.family": "DejaVu Sans", "svg.fonttype": "none"})


def label(ax, x, y, text, size=11, color=INK, weight="normal", ha="left", va="center"):
    return ax.text(
        x,
        y,
        text,
        fontsize=size,
        color=color,
        weight=weight,
        ha=ha,
        va=va,
        linespacing=1.45,
        zorder=5,
    )


def arrow(ax, start, end, color=MUTED, style="-|>", bend=0, lw=1.7):
    ax.add_patch(
        FancyArrowPatch(
            start,
            end,
            arrowstyle=style,
            connectionstyle=f"arc3,rad={bend}",
            mutation_scale=13,
            color=color,
            lw=lw,
            zorder=3,
        )
    )


def box(ax, x, y, w, h, title="", detail="", color=BLUE, title_size=13):
    ax.add_patch(
        FancyBboxPatch(
            (x, y),
            w,
            h,
            boxstyle="round,pad=0.0,rounding_size=0.8",
            facecolor=PALE[color],
            edgecolor=color,
            lw=1.2,
            zorder=2,
        )
    )
    if title:
        label(
            ax,
            x + w / 2,
            y + h * (0.69 if detail else 0.5),
            title,
            title_size,
            color,
            "bold",
            ha="center",
        )
    if detail:
        label(ax, x + w / 2, y + h * 0.30, detail, 10.5, ha="center")


def panel(ax, x, y, w, h):
    ax.add_patch(
        FancyBboxPatch(
            (x, y),
            w,
            h,
            boxstyle="round,pad=0.0,rounding_size=1.2",
            facecolor="white",
            edgecolor=LINE,
            lw=1.0,
            zorder=0,
        )
    )


def canvas(number, title, subtitle):
    fig, ax = plt.subplots(figsize=(16, 10.4), facecolor=BG)
    fig.subplots_adjust(left=0, right=1, top=1, bottom=0)
    ax.set(xlim=(0, 100), ylim=(0, 65))
    ax.axis("off")
    label(ax, 4, 61.7, f"0{number}  /  BASIN ARCHITECTURE", 10, MUTED, "bold")
    label(ax, 4, 58.3, title, 25, weight="bold")
    label(ax, 4, 54.7, subtitle, 11.5, MUTED)
    label(
        ax,
        4,
        2,
        "Implementation diagrams  •  next-patch objective  •  generated from configs/config.json",
        9,
        MUTED,
    )
    return fig, ax


def save(fig, stem):
    OUT.mkdir(parents=True, exist_ok=True)
    for suffix in ("png", "svg"):
        fig.savefig(OUT / f"{stem}.{suffix}", dpi=200, facecolor=BG)
    plt.close(fig)
    print(f"Saved {OUT / (stem + '.png')}")


def overview():
    fig, ax = canvas(
        1,
        "From observations to a forecast",
        f"One table has parallel variable streams. A token represents {P} consecutive observations of ONE variable.",
    )
    columns = [(3, 22), (28, 21), (52, 23), (78, 19)]
    for x, w in columns:
        panel(ax, x, 16, w, 35)
    for i, ((x, w), title) in enumerate(
        zip(
            columns,
            ("OBSERVATIONS", "PATCH EMBEDDINGS", "HYBRID BACKBONE", "PREDICTIONS"),
        ),
        1,
    ):
        label(ax, x + 1.4, 48.5, f"{i}   {title}", 10, MUTED, "bold")

    # Three illustrative aligned streams and patch boundaries.
    signal_x = np.linspace(4.6, 23.4, 96)
    for row, (name, color) in enumerate(
        (("Outcome", BLUE), ("Covariate A", TEAL), ("Covariate B", PURPLE))
    ):
        y = 41 - row * 8
        label(ax, 4.5, y + 3, name, 11, color, "bold")
        for j in range(3):
            ax.add_patch(
                Rectangle(
                    (4.6 + j * 6.27, y - 1.5),
                    6.0,
                    3.7,
                    facecolor=PALE[color],
                    edgecolor="none",
                    zorder=1,
                )
            )
        signal = np.sin(np.linspace(0, 3.8 * np.pi, 96) + row) * 0.95
        signal += np.cos(np.linspace(0, 8 * np.pi, 96)) * 0.23
        ax.plot(signal_x, y + signal, color=color, lw=1.4, zorder=2)
    for j in range(3):
        label(ax, 7.6 + j * 6.27, 20.7, f"P{j + 1}", 10, MUTED, ha="center")
    label(ax, 14, 18.7, f"{P} values per patch", 10, MUTED, ha="center")
    arrow(ax, (25.3, 34), (27.5, 34))

    box(
        ax,
        29.5,
        36.5,
        18,
        7.6,
        "Build patch features",
        f"{P} values + {P} masks + {P} times",
        PURPLE,
    )
    label(ax, 38.5, 33.6, f"{3 * P} numeric features / patch", 10.5, MUTED, ha="center")
    arrow(ax, (38.5, 32), (38.5, 30), PURPLE)
    box(
        ax,
        29.5,
        22,
        18,
        7.6,
        "Learned projection",
        f"{3 * P} → {D} dimensions\n+ target / covariate role",
        PURPLE,
    )
    label(ax, 38.5, 18.5, "Optimized during training", 9.8, PURPLE, ha="center")
    arrow(ax, (49.2, 34), (51.5, 34))

    label(ax, 63.5, 44.6, f"Repeat {L} times", 12, weight="bold", ha="center")
    box(ax, 54, 34.5, 19, 7.5, "RWKV-7 block", "Each variable through time", BLUE)
    arrow(ax, (63.5, 34), (63.5, 31.8), BLUE)
    box(
        ax,
        54,
        23.5,
        19,
        8,
        "Variable self-attention",
        f"Mix variables at each patch\n{H} heads × {D // H} dimensions",
        TEAL,
        11.5,
    )
    label(ax, 63.5, 19.3, "[batch, variables, patches, width]", 9.2, MUTED, ha="center")
    arrow(ax, (75.3, 34), (77.5, 34))

    box(
        ax,
        79.3,
        35,
        16.4,
        9,
        "Target output head",
        f"LayerNorm + projection\n{D} → {P * len(MODEL['quantiles'])} values",
        AMBER,
        11.5,
    )
    arrow(ax, (87.5, 34.5), (87.5, 31.5), AMBER)
    box(
        ax,
        79.3,
        22,
        16.4,
        9,
        "Next target patch",
        f"{P} observations\n{len(MODEL['quantiles'])} quantiles per observation",
        AMBER,
        11.5,
    )
    label(ax, 87.5, 18.8, "Median = point forecast", 9.8, AMBER, ha="center")

    panel(ax, 3, 5, 94, 8.5)
    label(ax, 5, 10.8, "Shapes in the example", 11, weight="bold")
    label(
        ax,
        5,
        7.6,
        f"3 variables × 3 real patches × {D} hidden values\nB = number of independent tables in a batch",
        10.5,
        MUTED,
    )
    label(
        ax,
        54,
        10.8,
        f"Context: final {MODEL['context_length']} observations",
        11,
        weight="bold",
    )
    label(
        ax,
        54,
        7.6,
        "Short inputs are padded.\nMissing labels do not contribute to loss.",
        10.5,
        MUTED,
    )
    save(fig, "01_observations_to_forecast")


def token(ax, x, y, text, color, w=7.8, h=4.4):
    box(ax, x, y, w, h, text, color=color, title_size=10)


def axes_of_mixing():
    fig, ax = canvas(
        2,
        "Two axes, two different operations",
        "The RWKV block moves information forward in time. Self-attention exchanges information between variables.",
    )
    panel(ax, 3, 22, 58, 29)
    panel(ax, 64, 22, 33, 29)
    label(ax, 5, 48, "A   RWKV: temporal processing", 15, BLUE, "bold")
    label(
        ax, 5, 44.8, "One causal stream per variable; shared RWKV weights.", 10.5, MUTED
    )
    xs = [15, 26, 37, 48]
    names = ["Outcome", "Covariate A", "Covariate B"]
    colors = [BLUE, TEAL, PURPLE]
    ys = [37, 31, 25]
    ax.add_patch(
        FancyBboxPatch(
            (47, 24),
            9.8,
            19,
            boxstyle="round,pad=0,rounding_size=.7",
            facecolor="#fff5df",
            edgecolor="#debd74",
            lw=1,
            zorder=1,
        )
    )
    for j, x in enumerate(xs):
        label(ax, x + 3.9, 42, f"Patch {j + 1}", 10, MUTED, ha="center")
    for name, color, y in zip(names, colors, ys):
        label(ax, 5, y + 2.2, name, 10, color, "bold")
        for j, x in enumerate(xs):
            token(ax, x, y, f"h{j + 1}", color)
            if j < len(xs) - 1:
                arrow(ax, (x + 8.1, y + 2.2), (xs[j + 1] - 0.3, y + 2.2), BLUE)
    arrow(ax, (56.8, 33.5), (65.3, 33.5), AMBER)

    label(ax, 66, 48, "B   Mix variables at patch 4", 14, TEAL, "bold")
    label(ax, 66, 44.8, "The same operation runs at every patch.", 10, MUTED)
    for color, y, name in zip(colors, (37.5, 31.5, 25.5), names):
        token(ax, 66, y, name, color, w=12.7, h=4)
        arrow(ax, (79, y + 2), (82, 33.5), TEAL, lw=1.1)
        arrow(ax, (91, 33.5), (94, y + 2), TEAL, lw=1.1)
    box(ax, 82, 28.2, 9, 10.6, "Q / K / V", "attention\n+ residual", TEAL, 11)
    label(ax, 94.3, 42, "Outputs", 9, MUTED, ha="center")
    for color, y in zip(colors, (37.5, 31.5, 25.5)):
        ax.add_patch(
            Rectangle(
                (94.1, y + 0.8), 1.1, 2.4, facecolor=color, edgecolor="none", zorder=4
            )
        )

    panel(ax, 3, 5.5, 94, 14)
    label(ax, 5, 16.7, "What the attention matrix connects", 12, weight="bold")
    values = np.array([[0.45, 0.4, 0.15], [0.2, 0.6, 0.2], [0.25, 0.15, 0.6]])
    # Colors show illustrative relevance, not measured or fitted attention weights.
    for r in range(3):
        for c in range(3):
            ax.add_patch(
                Rectangle(
                    (5 + c * 2.7, 7.1 + (2 - r) * 2.2),
                    2.45,
                    1.95,
                    facecolor=plt.cm.GnBu(0.2 + values[r, c]),
                    edgecolor="white",
                )
            )
    label(ax, 14.8, 12.9, "3 × 3 per head in this example", 11, TEAL, "bold")
    label(
        ax,
        14.8,
        9.8,
        "Q, K and V all come from these same three states.\nColors are illustrative; weights are learned during training.",
        10,
        MUTED,
    )
    label(ax, 54, 16.7, "Alternate A → B across all layers", 12, weight="bold")
    label(
        ax,
        54,
        12.3,
        "The next RWKV layer carries mixed information forward.\nVariables at a patch can summarize their earlier histories.\nNo attention across tables; no access to future patches.",
        10.5,
        MUTED,
    )
    save(fig, "02_time_and_variable_mixing")


def training_and_forecast():
    fig, ax = canvas(
        3,
        "How tokens become training targets",
        f"P1, P2, … are consecutive target patches of {P} observations. Aligned covariate streams accompany the context.",
    )
    panel(ax, 3, 29, 94, 22)
    label(
        ax,
        5,
        48,
        "A   Training: many next-patch predictions in one causal pass",
        15,
        BLUE,
        "bold",
    )
    xs = [24, 34, 44, 54, 64, 79]
    label(ax, 5, 41.5, "Observed table", 10.5, weight="bold")
    for j, x in enumerate(xs):
        token(ax, x, 39.3, f"P{j + 1}", AMBER if j == 5 else BLUE)
    label(ax, 89.2, 41.5, "Final patch\nreserved", 9.5, AMBER, ha="center")
    label(ax, 5, 34.6, "Training input", 10.5, BLUE, "bold")
    label(ax, 5, 30.9, "Expected output", 10.5, TEAL, "bold")
    for j in range(4):
        box(ax, xs[j], 33.2, 7.8, 2.9, f"P{j + 1}", color=BLUE, title_size=10)
        box(ax, xs[j], 29.5, 7.8, 2.9, f"P{j + 2}", color=TEAL, title_size=10)
        arrow(ax, (xs[j] + 3.9, 33.1), (xs[j] + 3.9, 32.5), TEAL, lw=1)
    label(
        ax,
        70.2,
        33.8,
        "At P4, the model has seen P1…P4.\nP6 contributes no training label.",
        10,
        MUTED,
    )

    panel(ax, 3, 17, 94, 10)
    label(ax, 5, 24, "B   Validation after each epoch", 13, AMBER, "bold")
    box(ax, 5, 18.5, 32, 3.4, "Context: P1  P2  P3  P4  P5", color=BLUE, title_size=11)
    arrow(ax, (37.5, 20.2), (42.5, 20.2), AMBER)
    box(ax, 43, 18.5, 18, 3.4, "Predict P6", color=AMBER, title_size=11)
    label(ax, 64, 21.7, "Score against the actual final patch.", 11, AMBER, "bold")
    label(ax, 64, 18.7, "Repeat for every non-test validation table.", 10, MUTED)

    panel(ax, 3, 5, 94, 10)
    label(ax, 5, 12.2, "C   Forecast on new data", 13, PURPLE, "bold")
    stages = [
        (5, 24, f"Keep final {MODEL['context_length']} values"),
        (34, 19, f"Predict next {P}"),
        (58, 15, "Append median"),
        (78, 17, "Crop and repeat"),
    ]
    for i, (x, w, text) in enumerate(stages):
        box(ax, x, 6.1, w, 3.5, text, color=PURPLE, title_size=10.5)
        if i < len(stages) - 1:
            arrow(ax, (x + w + 0.4, 7.9), (stages[i + 1][0] - 0.4, 7.9), PURPLE)
    label(
        ax,
        50,
        3.9,
        "A separate 10% of admitted tables is held out entirely for final rolling tests, outside A and B.",
        9.8,
        MUTED,
        ha="center",
    )
    save(fig, "03_training_validation_forecasting")


def main():
    overview()
    axes_of_mixing()
    training_and_forecast()
    sources = [
        "configs/config.json",
        "src/04_creating_embeddings/embed.py",
        "src/05_model_training/train.py",
        "src/05_model_training/training_protocol.py",
        "scripts/render_architecture_diagrams.py",
    ]
    (OUT / "manifest.json").write_text(
        json.dumps(
            {
                "model_settings": MODEL,
                "illustration": "Three variables and short sequences are illustrative; dimensions follow config.",
                "sources_sha256": {
                    p: hashlib.sha256((ROOT / p).read_bytes()).hexdigest()
                    for p in sources
                },
                "outputs": sorted(p.name for p in OUT.glob("0*.*")),
            },
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )


if __name__ == "__main__":
    main()

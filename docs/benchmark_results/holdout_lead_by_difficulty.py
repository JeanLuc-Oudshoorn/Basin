"""Basin's lead over the next-best model against table difficulty (holdout)."""

import json
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from scipy.ndimage import gaussian_filter

run, target = Path(sys.argv[1]), Path(sys.argv[2])
tables = json.loads((run / "table_metrics.json").read_text(encoding="utf-8"))
names = ["rwkv", "timestral", "timeh2o", "chronos", "naive"]
m = np.array(
    [[t.get(n, {}).get("mase", np.nan) or np.nan for n in names] for t in tables],
    dtype=float,
)
m = m[np.isfinite(m).all(axis=1)]
order = np.argsort(m[:, 4], kind="stable")
m = m[order]
count = len(m)
rank = (np.arange(count) + 0.5) / count * 100  # difficulty percentile
# Per table: how much lower Basin's MASE is than Timestral's, the runner-up.
lead = 100 * (1 - m[:, 0] / m[:, 1])
print("tables", count)

# Smoothed density of the tables, normalised within each difficulty column so
# that the sparse ends are as visible as the middle.
low, high = -60, 60
grid, _, _ = np.histogram2d(rank, lead, bins=(100, 120), range=((0, 100), (low, high)))
grid = gaussian_filter(grid, sigma=(1.5, 1.5), mode="nearest")
grid = grid / grid.max(axis=1, keepdims=True)

# Median and middle half of the tables, in a sliding window of 4 percentiles.
centres = np.arange(1, 100, 0.5)
bands = np.array(
    [
        np.percentile(lead[(rank >= c - 2) & (rank < c + 2)], [25, 50, 75])
        for c in centres
    ]
)
log_ratio = np.log(m[:, 0] / m[:, 1])
average = np.array(
    [
        100 * (1 - np.exp(log_ratio[(rank >= c - 2) & (rank < c + 2)].mean()))
        for c in centres
    ]
)

# The groups of the table in the document: lead of the group medians.
edges = [0, 10, 25, 50, 75, 90, 95, 100]
groups = []
for a, b in zip(edges, edges[1:]):
    rows = (rank >= a) & (rank < b)
    med = np.median(m[rows], axis=0)
    groups.append((a, b, 100 * (1 - med[0] / med[1:4].min())))
    print(
        a,
        b,
        rows.sum(),
        round(groups[-1][2], 1),
        "median of per-table lead",
        round(np.median(lead[rows]), 1),
        "Basin best on",
        round(100 * np.mean(lead[rows] > 0), 1),
    )

INK, MUTED, GRID, SURFACE, BLUE, ORANGE = (
    "#0b0b0b",
    "#52514e",
    "#e4e3df",
    "#fcfcfb",
    "#2a78d6",
    "#eb6834",
)
plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 10})
fig, ax = plt.subplots(figsize=(9.6, 5.2), dpi=160, facecolor=SURFACE)
ax.set_facecolor(SURFACE)
cmap = matplotlib.colors.LinearSegmentedColormap.from_list(
    "blue", [SURFACE, "#b9d3f1", "#6ea5e3"]
)
ax.imshow(
    grid.T,
    origin="lower",
    extent=(0, 100, low, high),
    aspect="auto",
    cmap=cmap,
    vmin=0.03,
    vmax=1.0,
    zorder=1,
)
ax.axhline(0, color=MUTED, linewidth=1, zorder=2)
ax.plot(centres, bands[:, 0], color=BLUE, linewidth=1, linestyle=(0, (4, 3)), zorder=3)
ax.plot(centres, bands[:, 2], color=BLUE, linewidth=1, linestyle=(0, (4, 3)), zorder=3)
ax.plot(centres, bands[:, 1], color=BLUE, linewidth=2.2, zorder=4)
ax.plot(centres, average, color=ORANGE, linewidth=2.2, zorder=4)
for edge in edges[1:-1]:
    ax.axvline(edge, color=GRID, linewidth=0.8, zorder=1.5)
ax.set_xlim(0, 100)
ax.set_ylim(low, high)
ax.set_xticks(edges)
ax.set_yticks(range(low, high + 1, 20))
ax.set_yticklabels([f"{v:+d}%" if v else "0%" for v in range(low, high + 1, 20)])
ax.set_xlabel(
    "Table difficulty: percentile of the naive forecast's MASE (0 = easiest, 100 = hardest)",
    color=MUTED,
)
ax.set_ylabel("Basin's MASE below Timestral's, per table", color=MUTED)
ax.tick_params(colors=MUTED, length=0)
for spine in ax.spines.values():
    spine.set_visible(False)
ax.text(1.5, 55, "Basin better", color=MUTED, fontsize=9, va="top")
ax.text(1.5, -55, "Basin worse", color=MUTED, fontsize=9, va="bottom")
handles = [
    plt.Line2D(
        [], [], color=BLUE, linewidth=2.2, label="Median table (sliding window)"
    ),
    plt.Line2D(
        [],
        [],
        color=BLUE,
        linewidth=1,
        linestyle=(0, (4, 3)),
        label="Middle half of the tables",
    ),
    plt.Line2D(
        [],
        [],
        color=ORANGE,
        linewidth=2.2,
        label="Average table (geometric mean, sliding window)",
    ),
    matplotlib.patches.Patch(
        color="#8fb9e9", label="Density of the tables at each difficulty"
    ),
]
ax.legend(
    handles=handles,
    loc="lower right",
    frameon=True,
    facecolor=SURFACE,
    edgecolor=GRID,
    fontsize=8.5,
    labelcolor=INK,
)
fig.suptitle(
    "Basin's lead over the runner-up grows with table difficulty",
    x=0.012,
    y=0.975,
    ha="left",
    fontsize=12,
    color=INK,
    fontweight="bold",
)
ax.set_title(
    "Holdout set, epoch 3: 49,306 tables. Each table is compared with the same table for Timestral.",
    loc="left",
    fontsize=9,
    color=MUTED,
    pad=8,
)
fig.tight_layout(rect=(0, 0, 1, 1.0))
target.parent.mkdir(parents=True, exist_ok=True)
fig.savefig(target, facecolor=SURFACE)
print("saved", target)

# ---- Second figure: Basin's lead over every other model, by difficulty.
# The average table: geometric mean of the per-table MASE ratios in a sliding
# window of 4 percentiles. Holt-Winters is read from where
# `add_classical.py` wrote it.
classical = run / "holtwinters" / "table_metrics.json"
holt_winters = {
    row["source"]: row["holtwinters"]["mase"]
    for row in json.loads(classical.read_text(encoding="utf-8"))
}
others = {
    "Timestral": "timestral",
    "TimeH2O": "timeh2o",
    "Chronos-2": "chronos",
    "Holt-Winters": None,
    "Naive": "naive",
}
COLOURS = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4"]
rows = []
for t in tables:
    values = [t.get(n, {}).get("mase", np.nan) or np.nan for n in names]
    rows.append([*values, holt_winters.get(t["source"]) or np.nan])
every = np.array(rows, dtype=float)
every = every[np.isfinite(every).all(axis=1)]
every = every[np.argsort(every[:, 4], kind="stable")]
place = (np.arange(len(every)) + 0.5) / len(every) * 100
columns = {"timestral": 1, "timeh2o": 2, "chronos": 3, "naive": 4, None: 5}
fig, ax = plt.subplots(figsize=(9.6, 5.2), dpi=160, facecolor=SURFACE)
ax.set_facecolor(SURFACE)
ax.axhline(0, color=MUTED, linewidth=1, zorder=2)
for edge in edges[1:-1]:
    ax.axvline(edge, color=GRID, linewidth=0.8, zorder=1)
ends, peak = [], 0.0
for (label, key), colour in zip(others.items(), COLOURS):
    ratio = np.log(every[:, 0] / every[:, columns[key]])
    line = np.array(
        [
            100 * (1 - np.exp(ratio[(place >= c - 2) & (place < c + 2)].mean()))
            for c in centres
        ]
    )
    ax.plot(centres, line, color=colour, linewidth=2, zorder=3, label=label)
    ends.append([line[-1], label, colour])
    peak = max(peak, line.max())
    print(
        label,
        "lead at percentiles 5, 50, 85, 97:",
        np.round(line[[8, 98, 168, 192]], 1),
    )
# Labels at the line ends, pushed apart where they would overlap.
ends.sort()
for i in range(1, len(ends)):
    ends[i][0] = max(ends[i][0], ends[i - 1][0] + 2.6)
for height, label, colour in ends:
    ax.plot([100.6], [height], marker="o", markersize=5, color=colour, clip_on=False)
    ax.text(101.6, height, label, color=INK, fontsize=9, va="center", clip_on=False)
ax.set_xlim(0, 100)
top = 10 * int(np.ceil(peak / 10)) + 10
ax.set_ylim(-10, top)
ax.set_xticks(edges)
ax.set_yticks(range(-10, top + 1, 10))
ax.set_yticklabels([f"{v:+d}%" if v else "0%" for v in range(-10, top + 1, 10)])
ax.yaxis.grid(True, color=GRID, linewidth=0.8)
ax.set_axisbelow(True)
ax.set_xlabel(
    "Table difficulty: percentile of the naive forecast's MASE (0 = easiest, 100 = hardest)",
    color=MUTED,
)
ax.set_ylabel("Basin's MASE below the other model's, average table", color=MUTED)
ax.tick_params(colors=MUTED, length=0)
for spine in ax.spines.values():
    spine.set_visible(False)
ax.legend(
    loc="upper left",
    frameon=True,
    facecolor=SURFACE,
    edgecolor=GRID,
    fontsize=8.5,
    labelcolor=INK,
    ncols=5,
)
fig.suptitle(
    "Basin's lead over every other model, by table difficulty",
    x=0.012,
    y=0.975,
    ha="left",
    fontsize=12,
    color=INK,
    fontweight="bold",
)
ax.set_title(
    f"Holdout set, epoch 3: {len(every):,} tables. Above 0%: Basin has the lower "
    "MASE. Geometric mean over a sliding window of tables.",
    loc="left",
    fontsize=9,
    color=MUTED,
    pad=8,
)
fig.tight_layout(rect=(0, 0, 0.9, 1.0))
second = target.with_name(target.stem + "_all_models.png")
fig.savefig(second, facecolor=SURFACE)
print("saved", second)

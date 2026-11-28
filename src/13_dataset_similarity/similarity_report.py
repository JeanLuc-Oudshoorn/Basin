"""Figures, tables and the interactive map of the similarity analysis.

Written to `docs/dataset_similarity/`: the figures under `figures/`, the
interactive map as `map_<points>.html` and every table of numbers as
`tables_<points>.md`.
"""

import base64
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import similarity_data
from matplotlib.colors import LinearSegmentedColormap
from series_features import DESCRIPTIONS, FAMILIES, FEATURES, prepare
from similarity_analysis import CONTROL, REAL, SIGNED_LOG, transformed

ROOT = Path(__file__).resolve().parents[2]
DOCS = ROOT / "docs" / "dataset_similarity"
# The synthetic series are the ground every benchmark is compared with, so
# they are drawn in a neutral grey; the three benchmarks take the three
# colours that stay apart from each other for colour-blind readers too.
COLOURS = {
    "synthetic": "#898781",
    CONTROL: "#52514e",
    "fakebench": "#2a78d6",
    "operational": "#eb6834",
    "ind_eval": "#1baf7a",
}
NAMES = {
    **similarity_data.GROUP_NAMES,
    "synthetic": "Synthetic",
    CONTROL: "Synthetic, set aside",
}
SURFACE, INK, MUTED, GRID = "#fcfcfb", "#0b0b0b", "#52514e", "#e1e0d9"
DIVERGING = LinearSegmentedColormap.from_list("gap", ["#2a78d6", "#f0efec", "#e34948"])
BENCHMARKS = ("fakebench", "operational", "ind_eval")


def style():
    plt.rcParams.update(
        {
            "figure.facecolor": SURFACE,
            "axes.facecolor": SURFACE,
            "savefig.facecolor": SURFACE,
            "axes.edgecolor": "#c3c2b7",
            "axes.labelcolor": MUTED,
            "axes.titlecolor": INK,
            "xtick.color": MUTED,
            "ytick.color": MUTED,
            "text.color": INK,
            "font.family": ["Segoe UI", "DejaVu Sans"],
            "font.size": 9,
            "axes.titlesize": 10,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "grid.color": GRID,
            "grid.linewidth": 0.6,
        }
    )


def save(figure, name):
    folder = DOCS / "figures"
    folder.mkdir(parents=True, exist_ok=True)
    figure.savefig(folder / name, dpi=150, bbox_inches="tight")
    plt.close(figure)


def group_share(summary, table="coverage"):
    rows = (
        summary[table]
        if isinstance(summary[table], list)
        else summary[table]["coverage"]
    )
    return {r["group"]: r for r in rows if r["dataset"] == "all"}


def map_figure(
    frame, summary, points, columns, name, title, axis_labels=None, source="coverage"
):
    """One panel per benchmark: its series over the synthetic ones."""
    drawn = frame[frame[columns[0]].notna()]
    synthetic = drawn[drawn["group"] == "synthetic"]
    shares = group_share(summary, source)
    figure, axes = plt.subplots(1, 3, figsize=(13.5, 4.6), sharex=True, sharey=True)
    for axis, group in zip(axes, BENCHMARKS):
        part = drawn[drawn["group"] == group]
        axis.scatter(
            synthetic[columns[0]],
            synthetic[columns[1]],
            s=2,
            c=COLOURS["synthetic"],
            alpha=0.25,
            linewidths=0,
            rasterized=True,
        )
        axis.scatter(
            part[columns[0]],
            part[columns[1]],
            s=3,
            c=COLOURS[group],
            alpha=0.45,
            linewidths=0,
            rasterized=True,
        )
        covered = shares.get(group, {}).get("covered")
        note = "" if covered is None else f": {covered:.0%} covered"
        axis.set_title(
            f"{NAMES[group]}{note}", color=COLOURS[group], loc="left", fontweight="bold"
        )
        if axis_labels:
            axis.set_xlabel(axis_labels[0])
            axis.grid(True)
        else:
            axis.set_xticks([])
            axis.set_yticks([])
            for side in axis.spines.values():
                side.set_visible(False)
    if axis_labels:
        axes[0].set_ylabel(axis_labels[1], fontsize=8)
    figure.suptitle(title, x=0.01, ha="left", fontsize=11.5, fontweight="bold")
    figure.text(
        0.01,
        0.905,
        f"Grey: synthetic series. Each point is one series of {points} points.",
        color=MUTED,
        fontsize=9,
    )
    figure.tight_layout(rect=(0, 0, 1, 0.9))
    save(figure, name)


def strongest(loadings, axis, count=3):
    """The descriptors that move a principal component most, with their sign."""
    weights = {name: values[axis] for name, values in loadings.items()}
    names = sorted(weights, key=lambda n: -abs(weights[n]))[:count]
    return ", ".join(f"{'+' if weights[n] > 0 else '−'}{n}" for n in names)


def coverage_figure(summary, points):
    rows = [
        r
        for r in summary["coverage"]
        # FakeBench-TS as one bar: its nineteen families differ little.
        if r["group"] != "synthetic"
        and (r["dataset"] == "all") == (r["group"] == "fakebench")
    ]
    for item in rows:
        if item["dataset"] == "all":
            item["dataset"] = "all walks"
    order = {g: i for i, g in enumerate((CONTROL,) + BENCHMARKS)}
    rows.sort(key=lambda r: (order[r["group"]], -r["covered"]))
    figure, axis = plt.subplots(figsize=(8.2, 0.3 * len(rows) + 1.3))
    for position, item in enumerate(rows):
        colour = COLOURS[item["group"]]
        axis.barh(position, item["close"], height=0.62, color=colour)
        axis.barh(
            position,
            max(0.0, item["covered"] - item["close"] - 0.004),
            left=item["close"] + 0.004,
            height=0.62,
            color=colour,
            alpha=0.4,
        )
        axis.text(
            item["covered"] + 0.012,
            position,
            f"{item['covered']:.0%}",
            va="center",
            fontsize=8.5,
            color=INK,
        )
    labels = [
        f"{NAMES[r['group']] if r['group'] == CONTROL else r['dataset']}  ({r['series']:,})"
        for r in rows
    ]
    axis.set_yticks(range(len(rows)), labels)
    axis.invert_yaxis()
    axis.set_xlim(0, 1.08)
    axis.set_xticks([0, 0.25, 0.5, 0.75, 1.0], ["0%", "25%", "50%", "75%", "100%"])
    axis.grid(True, axis="x")
    axis.set_axisbelow(True)
    handles = [
        plt.Rectangle((0, 0), 1, 1, color=COLOURS[g]) for g in (CONTROL,) + BENCHMARKS
    ] + [
        plt.Rectangle((0, 0), 1, 1, color=MUTED),
        plt.Rectangle((0, 0), 1, 1, color=MUTED, alpha=0.4),
    ]
    axis.legend(
        handles,
        [NAMES[g] for g in (CONTROL,) + BENCHMARKS]
        + ["as near as a typical synthetic neighbour", "farther, but within the limit"],
        frameon=False,
        ncols=3,
        loc="lower left",
        bbox_to_anchor=(-0.3, 1.0),
        fontsize=8.5,
    )
    axis.set_title(
        f"Share of series that have synthetic series nearby ({points} points)",
        loc="left",
        pad=40,
        fontsize=10.5,
        fontweight="bold",
        x=-0.3,
    )
    save(figure, f"coverage_{points}.png")


def gap_figure(summary, points):
    datasets = summary["separation"]["datasets"]
    datasets = {
        k: v for k, v in datasets.items() if not k.startswith(("fakebench", CONTROL))
    }
    walks = summary["separation"]["groups"].get("fakebench")
    if walks:
        datasets["fakebench|all walks"] = {
            "series": walks["series"],
            "auc": [walks["features"][name]["auc"] for name in FEATURES],
        }
    keys = [k for k in datasets if datasets[k]["series"] >= 30]
    order = {g: i for i, g in enumerate(BENCHMARKS)}
    keys.sort(key=lambda k: (order[k.split("|")[0]], k))
    values = np.array([datasets[k]["auc"] for k in keys]) - 0.5
    figure, axis = plt.subplots(figsize=(13.5, 0.36 * len(keys) + 3.4))
    image = axis.imshow(values, cmap=DIVERGING, vmin=-0.5, vmax=0.5, aspect="auto")
    axis.set_xticks(range(len(FEATURES)), FEATURES, rotation=60, ha="right", fontsize=8)
    axis.set_yticks(
        range(len(keys)),
        [f"{k.split('|')[1]}  ({datasets[k]['series']:,})" for k in keys],
    )
    for label, key in zip(axis.get_yticklabels(), keys):
        label.set_color(COLOURS[key.split("|")[0]])
    families = [FAMILIES[name] for name in FEATURES]
    for column in range(1, len(FEATURES)):
        if families[column] != families[column - 1]:
            axis.axvline(column - 0.5, color=SURFACE, linewidth=3)
    groups = [k.split("|")[0] for k in keys]
    for line in range(1, len(keys)):
        if groups[line] != groups[line - 1]:
            axis.axhline(line - 0.5, color=SURFACE, linewidth=3)
    start = 0
    for column in range(1, len(FEATURES) + 1):
        if column == len(FEATURES) or families[column] != families[start]:
            axis.text(
                (start + column - 1) / 2,
                -0.75,
                families[start].replace(" and complexity", "").replace(" ", "\n"),
                ha="center",
                va="bottom",
                fontsize=8,
                color=MUTED,
            )
            start = column
    axis.tick_params(length=0)
    for side in axis.spines.values():
        side.set_visible(False)
    bar = figure.colorbar(
        image, ax=axis, fraction=0.018, pad=0.01, ticks=[-0.5, 0, 0.5]
    )
    bar.ax.set_yticklabels(["always lower", "same", "always higher"])
    bar.outline.set_visible(False)
    axis.set_title(
        f"How each benchmark dataset differs from the synthetic series, per descriptor ({points} points)\n\n",
        loc="left",
        fontsize=11,
        fontweight="bold",
    )
    save(figure, f"descriptor_gaps_{points}.png")


def ranked_features(summary, groups=REAL):
    """Descriptors by how well each alone tells the real benchmarks from synthetic."""
    found = summary["separation"]["groups"]
    present = [g for g in groups if g in found]
    score = {
        name: np.mean([abs(found[g]["features"][name]["auc"] - 0.5) for g in present])
        for name in FEATURES
    }
    return sorted(FEATURES, key=lambda n: -score[n])


def distribution_figure(frame, summary, points, count=12):
    names = ranked_features(summary)[:count]
    table = transformed(frame)
    figure, axes = plt.subplots(3, 4, figsize=(13.5, 7.6))
    for axis, name in zip(axes.ravel(), names):
        column = FEATURES.index(name)
        low, high = np.quantile(table[:, column], [0.005, 0.995])
        edges = np.linspace(low, high, 41)
        for group in ("synthetic",) + BENCHMARKS:
            values = table[(frame["group"] == group).to_numpy(), column]
            if not len(values):
                continue
            height, _ = np.histogram(np.clip(values, low, high), bins=edges)
            height = height / height.sum()
            if group == "synthetic":
                axis.fill_between(
                    edges[:-1],
                    height,
                    step="post",
                    color=COLOURS[group],
                    alpha=0.35,
                    linewidth=0,
                )
            else:
                axis.step(
                    edges[:-1],
                    height,
                    where="post",
                    color=COLOURS[group],
                    linewidth=1.6,
                )
        scale = " (log scale)" if name in SIGNED_LOG else ""
        axis.set_title(f"{name}{scale}", loc="left", fontweight="bold")
        axis.set_xlabel(DESCRIPTIONS[name], fontsize=7.5)
        axis.set_yticks([])
        axis.spines["left"].set_visible(False)
    handles = [plt.Rectangle((0, 0), 1, 1, color=COLOURS["synthetic"], alpha=0.35)] + [
        plt.Line2D([0], [0], color=COLOURS[g], linewidth=1.6) for g in BENCHMARKS
    ]
    figure.legend(
        handles,
        [NAMES[g] for g in ("synthetic",) + BENCHMARKS],
        frameon=False,
        ncols=4,
        loc="upper left",
        bbox_to_anchor=(0.005, 0.955),
    )
    figure.suptitle(
        f"The twelve descriptors that best tell the real benchmarks from the synthetic series ({points} points)",
        x=0.01,
        ha="left",
        fontsize=11.5,
        fontweight="bold",
    )
    figure.tight_layout(rect=(0, 0, 1, 0.93))
    save(figure, f"distributions_{points}.png")


def standard(window):
    prepared = prepare(window)
    return None if prepared is None else prepared[1]


def arrow(shift):
    return f"{shift['feature']} {'higher' if shift['shift'] > 0 else 'lower'}"


def example_figure(clusters, index, windows, points, name, title, with_neighbour):
    """Example series of every cluster; beside each its nearest synthetic series."""
    if not clusters:
        return
    columns = 4
    figure, axes = plt.subplots(
        len(clusters),
        columns,
        figsize=(13.5, 2.0 * len(clusters) + 0.8),
        squeeze=False,
    )
    for line, cluster in enumerate(clusters):
        if with_neighbour:
            rows = [
                r
                for pair in zip(
                    cluster["examples"][:2], cluster["nearest_synthetic"][:2]
                )
                for r in pair
            ]
        else:
            rows = cluster["examples"][:columns]
        for column, axis in enumerate(axes[line]):
            axis.set_xticks([])
            axis.set_yticks([])
            for side in axis.spines.values():
                side.set_visible(False)
            if column >= len(rows) or rows[column] < 0:
                continue
            row = rows[column]
            values = standard(windows[row, -points:])
            group = index["group"].iat[row]
            axis.plot(values, color=COLOURS[group], linewidth=0.8)
            neighbour = with_neighbour and column % 2 == 1
            label = (
                "nearest synthetic series" if neighbour else index["dataset"].iat[row]
            )
            axis.set_title(label, fontsize=8, color=COLOURS[group], loc="left", pad=2)
        mostly = ", ".join(
            f"{d['dataset']} {d['series']:,}" for d in cluster["datasets"][:3]
        )
        differs = "; ".join(arrow(s) for s in cluster["differs_most"][:4])
        axes[line][0].text(
            0,
            1.3,
            f"{chr(65 + line)}. {cluster['series']:,} series ({mostly}). Against synthetic: {differs}",
            transform=axes[line][0].transAxes,
            fontsize=8.5,
            color=INK,
            fontweight="bold",
        )
    top = 1 - 1.0 / (2.0 * len(clusters) + 0.8)
    figure.subplots_adjust(left=0.02, right=0.98, bottom=0.02, top=top, hspace=0.75)
    figure.suptitle(title, x=0.02, y=0.995, ha="left", fontsize=11.5, fontweight="bold")
    save(figure, name)


def markdown_table(header, rows):
    lines = ["| " + " | ".join(header) + " |", "|" + "---|" * len(header)]
    lines += ["| " + " | ".join(str(c) for c in row) + " |" for row in rows]
    return "\n".join(lines)


def tables(frame, index, summary, points):
    """Every number of the analysis as markdown, for the written report."""
    parts = [f"# Tables for windows of {points} points\n"]
    counts = summary["counts"]
    parts.append("## Series per group\n")
    parts.append(
        markdown_table(
            [
                "Group",
                "Collected",
                f"With {points} points",
                "Without a shape",
                "Analysed",
            ],
            [
                [
                    NAMES[g],
                    f"{c['collected']:,}",
                    f"{c['long_enough']:,}",
                    f"{c['without_shape']:,}",
                    f"{c['long_enough'] - c['without_shape']:,}",
                ]
                for g, c in counts.items()
            ],
        )
    )
    lengths = index.groupby("group")["source_points"]
    parts.append("\n## Length of the collected series (points)\n")
    parts.append(
        markdown_table(
            ["Group", "25%", "Median", "75%", "Share with 512+", "Share with 2,048+"],
            [
                [
                    NAMES[g],
                    *(
                        f"{int(v):,}"
                        for v in lengths.get_group(g).quantile([0.25, 0.5, 0.75])
                    ),
                    f"{(lengths.get_group(g) >= 512).mean():.0%}",
                    f"{(lengths.get_group(g) >= 2048).mean():.0%}",
                ]
                for g in similarity_data.GROUPS
                if g in lengths.groups
            ],
        )
    )
    parts.append("\n## Coverage\n")
    parts.append(
        markdown_table(
            ["Group", "Dataset", "Series", "Close", "Covered", "Median distance ratio"],
            [
                [
                    NAMES[r["group"]],
                    r["dataset"],
                    f"{r['series']:,}",
                    f"{r['close']:.0%}",
                    f"{r['covered']:.0%}",
                    f"{r['median_ratio']:.2f}",
                ]
                for r in summary["coverage"]
            ],
        )
    )
    parts.append("\n## Coverage under other settings (neighbours k, quantile q)\n")
    labels = list(next(iter(summary["robustness"].values())))
    parts.append(
        markdown_table(
            ["Setting", *(NAMES[g] for g in labels)],
            [
                [name, *(f"{values[g]:.0%}" for g in labels)]
                for name, values in summary["robustness"].items()
            ],
        )
    )
    parts.append("\n## Share of synthetic series with benchmark series nearby\n")
    parts.append(
        markdown_table(
            ["Benchmark", "Share of synthetic series"],
            [[k, f"{v:.0%}"] for k, v in summary["synthetic_resembling"].items()],
        )
    )
    groups = summary["separation"]["groups"]
    mixes = dict(groups)
    if "real" in summary["separation"]:
        mixes["Operational and IND-Eval together"] = summary["separation"]["real"]
    parts.append(
        "\n## Classifier: telling a group from the synthetic series (AUC, 0.5 = cannot)\n"
    )
    parts.append(
        markdown_table(
            [
                "Group",
                "Series",
                "All descriptors",
                "Without sensor behaviour",
                "Synthetic series that resemble the group",
                "Group series that resemble synthetic",
                "Synthetic series that count after reweighting",
            ],
            [
                [
                    NAMES.get(g, g),
                    f"{v['series']:,}" if "series" in v else "-",
                    fmt(v["classifier_auc"]),
                    fmt(v.get("classifier_auc_without_sensor_behaviour")),
                    share(v.get("synthetic_resembling")),
                    share(v.get("benchmark_resembling_synthetic")),
                    share(v.get("effective")),
                ]
                for g, v in mixes.items()
            ],
        )
    )
    synthetic = dict(zip(FEATURES, summary["separation"]["synthetic"]))
    for group in BENCHMARKS:
        if group not in groups:
            continue
        found = groups[group]["features"]
        order = sorted(FEATURES, key=lambda n: -abs(found[n]["auc"] - 0.5))
        parts.append(
            f"\n## Descriptors: {NAMES[group]} against synthetic (10% / median / 90%)\n"
        )
        parts.append(
            markdown_table(
                [
                    "Descriptor",
                    "Synthetic",
                    NAMES[group],
                    "Chance higher",
                    "Above synthetic 99%",
                    "Below synthetic 1%",
                ],
                [
                    [
                        n,
                        three(synthetic[n]),
                        three(found[n]["quantiles"]),
                        f"{found[n]['auc']:.2f}",
                        f"{found[n]['above_synthetic_p99']:.0%}",
                        f"{found[n]['below_synthetic_p01']:.0%}",
                    ]
                    for n in order
                ],
            )
        )
    parts.append("\n## Cycles\n")
    rows = []
    for label in ("synthetic",) + BENCHMARKS:
        part = frame[frame["group"] == label]
        if part.empty:
            continue
        cyclic = part[part["season_period"] > 0]
        periods = (
            cyclic["season_period"].quantile([0.25, 0.5, 0.75])
            if len(cyclic)
            else [np.nan] * 3
        )
        rows.append(
            [
                NAMES[label],
                f"{len(cyclic) / len(part):.0%}",
                f"{(part['seasonal_strength'] > 0.6).mean():.0%}",
                *(f"{v:.0f}" for v in periods),
            ]
        )
    parts.append(
        markdown_table(
            [
                "Group",
                "Has a cycle",
                "Strong cycle (strength above 0.6)",
                "Cycle length 25%",
                "Median",
                "75%",
            ],
            rows,
        )
    )
    parts.append("\n### Cycles per dataset\n")
    rows = []
    for (group, dataset), part in frame[frame["group"] != "synthetic"].groupby(
        ["group", "dataset"], sort=False
    ):
        cyclic = part[part["season_period"] > 0]
        rows.append(
            [
                NAMES[group],
                dataset,
                f"{len(part):,}",
                f"{len(cyclic) / len(part):.0%}",
                f"{(part['seasonal_strength'] > 0.6).mean():.0%}",
                f"{cyclic['season_period'].median():.0f}" if len(cyclic) else "-",
            ]
        )
    parts.append(
        markdown_table(
            [
                "Group",
                "Dataset",
                "Series",
                "Has a cycle",
                "Strong cycle",
                "Median cycle length",
            ],
            rows,
        )
    )
    accuracy = summary.get("accuracy")
    if accuracy:
        parts.append(
            "\n## Basin's error against the other models (stage 11 forecast cases)\n"
        )
        parts.append(
            f"{accuracy['cases']:,} cases; Basin's error over the mean error of the three other models, "
            f"median {accuracy['basin_over_others_median']:.2f}. Rank correlation with the distance to the "
            f"synthetic series: {accuracy['distance']['overall']:.2f} overall, "
            f"{accuracy['distance']['within_datasets']:.2f} within datasets.\n"
        )
        parts.append(
            markdown_table(
                [
                    "Distance quarter",
                    "Distance ratio",
                    "Cases",
                    "Basin over others",
                    "Basin worse",
                ],
                [
                    [
                        i + 1,
                        f"{q['ratio_range'][0]:.2f} to {q['ratio_range'][1]:.2f}",
                        f"{q['cases']:,}",
                        f"{q['basin_over_others']:.2f}",
                        f"{q['basin_worse_share']:.0%}",
                    ]
                    for i, q in enumerate(accuracy["by_distance_quarter"])
                ],
            )
        )
        parts.append("")
        parts.append(
            markdown_table(
                ["Dataset", "Cases", "Basin over others", "Median distance ratio"],
                [
                    [
                        k,
                        f"{v['cases']:,}",
                        f"{v['basin_over_others']:.2f}",
                        f"{v['median_ratio']:.2f}",
                    ]
                    for k, v in accuracy["by_dataset"].items()
                ],
            )
        )
        parts.append("\n### Descriptors that go with Basin losing (rank correlation)\n")
        order = sorted(
            FEATURES, key=lambda n: -abs(accuracy["features"][n]["within_datasets"])
        )
        parts.append(
            markdown_table(
                ["Descriptor", "Overall", "Within datasets"],
                [
                    [
                        n,
                        f"{accuracy['features'][n]['overall']:+.2f}",
                        f"{accuracy['features'][n]['within_datasets']:+.2f}",
                    ]
                    for n in order
                ],
            )
        )
    for key, heading in (
        ("uncovered_clusters", "Benchmark series without synthetic neighbours"),
        ("synthetic_alone_clusters", "Synthetic series without benchmark neighbours"),
    ):
        parts.append(f"\n## {heading}, by kind\n")
        parts.append(
            markdown_table(
                [
                    "Kind",
                    "Series",
                    "Mostly",
                    "Differs from the typical synthetic series",
                ],
                [
                    [
                        chr(65 + i),
                        f"{c['series']:,}",
                        ", ".join(
                            f"{d['dataset']} {d['series']:,}" for d in c["datasets"]
                        ),
                        "; ".join(
                            f"{arrow(s)} ({s['shift']:+.1f})" for s in c["differs_most"]
                        ),
                    ]
                    for i, c in enumerate(summary.get(key, []))
                ],
            )
        )
    embeddings = summary.get("embeddings")
    if embeddings:
        parts.append("\n## Second opinion: Chronos-2 embeddings\n")
        parts.append(
            markdown_table(
                [
                    "Group",
                    "Dataset",
                    "Series",
                    "Close",
                    "Covered",
                    "Median distance ratio",
                ],
                [
                    [
                        NAMES[r["group"]],
                        r["dataset"],
                        f"{r['series']:,}",
                        f"{r['close']:.0%}",
                        f"{r['covered']:.0%}",
                        f"{r['median_ratio']:.2f}",
                    ]
                    for r in embeddings["coverage"]
                ],
            )
        )
        parts.append("")
        parts.append(
            markdown_table(
                [
                    "Group",
                    "Classifier AUC",
                    "Rank correlation of the two distances",
                    "Same verdict on covered",
                ],
                [
                    [
                        NAMES[g],
                        fmt(embeddings["classifier_auc"][g]),
                        f"{a['rank_correlation_of_distances']:.2f}",
                        f"{a['same_verdict']:.0%}",
                    ]
                    for g, a in embeddings["agreement"].items()
                ],
            )
        )
    parts.append("\n## Principal components\n")
    parts.append(
        markdown_table(
            ["Component", "Share of variation", "Moves most with"],
            [
                [i + 1, f"{share:.0%}", strongest(summary["pca"]["loadings"], i, 5)]
                for i, share in enumerate(summary["pca"]["explained"])
            ],
        )
    )
    (DOCS / f"tables_{points}.md").write_text("\n".join(parts) + "\n", encoding="utf-8")


def fmt(value):
    return "-" if value is None else f"{value:.3f}"


def share(value):
    return "-" if value is None else f"{value:.1%}"


def three(values):
    return " / ".join(f"{v:.2f}" for v in values)


def sparkline(window, count=128):
    """A standardised window as `count` bytes: per stretch its most extreme point."""
    values = standard(window)
    if values is None:
        return bytes(count)
    parts = values[: len(values) // count * count].reshape(count, -1)
    pick = np.abs(parts - parts.mean(axis=1, keepdims=True)).argmax(axis=1)
    chosen = parts[np.arange(count), pick]
    return (np.clip(chosen, -4, 4) * 31.75 + 127.5).astype(np.uint8).tobytes()


def interactive_map(frame, index, windows, summary, points, options):
    """A self-contained page: the 3-D map, and the series behind a clicked point."""
    limit = options["map"]
    drawn = frame[frame["in_map"]]
    keep = []
    for group, part in drawn.groupby("group", sort=False):
        most = 5000 if group == "synthetic" else 3000
        order = similarity_data.ranked(part["series"], options["seed"])
        keep.append(part.index.to_numpy()[order[: min(most, limit["per_group"])]])
    drawn = frame.loc[np.sort(np.concatenate(keep))]
    views = {
        "Descriptors, t-SNE": ["tsne3_1", "tsne3_2", "tsne3_3"],
        "Descriptors, principal components": ["pca_1", "pca_2", "pca_3"],
    }
    if "embedding_tsne3_1" in drawn:
        views["Chronos-2 embeddings, t-SNE"] = [
            f"embedding_tsne3_{i}" for i in (1, 2, 3)
        ]
    rows = drawn["row"].to_numpy()
    neighbours = drawn["nearest_synthetic"].to_numpy()
    extra = sorted(set(neighbours[neighbours >= 0].tolist()) - set(rows.tolist()))
    spark_rows = np.concatenate([rows, np.array(extra, dtype=int)])
    position = {int(row): i for i, row in enumerate(spark_rows)}
    sparks = b"".join(sparkline(windows[row, -points:]) for row in spark_rows)
    shown = [
        "seasonal_strength",
        "season_period",
        "trend_strength",
        "spectral_entropy",
        "long_memory",
        "noise_share",
        "flat_share",
        "jump_share",
    ]
    data = {
        "points": points,
        "groups": [NAMES[g] for g in ("synthetic",) + BENCHMARKS],
        "colours": [COLOURS[g] for g in ("synthetic",) + BENCHMARKS],
        "coloursDark": ["#898781", "#3987e5", "#d95926", "#199e70"],
        "group": [(("synthetic",) + BENCHMARKS).index(g) for g in drawn["group"]],
        "dataset": drawn["dataset"].tolist(),
        "series": [s[-60:] for s in drawn["series"]],
        "covered": drawn["covered"].astype(int).tolist(),
        "resembles": drawn["resembles_real"].astype(int).tolist(),
        "ratio": np.round(drawn["ratio"], 2).tolist(),
        "neighbour": [position.get(int(n), -1) for n in neighbours],
        "views": {
            name: [np.round(drawn[c].fillna(0), 2).tolist() for c in columns]
            for name, columns in views.items()
        },
        "has": {
            name: drawn[columns[0]].notna().astype(int).tolist()
            for name, columns in views.items()
        },
        "shown": shown,
        "features": {name: np.round(drawn[name], 2).tolist() for name in shown},
        "sparks": base64.b64encode(sparks).decode(),
    }
    template = (Path(__file__).parent / "map_template.html").read_text(encoding="utf-8")
    page = template.replace("/*DATA*/", json.dumps(data, separators=(",", ":")))
    (DOCS / f"map_{points}.html").write_text(page, encoding="utf-8")
    # The page loads the plotting library from the file beside it, so the
    # map opens without a network connection.
    library = DOCS / "plotly.min.js"
    if not library.is_file():
        from plotly.offline import get_plotlyjs

        library.write_text(get_plotlyjs(), encoding="utf-8")


def report_view(directory, points, options, log):
    path = directory / f"analysis_{points}.parquet"
    if not path.is_file():
        return
    frame = pd.read_parquet(path)
    summary = json.loads(
        (directory / f"analysis_{points}.json").read_text(encoding="utf-8")
    )
    index, windows = similarity_data.load(directory)
    map_figure(
        frame,
        summary,
        points,
        ["tsne2_1", "tsne2_2"],
        f"map_tsne_{points}.png",
        "Neighbourhood map of the descriptors (t-SNE): where benchmark series lie among the synthetic ones",
    )
    loadings, explained = summary["pca"]["loadings"], summary["pca"]["explained"]
    map_figure(
        frame,
        summary,
        points,
        ["pca_1", "pca_2"],
        f"map_pca_{points}.png",
        "The two main directions of variation of the descriptors (principal components)",
        axis_labels=[
            f"Component {i + 1} ({explained[i]:.0%}): {strongest(loadings, i, count)}"
            for i, count in ((0, 3), (1, 2))
        ],
    )
    if "embedding_tsne2_1" in frame:
        map_figure(
            frame,
            summary,
            points,
            ["embedding_tsne2_1", "embedding_tsne2_2"],
            f"map_embeddings_{points}.png",
            "Second opinion: neighbourhood map of the Chronos-2 embeddings (t-SNE)",
            source="embeddings",
        )
    coverage_figure(summary, points)
    gap_figure(summary, points)
    distribution_figure(frame, summary, points)
    example_figure(
        summary["uncovered_clusters"],
        index,
        windows,
        points,
        f"uncovered_{points}.png",
        "Benchmark series without synthetic neighbours, by kind; beside each example its nearest synthetic series",
        True,
    )
    example_figure(
        summary.get("synthetic_alone_clusters", []),
        index,
        windows,
        points,
        f"synthetic_alone_{points}.png",
        "Synthetic series without any benchmark series nearby, by kind",
        False,
    )
    tables(frame, index, summary, points)
    interactive_map(frame, index, windows, summary, points, options)
    log(f"report {points}: figures, tables and map written to {DOCS}")


def report(options, log=print):
    style()
    DOCS.mkdir(parents=True, exist_ok=True)
    directory = similarity_data.output_directory(options)
    for points in options["window_points"]:
        report_view(directory, points, options, log)

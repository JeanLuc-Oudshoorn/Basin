"""Measure how far the benchmark series are from the synthetic ones.

All of it works on the descriptors of series_features.py (and, as a second
opinion, on the embeddings of series_embeddings.py):

- distances: every series is a point; the distance between two points is the
  difference of their standardised descriptors, every family counting equally,
- coverage: a benchmark series is covered when synthetic series lie as near
  to it as synthetic series normally lie to each other,
- separation: per descriptor, how well it alone tells a group from the
  synthetic series, and how well a classifier does with all of them,
- accuracy: whether Basin loses more against the other models on series far
  from the synthetic ones,
- maps: the points in two and three dimensions, for looking at.

A fifth of the synthetic series is set aside as a control group and treated
like a benchmark. It must come out as covered and inseparable; if it does
not, the method is at fault, not the data.
"""

import hashlib
import json

import numpy as np
import pandas as pd
import similarity_data
from lightgbm import LGBMClassifier
from scipy.stats import rankdata, spearmanr
from series_features import FAMILIES, FEATURES
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.neighbors import NearestNeighbors

CONTROL = "control"
REAL = ("operational", "ind_eval")
OTHER_MODELS = ("chronos2", "timestral", "timeh2o")
NEIGHBOUR_COUNTS, QUANTILES = (1, 5, 20), (0.50, 0.90, 0.95, 0.99)
# Heavy-tailed descriptors are compressed before standardising.
SIGNED_LOG = ("kurtosis", "diff_kurtosis", "skewness", "diff_skewness", "lumpiness")


def signed_log(values):
    return np.sign(values) * np.log1p(np.abs(values))


def transformed(frame):
    table = frame[list(FEATURES)].to_numpy(dtype=float, copy=True)
    for name in SIGNED_LOG:
        column = FEATURES.index(name)
        table[:, column] = signed_log(table[:, column])
    return table


def weighted_quantile(values, weights, quantiles):
    order = np.argsort(values)
    cumulative = np.cumsum(weights[order])
    return np.interp(quantiles, cumulative / cumulative[-1], values[order])


def standardise(table, groups):
    """Standardised descriptors in which every group and family counts equally.

    Each of the groups gets the same total weight in the centre and spread of
    a descriptor, whatever its number of series. The extreme half percent on
    either side is pulled in, so that one odd value cannot decide a distance.
    A descriptor's weight is 1 / sqrt(size of its family): the squared
    distance is the sum over families of the mean squared difference.
    """
    names, counts = np.unique(groups, return_counts=True)
    weights = (1.0 / (len(names) * counts))[np.searchsorted(names, groups)]
    scaled = np.empty_like(table)
    limits = []
    for column in range(table.shape[1]):
        values = table[:, column]
        low, high = weighted_quantile(values, weights, [0.005, 0.995])
        values = np.clip(values, low, high)
        mean = float((weights * values).sum())
        spread = float(np.sqrt((weights * (values - mean) ** 2).sum()))
        scaled[:, column] = (values - mean) / spread if spread > 0 else 0.0
        limits.append((low, high, mean, spread))
    sizes = pd.Series(FAMILIES).value_counts()
    family_weight = np.array(
        [1.0 / np.sqrt(sizes[FAMILIES[name]]) for name in FEATURES]
    )
    return scaled, scaled * family_weight, limits


def load_view(directory, points):
    """Facts and descriptors of the series of one view; (frame, counts)."""
    index, _ = similarity_data.load(directory)
    features = pd.read_parquet(similarity_data.features_path(directory, points))
    frame = index.iloc[features["row"].to_numpy()].reset_index(drop=True)
    frame = pd.concat([frame, features], axis=1)
    counts = {}
    for group in similarity_data.GROUPS:
        collected = index[index["group"] == group]
        in_view = frame[frame["group"] == group]
        counts[group] = {
            "collected": int(len(collected)),
            "long_enough": int(len(in_view)),
            "without_shape": int(in_view[FEATURES[0]].isna().sum()),
        }
    frame = frame[frame[FEATURES[0]].notna()].reset_index(drop=True)
    return frame, counts


def control_rows(frame, seed, share=0.2):
    """A fixed fifth of the synthetic series, by name."""
    cut = int(share * 2**32)
    digest = [
        int(hashlib.sha256(f"control:{seed}:{name}".encode()).hexdigest()[:8], 16)
        for name in frame["series"]
    ]
    return (frame["group"].to_numpy() == "synthetic") & (np.array(digest) < cut)


def underlying(frame):
    """What a window was cut from, so that its windows stay in one fold."""
    keys = []
    for group, dataset, series in zip(
        frame["group"], frame["dataset"], frame["series"]
    ):
        if group == "ind_eval":
            parts = series.split("|")
            series = "|".join(parts[1:2] + parts[3:])
        keys.append(f"{group}|{dataset}|{series}")
    return np.array(keys)


def neighbour_distances(reference, query, count, *, own=False):
    """Distances and positions of the `count` nearest reference points."""
    search = NearestNeighbors(n_neighbors=count + int(own), algorithm="brute")
    search.fit(reference)
    distance, position = search.kneighbors(query)
    return (distance[:, 1:], position[:, 1:]) if own else (distance, position)


def coverage(points, role, options):
    """Distance of every point to the synthetic reference, and what is covered."""
    k, quantile = options["neighbours"], options["covered_quantile"]
    most = max(NEIGHBOUR_COUNTS + (k,))
    reference = role == "reference"
    own, _ = neighbour_distances(points[reference], points[reference], most, own=True)
    distance = np.empty((len(points), most))
    nearest = np.empty(len(points), dtype=int)
    distance[reference] = own
    other, position = neighbour_distances(points[reference], points[~reference], most)
    distance[~reference] = other
    reference_rows = np.flatnonzero(reference)
    nearest[~reference] = reference_rows[position[:, 0]]
    nearest[reference] = -1
    limit = float(np.quantile(own[:, k - 1], quantile))
    typical = float(np.median(own[:, k - 1]))
    grid = {
        f"k{count}_q{int(q * 100)}": float(np.quantile(own[:, count - 1], q))
        for count in NEIGHBOUR_COUNTS
        for q in QUANTILES
    }
    return {
        "distance": distance,
        "ratio": distance[:, k - 1] / typical,
        "covered": distance[:, k - 1] <= limit,
        "nearest": nearest,
        "limit": limit,
        "typical": typical,
        "grid": grid,
    }


def share_table(frame, covered, ratio, role):
    """Coverage per group and per dataset."""
    rows = []
    for (group, dataset), part in frame.groupby(["label", "dataset"], sort=False):
        chosen = part.index.to_numpy()
        rows.append(
            {
                "group": group,
                "dataset": dataset,
                "series": int(len(chosen)),
                "close": float((ratio[chosen] <= 1.0).mean()),
                "covered": float(covered[chosen].mean()),
                "median_ratio": float(np.median(ratio[chosen])),
            }
        )
    for group, part in frame.groupby("label", sort=False):
        chosen = part.index.to_numpy()
        rows.append(
            {
                "group": group,
                "dataset": "all",
                "series": int(len(chosen)),
                "close": float((ratio[chosen] <= 1.0).mean()),
                "covered": float(covered[chosen].mean()),
                "median_ratio": float(np.median(ratio[chosen])),
            }
        )
    return rows


def resembled(points, frame, role, found, options):
    """Share of the synthetic reference that has benchmark series nearby."""
    k = options["neighbours"]
    reference = role == "reference"
    result = {}
    choices = {name: [name] for name in similarity_data.GROUPS[1:]}
    choices["operational or ind_eval"] = list(REAL)
    near = {}
    for name, groups in choices.items():
        chosen = frame["group"].isin(groups).to_numpy()
        if chosen.sum() < k:
            continue
        distance, _ = neighbour_distances(points[chosen], points[reference], k)
        near[name] = distance[:, k - 1] <= found["limit"]
        result[name] = float(near[name].mean())
    return result, near.get("operational or ind_eval")


def auc_higher(benchmark, synthetic):
    """Chance that a benchmark value is above a synthetic one (ties count half)."""
    ranks = rankdata(np.concatenate([benchmark, synthetic]))
    n, m = len(benchmark), len(synthetic)
    return float((ranks[:n].sum() - n * (n + 1) / 2) / (n * m))


def classifier_scores(table, positive, keys, seed, columns=None, weights=None):
    """Cross-validated chance of being a positive row, with both sides equal.

    The two sides get the same total weight, so a score of 0.5 means "as
    likely the one as the other" whatever their numbers of rows. Windows of
    one recording stay in one fold. None where there is too little to learn.
    """
    data = table if columns is None else table[:, columns]
    if positive.sum() < 10 or len(np.unique(keys[positive])) < 5:
        return None
    weights = np.ones(len(positive)) if weights is None else weights.astype(float)
    weights = np.where(
        positive, weights / weights[positive].sum(), weights / weights[~positive].sum()
    )
    folds = StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=seed)
    scores = np.zeros(len(positive))
    for train, test in folds.split(data, positive, keys):
        model = LGBMClassifier(
            n_estimators=200,
            learning_rate=0.08,
            subsample=0.8,
            subsample_freq=1,
            colsample_bytree=0.8,
            random_state=seed,
            verbose=-1,
        )
        model.fit(
            data[train], positive[train], sample_weight=weights[train] * len(train)
        )
        scores[test] = model.predict_proba(data[test])[:, 1]
    return scores


def classifier_auc(table, positive, keys, seed, columns=None):
    scores = classifier_scores(table, positive, keys, seed, columns)
    return None if scores is None else float(roc_auc_score(positive, scores))


def mix_difference(scores, positive):
    """What a classifier's scores say about how differently two sets are mixed.

    `synthetic_resembling`: the share of the synthetic series that are at
    least as likely to be benchmark series as synthetic ones. `effective`:
    if the synthetic series were reweighted to the mix of the benchmark, the
    share of them that would still count (1: the mixes are the same).
    """
    synthetic = np.clip(scores[~positive], 1e-3, 1 - 1e-3)
    weight = synthetic / (1 - synthetic)
    return {
        "classifier_auc": float(roc_auc_score(positive, scores)),
        "synthetic_resembling": float((scores[~positive] >= 0.5).mean()),
        "benchmark_resembling_synthetic": float((scores[positive] < 0.5).mean()),
        "effective": float(weight.sum() ** 2 / (len(weight) * (weight**2).sum())),
    }


def separation(frame, table, role, seed, log):
    """Per group: what tells it from the synthetic reference."""
    reference = np.flatnonzero(role == "reference")
    keys = underlying(frame)
    low, high = np.quantile(table[reference], [0.01, 0.99], axis=0)
    quantiles = [0.1, 0.5, 0.9]
    result = {"synthetic": np.quantile(table[reference], quantiles, axis=0).T.tolist()}
    without_sensor = [
        i for i, name in enumerate(FEATURES) if FAMILIES[name] != "sensor behaviour"
    ]
    groups, datasets = {}, {}
    likeness = np.full(len(frame), np.nan)
    # The real benchmarks together, each counting equally.
    real = np.flatnonzero(frame["group"].isin(REAL).to_numpy())
    if len(real):
        rows = np.concatenate([real, reference])
        positive = np.arange(len(rows)) < len(real)
        sizes = frame["group"].iloc[rows].map(frame["group"].iloc[real].value_counts())
        weights = np.where(positive, 1.0 / sizes.fillna(1).to_numpy(), 1.0)
        scores = classifier_scores(
            table[rows], positive, keys[rows], seed, weights=weights
        )
        if scores is not None:
            result["real"] = mix_difference(scores, positive)
            likeness[rows] = scores
    for label, part in frame.groupby("label", sort=False):
        if label == "synthetic":
            continue
        chosen = part.index.to_numpy()
        rows = np.concatenate([chosen, reference])
        positive = np.arange(len(rows)) < len(chosen)
        scores = classifier_scores(table[rows], positive, keys[rows], seed)
        groups[label] = {
            "series": int(len(chosen)),
            "classifier_auc": None,
            **({} if scores is None else mix_difference(scores, positive)),
            "classifier_auc_without_sensor_behaviour": classifier_auc(
                table[rows], positive, keys[rows], seed, without_sensor
            ),
            "features": feature_gaps(table, chosen, reference, low, high, quantiles),
        }
        log(f"separation: {label} classifier AUC {groups[label]['classifier_auc']}")
        for dataset, piece in part.groupby("dataset", sort=False):
            rows = piece.index.to_numpy()
            datasets[f"{label}|{dataset}"] = {
                "series": int(len(rows)),
                "auc": [
                    auc_higher(table[rows, column], table[reference, column])
                    for column in range(table.shape[1])
                ],
            }
    result.update(groups=groups, datasets=datasets)
    return result, likeness


def feature_gaps(table, chosen, reference, low, high, quantiles):
    rows = {}
    for column, name in enumerate(FEATURES):
        values = table[chosen, column]
        rows[name] = {
            "quantiles": np.quantile(values, quantiles).tolist(),
            "auc": auc_higher(values, table[reference, column]),
            "above_synthetic_p99": float((values > high[column]).mean()),
            "below_synthetic_p01": float((values < low[column]).mean()),
        }
    return rows


def accuracy_link(frame, table, ratio, options):
    """Does Basin lose more where the series is far from the synthetic ones?

    For the forecast cases of stage 11: Basin's error over the mean error of
    the three other pretrained models (above 1: Basin is worse), against the
    distance to the synthetic series and against every descriptor. The same
    is repeated within datasets, because a few datasets differ in both.
    """
    scores = similarity_data.latest_operational_scores(options)
    cases = frame[(frame["group"] == "operational") & (frame["case"] >= 0)]
    if scores is None or cases.empty:
        return None
    scores = scores.set_index("case").reindex(cases["case"].to_numpy())
    others = np.exp(
        np.mean(
            [np.log(scores[f"{m}_mase_112"].to_numpy()) for m in OTHER_MODELS], axis=0
        )
    )
    with np.errstate(divide="ignore", invalid="ignore"):
        gap = np.log(scores["basin_mase_112"].to_numpy() / others)
    usable = np.isfinite(gap)
    rows, gap = cases.index.to_numpy()[usable], gap[usable]
    datasets = cases["dataset"].to_numpy()[usable]
    within = gap - pd.Series(gap).groupby(datasets).transform("median").to_numpy()

    def rank(a, b):
        # Undefined when a descriptor has one value for every case.
        if np.ptp(a) == 0 or np.ptp(b) == 0:
            return 0.0
        return float(spearmanr(a, b).statistic)

    def rank_correlation(values):
        centred = (
            values - pd.Series(values).groupby(datasets).transform("median").to_numpy()
        )
        return {
            "overall": rank(values, gap),
            "within_datasets": rank(centred, within),
        }

    edges = np.quantile(ratio[rows], [0.25, 0.5, 0.75])
    quarter = np.searchsorted(edges, ratio[rows])
    return {
        "cases": int(len(rows)),
        "basin_over_others_median": float(np.exp(np.median(gap))),
        "distance": rank_correlation(ratio[rows]),
        "by_distance_quarter": [
            {
                "ratio_range": [
                    float(ratio[rows][quarter == q].min()),
                    float(ratio[rows][quarter == q].max()),
                ],
                "cases": int((quarter == q).sum()),
                "basin_over_others": float(np.exp(np.median(gap[quarter == q]))),
                "basin_worse_share": float((gap[quarter == q] > 0).mean()),
            }
            for q in range(4)
        ],
        "by_dataset": {
            name: {
                "cases": int((datasets == name).sum()),
                "basin_over_others": float(np.exp(np.median(gap[datasets == name]))),
                "median_ratio": float(np.median(ratio[rows][datasets == name])),
            }
            for name in np.unique(datasets)
        },
        "features": {
            name: rank_correlation(table[rows, column])
            for column, name in enumerate(FEATURES)
        },
    }


def describe_clusters(points, scaled, frame, chosen, reference, nearest, seed, count=8):
    """Groups of similar series among `chosen`, each against the synthetic ones."""
    count = int(min(count, max(1, len(chosen) // 50)))
    if len(chosen) < 20:
        return [], np.full(len(frame), -1)
    model = KMeans(n_clusters=count, n_init=5, random_state=seed).fit(points[chosen])
    centre = np.median(scaled[reference], axis=0)
    labels = np.full(len(frame), -1)
    labels[chosen] = model.labels_
    clusters = []
    for cluster in np.argsort(-np.bincount(model.labels_, minlength=count)):
        rows = chosen[model.labels_ == cluster]
        shift = np.median(scaled[rows], axis=0) - centre
        distance = np.linalg.norm(
            points[rows] - model.cluster_centers_[cluster], axis=1
        )
        examples = rows[np.argsort(distance)]
        # Examples from different datasets where the cluster has several.
        picked, used = [], set()
        for row in examples:
            dataset = frame["dataset"].iat[row]
            if dataset not in used:
                picked.append(int(row))
                used.add(dataset)
        picked = picked[:4]
        for row in examples:
            if len(picked) == 4:
                break
            if int(row) not in picked:
                picked.append(int(row))
        clusters.append(
            {
                "cluster": int(cluster),
                "series": int(len(rows)),
                "datasets": (
                    frame.iloc[rows]
                    .groupby(["group", "dataset"])
                    .size()
                    .sort_values(ascending=False)
                    .head(4)
                    .reset_index(name="series")
                    .to_dict("records")
                ),
                "differs_most": [
                    {"feature": FEATURES[i], "shift": float(shift[i])}
                    for i in np.argsort(-np.abs(shift))[:6]
                ],
                "examples": picked,
                "nearest_synthetic": [int(nearest[row]) for row in picked],
            }
        )
    return clusters, labels


def map_sample(frame, options):
    """The series that are drawn: a fixed random subset of every group."""
    limits = options["map"]
    chosen = []
    for group, part in frame.groupby("group", sort=False):
        limit = limits["synthetic"] if group == "synthetic" else limits["per_group"]
        order = similarity_data.ranked(part["series"], options["seed"])
        chosen.append(part.index.to_numpy()[order[:limit]])
    return np.sort(np.concatenate(chosen))


def maps(points, sample, options, log, prefix=""):
    """Coordinates for drawing: principal components and t-SNE, 2-D and 3-D.

    The principal components keep distances as far as two or three straight
    axes can, and the axes can be read (their loadings are saved). t-SNE
    keeps neighbours together but not distances: islands on a t-SNE map show
    that neighbours differ, their size and spacing mean nothing.
    """
    pca = PCA(n_components=min(3, points.shape[1])).fit(points[sample])
    columns = {
        f"{prefix}pca_{axis + 1}": values
        for axis, values in enumerate(pca.transform(points[sample]).T)
    }
    for dimensions in (2, 3):
        embedding = TSNE(
            n_components=dimensions,
            perplexity=options["map"]["perplexity"],
            init="pca",
            random_state=options["seed"],
            n_jobs=options["workers"],
        ).fit_transform(points[sample])
        log(f"maps: {prefix}t-SNE in {dimensions} dimensions done")
        for axis in range(dimensions):
            columns[f"{prefix}tsne{dimensions}_{axis + 1}"] = embedding[:, axis]
    return columns, pca


def embedding_points(directory, points, frame, seed):
    """Embeddings reduced to 32 uncorrelated numbers, for the rows that have one."""
    path = similarity_data.embeddings_path(directory, points)
    if not path.is_file():
        return None, None
    saved = np.load(path)
    position = pd.Series(np.arange(len(saved["row"])), index=saved["row"])
    present = frame["row"].isin(position.index).to_numpy()
    vectors = saved["vectors"][position.loc[frame["row"][present]].to_numpy()]
    vectors = (vectors - vectors.mean(axis=0)) / (vectors.std(axis=0) + 1e-9)
    reduced = PCA(n_components=32, random_state=seed).fit_transform(vectors)
    return np.flatnonzero(present), reduced


def analyse_view(directory, points, options, log):
    frame, counts = load_view(directory, points)
    seed = options["seed"]
    control = control_rows(frame, seed)
    group = frame["group"].to_numpy()
    role = np.where(
        control, CONTROL, np.where(group == "synthetic", "reference", "benchmark")
    )
    frame["label"] = np.where(control, CONTROL, group)
    table = transformed(frame)
    scaled, points_, limits = standardise(table, group)
    log(f"view {points}: {len(frame):,} series, {dict(frame['label'].value_counts())}")

    found = coverage(points_, role, options)
    frame["ratio"], frame["covered"] = found["ratio"], found["covered"]
    frame["nearest_synthetic"] = np.where(
        found["nearest"] >= 0, frame["row"].to_numpy()[found["nearest"]], -1
    )
    robustness = {
        name: {
            label: float(
                (
                    found["distance"][
                        part.index.to_numpy(), int(name[1:].split("_")[0]) - 1
                    ]
                    <= limit
                ).mean()
            )
            for label, part in frame.groupby("label", sort=False)
        }
        for name, limit in found["grid"].items()
    }
    resembles, near_real = resembled(points_, frame, role, found, options)
    separated, frame["benchmark_likeness"] = separation(frame, table, role, seed, log)
    summary = {
        "points": points,
        "counts": counts,
        "limit": found["limit"],
        "typical_distance": found["typical"],
        "coverage": share_table(frame, found["covered"], found["ratio"], role),
        "robustness": robustness,
        "synthetic_resembling": resembles,
        "separation": separated,
        "accuracy": accuracy_link(frame, table, found["ratio"], options),
    }
    log(f"view {points}: coverage and separation done")

    reference = np.flatnonzero(role == "reference")
    real = frame["group"].isin(REAL).to_numpy()
    uncovered = np.flatnonzero(real & ~found["covered"])
    summary["uncovered_clusters"], labels = describe_clusters(
        points_, scaled, frame, uncovered, reference, found["nearest"], seed
    )
    frame["uncovered_cluster"] = labels
    frame["resembles_real"] = False
    if near_real is not None:
        frame.loc[reference, "resembles_real"] = near_real
        alone = reference[~near_real]
        summary["synthetic_alone_clusters"], labels = describe_clusters(
            points_,
            scaled,
            frame,
            alone,
            reference,
            np.full(len(frame), -1),
            seed,
            count=6,
        )
        frame["alone_cluster"] = labels
    # Clusters name positions in the frame; save rows of the collection.
    for cluster in summary["uncovered_clusters"] + summary.get(
        "synthetic_alone_clusters", []
    ):
        for key in ("examples", "nearest_synthetic"):
            cluster[key] = [
                int(frame["row"].iat[r]) if r >= 0 else -1 for r in cluster[key]
            ]

    sample = map_sample(frame, options)
    columns, pca = maps(points_, sample, options, log)
    summary["pca"] = {
        "explained": pca.explained_variance_ratio_.tolist(),
        "loadings": {
            name: pca.components_[:, column].tolist()
            for column, name in enumerate(FEATURES)
        },
    }
    present, reduced = embedding_points(directory, points, frame, seed)
    if present is not None:
        second = coverage(reduced, role[present], options)
        frame["embedding_ratio"], frame["embedding_covered"] = np.nan, False
        frame.loc[present, "embedding_ratio"] = second["ratio"]
        frame.loc[present, "embedding_covered"] = second["covered"]
        part = frame.iloc[present].reset_index(drop=True)
        keys = underlying(part)
        summary["embeddings"] = {
            "series": int(len(present)),
            "coverage": share_table(
                part, second["covered"], second["ratio"], role[present]
            ),
            "classifier_auc": {},
            "agreement": {},
        }
        reference_part = np.flatnonzero(part["label"] == "synthetic")
        for label, piece in part.groupby("label", sort=False):
            if label == "synthetic":
                continue
            chosen = piece.index.to_numpy()
            rows = np.concatenate([chosen, reference_part])
            positive = np.arange(len(rows)) < len(chosen)
            summary["embeddings"]["classifier_auc"][label] = classifier_auc(
                reduced[rows], positive, keys[rows], seed
            )
            summary["embeddings"]["agreement"][label] = {
                "rank_correlation_of_distances": float(
                    spearmanr(piece["ratio"], piece["embedding_ratio"]).statistic
                ),
                "same_verdict": float(
                    (piece["covered"] == piece["embedding_covered"]).mean()
                ),
            }
        in_both = np.flatnonzero(np.isin(present, sample))
        more, _ = maps(reduced, in_both, options, log, prefix="embedding_")
        for name, values in more.items():
            frame[name] = np.nan
            frame.loc[present[in_both], name] = values
        log(f"view {points}: embeddings done")
    frame["in_map"] = False
    frame.loc[sample, "in_map"] = True
    for name, values in columns.items():
        frame[name] = np.nan
        frame.loc[sample, name] = values
    scaled_frame = pd.DataFrame(scaled, columns=[f"z_{name}" for name in FEATURES])
    pd.concat([frame, scaled_frame], axis=1).to_parquet(
        directory / f"analysis_{points}.parquet", index=False
    )
    (directory / f"analysis_{points}.json").write_text(
        json.dumps(summary, indent=1), encoding="utf-8"
    )
    return summary


def analyse(options, log=print):
    directory = similarity_data.output_directory(options)
    return {
        points: analyse_view(directory, points, options, log)
        for points in options["window_points"]
    }

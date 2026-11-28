"""Chronos-2 encoder embeddings of the collected windows, as a second opinion.

The descriptors of series_features.py are chosen by hand, so they could miss
a difference between the groups. An embedding from a forecasting model that
was trained on real data is an independent description of the same windows:
the average of the encoder's outputs over the patches of the window, one
vector of 768 numbers per series.

Basin's own embedding is deliberately not used for this. Basin has only seen
synthetic series, so where it places a series unlike those is not
informative about how unlike it is.

Only the shortest view is embedded; it holds the most series and costs the
least. Runs on the CPU.
"""

from pathlib import Path

import numpy as np
import similarity_data

ROOT = Path(__file__).resolve().parents[2]


def load_model(config, options):
    from chronos import Chronos2Pipeline

    chosen = options["embeddings"]["model_path"]
    if chosen:
        return Chronos2Pipeline.from_pretrained(chosen, device_map="cpu")
    holdout = config["holdout"]
    return Chronos2Pipeline.from_pretrained(
        holdout["chronos"]["model_id"],
        revision=holdout["chronos"]["revision"],
        cache_dir=str(ROOT / holdout["model_cache_dir"]),
        device_map="cpu",
    )


def filled(window):
    """Gaps filled the way the descriptors fill them, or None without a shape."""
    import series_features

    prepared = series_features.prepare(window)
    return None if prepared is None else prepared[1].astype(np.float32)


def embed(config, options, log=print):
    import torch

    if not options["embeddings"]["enabled"]:
        log("embed: switched off (dataset_similarity.embeddings.enabled)")
        return
    torch.set_num_threads(options["embeddings"]["threads"])
    directory = similarity_data.output_directory(options)
    index, windows = similarity_data.load(directory)
    points = min(options["window_points"])
    in_view = similarity_data.view_rows(index, points)
    chosen = []
    for group in similarity_data.GROUPS:
        part = in_view[(index["group"] == group).to_numpy()[in_view]]
        limit = options["embeddings"][
            "synthetic" if group == "synthetic" else "per_group"
        ]
        # The same fixed order as the maps use, so that what is drawn has one.
        order = similarity_data.ranked(
            index["series"].to_numpy()[part], options["seed"]
        )
        chosen.append(part[order[:limit]])
    rows, series = [], []
    for position in np.sort(np.concatenate(chosen)):
        values = filled(windows[position, -points:])
        if values is not None:
            rows.append(position)
            series.append(values)
    model = load_model(config, options)
    batch, vectors = options["embeddings"]["batch_size"], []
    for start in range(0, len(series), batch * 20):
        part = np.stack(series[start : start + batch * 20])[:, None, :]
        outputs, _ = model.embed(part, batch_size=batch, context_length=points)
        # The last two positions are not patches of the window.
        vectors += [output[0, :-2].mean(dim=0).numpy() for output in outputs]
        log(f"embed {points}: {len(vectors):,} of {len(series):,} series")
    np.savez(
        similarity_data.embeddings_path(directory, points),
        row=np.asarray(rows),
        vectors=np.stack(vectors).astype(np.float32),
    )

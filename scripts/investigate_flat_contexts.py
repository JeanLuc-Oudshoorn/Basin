"""Controlled normalization/flat-context probes using frozen trained weights."""
# ruff: noqa: E402

import hashlib
import json
import sys
from pathlib import Path
from types import MethodType

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import torch
from plotly.subplots import make_subplots

ROOT = Path(__file__).resolve().parents[1]
for stage in (
    "03_transforming_input_data",
    "04_creating_embeddings",
    "05_model_training",
):
    sys.path.insert(0, str(ROOT / "src" / stage))
from fast_inference import configure_fast_inference
from inference import forecast_contexts
from train import RWKVForecaster
from transform import file_digest, load_config, run_context, write_json


def natural_tokenize(original, context, **kwargs):
    """Diagnostic only: use each positive prefix standard deviation as scale."""
    prepared = original(context, **kwargs)
    past = kwargs.get("past_covariates")
    history = context if past is None else torch.cat([context, past], dim=1)
    patch = 16
    history = torch.nn.functional.pad(
        history.float(), ((-history.shape[-1]) % patch, 0), value=float("nan")
    )
    values = history.reshape(1, history.shape[1], -1, patch)
    valid = torch.isfinite(values)
    clean = values.nan_to_num().double()
    count = valid.sum(-1, keepdim=True).cumsum(-2).clamp_min(1)
    loc = clean.sum(-1, keepdim=True).cumsum(-2) / count
    std = (
        (clean.square().sum(-1, keepdim=True).cumsum(-2) / count - loc.square())
        .clamp_min(0)
        .sqrt()
    )
    scale = torch.where(std > 0, std, torch.ones_like(std)).float()
    prepared["scale"] = scale
    prepared["features"] = prepared["features"].clone()
    prepared["features"][..., :patch] = (
        ((values - loc.float()) / scale).nan_to_num().clamp(-1e4, 1e4)
    )
    return prepared


@torch.inference_mode()
def investigate(config):
    output = ROOT / "docs/audits/2026-11-04/flat_contexts"
    output.mkdir(parents=True, exist_ok=True)
    checkpoint = ROOT / "results/model/training/model.pt"
    metadata = torch.load(checkpoint, mmap=True, map_location="cpu", weights_only=True)
    torch.set_num_threads(4)
    model = RWKVForecaster.load(checkpoint, checkpoint=metadata).eval()
    # Keep graph capture and compilation out of the normalization experiment.
    configure_fast_inference(model, {"inference_cuda_graphs": True})
    if model.fast_inference is not None:
        model.fast_inference.graphs.enabled = False
    original = model.tokenize_next
    evidence = {
        "checkpoint_sha256": file_digest(checkpoint),
        "training_summary": metadata.get("training_summary"),
        "cases": [],
    }
    records = []
    for table in ("019943", "072778"):
        source = f"data/01_generated/2026-10-31/table_285516885900_{table}/data.parquet"
        raw = pd.read_parquet(ROOT / source)
        for origin in (32, 144) if table == "019943" else (32, 48):
            observed = raw.outcome.iloc[:origin].to_numpy(dtype=np.float32)
            covariates = [c for c in raw if c not in ("time_index", "outcome")]
            past = (
                torch.tensor(
                    raw[covariates].iloc[:origin].to_numpy(dtype=np.float32).T.copy()
                )[None]
                if covariates
                else None
            )
            experiments = [
                ("baseline", observed, past, False),
                ("natural_positive_scale", observed, past, True),
                ("target_only", observed, None, False),
            ]
            if table == "019943" and origin == 32:
                experiments += [
                    (
                        f"amplitude_x{factor}",
                        observed.mean() + (observed - observed.mean()) * factor,
                        past,
                        False,
                    )
                    for factor in (0.1, 10.0, 1000.0)
                ]
                experiments += [
                    ("exact_zero_with_covariate", np.zeros_like(observed), past, False),
                    ("exact_zero_target_only", np.zeros_like(observed), None, False),
                ]
            for variant, values, cov, natural in experiments:
                context = torch.tensor(values.copy())[None, None]
                if natural:
                    model.tokenize_next = MethodType(
                        lambda self, context, **kw: natural_tokenize(
                            original, context, **kw
                        ),
                        model,
                    )
                else:
                    model.tokenize_next = original
                prepared = model.tokenize_next(context, past_covariates=cov)
                prediction = list(
                    forecast_contexts(
                        model,
                        [{"context": context, "past_covariates": cov}],
                        1,
                        prediction_patches=4,
                    )
                )[0][0, :, 4].numpy()
                last_loc = float(prepared["loc"][0, 0, -1, 0])
                scale = float(prepared["scale"][0, 0, -1, 0])
                actual = raw.outcome.iloc[origin : origin + 64].to_numpy()
                weights = np.repeat([0.8, 0.16, 0.03, 0.01], 16)[: len(actual)]
                row = {
                    "table": table,
                    "origin": origin,
                    "variant": variant,
                    "context_std": float(np.std(values)),
                    "encoder_scale": scale,
                    "location": last_loc,
                    "prediction_min": float(prediction.min()),
                    "prediction_max": float(prediction.max()),
                    "prediction_std": float(prediction.std()),
                    "first_patch_min": float(prediction[:16].min()),
                    "first_patch_max": float(prediction[:16].max()),
                    "weighted_mae": float(
                        np.average(
                            np.abs(prediction[: len(actual)] - actual), weights=weights
                        )
                    )
                    if variant in ("baseline", "natural_positive_scale", "target_only")
                    else None,
                }
                if variant == "baseline":
                    row["output_only_rescaling_std"] = float(
                        prediction.std() * np.std(values) / scale
                    )
                evidence["cases"].append(row)
                records.extend(
                    {
                        "table": table,
                        "origin": origin,
                        "variant": variant,
                        "time_index": origin + i,
                        "prediction": float(value),
                    }
                    for i, value in enumerate(prediction)
                )
                print(json.dumps(row), flush=True)
    model.tokenize_next = original
    # A reproducible sample of actual training sources; no holdout sources here.
    sources = metadata["dataset_split"]["train_sources"]
    ranked = sorted(
        sources, key=lambda s: hashlib.sha256(f"flat-audit:{s}".encode()).digest()
    )[:512]
    sample = []
    for source in ranked:
        raw = pd.read_parquet(ROOT / source, columns=["outcome"])
        values = raw.outcome.to_numpy(dtype=np.float32)
        stop = len(values) - 64
        prefix_stds = [
            float(np.std(values[:end].astype(np.float64)))
            for end in range(16, stop, 16)
        ]
        sample.append(
            {
                "source": source,
                "training_tokens": len(prefix_stds),
                "near_flat_tokens": sum(s <= 1e-5 for s in prefix_stds),
                "first_two_token_std": float(np.std(values[:32].astype(np.float64))),
            }
        )
    evidence["training_sample"] = {
        "sources_in_checkpoint": len(sources),
        "sampled_tables": len(sample),
        "tables_with_near_flat_training_prefix": sum(
            s["near_flat_tokens"] > 0 for s in sample
        ),
        "near_flat_first_two_token_contexts": sum(
            s["first_two_token_std"] <= 1e-5 for s in sample
        ),
        "near_flat_prefix_tokens": sum(s["near_flat_tokens"] for s in sample),
        "prefix_tokens": sum(s["training_tokens"] for s in sample),
        "selection": "512 smallest SHA256(flat-audit:source), training partition only; global prefix target std; reserved final 64 points excluded",
    }
    write_json(output / "evidence.json", evidence)
    pd.DataFrame(sample).to_parquet(output / "training_sample.parquet", index=False)
    pd.DataFrame(records).to_parquet(output / "probes.parquet", index=False)
    figure = make_subplots(
        rows=2,
        cols=1,
        subplot_titles=[
            "019943: first 32 observations (baseline vs scale probe)",
            "072778: first 32 observations (with and without covariates)",
        ],
    )
    for panel, table in enumerate(("019943", "072778"), 1):
        raw = pd.read_parquet(
            ROOT
            / f"data/01_generated/2026-10-31/table_285516885900_{table}/data.parquet"
        )
        figure.add_trace(
            go.Scatter(
                x=np.arange(96),
                y=raw.outcome.iloc[:96],
                name=f"{table} actual",
                line={"color": "black"},
            ),
            row=panel,
            col=1,
        )
        saved = pd.DataFrame(records)
        for variant in ("baseline", "natural_positive_scale", "target_only"):
            segment = saved[
                (saved.table == table)
                & (saved.origin == 32)
                & (saved.variant == variant)
            ]
            figure.add_trace(
                go.Scatter(
                    x=segment.time_index,
                    y=segment.prediction,
                    name=f"{table} {variant}",
                ),
                row=panel,
                col=1,
            )
    figure.update_layout(
        height=850,
        template="plotly_white",
        title="Frozen-weight probes; scale variants are diagnostic, not deployed fixes",
    )
    figure.write_html(
        output / "normalization_probes.html",
        include_plotlyjs=True,
        config={"scrollZoom": True},
    )
    print(json.dumps(evidence["training_sample"]), flush=True)


def main():
    config = load_config()
    with run_context(config, "flat_context_investigation"):
        investigate(config)


if __name__ == "__main__":
    main()

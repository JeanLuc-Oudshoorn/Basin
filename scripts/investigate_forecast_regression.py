"""Matched trained-checkpoint inference ablations; production artifacts are read-only."""

# ruff: noqa: E402
# isort: skip_file
import argparse
import gc
import json
import sys
from pathlib import Path
from time import perf_counter

ROOT = Path(__file__).resolve().parents[1]
for stage in ("03_transforming_input_data", "05_model_training", "06_holdout_test"):
    sys.path.insert(0, str(ROOT / "src" / stage))
sys.path.insert(0, str(ROOT))
from train import RWKVForecaster
import numpy as np
import pandas as pd
import torch
from fast_inference import FastInference
from inference import forecast_contexts
from performance import configure_cuda_memory
from variate_attention import configure_variate_execution
from transform import load_config, logger, run_context, write_json

OUT = ROOT / "docs/audits/2026-11-02/forecast_regression"
MODES = ("native", "compiled_native", "fast_no_replay", "fast_replay")


def metrics(frame, name):
    error = frame[name].to_numpy() - frame.actual.to_numpy()
    denominator = frame.mase_scale.to_numpy()
    valid = np.isfinite(denominator) & (denominator > 0)
    return dict(
        rmse=float(np.sqrt(np.mean(error**2))),
        mae=float(np.abs(error).mean()),
        mase=float(np.mean(np.abs(error[valid]) / denominator[valid])),
        mase_points=int(valid.sum()),
        points=len(frame),
    )


@torch.inference_mode()
def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", choices=["model", "447m_model"], required=True)
    parser.add_argument("--limit", type=int)
    parser.add_argument("--cohorts", nargs="+", default=["model", "447m_model"])
    parser.add_argument("--skip-completed", action="store_true")
    args = parser.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)
    config = load_config()
    torch.set_num_threads(4)
    configure_cuda_memory(config["training"])
    with run_context(config, "forecast_regression_" + args.model) as logs:
        path = ROOT / "results" / args.model / "training/model.pt"
        checkpoint = torch.load(path, map_location="cpu", weights_only=True, mmap=True)
        model = RWKVForecaster.load(path, checkpoint=checkpoint).eval()
        metadata = {
            k: checkpoint.get(k)
            for k in ("config", "training_summary", "pipeline_config")
        }
        write_json(OUT / f"{args.model}_checkpoint_metadata.json", metadata)
        del checkpoint
        configure_variate_execution(model, config["training"], ROOT)
        compiled = [mix.compiled_forward for mix in model.variate]
        summaries = []
        for cohort in args.cohorts:
            result_name = f"{args.model}_on_{cohort}" + (
                f"_limit{args.limit}" if args.limit else ""
            )
            if args.skip_completed and (OUT / (result_name + ".json")).exists():
                summaries.append(
                    json.loads((OUT / (result_name + ".json")).read_text())
                )
                continue
            base = ROOT / "results" / cohort / "holdout_test"
            saved = pd.read_parquet(base / "test_predictions.parquet")
            policy = json.loads((base / "sample_selection.json").read_text())
            sources = policy["selected_sources"]
            recovered = (
                torch.load(OUT / "recovered_old_inputs.pt", weights_only=True)
                if cohort == "447m_model"
                else None
            )
            if args.limit:
                indices = np.linspace(
                    0, len(sources) - 1, min(args.limit, len(sources)), dtype=int
                )
                sources = [sources[i] for i in indices]
            origin_rows, predictions = [], []
            skipped_origins = []
            modes = {
                name: FastInference(
                    model, cuda_graphs=name == "fast_replay", max_graphs=8
                )
                for name in ("fast_no_replay", "fast_replay")
            }
            started = perf_counter()
            for table_index, source in enumerate(sources):
                if recovered is None:
                    raw = pd.read_parquet(ROOT / source)
                else:
                    archive = recovered[source]
                    raw = pd.DataFrame(
                        archive["values"].numpy().T, columns=archive["columns"]
                    )
                selected = saved[saved.source.eq(source)]
                targets = list(selected.target.unique())
                data_policy = metadata["pipeline_config"]["data"]
                import re

                covariates = [
                    c
                    for c in raw.columns
                    if c not in targets + [data_policy["time_column"]]
                    and not any(
                        re.search(p, c) for p in data_policy["excluded_column_patterns"]
                    )
                ]
                for origin in sorted(selected.origin.unique()):
                    if (
                        recovered is not None
                        and covariates
                        and origin > archive["cutoff"]
                    ):
                        skipped_origins.append(
                            dict(
                                source=source,
                                origin=int(origin),
                                reason="Archived covariates stop before final partial-patch origin",
                            )
                        )
                        continue
                    history = raw.iloc[max(0, origin - 2048) : origin]
                    context = dict(
                        context=torch.tensor(
                            history[targets].to_numpy(dtype=np.float32).T.copy()
                        )[None],
                        past_covariates=torch.tensor(
                            history[covariates].to_numpy(dtype=np.float32).T.copy()
                        )[None]
                        if covariates
                        else None,
                    )
                    outputs = {}
                    for mode in MODES:
                        for mix, forward in zip(model.variate, compiled):
                            mix.compiled_forward = None if mode == "native" else forward
                        model.fast_inference = modes.get(mode)
                        outputs[mode] = list(forecast_contexts(model, [context], 1))[
                            0
                        ].numpy()
                    rows = selected[selected.origin.eq(origin)].copy()
                    for mode, quantiles in outputs.items():
                        values = []
                        for _, row in rows.iterrows():
                            values.append(
                                float(
                                    quantiles[
                                        targets.index(row.target),
                                        int(row.time_index - origin),
                                        4,
                                    ]
                                )
                            )
                        rows[mode] = values
                        normalized_error = np.abs(outputs[mode] - outputs["native"])
                        scales = max(
                            float(np.nanstd(history[targets].to_numpy())), 1e-5
                        )
                        origin_rows.append(
                            dict(
                                source=source,
                                origin=int(origin),
                                mode=mode,
                                max_quantile_difference=float(normalized_error.max()),
                                max_difference_over_context_std=float(
                                    normalized_error.max() / scales
                                ),
                                max_quantile_difference_from_compiled=float(
                                    np.max(
                                        np.abs(
                                            outputs[mode] - outputs["compiled_native"]
                                        )
                                    )
                                ),
                            )
                        )
                    predictions.append(rows)
                if (table_index + 1) % 20 == 0 or table_index == 0:
                    logger.info(
                        "Regression model={} cohort={} tables={}/{} elapsed={:.1f}s",
                        args.model,
                        cohort,
                        table_index + 1,
                        len(sources),
                        perf_counter() - started,
                    )
            frame = pd.concat(predictions, ignore_index=True)
            name = f"{args.model}_on_{cohort}" + (
                f"_limit{args.limit}" if args.limit else ""
            )
            frame.to_parquet(OUT / (name + ".parquet"), index=False)
            pd.DataFrame(origin_rows).to_parquet(
                OUT / (name + "_equivalence.parquet"), index=False
            )
            result = dict(
                model=args.model,
                cohort=cohort,
                tables=len(sources),
                seconds=perf_counter() - started,
                metrics={
                    mode: metrics(frame, mode)
                    for mode in (*MODES, "rwkv", "naive", "chronos", "timestral")
                    if mode in frame
                },
                max_quantile_difference={
                    mode: max(
                        r["max_quantile_difference"]
                        for r in origin_rows
                        if r["mode"] == mode
                    )
                    for mode in MODES
                },
                graph_profiles=len(modes["fast_replay"].graphs.entries),
                skipped_origins=skipped_origins,
                logs=str(logs),
            )
            summaries.append(result)
            write_json(OUT / (name + ".json"), result)
            logger.info("Regression result {}", result)
            model.fast_inference = None
            for engine in modes.values():
                engine.clear()
            del modes, frame, predictions, saved
            gc.collect()
            torch.cuda.empty_cache()
        write_json(OUT / f"{args.model}_complete.json", summaries)


if __name__ == "__main__":
    main()

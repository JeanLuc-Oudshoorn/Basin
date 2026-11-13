"""Short controlled fine-tuning diagnostic; no production checkpoint is modified."""

# ruff: noqa: E402
# isort: skip_file
import gc
import random
import sys
from dataclasses import replace
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
from batching import make_batches
from epoch_training import accumulated_update
from inference import forecast_contexts
from performance import configure_cuda_memory, make_optimizer, prepared_updates
from variate_attention import configure_variate_execution
from transform import load_config, logger, run_context, write_json
from investigate_forecast_regression import OUT, metrics


@torch.inference_mode()
def evaluate(model, cohort):
    saved = pd.read_parquet(
        ROOT / "results" / cohort / "holdout_test/test_predictions.parquet"
    )
    recovered = (
        torch.load(OUT / "recovered_old_inputs.pt", weights_only=True)
        if cohort == "447m_model"
        else None
    )
    output = []
    for source, rows in saved.groupby("source", sort=False):
        if recovered is None:
            raw = pd.read_parquet(ROOT / source)
            covariates = [c for c in raw if c not in ("outcome", "time_index")]
        else:
            item = recovered[source]
            raw = pd.DataFrame(item["values"].numpy().T, columns=item["columns"])
            covariates = [c for c in raw if c != "outcome"]
        for origin, case in rows.groupby("origin", sort=False):
            if recovered is not None and covariates and origin > item["cutoff"]:
                continue
            history = raw.iloc[max(0, origin - 2048) : origin]
            context = dict(
                context=torch.tensor(
                    history[["outcome"]].to_numpy(dtype=np.float32).T.copy()
                )[None],
                past_covariates=torch.tensor(
                    history[covariates].to_numpy(dtype=np.float32).T.copy()
                )[None]
                if covariates
                else None,
            )
            q = list(forecast_contexts(model, [context], 1))[0].numpy()
            case = case.copy()
            case["diagnostic"] = [
                float(q[0, int(t - origin), 4]) for t in case.time_index
            ]
            output.append(case)
    frame = pd.concat(output, ignore_index=True)
    return metrics(frame, "diagnostic"), frame


def main():
    config = load_config()
    torch.set_num_threads(4)
    configure_cuda_memory(config["training"])
    with run_context(config, "regression_objective_ablation") as logs:
        checkpoint = torch.load(
            ROOT / "results/model/training/model.pt",
            map_location="cpu",
            weights_only=True,
            mmap=True,
        )
        model = RWKVForecaster.load(None, checkpoint=checkpoint).train()
        configure_variate_execution(
            model, dict(config["training"], training_cuda_graphs=False), ROOT
        )
        model.checkpoint_token_threshold = config["training"][
            "checkpoint_token_threshold"
        ]
        model.fast_inference = None
        initial = {k: v.detach().cpu() for k, v in model.state_dict().items()}
        sources = list(checkpoint["dataset_split"]["train_sources"])
        random.Random(123).shuffle(sources)  # nosec B311
        sources = sources[:8192]
        if set(sources) & set(checkpoint["dataset_split"]["test_sources"]):
            raise ValueError("Holdout leakage")
        records = []
        for index, source in enumerate(sources):
            frame = pd.read_parquet(ROOT / source).iloc[:-16]
            covariates = [c for c in frame if c not in ("outcome", "time_index")]
            prepared = model.tokenize_next(
                torch.tensor(frame[["outcome"]].to_numpy(dtype=np.float32).T.copy())[
                    None
                ],
                past_covariates=torch.tensor(
                    frame[covariates].to_numpy(dtype=np.float32).T.copy()
                )[None]
                if covariates
                else None,
                training=True,
            )
            records.append(
                dict(source=source, id=source, split="train", prepared=prepared)
            )
            if (index + 1) % 2000 == 0:
                logger.info(
                    "Prepared controlled training inputs {}/{}", index + 1, len(sources)
                )
        del checkpoint
        batches = make_batches(records, 8, 123, bucket_by_length=True)[:1024]
        options = dict(
            config["training"], learning_rate=1e-5, training_cuda_graphs=False
        )
        write_json(
            OUT / "objective_ablation_settings.json",
            dict(
                sources=sources,
                batches=len(batches),
                learning_rate=1e-5,
                microbatch=8,
                gradient_clip=1.0,
                seed=123,
                weights=[0.5, 0.0],
                description="Fresh AdamW, same best 307M state and same batches for each arm; short fine-tuning diagnostic, not a from-scratch size ablation",
                logs=str(logs),
            ),
        )
        results = []
        for weight in (0.5, 0.0):
            model.zero_grad(set_to_none=True)
            gc.collect()
            torch.cuda.empty_cache()
            model.load_state_dict(initial)
            model.config = replace(model.config, multi_horizon_weight=weight)
            model.train()
            torch.manual_seed(123)
            optimizer = make_optimizer(model, options)
            updates = prepared_updates(
                batches,
                1,
                prefetch=True,
                pin_memory=True,
                microbatch_size=8,
                microbatch_token_limit=1536,
            )
            losses = []
            started = perf_counter()
            for index, (_, group, prepared) in enumerate(updates, 1):
                result = accumulated_update(model, group, optimizer, 1.0, prepared)
                losses.append(result["loss"])
                if index % 100 == 0:
                    logger.info(
                        "Objective weight={} updates={}/{} loss={} elapsed={:.1f}s",
                        weight,
                        index,
                        len(batches),
                        result["loss"],
                        perf_counter() - started,
                    )
            del optimizer
            model.eval()
            scores = {}
            for cohort in ("model", "447m_model"):
                scores[cohort], predictions = evaluate(model, cohort)
                predictions.to_parquet(
                    OUT / f"objective_weight_{weight}_on_{cohort}.parquet", index=False
                )
            results.append(
                dict(
                    weight=weight,
                    scores=scores,
                    losses=losses,
                    seconds=perf_counter() - started,
                )
            )
            write_json(OUT / "objective_ablation_results.json", results)
            logger.info(
                "Objective arm completed {}",
                {k: v for k, v in results[-1].items() if k != "losses"},
            )


if __name__ == "__main__":
    main()

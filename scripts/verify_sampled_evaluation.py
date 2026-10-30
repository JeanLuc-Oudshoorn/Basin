"""Check compiled inference on trained weights and run an isolated holdout smoke."""

# ruff: noqa: E402
import copy
import json
import sys
from pathlib import Path

import torch
from loguru import logger

ROOT = Path(__file__).resolve().parents[1]
for stage in (
    "03_transforming_input_data",
    "04_creating_embeddings",
    "05_model_training",
    "06_holdout_test",
    "07_forecasting",
):
    sys.path.insert(0, str(ROOT / "src" / stage))
from embed import LazyPatchRecord, PatchEncoder
from holdout import evaluate_holdout
from inference import forecast_contexts
from performance import configure_cuda_memory
from train import RWKVForecaster
from training_log import training_run
from training_protocol import validate
from transform import file_digest, load_config, location, write_json
from variate_attention import configure_variate_execution


def main():
    config = load_config()
    output = ROOT / "docs/audits/2026-10-29/sampled_evaluation"
    output.mkdir(parents=True, exist_ok=True)
    logger.remove()
    logger.add(ROOT / "logs/20261029_sampled_gpu_verification.log", level="INFO")
    torch.set_num_threads(4)
    configure_cuda_memory(config["training"])
    checkpoint = location(config, "training") / "model.pt"
    model = RWKVForecaster.load(checkpoint).eval()
    raw = torch.load(
        ROOT / "docs/audits/2026-10-28/training_speed/sample.pt",
        map_location="cpu",
        weights_only=False,
    )  # nosec B614
    encoder = PatchEncoder(model.config)
    records = [LazyPatchRecord(r, encoder) for r in raw["validation"]]
    eager = validate(model, records, 16)
    configure_variate_execution(model, config["training"], ROOT)
    compiled = validate(model, records, 16)
    relative_loss = abs(eager["loss"] - compiled["loss"]) / max(
        abs(eager["loss"]), 1e-9
    )
    if relative_loss > 0.01:
        raise AssertionError(
            f"Compiled inference validation loss changed: {relative_loss}"
        )
    contexts = [
        {k: r["batch"][k] for k in ("context", "past_covariates") if k in r["batch"]}
        for r in raw["validation"][:8]
    ]
    errors = []
    for c in contexts:
        expected = (
            model.forecast(
                c["context"].cuda(),
                16,
                past_covariates=c["past_covariates"].cuda()
                if c.get("past_covariates") is not None
                else None,
            )
            .quantiles.float()
            .cpu()[0]
        )
        actual = list(forecast_contexts(model, [c, c], 2))[0]
        error = float((actual - expected).norm() / expected.norm().clamp_min(1e-9))
        if error > 0.02:
            raise AssertionError(f"Batched prediction changed: {error}")
        errors.append(error)
    split = json.loads(
        (location(config, "training") / "dataset_split.json").read_text()
    )
    sources = split["test_sources"][:3]
    sample = [
        {
            "source": s,
            "source_sha256": file_digest(ROOT / s),
            "targets": config["data"]["targets"],
            "roles": {"past": []},
        }
        for s in sources
    ]
    import pandas as pd

    for r in sample:
        frame = pd.read_parquet(ROOT / r["source"])
        r["roles"]["past"] = [
            c
            for c in frame
            if c not in config["data"]["targets"] + [config["data"]["time_column"]]
        ]
    smoke = copy.deepcopy(config)
    smoke["paths"]["holdout_test"] = str(output / "smoke" / "holdout_test")
    smoke["plot"]["max_plots"] = 2
    with training_run(smoke, "sampled_evaluation_smoke"):
        result = evaluate_holdout(model, sample, smoke)
    report = {
        "status": "passed",
        "checkpoint": str(checkpoint),
        "parameters": sum(p.numel() for p in model.parameters()),
        "validation_tables": len(records),
        "eager_loss": eager["loss"],
        "compiled_loss": compiled["loss"],
        "relative_loss_difference": relative_loss,
        "max_batch_relative_prediction_error": max(errors),
        "smoke_tables": result["table_count"],
        "smoke_origins": result["origin_target_count"],
        "smoke_seconds": result["seconds"],
        "pooled": result["pooled"],
    }
    write_json(output / "gpu_verification.json", report)
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()

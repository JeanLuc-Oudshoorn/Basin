"""Bounded inference experiments on the full model and a saved real-data sample."""

# ruff: noqa: E402
import json
import sys
from pathlib import Path
from time import perf_counter
from types import MethodType

import torch
from loguru import logger

ROOT = Path(__file__).resolve().parents[1]
for stage in (
    "03_transforming_input_data",
    "04_creating_embeddings",
    "05_model_training",
):
    sys.path.insert(0, str(ROOT / "src" / stage))
from embed import ForecastConfig, LazyPatchRecord, PatchEncoder
from inference import forecast_contexts
from performance import configure_cuda_memory
from train import RWKVForecaster
from training_protocol import validate_batched
from transform import load_config, write_json
from variate_attention import compiled_attention, configure_variate_execution


def main():
    config = load_config()
    torch.set_num_threads(4)
    configure_cuda_memory(config["training"])
    output = ROOT / "docs/audits/2026-10-29/sampled_evaluation"
    output.mkdir(parents=True, exist_ok=True)
    logger.remove()
    logger.add(ROOT / "logs/20261029_inference_benchmark.log", level="INFO")
    raw = torch.load(
        ROOT / "docs/audits/2026-10-28/training_speed/sample.pt",
        weights_only=False,
        map_location="cpu",
    )  # nosec B614
    settings = ForecastConfig(
        **dict(config["model"], quantiles=tuple(config["model"]["quantiles"]))
    )
    torch.manual_seed(7)
    model = RWKVForecaster(settings).cuda().bfloat16().eval()
    configure_variate_execution(
        model, dict(config["training"], compile_variate_inference=False), ROOT
    )
    encoder = PatchEncoder(settings)
    records = [LazyPatchRecord(r, encoder) for r in raw["validation"]]
    report = {
        "parameters": sum(p.numel() for p in model.parameters()),
        "tables": len(records),
        "validation": {},
    }
    original = [m.forward for m in model.variate]
    for case in ("no_grad", "inference_mode", "compiled_inference"):
        if case == "compiled_inference":
            fn = compiled_attention()
            for mix in model.variate:
                mix.forward = MethodType(lambda self, x: fn(self, x), mix)
        timings = []
        for repeat in range(3):
            context = torch.no_grad() if case == "no_grad" else torch.inference_mode()
            with context:
                torch.cuda.synchronize()
                started = perf_counter()
                result = validate_batched.__wrapped__(model, records, 16)
                torch.cuda.synchronize()
                elapsed = perf_counter() - started
            if repeat:
                timings.append(elapsed)
        report["validation"][case] = {
            "seconds": timings,
            "tables_per_second": len(records) / (sum(timings) / len(timings)),
            "loss": result["loss"],
            "rmse": result["rmse"],
            "peak_gib": torch.cuda.max_memory_allocated() / 2**30,
        }
        print(case, report["validation"][case], flush=True)
    for mix, forward in zip(model.variate, original):
        mix.forward = forward
    record = max(raw["validation"], key=lambda r: r["batch"]["context"].shape[-1])
    batch = record["batch"]
    length = batch["context"].shape[-1]
    sizes = [
        32,
        max(32, length // 4),
        max(32, length // 2),
        max(32, 3 * length // 4),
        length,
    ]
    contexts = [
        {
            "context": batch["context"][..., :n],
            "past_covariates": batch.get("past_covariates")[..., :n]
            if batch.get("past_covariates") is not None
            else None,
        }
        for n in sizes
    ]
    forecasts = {}
    for case in ("sequential", "batched"):
        timings = []
        for repeat in range(11):
            torch.cuda.synchronize()
            started = perf_counter()
            if case == "batched":
                values = list(forecast_contexts(model, contexts, 5))
            else:
                values = [
                    model.forecast(
                        c["context"].cuda(),
                        16,
                        past_covariates=c["past_covariates"].cuda()
                        if c["past_covariates"] is not None
                        else None,
                    )
                    .quantiles.float()
                    .cpu()[0]
                    for c in contexts
                ]
            torch.cuda.synchronize()
            if repeat:
                timings.append(perf_counter() - started)
        forecasts[case] = torch.stack(values)
        report[case] = {"seconds_per_five_origins": sum(timings) / len(timings)}
    error = (forecasts["batched"] - forecasts["sequential"]).norm() / forecasts[
        "sequential"
    ].norm().clamp_min(1e-9)
    report["batch_relative_prediction_error"] = float(error)
    write_json(output / "inference_benchmark.json", report)
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()

"""Exercise two complete GPU epochs, prefetch, validation, JSONL and checkpoint restore."""
# ruff: noqa: E402

import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
for stage in (
    "03_transforming_input_data",
    "04_creating_embeddings",
    "05_model_training",
):
    sys.path.insert(0, str(ROOT / "src" / stage))
from embed import ForecastConfig, LazyPatchRecord, PatchEncoder
from epoch_training import fit_epochs
from performance import make_optimizer
from train import RWKVForecaster
from training_protocol import validate
from transform import load_config, run_context, write_json


def main():
    config = load_config()
    directory = ROOT / "docs/audits/2026-10-28/training_speed/gpu_smoke"
    config["paths"]["training"] = str(directory)
    config["model"].update(width=128, layers=2, variate_heads=4)
    config["training"].update(
        batch_size=4,
        gradient_accumulation_steps=2,
        validation_batch_size=4,
        epochs=2,
        steps=None,
        checkpoint_every=0,
        checkpoint_token_threshold=128,
    )
    torch.set_num_threads(4)
    torch.manual_seed(config["seed"])
    with run_context(config, "train"):
        raw = torch.load(
            directory.parent / "sample.pt", map_location="cpu", weights_only=False
        )  # nosec B614
        settings = ForecastConfig(
            **dict(config["model"], quantiles=tuple(config["model"]["quantiles"]))
        )
        encoder = PatchEncoder(settings)
        records = [LazyPatchRecord(r, encoder) for r in raw["train"][:32]]
        validation = [LazyPatchRecord(r, encoder) for r in raw["validation"][:16]]
        model = RWKVForecaster(settings).to(device="cuda", dtype=torch.bfloat16)
        model.checkpoint_token_threshold = config["training"][
            "checkpoint_token_threshold"
        ]
        optimizer = make_optimizer(model, config["training"])
        history, summary, _ = fit_epochs(model, records, validation, config, optimizer)
        assert summary["completed_epochs"] == 2
        assert summary["training_windows_processed"] == 64
        assert summary["restored_best_weights"]
        a = validate(model, validation, batch_size=1)
        b = validate(model, validation, batch_size=4)
        assert a["points"] == b["points"] == 256
        assert abs(a["loss"] - b["loss"]) < 0.003
        assert abs(a["rmse"] - b["rmse"]) / max(a["rmse"], 1e-6) < 0.02
        report = {
            "status": "passed",
            "summary": summary,
            "parameters": sum(p.numel() for p in model.parameters()),
            "optimizer_fused": optimizer.defaults["fused"],
            "validation_single": a,
            "validation_batched": b,
            "embedding_gradient_nonzero": any(
                r["embedding_skip_gradient_norm"] > 0 for r in history
            ),
        }
        assert report["embedding_gradient_nonzero"]
        write_json(directory / "verification.json", report)


if __name__ == "__main__":
    main()

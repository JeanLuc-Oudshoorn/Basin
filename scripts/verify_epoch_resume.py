"""Verify interruption and GPU optimizer recovery with the actual RWKV backend."""
# ruff: noqa: E402

import argparse
import copy
import sys
from pathlib import Path
from unittest.mock import patch
from uuid import uuid4

import torch

ROOT = Path(__file__).resolve().parents[1]
for stage in (
    "03_transforming_input_data",
    "04_creating_embeddings",
    "05_model_training",
):
    sys.path.insert(0, str(ROOT / "src" / stage))
from embed import ForecastConfig, LazyPatchRecord, PatchEncoder
from epoch_training import accumulated_update, fit_epochs
from performance import configure_cuda_memory, make_optimizer
from train import RWKVForecaster
from transform import load_config, run_context, write_json
from variate_attention import configure_variate_execution


def initialize(config):
    configure_cuda_memory(config["training"])
    torch.manual_seed(23)
    settings = ForecastConfig(
        **dict(config["model"], quantiles=tuple(config["model"]["quantiles"]))
    )
    encoder = PatchEncoder(settings)
    train, validation = [], []
    for i, length in enumerate((48, 64, 80, 96)):
        series = torch.sin(torch.arange(length + 16).float() / 5)[None, None]
        cov = series * 2
        for split, destination in (("train", train), ("evaluation", validation)):
            context = (
                series[..., :length]
                if split == "evaluation"
                else series[..., : length - 16]
            )
            batch = {
                "context": context,
                "past_covariates": cov[..., : context.shape[-1]],
                "horizon": 16,
                "future_target": series[..., length : length + 16],
            }
            destination.append(
                LazyPatchRecord(
                    {"id": str(i), "source": str(i), "split": split, "batch": batch},
                    encoder,
                )
            )
    model = RWKVForecaster(settings).to(device="cuda", dtype=torch.bfloat16)
    model.checkpoint_token_threshold = config["training"]["checkpoint_token_threshold"]
    configure_variate_execution(model, config["training"], ROOT)
    return model, train, validation, make_optimizer(model, config["training"])


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output-dir", default="docs/audits/2026-10-28/epoch_checkpoints"
    )
    parser.add_argument("--compile-variate", action="store_true")
    parser.add_argument("--microbatch-size", type=int)
    args = parser.parse_args()
    config = load_config()
    base = ROOT / args.output_dir
    config["training"]["compile_variate"] = args.compile_variate
    if args.microbatch_size is not None:
        config["training"]["microbatch_size"] = args.microbatch_size
    directory = base / f"gpu_smoke_{uuid4().hex[:8]}"
    config["model"].update(
        width=128, layers=2, variate_heads=4, context_length=128, max_patches=16
    )
    config["training"].update(
        epochs=2,
        steps=None,
        batch_size=2,
        gradient_accumulation_steps=1,
        validation_batch_size=2,
        checkpoint_every=0,
        checkpoint_token_threshold=32,
    )
    torch.set_num_threads(4)
    config["paths"]["training"] = str(directory / "reference")
    with run_context(config, "train"):
        model, records, validation, optimizer = initialize(config)
        _, reference, _ = fit_epochs(model, records, validation, config, optimizer)
        expected = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
    del model, optimizer
    torch.cuda.empty_cache()
    config = copy.deepcopy(config)
    config["paths"]["training"] = str(directory / "interrupted")
    try:
        with run_context(config, "train"):
            model, records, validation, optimizer = initialize(config)
            calls = 0

            def interrupt(*args, **kwargs):
                nonlocal calls
                calls += 1
                if calls == 4:
                    raise RuntimeError("intentional GPU recovery test interruption")
                return accumulated_update(*args, **kwargs)

            with patch("epoch_training.accumulated_update", side_effect=interrupt):
                fit_epochs(model, records, validation, config, optimizer)
    except RuntimeError as error:
        if str(error) != "intentional GPU recovery test interruption":
            raise
    else:
        raise AssertionError("Expected interruption")
    del model, optimizer
    torch.cuda.empty_cache()
    with run_context(config, "train"):
        model, records, validation, optimizer = initialize(config)
        history, resumed, scheduler = fit_epochs(
            model, records, validation, config, optimizer
        )
        assert reference["completed_updates"] == resumed["completed_updates"] == 4
        assert reference["completed_epochs"] == resumed["completed_epochs"] == 2
        assert resumed["resumed_from"]
        for name, value in model.state_dict().items():
            torch.testing.assert_close(
                value.cpu(), expected[name], rtol=1e-3, atol=1e-3
            )
        assert optimizer.state
        assert all(s["exp_avg"].device.type == "cuda" for s in optimizer.state.values())
        assert all(int(s["step"]) == 4 for s in optimizer.state.values())
        assert scheduler["completed_validation_epochs"] == 2
        report = {
            "status": "passed",
            "parameters": sum(p.numel() for p in model.parameters()),
            "reference": reference,
            "resumed": resumed,
            "optimizer_state_on_cuda": True,
            "updates": len(history),
            "directory": str(directory),
        }
        write_json(base / "gpu_verification.json", report)


if __name__ == "__main__":
    main()

"""Count real model parameters on meta tensors, including all non-backbone heads."""
# ruff: noqa: E402

import sys
from dataclasses import replace
from pathlib import Path

import torch
from loguru import logger

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src/05_model_training"))
from train import ForecastConfig, RWKVForecaster
from transform import load_config, run_context, write_json


def main():
    config = load_config()
    settings = ForecastConfig(
        **dict(config["model"], quantiles=tuple(config["model"]["quantiles"]))
    )
    budget = 400_000_000
    with run_context(config, "model_capacity_verification"):
        counts = {}
        groups = {}
        for layers in (17, settings.layers, settings.layers + 1):
            # Uses the actual third-party RWKV classes. Meta avoids allocating
            # hundreds of millions of weights in host RAM or VRAM.
            with torch.device("meta"):
                model = RWKVForecaster(replace(settings, layers=layers))
            count = sum(parameter.numel() for parameter in model.parameters())
            counts[str(layers)] = count
            logger.info(
                "Actual architecture layers={} total_parameters={:,}", layers, count
            )
            if layers == settings.layers:
                for name, parameter in model.named_parameters():
                    key = name.split(".")[0]
                    groups[key] = groups.get(key, 0) + parameter.numel()
            del model
        if (
            not counts[str(settings.layers)]
            < budget
            <= counts[str(settings.layers + 1)]
        ):
            raise RuntimeError(
                "Default depth must be the largest whole layer count below 400M"
            )
        write_json(
            ROOT / "docs/model_training/capacity_400m_verification.json",
            {
                "budget": budget,
                "layers": settings.layers,
                "total_parameters": counts[str(settings.layers)],
                "parameter_counts_by_layers": counts,
                "parameter_groups": groups,
                "method": "real RWKVForecaster and third-party RWKV backend on meta tensors; no weights allocated or trained",
                "width": settings.width,
                "output_patches": settings.output_patches,
                "horizon_loss_weights": settings.horizon_loss_weights,
            },
        )


if __name__ == "__main__":
    main()

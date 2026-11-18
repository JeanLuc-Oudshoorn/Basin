"""Bounded GPU regression for validation followed by changing training shapes.

Uses synthetic data, the configured model and production optimizer/update path.
Writes only audit artifacts; never changes production training checkpoints.
"""

# ruff: noqa: E402
import argparse
import json
import os
import sys
from pathlib import Path
from time import perf_counter
from unittest.mock import patch

os.environ.setdefault("MAX_JOBS", "1")
os.environ.setdefault("TORCHINDUCTOR_COMPILE_THREADS", "1")
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "src/05_model_training"))
sys.path.insert(0, str(ROOT / "src/03_transforming_input_data"))

import torch
from batching import make_batches
from epoch_training import accumulated_update
from performance import configure_cuda_memory, make_optimizer, prepared_updates
from train import ForecastConfig, RWKVForecaster
from training_protocol import split_datasets
from transform import load_config, location, read_artifact
from variate_attention import configure_variate_execution


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--graphs", action="store_true")
    parser.add_argument("--inject-capture-failure", action="store_true")
    parser.add_argument(
        "--real-data", action="store_true", help="First 40 actual epoch-1 batch groups"
    )
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    config = load_config()
    torch.set_num_threads(4)
    torch.manual_seed(7)
    options = dict(config["training"], training_cuda_graphs=args.graphs)
    configure_cuda_memory(options)
    model = (
        RWKVForecaster(
            ForecastConfig(
                **dict(config["model"], quantiles=tuple(config["model"]["quantiles"]))
            )
        )
        .cuda()
        .bfloat16()
        .train()
    )
    model.checkpoint_token_threshold = options["checkpoint_token_threshold"]
    configure_variate_execution(model, options, ROOT)
    optimizer = make_optimizer(model, options)
    model.eval()
    with torch.no_grad():
        for variables, tokens in ((1, 48), (2, 80), (3, 48), (4, 16)):
            model.forecast(
                torch.randn(16, 1, tokens * 16, device="cuda"),
                16,
                past_covariates=(
                    torch.randn(16, variables - 1, tokens * 16, device="cuda")
                    if variables > 1
                    else None
                ),
            )
    model.train()
    rows = []
    factory = torch.cuda.make_graphed_callables
    injected = False

    def capture(proxy, args_):
        nonlocal injected
        if (
            args.inject_capture_failure
            and not injected
            and tuple(args_[0].shape) == (4, 4, 16, 1024)
        ):
            injected = True
            execute = proxy.execute

            def invalid_capture(mix, x):
                if torch.cuda.is_current_stream_capturing():
                    torch.cuda.synchronize()
                return execute(mix, x)

            proxy.execute = invalid_capture
        return factory(proxy, args_)

    groups = []
    for step, (variables, tokens) in enumerate(
        (
            (1, 48),
            (2, 80),
            (3, 48),
            (3, 48),
            (4, 16),
            (4, 16),
            (6, 128),
            (4, 16),
            (1, 48),
            (3, 48),
        ),
        1,
    ):
        generator = torch.Generator().manual_seed(step)
        group = []
        for _ in range(8):
            length = (tokens + 1) * 16
            prepared = model.tokenize_next(
                torch.randn(1, 1, length, generator=generator),
                past_covariates=(
                    torch.randn(1, variables - 1, length, generator=generator)
                    if variables > 1
                    else None
                ),
                training=True,
            )
            group.append(dict(split="train", prepared=prepared))
        groups.append(group)
    if args.real_data:
        data = read_artifact(location(config, "patch_index") / "patch_index.pt", config)
        training, validation, holdout, split = split_datasets(data["records"], config)
        groups = make_batches(
            training,
            options["batch_size"],
            config["seed"],
            bucket_by_length=options["bucket_by_length"],
        )[:40]
        for name in ("embedding", "roles"):
            getattr(model, name).load_state_dict(
                {
                    k.removeprefix(name + "."): v
                    for k, v in data["encoder_state"].items()
                    if k.startswith(name + ".")
                }
            )
        del data, training, validation, holdout, split
    updates = prepared_updates(
        groups,
        1,
        prefetch=True,
        pin_memory=True,
        microbatch_size=options["microbatch_size"],
        microbatch_token_limit=options["microbatch_token_limit"],
    )
    for step, (_, group, batches) in enumerate(updates, 1):
        torch.cuda.synchronize()
        start = perf_counter()
        torch.manual_seed(100 + step)
        with patch("torch.cuda.make_graphed_callables", side_effect=capture):
            result = accumulated_update(
                model, group, optimizer, 1.0, prepared_batches=batches
            )
        torch.cuda.synchronize()
        row = dict(step=step, seconds=perf_counter() - start, **result)
        rows.append(row)
        print(json.dumps(row), flush=True)
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(
            json.dumps(
                dict(
                    torch=torch.__version__,
                    gpu=torch.cuda.get_device_name(),
                    parameters=sum(p.numel() for p in model.parameters()),
                    graphs=args.graphs,
                    prefetch=True,
                    real_data=args.real_data,
                    injected_capture_failure=injected,
                    peak_allocated_gib=torch.cuda.max_memory_allocated() / 2**30,
                    rows=rows,
                ),
                indent=2,
            )
            + "\n",
            encoding="utf-8",
        )


if __name__ == "__main__":
    main()

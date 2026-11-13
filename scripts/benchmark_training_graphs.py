"""Compare complete 307M training updates and VRAM, one model resident at a time."""

# ruff: noqa: E402
import hashlib
import os
import sys
from pathlib import Path
from time import perf_counter

os.environ["MAX_JOBS"] = "1"
os.environ["TORCHINDUCTOR_COMPILE_THREADS"] = "1"
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "src/05_model_training"))
sys.path.insert(0, str(ROOT / "src/03_transforming_input_data"))

import torch
from epoch_training import accumulated_update
from performance import configure_cuda_memory, make_optimizer, prepare_update
from train import ForecastConfig, RWKVForecaster
from training_graphs import configure_training_graphs
from transform import load_config, logger, run_context, write_json
from variate_attention import configure_variate_execution


def digest(tensors):
    value = hashlib.sha256()
    for tensor in tensors:
        if tensor is not None:
            value.update(
                tensor.detach().contiguous().cpu().view(torch.uint8).numpy().tobytes()
            )
    return value.hexdigest()


def records(model, count, variables, tokens):
    generator = torch.Generator().manual_seed(9)
    rows = []
    for _ in range(count):
        context = torch.randn(1, 1, (tokens + 1) * 16, generator=generator)
        past = torch.randn(1, variables - 1, context.shape[-1], generator=generator)
        prepared = model.tokenize_next(context, past_covariates=past, training=True)
        rows.append(dict(split="train", prepared=prepared))
    return rows


def compare_snapshot(model, baseline):
    gradient_error = gradient_size = weight_error = weight_size = 0.0
    for parameter, (a, ga) in zip(model.parameters(), baseline):
        b = parameter.detach().cpu()
        weight_error += (a.float() - b.float()).square().sum().item()
        weight_size += a.float().square().sum().item()
        if ga is not None:
            gb = parameter.grad.detach().cpu()
            gradient_error += (ga.float() - gb.float()).square().sum().item()
            gradient_size += ga.float().square().sum().item()
    return dict(
        gradient_relative_l2=(gradient_error / max(gradient_size, 1e-30)) ** 0.5,
        weight_relative_l2=(weight_error / max(weight_size, 1e-30)) ** 0.5,
    )


def main():
    config = load_config()
    torch.set_num_threads(4)
    configure_cuda_memory(config["training"])
    directory = ROOT / "docs/time_benchmark/training_graphs_integration"
    directory.mkdir(parents=True, exist_ok=True)
    with run_context(config, "training_graphs_integration"):
        torch.manual_seed(7)
        model = (
            RWKVForecaster(
                ForecastConfig(
                    **dict(
                        config["model"], quantiles=tuple(config["model"]["quantiles"])
                    )
                )
            )
            .cuda()
            .bfloat16()
            .train()
        )
        model.checkpoint_token_threshold = config["training"][
            "checkpoint_token_threshold"
        ]
        options = dict(
            config["training"],
            training_cuda_graphs=False,
            training_graph_memory_gib=1.5,
            training_graph_cache_shapes=2,
        )
        configure_variate_execution(model, options, ROOT)
        initial = {
            key: value.detach().cpu() for key, value in model.state_dict().items()
        }
        rows = []
        # Each comparison keeps the effective batch fixed at eight tables.
        for variables, tokens, microbatch in (
            (4, 32, 1),
            (4, 64, 4),
            (4, 32, 8),
            (6, 128, 4),
        ):
            group = records(model, 8, variables, tokens)
            prepared = prepare_update(
                [group],
                microbatch_size=microbatch,
                microbatch_token_limit=options["microbatch_token_limit"],
                pin_memory=True,
            )
            baseline_snapshot = None
            for enabled in (False, True):
                model.load_state_dict(initial)
                model.zero_grad(set_to_none=True)
                configure_training_graphs(
                    model, dict(options, training_cuda_graphs=enabled)
                )
                torch.cuda.empty_cache()
                optimizer = make_optimizer(model, options)
                torch.manual_seed(101)
                torch.cuda.reset_peak_memory_stats()
                times, losses, cold = [], [], None
                first_grad = first_weights = None
                parity = None
                for step in range(8):
                    torch.cuda.synchronize()
                    start = perf_counter()
                    result = accumulated_update(
                        model, [group], optimizer, 1.0, prepared_batches=prepared
                    )
                    torch.cuda.synchronize()
                    seconds = perf_counter() - start
                    if step == 0:
                        cold = seconds
                        first_grad = digest(p.grad for p in model.parameters())
                        first_weights = digest(model.parameters())
                        if not enabled:
                            baseline_snapshot = [
                                (
                                    p.detach().cpu(),
                                    None if p.grad is None else p.grad.detach().cpu(),
                                )
                                for p in model.parameters()
                            ]
                        else:
                            parity = compare_snapshot(model, baseline_snapshot)
                    if step >= 2:
                        times.append(seconds * 1000)
                    losses.append(result["loss"])
                    logger.info(
                        "Training graph test shape=({}, {}, {}, 1024) enabled={} update={}/8 seconds={:.3f} allocated_GiB={:.2f}",
                        microbatch,
                        variables,
                        tokens,
                        enabled,
                        step + 1,
                        seconds,
                        torch.cuda.memory_allocated() / 2**30,
                    )
                row = dict(
                    graphs=enabled,
                    variables=variables,
                    tokens=tokens,
                    microbatch=microbatch,
                    physical_microbatch_sizes=[p[0]["b"] for p in prepared],
                    optimizer_updates=8,
                    cold_seconds=cold,
                    warm_times_ms=times,
                    losses=losses,
                    first_gradient_sha256=first_grad,
                    first_weights_sha256=first_weights,
                    peak_allocated_gib=torch.cuda.max_memory_allocated() / 2**30,
                    peak_reserved_gib=torch.cuda.max_memory_reserved() / 2**30,
                    graph_stats=model.training_graphs.summary() if enabled else None,
                    first_update_parity=parity,
                )
                rows.append(row)
                write_json(directory / "experiment.json", rows)
                if enabled:
                    baseline = rows[-2]
                    loss_error = max(
                        abs(a - b) / max(abs(a), 1e-6)
                        for a, b in zip(row["losses"], baseline["losses"])
                    )
                    row["loss_max_relative_error"] = loss_error
                    write_json(directory / "experiment.json", rows)
                    if (
                        parity["gradient_relative_l2"] > 0.02
                        or parity["weight_relative_l2"] > 0.001
                        or loss_error > 0.01
                    ):
                        raise AssertionError(
                            "Training graph update differs from the compiled reference"
                        )
                configure_training_graphs(
                    model, dict(options, training_cuda_graphs=False)
                )
                del optimizer
                model.zero_grad(set_to_none=True)
                torch.cuda.empty_cache()
            del baseline_snapshot
        write_json(
            directory / "environment.json",
            dict(
                torch=torch.__version__,
                cuda=torch.version.cuda,
                gpu=torch.cuda.get_device_name(),
                precision="bf16",
                parameters=sum(p.numel() for p in model.parameters()),
                allocator_limit_gib=8,
                graph_budget_gib=1.5,
                cpu_threads=4,
            ),
        )


if __name__ == "__main__":
    main()

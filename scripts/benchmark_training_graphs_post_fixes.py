"""Paired full-model timing after graph lifetime, prefetch and storage fixes.

Measures warm repeated shapes separately from capture-inclusive mixed real data.
One GPU model is resident; no production checkpoint is written or overwritten.
"""

# ruff: noqa: E402
import gc
import json
import os
import statistics
import sys
from pathlib import Path
from time import perf_counter

os.environ["MAX_JOBS"] = "1"
os.environ["TORCHINDUCTOR_COMPILE_THREADS"] = "1"
ROOT = Path(__file__).resolve().parents[1]
for path in (
    ROOT,
    ROOT / "src/05_model_training",
    ROOT / "src/03_transforming_input_data",
):
    sys.path.insert(0, str(path))

import torch
from batching import make_batches
from epoch_training import accumulated_update
from performance import configure_cuda_memory, make_optimizer, prepared_updates
from train import ForecastConfig, RWKVForecaster
from training_graphs import configure_training_graphs
from training_protocol import split_datasets
from transform import (
    load_config,
    location,
    logger,
    read_artifact,
    run_context,
    write_json,
)
from variate_attention import configure_variate_execution


def synthetic_group(model, variables, patches, count):
    rng = torch.Generator().manual_seed(9)
    length = (patches + 1) * model.config.patch_size
    return [
        dict(
            split="train",
            prepared=model.tokenize_next(
                torch.randn(1, 1, length, generator=rng),
                past_covariates=torch.randn(1, variables - 1, length, generator=rng),
                training=True,
            ),
        )
        for _ in range(count)
    ]


def execute(model, options, groups, microbatch, *, seed):
    """Time consuming prefetch, pinning, full update, and optimizer completion."""
    torch.manual_seed(seed)
    optimizer = make_optimizer(model, options)
    updates = prepared_updates(
        groups,
        1,
        prefetch=options.get("prefetch", True),
        pin_memory=True,
        microbatch_size=microbatch,
        microbatch_token_limit=options["microbatch_token_limit"],
    )
    rows = []
    torch.cuda.reset_peak_memory_stats()
    try:
        for _ in range(len(groups)):
            torch.cuda.synchronize()
            started = perf_counter()
            _, group, prepared = next(updates)
            result = accumulated_update(
                model, group, optimizer, options["gradient_clip"], prepared
            )
            torch.cuda.synchronize()
            rows.append(
                dict(
                    milliseconds=(perf_counter() - started) * 1000,
                    loss=result["loss"],
                    gradient_norm=result["gradient_norm"],
                    windows=result["windows"],
                    embedding_shapes=result["embedding_shapes"],
                )
            )
    finally:
        updates.close()
    return dict(
        updates=rows,
        peak_allocated_gib=torch.cuda.max_memory_allocated() / 2**30,
        peak_reserved_gib=torch.cuda.max_memory_reserved() / 2**30,
        graphs=model.training_graphs.summary() if model.training_graphs else None,
    )


def main():
    spec = json.loads((ROOT / "configs/training_graph_benchmark.json").read_text())
    config = load_config(ROOT / spec["model_config"])
    options = dict(config["training"])
    output = ROOT / "docs/time_benchmark/training_graphs_post_fixes"
    output.mkdir(parents=True, exist_ok=True)
    torch.set_num_threads(options["cpu_threads"])
    configure_cuda_memory(options)
    with run_context(config, "training_graph_speed_post_fixes") as log_directory:
        write_json(
            output / "settings.json",
            dict(benchmark=spec, model=config["model"], training=options),
        )
        torch.manual_seed(config["seed"])
        model = (
            RWKVForecaster(
                ForecastConfig(
                    **dict(
                        config["model"],
                        quantiles=tuple(config["model"]["quantiles"]),
                    )
                )
            )
            .cuda()
            .bfloat16()
            .train()
        )
        model.checkpoint_token_threshold = options["checkpoint_token_threshold"]
        configure_variate_execution(
            model, dict(options, training_cuda_graphs=False), ROOT
        )
        initial = {
            key: tensor.detach().cpu() for key, tensor in model.state_dict().items()
        }
        all_rows = []

        def reset(enabled):
            configure_training_graphs(model, dict(options, training_cuda_graphs=False))
            model.zero_grad(set_to_none=True)
            gc.collect()
            torch.cuda.empty_cache()
            model.load_state_dict(initial)
            configure_training_graphs(
                model, dict(options, training_cuda_graphs=enabled)
            )

        def compare(name, groups, microbatch, warmup, repeats, metadata):
            # Compile/warm the common reference outside BOTH timed arms. Timed
            # graph arms still start from an empty graph cache on every repeat.
            reset(False)
            execute(model, options, groups, microbatch, seed=spec["seed"])
            for repeat in range(repeats):
                paired = {}
                for enabled in (False, True) if repeat % 2 == 0 else (True, False):
                    reset(enabled)
                    result = execute(
                        model, options, groups, microbatch, seed=spec["seed"]
                    )
                    measured = [r["milliseconds"] for r in result["updates"][warmup:]]
                    result.update(
                        name=name,
                        repeat=repeat,
                        enabled=enabled,
                        **metadata,
                        warmup_updates_excluded=warmup,
                        median_ms=statistics.median(measured),
                        measured_total_ms=sum(measured),
                        all_updates_total_ms=sum(
                            r["milliseconds"] for r in result["updates"]
                        ),
                    )
                    paired[enabled] = result
                    all_rows.append(result)
                    write_json(output / "results.json", all_rows)
                    logger.info(
                        "BENCH {} repeat={}/{} replay={} measured_median_ms={:.2f} total_ms={:.1f} peak_GiB={:.2f}",
                        name,
                        repeat + 1,
                        repeats,
                        enabled,
                        result["median_ms"],
                        result["measured_total_ms"],
                        result["peak_allocated_gib"],
                    )
                a, b = paired[False], paired[True]
                error = max(
                    abs(x["loss"] - y["loss"]) / max(abs(x["loss"]), 1e-6)
                    for x, y in zip(a["updates"], b["updates"])
                )
                b["max_loss_relative_difference"] = error
                b["loss_agreement_within_one_percent"] = error <= 0.01
                if b["graphs"]["disabled_reason"] is not None:
                    raise RuntimeError("Replay disabled during benchmark")
                write_json(output / "results.json", all_rows)

        for case in spec["cases"]:
            group = synthetic_group(
                model, case["variables"], case["patches"], options["batch_size"]
            )
            groups = [group] * (spec["warmup_updates"] + spec["measured_updates"])
            compare(
                case["name"],
                groups,
                case["microbatch"],
                spec["warmup_updates"],
                spec["repeats"],
                dict(
                    kind="repeated_shape",
                    **{k: v for k, v in case.items() if k != "name"},
                ),
            )

        data = read_artifact(location(config, "embedded") / "embeddings.pt", config)
        training, validation, test, split = split_datasets(data["records"], config)
        mixed = make_batches(
            training,
            options["batch_size"],
            config["seed"],
            bucket_by_length=options["bucket_by_length"],
        )[: spec["mixed_batches"]]
        sources = [[r["source"] for r in group] for group in mixed]
        write_json(output / "mixed_sources.json", sources)
        del data, training, validation, test, split
        # Every update is measured here, including first capture and all evictions.
        compare(
            "mixed_real_tables",
            mixed,
            options["microbatch_size"],
            0,
            spec["mixed_repeats"],
            dict(kind="mixed_shapes", tables=sum(map(len, mixed))),
        )
        reset(False)
        write_json(
            output / "environment.json",
            dict(
                torch=torch.__version__,
                cuda=torch.version.cuda,
                gpu=torch.cuda.get_device_name(),
                parameters=sum(p.numel() for p in model.parameters()),
                precision="bf16",
                allocator_limit_gib=options["cuda_memory_limit_gib"],
                graph_budget_gib=options["training_graph_memory_gib"],
                cpu_threads=torch.get_num_threads(),
                logs=str(log_directory),
            ),
        )


if __name__ == "__main__":
    main()

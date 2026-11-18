"""Bounded full-model CUDA stress test; never touches production checkpoints."""

# ruff: noqa: E402
import argparse
import gc
import os
import sys
from pathlib import Path
from time import perf_counter

ROOT = Path(__file__).resolve().parents[1]
for stage in (
    "03_transforming_input_data",
    "04_patch_indexing",
    "05_model_training",
):
    sys.path.insert(0, str(ROOT / "src" / stage))
sys.path.insert(0, str(ROOT))
from runtime_storage import configure_runtime_storage

configure_runtime_storage(ROOT)
os.environ["MAX_JOBS"] = "1"
os.environ["TORCHINDUCTOR_COMPILE_THREADS"] = "1"

import torch
from epoch_training import accumulated_update
from performance import configure_cuda_memory, make_optimizer, prepare_update
from train import ForecastConfig, RWKVForecaster
from transform import load_config, run_context, write_json
from variate_attention import configure_variate_execution


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default=str(ROOT / "configs/config.json"))
    parser.add_argument("--updates", type=int, default=600)
    parser.add_argument("--reproduce-old", action="store_true")
    parser.add_argument("--exercise-recovery", action="store_true")
    parser.add_argument("--microbatch-size", type=int)
    parser.add_argument(
        "--all-shapes",
        action="store_true",
        help="Exercise all 48 variable-count/patch-length buckets",
    )
    parser.add_argument(
        "--output", default="docs/audits/2026-11-08/training_memory_stress.json"
    )
    args = parser.parse_args()
    config = load_config(args.config)
    options = config["training"]
    if args.microbatch_size is not None:
        if args.microbatch_size < 1:
            parser.error("microbatch-size must be positive")
        options["microbatch_size"] = args.microbatch_size
    torch.set_num_threads(options["cpu_threads"])
    torch.manual_seed(config["seed"])
    configure_cuda_memory(options)
    output = ROOT / args.output
    with run_context(config, "training_memory_stress") as log_directory:
        model = (
            RWKVForecaster(
                ForecastConfig(
                    **dict(
                        config["model"], quantiles=tuple(config["model"]["quantiles"])
                    )
                )
            )
            .to(device="cuda", dtype=torch.bfloat16)
            .train()
        )
        model.checkpoint_token_threshold = options["checkpoint_token_threshold"]
        model.auto_microbatch = options.get("auto_microbatch", True)
        configure_variate_execution(model, options, ROOT)
        optimizer = make_optimizer(model, options)
        generator = torch.Generator().manual_seed(9)

        def group_for(length, variables):
            return [
                [
                    dict(
                        split="train",
                        prepared=model.tokenize_next(
                            torch.randn(1, 1, length, generator=generator),
                            past_covariates=(
                                torch.randn(
                                    1, variables - 1, length, generator=generator
                                )
                                if variables > 1
                                else None
                            ),
                            training=True,
                        ),
                    )
                    for _ in range(options["batch_size"])
                ]
            ]

        report = dict(
            parameters=sum(p.numel() for p in model.parameters()),
            gpu=torch.cuda.get_device_name(),
            torch_version=str(torch.__version__),
            model=config["model"],
            training=options,
            log_directory=str(log_directory),
            updates=[],
            old_settings=None,
        )
        if args.reproduce_old:
            # Catch only outside the update frame so its failed activations can
            # be collected. No persistent model checkpoint is changed.
            model.auto_microbatch = False
            model.checkpoint_token_threshold = 2048
            configure_cuda_memory(dict(options, cuda_memory_limit_gib=8.0))
            old_group = group_for(1024, 6)
            old_prepared = prepare_update(
                old_group, microbatch_size=8, microbatch_token_limit=1536
            )
            try:
                # The first update allocates AdamW moments. The second has those
                # moments resident during its forward, as production did at step 200.
                for _ in range(2):
                    accumulated_update(
                        model,
                        old_group,
                        optimizer,
                        options["gradient_clip"],
                        old_prepared,
                    )
                report["old_settings"] = {"oom": False}
            except torch.OutOfMemoryError as error:
                report["old_settings"] = {"oom": True, "error": str(error)}
            optimizer.zero_grad(set_to_none=True)
            del old_prepared, old_group
            gc.collect()
            torch.cuda.empty_cache()
            configure_cuda_memory(options)
            model.auto_microbatch = True
            model.checkpoint_token_threshold = options["checkpoint_token_threshold"]
        if args.exercise_recovery:
            recovery_group = group_for(1024, 6)
            recovery_prepared = prepare_update(
                recovery_group, microbatch_size=4, microbatch_token_limit=1536
            )
            configure_cuda_memory(dict(options, cuda_memory_limit_gib=5.5))
            recovered = accumulated_update(
                model,
                recovery_group,
                optimizer,
                options["gradient_clip"],
                recovery_prepared,
            )
            report["forced_recovery"] = {
                key: recovered[key]
                for key in (
                    "oom_retries",
                    "microbatches",
                    "windows",
                    "loss",
                    "embedding_shapes",
                )
            }
            del recovery_group, recovery_prepared
            optimizer.zero_grad(set_to_none=True)
            gc.collect()
            torch.cuda.empty_cache()
            configure_cuda_memory(options)
        cases = [(256, 1), (1024, 6), (2048, 6), (2048, 1), (512, 3), (1536, 5)]
        if args.all_shapes:
            cases = [
                (length, variables)
                for length in range(256, 2049, 256)
                for variables in range(1, 7)
            ]
        report["case_count"] = len(cases)
        started = perf_counter()
        write_json(output, report)
        for index in range(args.updates):
            length, variables = cases[index % len(cases)]
            group = group_for(length, variables)
            prepared = prepare_update(
                group,
                microbatch_size=options["microbatch_size"],
                microbatch_token_limit=options["microbatch_token_limit"],
                pin_memory=True,
            )
            torch.cuda.reset_peak_memory_stats()
            before = perf_counter()
            result = accumulated_update(
                model, group, optimizer, options["gradient_clip"], prepared
            )
            torch.cuda.synchronize()
            row = dict(
                update=index + 1,
                timesteps=length,
                variables=variables,
                seconds=perf_counter() - before,
                loss=result["loss"],
                microbatches=result["microbatches"],
                shapes=result["embedding_shapes"],
                oom_retries=result["oom_retries"],
                allocated_gib=result["cuda_allocated_gib"],
                peak_allocated_gib=torch.cuda.max_memory_allocated() / 2**30,
                peak_reserved_gib=torch.cuda.max_memory_reserved() / 2**30,
            )
            report["updates"].append(row)
            print(row, flush=True)
            if (index + 1) % 12 == 0:
                report["seconds"] = perf_counter() - started
                write_json(output, report)
        report["seconds"] = perf_counter() - started
        report["completed"] = True
        report["final_checkpoint_token_threshold"] = model.checkpoint_token_threshold
        report["learned_shape_limits"] = [
            {"shapes": key, "microbatch_size": value}
            for key, value in getattr(model, "_microbatch_limits", {}).items()
        ]
        write_json(output, report)


if __name__ == "__main__":
    main()

"""Capture-inclusive real-table benchmark of the production shape schedule."""

# ruff: noqa: E402
# isort: skip_file
import argparse
import gc
import json
import random
from collections import Counter

from benchmark_training_graphs_post_fixes import (
    ROOT,
    ForecastConfig,
    RWKVForecaster,
    configure_cuda_memory,
    configure_training_graphs,
    configure_variate_execution,
    execute,
    load_config,
    location,
    logger,
    make_batches,
    read_artifact,
    run_context,
    split_datasets,
    torch,
    write_json,
)
from batching import pack_records


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--microbatch", type=int, default=4)
    parser.add_argument("--skip-shuffled", action="store_true")
    parser.add_argument(
        "--resume",
        action="store_true",
        help="Keep completed trial arms and run only missing arms",
    )
    args = parser.parse_args()
    config = load_config()
    if args.microbatch is not None:
        config["training"]["microbatch_size"] = args.microbatch
    options = dict(config["training"])
    suffix = f"_micro{args.microbatch}" if args.microbatch != 4 else ""
    output = ROOT / f"docs/time_benchmark/training_graph_schedule{suffix}"
    output.mkdir(parents=True, exist_ok=True)
    previous_settings = (
        json.loads((output / "settings.json").read_text()) if args.resume else None
    )
    results = json.loads((output / "results.json").read_text()) if args.resume else []
    if previous_settings is not None and (
        previous_settings["training"] != options
        or previous_settings["model"] != config["model"]
    ):
        raise ValueError(
            "Cannot resume benchmark with changed model or training settings"
        )
    torch.set_num_threads(options["cpu_threads"])
    configure_cuda_memory(options)
    with run_context(config, "training_graph_schedule") as logs:
        torch.manual_seed(config["seed"])
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
        model.checkpoint_token_threshold = options["checkpoint_token_threshold"]
        configure_variate_execution(
            model, dict(options, training_cuda_graphs=False), ROOT
        )
        initial = {k: v.detach().cpu() for k, v in model.state_dict().items()}
        data = read_artifact(location(config, "patch_index") / "patch_index.pt", config)
        training, validation, test, split = split_datasets(data["records"], config)
        epoch = make_batches(
            training,
            options["batch_size"],
            config["seed"],
            bucket_by_length=True,
            shape_run_batches=options["shape_run_batches"],
        )
        # Eight consecutive production blocks, including their capture costs.
        groups = epoch[:1024]
        shuffled = list(groups)
        random.Random(101).shuffle(shuffled)  # nosec B311
        sources = [[r["source"] for r in batch] for batch in groups]
        if args.resume and json.loads((output / "sources.json").read_text()) != sources:
            raise ValueError("Cannot resume benchmark with changed source batches")
        write_json(output / "sources.json", sources)
        # Inventory uses raw metadata only, never tokenizes the whole corpus.
        inventory = Counter()
        for batch in epoch:
            record = batch[0]
            raw = record["batch"]
            variables = raw["context"].shape[1] + sum(
                raw[k].shape[1] if raw.get(k) is not None else 0
                for k in ("past_covariates", "known_covariates")
            )
            patches = max(
                (r["batch"]["context"].shape[-1] + 15) // 16 - 1 for r in batch
            )
            padded = (patches + 15) // 16 * 16
            inventory[str((len(batch), variables, padded))] += 1
        del epoch, data, training, validation, test, split
        write_json(
            output / "settings.json",
            dict(
                training=options,
                model=config["model"],
                epoch_shapes=inventory,
                groups=len(groups),
                logs=str(logs),
                resumed_from_logs=previous_settings["logs"]
                if previous_settings
                else None,
            ),
        )

        def reset(enabled):
            configure_training_graphs(model, dict(options, training_cuda_graphs=False))
            model.zero_grad(set_to_none=True)
            gc.collect()
            torch.cuda.empty_cache()
            model.load_state_dict(initial)
            configure_training_graphs(
                model, dict(options, training_cuda_graphs=enabled)
            )

        unique = {}
        for batch in groups:
            packed = pack_records(batch)
            unique.setdefault(
                (len(batch), len(packed["roles"]), packed["padded_tokens"]), batch
            )
        logger.info(
            "Benchmark {} updates, {} shapes; compilation warmup outside all arms",
            len(groups),
            len(unique),
        )
        reset(False)
        execute(
            model,
            options,
            list(unique.values()) * 2,
            options["microbatch_size"],
            seed=101,
        )
        for repeat in range(2):
            arms = [
                ("shuffled_off", shuffled, False),
                ("grouped_off", groups, False),
                ("grouped_replay", groups, True),
            ]
            if args.skip_shuffled:
                arms = [arm for arm in arms if arm[0] != "shuffled_off"]
            if repeat:
                arms.reverse()
            for name, batches, enabled in arms:
                if any(r["name"] == name and r["repeat"] == repeat for r in results):
                    logger.info("Retaining completed {} repeat {}", name, repeat)
                    continue
                reset(enabled)
                logger.info(
                    "Starting {} repeat {}: {} complete optimizer updates",
                    name,
                    repeat,
                    len(batches),
                )
                result = execute(
                    model, options, batches, options["microbatch_size"], seed=101
                )
                result.update(
                    name=name,
                    repeat=repeat,
                    total_ms=sum(r["milliseconds"] for r in result["updates"]),
                )
                results.append(result)
                write_json(output / "results.json", results)
                logger.info(
                    "Finished {} repeat {}: {:.2f} ms/update, graphs={}",
                    name,
                    repeat,
                    result["total_ms"] / len(batches),
                    result["graphs"],
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
                logs=str(logs),
            ),
        )


if __name__ == "__main__":
    main()

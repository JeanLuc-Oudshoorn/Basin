"""Stage 9: FakeBench-TS. Forecast random walks, where the best forecast is known.

Runs Basin and the holdout stage's foundation models on the FakeBench-TS
series (benchmarks/fakebench-ts): random walks whose steps have mean zero,
half of them with symmetric and half with skewed steps. The best forecast of
every series is its last value, and the stage scores how much error each
model adds to that (RMSE against the naive forecast) and how far its
forecasts stray from it. Nothing here touches training data or the holdout.
"""
# ruff: noqa: E402

import argparse
import json
import math
import shutil
import sys
from datetime import datetime, timezone
from itertools import count
from pathlib import Path

import numpy as np
import torch
from loguru import logger

ROOT = Path(__file__).resolve().parents[2]
PACKAGE = ROOT / "benchmarks" / "fakebench-ts" / "src"
if not (PACKAGE / "fakebench_ts").is_dir():
    raise SystemExit(
        "FakeBench-TS is missing. Fetch it with: "
        "git submodule update --init benchmarks/fakebench-ts"
    )
for folder in (
    ROOT / "src" / "03_transforming_input_data",
    ROOT / "src" / "04_patch_indexing",
    ROOT / "src" / "05_model_training",
    ROOT / "src" / "06_holdout_test",
    PACKAGE,
):
    sys.path.insert(0, str(folder))
import fakebench_ts
from checkpoint_upload import read_environment_file, upload_options
from fakebench_ts import cli as fakebench_cli
from fakebench_ts import metrics as fakebench_metrics
from fakebench_ts import models as fakebench_models
from fast_inference import configure_fast_inference
from holdout_test import holdout_checkpoint
from inference import forecast_contexts
from performance import configure_cuda_memory
from result_upload import HoldoutResultStore, first_line
from train import RWKVForecaster, launch_wsl
from transform import load_config, run_context, timed, write_json
from variate_attention import configure_variate_execution

MODELS = ("basin", "chronos2", "timestral", "timeh2o")
# v1 (2,100 walks) is published in the FakeBench-TS repository. v3 has 10,000
# walks, covariates and true quantiles; it is generated locally:
#   python -m fakebench_ts generate   (in benchmarks/fakebench-ts, a few minutes)
DATASETS = {
    name: PACKAGE.parent / "data" / f"fakebench_{name}.parquet" for name in ("v1", "v3")
}
DEFAULT_DATASET = "v3"
COMPLETED = "results.json"  # written by scoring, so it marks a finished run


def basin(config, checkpoint=None, batch_size=16, quantiles=False, covariates=False):
    """Basin as a FakeBench-TS forecaster: the weights the holdout test would use.

    The returned function carries `details`: the checkpoint and, where the
    checkpoint records them, the epoch and step of the weights. It returns the
    median, or with `quantiles` the FakeBench-TS quantile levels,
    [contexts, levels, horizon]. With `covariates` it also takes the history
    of covariates, [contexts, count, context length], as every model does.
    """
    path = Path(checkpoint) if checkpoint else holdout_checkpoint(config)
    saved = torch.load(path, map_location="cpu", weights_only=True, mmap=True)
    summary = saved.get("training_summary", {})
    details = {
        "checkpoint": str(path),
        "epoch": summary.get("exported_epoch", summary.get("best_epoch")),
        "step": summary.get("exported_step", summary.get("best_step")),
    }
    if "best_state_dict" in saved:
        # A training recovery checkpoint: take its best validation weights.
        details["epoch"] = saved["early_stopping"]["best_epoch"]
        details["step"] = saved["best_step"]
        saved = {
            "config": saved["identity"]["model"],
            "state_dict": saved["best_state_dict"],
        }
    model = RWKVForecaster.load(path, checkpoint=saved).eval()
    del saved
    configure_variate_execution(model, config["training"], ROOT)
    configure_fast_inference(model, config["training"])
    median = model.config.quantiles.index(0.5)
    patch = model.config.patch_size
    details["parameters"] = sum(p.numel() for p in model.parameters())
    logger.info("Basin {}", details)

    def forecast(contexts, horizon, past_covariates=None):
        # [contexts, count, context length], or None without covariates.
        given = (
            None
            if past_covariates is None
            else np.asarray(past_covariates, dtype=np.float32)
        )
        requests = [
            {
                "context": torch.tensor(np.asarray(c, dtype=np.float32))[None, None],
                "past_covariates": None
                if given is None
                else torch.tensor(given[i])[None],
            }
            for i, c in enumerate(contexts)
        ]
        forecasts = forecast_contexts(
            model, requests, batch_size, prediction_patches=math.ceil(horizon / patch)
        )
        if not quantiles:
            return np.stack([f[0, :horizon, median].numpy() for f in forecasts])
        # Each forecast is [target, step, level].
        return np.stack(
            [
                fakebench_models.at_quantile_levels(
                    f[0, :horizon].float().numpy(), model.config.quantiles
                ).T
                for f in forecasts
            ]
        )

    forecast.details = details
    if covariates:
        forecast.uses_covariates = True
    return forecast


def build(name, config, *, batch_size=16, probabilistic=False):
    """The same published models, revisions and cache as the holdout stage.

    `probabilistic` asks Basin for its quantiles and gives it the covariates,
    as the published models get them; it suits a dataset that has both (v3).
    """
    settings = config.get("holdout", {})
    shared = {
        "device": settings.get("foundation_device"),
        "cache_dir": str(ROOT / settings.get("model_cache_dir", ".cache/huggingface")),
    }
    if name == "basin":
        return basin(
            config,
            batch_size=batch_size,
            quantiles=probabilistic,
            covariates=probabilistic,
        )
    # A Hugging Face token for gated weights may be kept in .env (HF_TOKEN).
    read_environment_file(ROOT / ".env")
    if name == "chronos2":
        return fakebench_models.chronos2(**settings.get("chronos", {}), **shared)
    if name == "timestral":
        return fakebench_models.timestral(
            **settings.get("timestral", {}),
            max_context=config["model"]["context_length"],
            **shared,
        )
    if name == "timeh2o":
        return fakebench_models.timeh2o(**settings.get("timeh2o", {}), **shared)
    return fakebench_models.load_factory(name)()


def results_root(config):
    return ROOT / config["paths"].get("fakebench", "results/model/fakebench")


def new_run(config, *, epoch=None, step=None, root=None):
    """A new timestamped folder for one run; an existing one is never written to."""
    root = root or results_root(config)
    name = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    if epoch is not None:
        name += f"_epoch{epoch}"
    if step is not None:
        name += f"_step{step}"
    for attempt in count(1):
        directory = root / (name if attempt == 1 else f"{name}_{attempt}")
        try:
            directory.mkdir(parents=True)
            return directory
        except FileExistsError:
            continue


def earlier_forecasts(config, name, dataset, wanted):
    """The newest completed run's forecasts of a published model on this data.

    Only forecasts made with the covariates given as they are now: earlier
    runs also gave some models the future values of covariates.
    """
    root = results_root(config)
    runs = sorted(p for p in root.iterdir() if (p / COMPLETED).is_file())
    for directory in reversed(runs):
        path = directory / "forecasts" / f"{name}.parquet"
        if path.is_file():
            saved, forecasts = fakebench_models.load_forecasts(path)
            # A series forecast a second time, with its covariates, counts once.
            series = sum(
                not key.endswith(fakebench_metrics.COVARIATE_SUFFIX)
                for key in forecasts
            )
            if (
                saved.get("dataset_sha256") == dataset.sha256
                and saved.get("covariates") == fakebench_models.COVARIATES_GIVEN
                and series == wanted
            ):
                return path
    return None


def headline(output):
    """{track: {model: RMSE against the naive forecast}}, best first."""
    results = json.loads((output / COMPLETED).read_text(encoding="utf-8"))
    table = {}
    for model, scored in results["models"].items():
        if model == fakebench_metrics.TRUE_MEDIAN:
            continue
        for track, rows in scored["tracks"].items():
            table.setdefault(track, {})[model] = rows["overall"]["rmse_ratio"]
    return {
        track: dict(sorted(models.items(), key=lambda item: item[1]))
        for track, models in table.items()
    }


class FakebenchResultStore(HoldoutResultStore):
    """A copy of every run folder in Azure Blob Storage (`fakebench.result_upload`).

    The same settings and account as `holdout.result_upload`; a run folder is
    `<results folder>/<run folder>/` in the container.
    """

    setting = "fakebench.result_upload"
    label = "FakeBench result upload"
    completed = COMPLETED
    if_unavailable = (
        "The benchmark has not started. Fix the storage settings, or remove "
        "fakebench.result_upload to keep the results on this disk only."
    )


def make_result_store(config):
    """The configured store, or None when `fakebench.result_upload` is off."""
    options = upload_options(
        config.get("fakebench", {}), key="result_upload", name="fakebench"
    )
    if options is None:
        return None
    directory = results_root(config)
    if directory.is_relative_to(ROOT):
        prefix = directory.relative_to(ROOT).as_posix()
    elif options["prefix"] is None:
        raise ValueError(
            f"FakeBench results are kept outside the project ({directory}); "
            "set fakebench.result_upload.prefix to name their place in the container"
        )
    else:
        prefix = options["prefix"]
    return FakebenchResultStore(options, prefix=prefix, root=ROOT)


def upload_run(store, directory):
    """Copy one finished run folder to the container; a failure loses nothing."""
    destination = f"{store.describe_container()}/{store.base(directory.name)}"
    logger.info("FAKEBENCH upload of {} to {}", directory, destination)
    try:
        summary = store.sync(directory, directory.name)
    except Exception as error:
        raise RuntimeError(
            f"The FakeBench results are complete in {directory}; only their copy "
            f"to {destination} failed ({first_line(error)}). Send them again with: "
            f"python src/09_fakebench/fakebench.py --upload {directory.name}"
        ) from error
    logger.success(
        "FakeBench results copied to {}: {:,} files, {:.0f} MiB ({:,} sent, {:,} "
        "already there), verified, {:.0f}s",
        summary["destination"],
        summary["files"],
        summary["bytes"] / 2**20,
        summary["uploaded_files"],
        summary["unchanged_files"],
        summary["seconds"],
    )
    return summary


def run(config, args, store=None):
    """Forecast and score; with a `store`, copy the finished run folder to it."""
    torch.set_num_threads(config["training"].get("cpu_threads", 4))
    configure_cuda_memory(config["training"])
    data = DATASETS[args.dataset]
    dataset = fakebench_ts.load_dataset(data)
    wanted = min(args.limit or len(dataset), len(dataset))
    # With true quantiles in the data, Basin gives its quantiles and takes
    # the covariates, as the published models do.
    probabilistic = all(series.quantile_offset is not None for series in dataset)
    # Basin first: its checkpoint names the run folder.
    names = sorted(args.models, key=lambda name: name != "basin")
    loaded = {}
    record = {
        "dataset": dataset.spec["name"],
        "data": str(data),
        "series": wanted,
        "models": {},
    }

    def load(name):
        return build(
            name, config, batch_size=args.batch_size, probabilistic=probabilistic
        )

    if "basin" in names:
        with timed("load basin", cuda=True):
            loaded["basin"] = load("basin")
        record["basin"] = loaded["basin"].details
    # A trial on a subset is kept apart from the complete runs.
    trial = wanted < len(dataset)
    output = new_run(
        config,
        epoch=record.get("basin", {}).get("epoch"),
        step=record.get("basin", {}).get("step"),
        root=results_root(config) / "trials" if trial else None,
    )
    logger.info("FAKEBENCH {} series from {}; output={}", wanted, data, output)
    for name in names:
        target = output / "forecasts" / f"{name}.parquet"
        # A published model's forecasts do not change between runs; Basin's do.
        earlier = (
            None
            if args.rerun or trial or name == "basin"
            else earlier_forecasts(config, name, dataset, wanted)
        )
        if earlier is not None:
            target.parent.mkdir(exist_ok=True)
            shutil.copy2(earlier, target)
            record["models"][name] = f"copied from {earlier.parent.parent.name}"
            logger.info("Copied {} forecasts from {}", name, earlier)
            continue
        if name not in loaded:
            with timed(f"load {name}", cuda=True):
                loaded[name] = load(name)
        with timed(f"forecast {name}", cuda=True):
            forecasts, timing = fakebench_models.timed_run(
                loaded[name], dataset, batch_size=args.batch_size, limit=args.limit
            )
        # The models run on the GPU; say so next to the timings.
        timing["latency"]["machine"] = (
            torch.cuda.get_device_name()
            if torch.cuda.is_available()
            else timing["latency"]["machine"]
        )
        fakebench_models.save_forecasts(
            forecasts, target, model=name, dataset=dataset, details=timing
        )
        record["models"][name] = (
            f"forecast in this run in {timing['seconds']:.1f} s; one forecast "
            f"at a time: median {timing['latency']['median_ms']:.1f} ms"
        )
        del loaded[name]
        torch.cuda.empty_cache()
    write_json(output / "run.json", record)
    fakebench_cli.main(
        [
            "score",
            "--data",
            str(data),
            "--forecasts",
            str(output / "forecasts"),
            "--out",
            str(output),
        ]
    )
    for track, models in headline(output).items():
        logger.success(
            "FAKEBENCH {} RMSE vs naive: {}",
            track,
            {name: round(ratio, 3) for name, ratio in models.items()},
        )
    logger.success("FakeBench-TS complete: {}", output / "leaderboard.md")
    if store is not None and not trial:
        # After everything is on the local disk: a failed copy loses nothing.
        upload_run(store, output)
    return output


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default=str(ROOT / "configs/config.json"))
    parser.add_argument(
        "--models",
        nargs="+",
        default=list(MODELS),
        help=f"default: {' '.join(MODELS)}; FakeBench-TS baselines also work",
    )
    parser.add_argument(
        "--dataset",
        choices=sorted(DATASETS),
        default=DEFAULT_DATASET,
        help=f"default: {DEFAULT_DATASET} (10,000 walks); v1 is the earlier set "
        "of 2,100 walks",
    )
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument(
        "--limit",
        type=int,
        help="forecast an even subset (trial); written under trials/, not uploaded",
    )
    parser.add_argument(
        "--upload",
        metavar="RUN",
        help="only copy this finished run folder to the container again",
    )
    parser.add_argument(
        "--rerun",
        action="store_true",
        help="forecast the published models again instead of copying their forecasts",
    )
    args = parser.parse_args()
    config = load_config(args.config)
    extra = ["--models", *args.models, "--batch-size", str(args.batch_size)]
    extra += ["--dataset", args.dataset]
    extra += ["--limit", str(args.limit)] if args.limit else []
    extra += ["--rerun"] if args.rerun else []
    with run_context(config, "fakebench"):
        if args.upload:
            directory = results_root(config) / args.upload
            if not (directory / COMPLETED).is_file():
                raise SystemExit(f"No finished FakeBench run at {directory}")
            store = make_result_store(config)
            if store is None:
                raise SystemExit("fakebench.result_upload is not configured")
            upload_run(store, directory)
            return
        if launch_wsl(__file__, args.config, extra_args=extra):
            return
        if not torch.cuda.is_available() or not torch.cuda.is_bf16_supported():
            raise RuntimeError("Use the configured WSL CUDA bf16 runtime")
        with timed("FAKEBENCH stage", cuda=True):
            # Only this entry point uploads; find out now, not after the
            # forecasts, that the container cannot be written.
            store = None if args.limit else make_result_store(config)
            if store is not None:
                store.preflight()
            run(config, args, store)


if __name__ == "__main__":
    main()

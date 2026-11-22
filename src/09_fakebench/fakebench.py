"""Stage 9: FakeBench-TS. Forecast random walks, where the best forecast is known.

Runs Basin and the holdout stage's foundation models on the FakeBench-TS
series (benchmarks/fakebench-ts) and scores how far each forecast strays
from the optimal one. Nothing here touches training data or the holdout.
"""
# ruff: noqa: E402

import argparse
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
from checkpoint_upload import read_environment_file
from fakebench_ts import cli as fakebench_cli
from fakebench_ts import models as fakebench_models
from fast_inference import configure_fast_inference
from holdout_test import holdout_checkpoint
from inference import forecast_contexts
from performance import configure_cuda_memory
from train import RWKVForecaster, launch_wsl
from transform import load_config, run_context, timed, write_json
from variate_attention import configure_variate_execution

MODELS = ("basin", "chronos2", "timestral", "timeh2o", "timesfm")
# The dataset published in the FakeBench-TS repository.
DATA = PACKAGE.parent / "data" / "fakebench_v1.parquet"
COMPLETED = "results.json"  # written by scoring, so it marks a finished run


def basin(config, checkpoint=None, batch_size=16):
    """Basin as a FakeBench-TS forecaster: the weights the holdout test would use.

    The returned function carries `details`: the checkpoint and, where the
    checkpoint records them, the epoch and step of the weights.
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

    def forecast(contexts, horizon):
        requests = [
            {
                "context": torch.tensor(np.asarray(c, dtype=np.float32))[None, None],
                "past_covariates": None,
            }
            for c in contexts
        ]
        quantiles = forecast_contexts(
            model, requests, batch_size, prediction_patches=math.ceil(horizon / patch)
        )
        return np.stack([q[0, :horizon, median].numpy() for q in quantiles])

    forecast.details = details
    return forecast


def build(name, config):
    """The same published models, revisions and cache as the holdout stage."""
    settings = config.get("holdout", {})
    shared = {
        "device": settings.get("foundation_device"),
        "cache_dir": str(ROOT / settings.get("model_cache_dir", ".cache/huggingface")),
    }
    if name == "basin":
        return basin(config)
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
    if name == "timesfm":
        return fakebench_models.timesfm(**settings.get("timesfm", {}), **shared)
    return fakebench_models.load_factory(name)()


def results_root(config):
    return ROOT / config["paths"].get("fakebench", "results/model/fakebench")


def new_run(config, *, epoch=None, step=None):
    """A new timestamped folder for one run; an existing one is never written to."""
    name = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    if epoch is not None:
        name += f"_epoch{epoch}"
    if step is not None:
        name += f"_step{step}"
    for attempt in count(1):
        directory = results_root(config) / (
            name if attempt == 1 else f"{name}_{attempt}"
        )
        try:
            directory.mkdir(parents=True)
            return directory
        except FileExistsError:
            continue


def earlier_forecasts(config, name, dataset, wanted):
    """The newest completed run's forecasts of a published model on this data."""
    root = results_root(config)
    runs = sorted(p for p in root.iterdir() if (p / COMPLETED).is_file())
    for directory in reversed(runs):
        path = directory / "forecasts" / f"{name}.parquet"
        if path.is_file():
            saved, forecasts = fakebench_models.load_forecasts(path)
            if (
                saved.get("dataset_sha256") == dataset.sha256
                and len(forecasts) == wanted
            ):
                return path
    return None


def run(config, args):
    torch.set_num_threads(config["training"].get("cpu_threads", 4))
    configure_cuda_memory(config["training"])
    dataset = fakebench_ts.load_dataset(DATA)
    wanted = min(args.limit or len(dataset), len(dataset))
    # Basin first: its checkpoint names the run folder.
    names = sorted(args.models, key=lambda name: name != "basin")
    loaded, record = {}, {"data": str(DATA), "series": wanted, "models": {}}
    if "basin" in names:
        with timed("load basin", cuda=True):
            loaded["basin"] = build("basin", config)
        record["basin"] = loaded["basin"].details
    output = new_run(
        config,
        epoch=record.get("basin", {}).get("epoch"),
        step=record.get("basin", {}).get("step"),
    )
    logger.info("FAKEBENCH {} series from {}; output={}", wanted, DATA, output)
    for name in names:
        target = output / "forecasts" / f"{name}.parquet"
        # A published model's forecasts do not change between runs; Basin's do.
        earlier = (
            None
            if args.rerun or name == "basin"
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
                loaded[name] = build(name, config)
        with timed(f"forecast {name}", cuda=True):
            forecasts, seconds = fakebench_models.timed_run(
                loaded[name], dataset, batch_size=args.batch_size, limit=args.limit
            )
        fakebench_models.save_forecasts(
            forecasts, target, model=name, dataset=dataset, details={"seconds": seconds}
        )
        record["models"][name] = f"forecast in this run in {seconds:.1f} s"
        del loaded[name]
        torch.cuda.empty_cache()
    write_json(output / "run.json", record)
    fakebench_cli.main(
        [
            "score",
            "--data",
            str(DATA),
            "--forecasts",
            str(output / "forecasts"),
            "--out",
            str(output),
        ]
    )
    logger.success("FakeBench-TS complete: {}", output / "leaderboard.md")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default=str(ROOT / "configs/config.json"))
    parser.add_argument(
        "--models",
        nargs="+",
        default=list(MODELS),
        help=f"default: {' '.join(MODELS)}; FakeBench-TS baselines also work",
    )
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--limit", type=int, help="forecast an even subset (trial)")
    parser.add_argument(
        "--rerun",
        action="store_true",
        help="forecast the published models again instead of copying their forecasts",
    )
    args = parser.parse_args()
    config = load_config(args.config)
    extra = ["--models", *args.models, "--batch-size", str(args.batch_size)]
    extra += ["--limit", str(args.limit)] if args.limit else []
    extra += ["--rerun"] if args.rerun else []
    with run_context(config, "fakebench"):
        if launch_wsl(__file__, args.config, extra_args=extra):
            return
        if not torch.cuda.is_available() or not torch.cuda.is_bf16_supported():
            raise RuntimeError("Use the configured WSL CUDA bf16 runtime")
        with timed("FAKEBENCH stage", cuda=True):
            run(config, args)


if __name__ == "__main__":
    main()

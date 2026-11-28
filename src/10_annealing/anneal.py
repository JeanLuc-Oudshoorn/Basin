"""Run an annealing experiment: train on its set, starting from the main run's weights.

    python src/10_annealing/anneal.py --experiment flat25_lr1e-5

The experiment must have been created by build_annealing_set.py, and its set
transformed and indexed. Like train.py this routes itself to WSL and needs the
GPU. A first start takes the weights of the main run's final checkpoint; a
later start of the same experiment continues the experiment's own checkpoint.
All output goes to the experiment's folder; see docs/annealing/ANNEALING.md.
"""

import argparse
import sys
from dataclasses import asdict
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
for directory in (HERE, ROOT / "src/05_model_training"):
    sys.path.insert(0, str(directory))

import train as training_stage  # noqa: E402
from annealing_experiment import EXPERIMENT_ROOT, experiment_config_path  # noqa: E402
from annealing_start import load_start_weights, starting_from  # noqa: E402
from checkpoints import resume_path  # noqa: E402
from index_patches import ForecastConfig  # noqa: E402
from loguru import logger  # noqa: E402
from transform import load_config, location, run_context, timed, write_json  # noqa: E402


def anneal(config):
    if "annealing" not in config:
        raise ValueError(
            "Not an annealing experiment configuration; use the config.json that "
            "build_annealing_set.py wrote into the experiment folder"
        )
    directory = location(config, "training")
    if (ROOT / EXPERIMENT_ROOT).resolve() not in directory.resolve().parents:
        raise ValueError(
            f"An annealing experiment trains under {EXPERIMENT_ROOT}, not {directory}"
        )
    if resume_path(config) is not None:
        # Interrupted earlier: its own checkpoint holds weights, optimizer,
        # schedule and position. The main run's checkpoint is not needed again.
        logger.info(
            "ANNEALING {} continues from its own checkpoint",
            config["annealing"]["experiment"],
        )
        return training_stage.train(config)
    settings = asdict(
        ForecastConfig(
            **dict(config["model"], quantiles=tuple(config["model"]["quantiles"]))
        )
    )
    with timed("read annealing start weights"):
        weights, start = load_start_weights(config, settings)
    logger.info(
        "ANNEALING {} starts from {} ({} weights of update {:,}, sha256={})",
        config["annealing"]["experiment"],
        start["checkpoint"],
        start["weights"],
        start["step"],
        start["weights_sha256"],
    )
    directory.mkdir(parents=True, exist_ok=True)
    write_json(directory / "annealing_start.json", start)
    applied = []
    with starting_from(training_stage, weights, applied):
        result = training_stage.train(config)
    if not applied:
        raise RuntimeError(
            "Training finished without loading the start weights: train() no longer "
            "builds its optimizer through train.make_optimizer. This run trained a "
            "freshly initialised model; do not use its results."
        )
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--experiment", help="name of the annealing experiment")
    group.add_argument("--config", help="the experiment's config.json")
    args = parser.parse_args()
    path = args.config or str(
        experiment_config_path(args.experiment).relative_to(ROOT).as_posix()
    )
    config = load_config(path)
    with run_context(config, "train"):
        if training_stage.launch_wsl(__file__, path):
            return
        import torch

        if not torch.cuda.is_available() or not torch.cuda.is_bf16_supported():
            raise RuntimeError("Use the configured WSL CUDA bf16 runtime")
        with timed("ANNEALING stage", cuda=True):
            anneal(config)


if __name__ == "__main__":
    main()

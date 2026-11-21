"""Where an annealing experiment starts: weights read from the main run's checkpoint.

Only the model weights are taken. The optimizer starts fresh (its float32
master weights are copied from these weights, its moment estimates are empty),
and the weight average starts as a copy of them. The main run's checkpoint is
opened read-only and never written.
"""

import hashlib
import json
from contextlib import contextmanager
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[2]
FORMATS = ("rwkv_epoch_checkpoint_v1", "rwkv_timed_checkpoint_v1")
FIELDS = {"best": "best_state_dict", "last": "state_dict"}


def weights_digest(weights):
    """Same fingerprint as the holdout stage's `evaluated_weights_sha256`."""
    digest = hashlib.sha256()
    for name, tensor in sorted(weights.items()):
        header = json.dumps(
            [name, str(tensor.dtype), list(tensor.shape)], separators=(",", ":")
        ).encode()
        digest.update(len(header).to_bytes(8, "little"))
        digest.update(header)
        raw = tensor.detach().cpu().contiguous().reshape(-1).view(torch.uint8).numpy()
        digest.update(memoryview(raw))
    return digest.hexdigest()


def _plain(value):
    """Tuples and lists compare equal after a round trip through JSON."""
    return json.loads(json.dumps(value, sort_keys=True))


def load_start_weights(config, model_settings):
    """(weights, description) for the start named in `config["annealing"]`.

    `model_settings` is the model configuration training is about to build, as
    a plain mapping; the checkpoint must describe the same architecture.
    """
    start = config["annealing"]["start"]
    path = ROOT / start["checkpoint"]
    if not path.is_file():
        raise FileNotFoundError(f"Annealing start checkpoint not found: {path}")
    training = (ROOT / config["paths"]["training"]).resolve()
    if path.resolve().parent == training:
        raise ValueError(
            "An annealing experiment cannot start from a checkpoint in its own "
            "training folder"
        )
    state = torch.load(path, map_location="cpu", weights_only=True, mmap=True)
    if state.get("format") not in FORMATS:
        raise ValueError(
            f"{path} is not a training checkpoint; annealing starts from "
            "latest_checkpoint.pt of a training run"
        )
    saved = _plain(state["identity"]["model"])
    wanted = _plain(model_settings)
    different = sorted(
        key for key in wanted if key in saved and saved[key] != wanted[key]
    )
    if different:
        raise ValueError(
            f"Start checkpoint {path} holds another model than the configuration "
            f"describes; differing settings: {different}"
        )
    reason = state.get("stop_reason")
    if reason is None and not start.get("allow_unfinished", False):
        raise ValueError(
            f"{path} was written while its training run was still going, after "
            f"{state['step']:,} optimizer steps, so it is not the final checkpoint. "
            "Wait for the run to finish, or use start.allow_unfinished=true in the "
            "recipe for a trial on intermediate weights."
        )
    field = FIELDS[start["weights"]]
    # Copies in memory, so the checkpoint file is released again right away
    # instead of staying mapped for the whole annealing run.
    weights = {name: tensor.clone() for name, tensor in state[field].items()}
    description = {
        "checkpoint": start["checkpoint"],
        "weights": start["weights"],
        "field": field,
        # The update the weights are from: the selected (best validated) one,
        # or the last one of the run.
        "step": state["best_step"] if start["weights"] == "best" else state["step"],
        "checkpoint_step": state["step"],
        "checkpoint_epoch": state["epoch"],
        "checkpoint_stop_reason": reason,
        "weights_sha256": weights_digest(weights),
        "tensors": len(weights),
        "parameters": sum(tensor.numel() for tensor in weights.values()),
    }
    del state
    return weights, description


@contextmanager
def starting_from(stage, weights, applied=None):
    """Within the block, `stage.train` starts at `weights` instead of a fresh model.

    train() builds the model, then calls the module's `make_optimizer`, then
    fits. The weights are loaded at that call: after initialisation, before
    the optimizer copies them into its float32 masters and before the weight
    average takes its starting copy. `applied` (a list) receives one entry per
    load, so the caller can tell that it happened.
    """
    make_optimizer = stage.make_optimizer

    def make_optimizer_at_start(model, options):
        model.load_state_dict(weights)
        if applied is not None:
            applied.append(True)
        return make_optimizer(model, options)

    stage.make_optimizer = make_optimizer_at_start
    try:
        yield
    finally:
        stage.make_optimizer = make_optimizer

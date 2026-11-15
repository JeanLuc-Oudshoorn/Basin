"""Exponential moving average (EMA) of the trained weights.

Each optimizer update uses only a handful of tables, so the weights jitter
around the point they are converging to. The average of the recent weights sits
closer to that point than any single step does: it gives much of the benefit of
a larger batch without spending compute on one. Training itself is unchanged.
Validation, best-weight selection and the exported model use the average.

The average is kept in float32 and follows the optimizer's float32 master
weights where they exist, so it is not limited to the BF16 grid.
"""

import math
from contextlib import contextmanager

import torch


def weight_average_options(training):
    """Validated `training.weight_average`; None when averaging is off."""
    options = training.get("weight_average")
    if options is None:
        return None
    if not isinstance(options, dict):
        raise ValueError("training.weight_average must be an object or null")
    unknown = set(options) - {"decay", "update_every", "warmup_updates"}
    if unknown:
        raise ValueError(f"Unknown training.weight_average fields: {sorted(unknown)}")
    decay = options.get("decay", 0.9999)
    every = options.get("update_every", 4)
    warmup = options.get("warmup_updates", 10)
    if (
        isinstance(decay, bool)
        or not isinstance(decay, (int, float))
        or not 0 < decay < 1
    ):
        raise ValueError("training.weight_average.decay must be between zero and one")
    for name, value, minimum in (
        ("update_every", every, 1),
        ("warmup_updates", warmup, 0),
    ):
        if isinstance(value, bool) or not isinstance(value, int) or value < minimum:
            raise ValueError(
                f"training.weight_average.{name} must be an integer >= {minimum}"
            )
    return {"decay": float(decay), "update_every": every, "warmup_updates": warmup}


class WeightAverage:
    def __init__(self, model, optimizer, *, decay, update_every=1, warmup_updates=10):
        self.decay = decay
        self.update_every = update_every
        self.warmup_updates = warmup_updates
        self.parameters = [p for p in model.parameters() if p.requires_grad]
        masters = getattr(optimizer, "master_parameters", None)
        expected = getattr(optimizer, "model_parameters", None)
        if masters is not None:
            # Follow the float32 masters, matched to the model's parameters by
            # identity: optimizer groups reorder them.
            by_parameter = {id(p): m for p, m in zip(expected, masters)}
            self.sources = [by_parameter[id(p)] for p in self.parameters]
        else:
            self.sources = self.parameters
        self.masters = masters is not None
        self.average = [s.detach().float().clone() for s in self.sources]
        self.updates = 0  # optimizer updates seen
        self.averaged = 0  # averaging steps applied

    def update(self):
        """Call once after every optimizer update."""
        self.updates += 1
        if self.updates % self.update_every:
            return
        self.averaged += 1
        # Per averaging step, the decay that matches `decay` per optimizer
        # update. Early on, follow the weights more closely: an average that
        # still remembers the random initialisation is worse than no average.
        decay = min(
            self.decay**self.update_every,
            (1 + self.averaged) / (self.warmup_updates + self.averaged)
            if self.warmup_updates
            else 1.0,
        )
        sources = [s.detach().float() for s in self.sources]
        torch._foreach_lerp_(self.average, sources, 1.0 - decay)

    @contextmanager
    def applied(self):
        """Run a block with the averaged weights in the model, then restore training's."""
        backup = None if self.masters else [p.detach().clone() for p in self.parameters]
        with torch.no_grad():
            for parameter, value in zip(self.parameters, self.average):
                parameter.copy_(value)
        try:
            yield
        finally:
            with torch.no_grad():
                # Model weights are the rounded masters, so the masters restore
                # them exactly. Without masters, restore the saved copy.
                for parameter, value in zip(
                    self.parameters, self.sources if self.masters else backup
                ):
                    parameter.copy_(value)

    def state_dict(self):
        return {
            "average": self.average,
            "updates": self.updates,
            "averaged": self.averaged,
            "decay": self.decay,
            "update_every": self.update_every,
            "warmup_updates": self.warmup_updates,
        }

    def load_state_dict(self, state):
        if len(state["average"]) != len(self.average) or any(
            saved.shape != current.shape
            for saved, current in zip(state["average"], self.average)
        ):
            raise ValueError("Saved weight average does not match model parameters")
        if not all(
            math.isclose(state[key], getattr(self, key))
            for key in ("decay", "update_every", "warmup_updates")
        ):
            raise ValueError("Saved weight average uses different settings")
        with torch.no_grad():
            for current, saved in zip(self.average, state["average"]):
                current.copy_(saved)
        self.updates, self.averaged = state["updates"], state["averaged"]

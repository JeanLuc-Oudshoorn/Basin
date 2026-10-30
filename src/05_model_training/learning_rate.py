"""Optimizer-update learning-rate schedules shared by both training entry points."""

import math
from dataclasses import asdict, dataclass

import torch


@dataclass
class Schedule:
    name: str
    total_updates: int | None
    warmup_updates: int
    peak_learning_rate: float
    min_learning_rate: float

    def __call__(self, completed_updates):
        # LambdaLR initializes at index 0, before the first optimizer update.
        if self.name == "constant":
            return 1.0
        update = min(completed_updates + 1, self.total_updates)
        if self.warmup_updates and update <= self.warmup_updates:
            return update / self.warmup_updates
        start = max(1, self.warmup_updates)
        progress = (update - start) / max(1, self.total_updates - start)
        floor = self.min_learning_rate / self.peak_learning_rate
        return floor + (1 - floor) * 0.5 * (1 + math.cos(math.pi * progress))


def make_scheduler(optimizer, training, total_updates):
    if total_updates is not None and (
        isinstance(total_updates, bool)
        or not isinstance(total_updates, int)
        or total_updates < 1
    ):
        raise ValueError("total_updates must be a positive integer")
    peak = training["learning_rate"]
    if not math.isfinite(peak) or peak <= 0:
        raise ValueError("learning_rate must be finite and positive")
    settings = training.get("scheduler", {"name": "constant"})
    name = settings["name"]
    if name == "reduce_on_plateau":
        allowed = {"name", "factor", "patience_epochs", "min_learning_rate"}
        if set(settings) - allowed:
            raise ValueError(
                f"Unknown plateau fields: {sorted(set(settings) - allowed)}"
            )
        factor = settings.get("factor", 0.5)
        patience = settings.get("patience_epochs", 1)
        minimum = settings.get("min_learning_rate", peak * 0.01)
        if (
            not math.isfinite(factor)
            or not 0 < factor < 1
            or isinstance(patience, bool)
            or not isinstance(patience, int)
            or patience < 0
            or not math.isfinite(minimum)
            or not 0 <= minimum <= peak
        ):
            raise ValueError(
                "Invalid plateau factor, patience_epochs, or min_learning_rate"
            )
        delta = training.get("early_stopping", {}).get("min_delta", 0.0)
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer,
            mode="min",
            factor=factor,
            patience=patience,
            threshold=delta,
            threshold_mode="abs",
            min_lr=minimum,
        )
        return scheduler, {
            "name": name,
            "total_updates": total_updates,
            "step_unit": "validated_epochs",
            "factor": factor,
            "patience_epochs": patience,
            "min_learning_rate": minimum,
            "threshold": delta,
        }
    if name not in ("constant", "warmup_cosine"):
        raise ValueError(f"Unknown learning-rate scheduler: {name}")
    allowed = {"name", "warmup_steps", "warmup_fraction", "min_learning_rate"}
    if set(settings) - allowed:
        raise ValueError(f"Unknown scheduler fields: {sorted(set(settings) - allowed)}")
    if name == "constant":
        policy = Schedule(name, total_updates, 0, peak, peak)
    else:
        if total_updates is None:
            raise ValueError(
                "warmup_cosine requires a finite update budget; use reduce_on_plateau for open-ended training"
            )
        minimum = settings.get("min_learning_rate", peak * 0.1)
        if not math.isfinite(minimum) or not 0 <= minimum <= peak:
            raise ValueError(
                "min_learning_rate must be finite and between zero and the peak learning_rate"
            )
        if "warmup_steps" in settings:
            if "warmup_fraction" in settings:
                raise ValueError("Choose warmup_steps or warmup_fraction, not both")
            warmup = settings["warmup_steps"]
            if (
                isinstance(warmup, bool)
                or not isinstance(warmup, int)
                or not 0 <= warmup < total_updates
            ):
                raise ValueError(
                    "warmup_steps must be an integer in [0, total_updates)"
                )
        else:
            fraction = settings.get("warmup_fraction", 0.05)
            if not math.isfinite(fraction) or not 0 <= fraction < 1:
                raise ValueError("warmup_fraction must be finite and in [0, 1)")
            warmup = min(math.ceil(total_updates * fraction), total_updates - 1)
        policy = Schedule(name, total_updates, warmup, peak, minimum)
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, policy)
    return scheduler, asdict(policy)


def scheduler_snapshot(scheduler, plan):
    state = scheduler.state_dict()
    if isinstance(scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau):
        # These infinity sentinels are reconstructed by a fresh scheduler.
        # Omit them so JSON remains strict, without altering live state.
        state.pop("mode_worse", None)
        if not math.isfinite(state["best"]):
            state.pop("best")
    return {
        "plan": plan,
        "completed_updates": scheduler.last_epoch
        if not isinstance(scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau)
        else None,
        "completed_validation_epochs": scheduler.last_epoch
        if isinstance(scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau)
        else None,
        "next_learning_rates": scheduler.get_last_lr(),
        "state_dict": state,
    }

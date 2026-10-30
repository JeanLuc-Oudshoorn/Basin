"""Test actual optimizer-used rates, schedule endpoints and state restoration."""

import copy
import json
import math
import sys
import unittest
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src/05_model_training"))
from learning_rate import make_scheduler, scheduler_snapshot


def setup(total=8, warmup=2):
    p = torch.nn.Parameter(torch.ones(1))
    o = torch.optim.AdamW([p], lr=0.1)
    settings = {
        "learning_rate": 0.1,
        "scheduler": {
            "name": "warmup_cosine",
            "warmup_steps": warmup,
            "min_learning_rate": 0.01,
        },
    }
    s, plan = make_scheduler(o, settings, total)
    return p, o, s, plan


def update(p, o, s):
    lr = o.param_groups[0]["lr"]
    o.zero_grad()
    p.grad = torch.ones_like(p)
    o.step()
    s.step()
    return lr


class ScheduleTests(unittest.TestCase):
    def test_open_ended_plateau_schedule_strict_json_and_resume(self):
        parameter = torch.nn.Parameter(torch.ones(()))
        optimizer = torch.optim.AdamW([parameter], lr=0.1)
        training = {
            "learning_rate": 0.1,
            "scheduler": {
                "name": "reduce_on_plateau",
                "factor": 0.5,
                "patience_epochs": 1,
                "min_learning_rate": 0.01,
            },
        }
        scheduler, plan = make_scheduler(optimizer, training, None)
        json.dumps(scheduler_snapshot(scheduler, plan), allow_nan=False)
        for metric in (1.0, 1.0, 1.0):
            scheduler.step(metric)
        self.assertAlmostEqual(optimizer.param_groups[0]["lr"], 0.05)
        state = scheduler_snapshot(scheduler, plan)
        json.dumps(state, allow_nan=False)
        self.assertEqual(state["completed_validation_epochs"], 3)
        second_optimizer = torch.optim.AdamW([parameter], lr=0.05)
        second, _ = make_scheduler(second_optimizer, training, None)
        second.load_state_dict(state["state_dict"])
        for metric in (1.0, 1.0):
            scheduler.step(metric)
            second.step(metric)
        self.assertEqual(scheduler.get_last_lr(), second.get_last_lr())

    def test_warmup_peak_decay_and_floor_are_used_by_optimizer(self):
        p, o, s, plan = setup()
        rates = [update(p, o, s) for _ in range(8)]
        self.assertAlmostEqual(rates[0], 0.05)
        self.assertAlmostEqual(rates[1], 0.1)
        self.assertAlmostEqual(rates[-1], 0.01)
        self.assertTrue(all(a >= b for a, b in zip(rates[1:], rates[2:])))
        self.assertAlmostEqual(update(p, o, s), 0.01)
        self.assertEqual(plan["warmup_updates"], 2)
        json.dumps(scheduler_snapshot(s, plan))

    def test_resume_preserves_remaining_rates(self):
        p, o, s, _ = setup()
        for _ in range(3):
            update(p, o, s)
        optimizer_state = copy.deepcopy(o.state_dict())
        scheduler_state = copy.deepcopy(s.state_dict())
        reference = [update(p, o, s) for _ in range(5)]
        p2, o2, s2, _ = setup()
        o2.load_state_dict(optimizer_state)
        s2.load_state_dict(scheduler_state)
        restored = [update(p2, o2, s2) for _ in range(5)]
        self.assertEqual(reference, restored)

    def test_no_warmup_short_run_and_legacy_constant(self):
        p, o, s, _ = setup(total=2, warmup=0)
        self.assertEqual([update(p, o, s) for _ in range(2)], [0.1, 0.01])
        o = torch.optim.AdamW([p], lr=0.1)
        s, plan = make_scheduler(
            o,
            {
                "learning_rate": 0.1,
                "scheduler": {
                    "name": "warmup_cosine",
                    "warmup_fraction": 0.05,
                    "min_learning_rate": 0.01,
                },
            },
            1,
        )
        self.assertEqual(update(p, o, s), 0.1)
        self.assertEqual(plan["warmup_updates"], 0)
        o = torch.optim.AdamW([p], lr=0.1)
        s, _ = make_scheduler(o, {"learning_rate": 0.1}, 4)
        self.assertEqual([update(p, o, s) for _ in range(4)], [0.1] * 4)

    def test_invalid_policies_fail_early(self):
        for overrides in [
            {"name": "unknown"},
            {"name": "warmup_cosine", "min_learning_rate": 0.2},
            {"name": "warmup_cosine", "min_learning_rate": math.nan},
            {"name": "warmup_cosine", "warmup_steps": 8},
            {"name": "warmup_cosine", "warmup_steps": True},
            {"name": "warmup_cosine", "warmup_fraction": 1},
            {"name": "warmup_cosine", "warmup_fraction": math.nan},
            {"name": "warmup_cosine", "warmup_fraction": 0.1, "warmup_steps": 2},
            {"name": "warmup_cosine", "warmup_typo": 2},
        ]:
            with self.subTest(overrides=overrides):
                p = torch.nn.Parameter(torch.ones(1))
                o = torch.optim.AdamW([p], lr=0.1)
                with self.assertRaises(ValueError):
                    make_scheduler(o, {"learning_rate": 0.1, "scheduler": overrides}, 8)

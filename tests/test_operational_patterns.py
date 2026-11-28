import json
import sys
import unittest
from pathlib import Path

import numpy as np

sys.path.insert(
    0, str(Path(__file__).resolve().parents[1] / "src" / "01_data_generation")
)
from measurement_layer import _apply_artifacts, sample_observation
from operational_patterns import (
    PATTERNS,
    WALK_LAWS,
    sample_pattern,
    simulate_pattern,
    walk_steps,
)
from plant_mechanisms import PLANT_FAMILIES, sample_plant, simulate_plant

LENGTH = 2000


def drawn(family, seed, **changes):
    spec = sample_pattern(np.random.default_rng(seed), LENGTH, family)
    spec.update(family=family, **changes)
    return spec


def signal(spec, seed=3, drive=None):
    drive = np.ones(LENGTH) if drive is None else drive
    return simulate_pattern(spec, drive, np.random.default_rng(seed))[0]


class PatternTests(unittest.TestCase):
    def test_every_pattern_is_a_unit_family_and_replays(self):
        for family in PATTERNS:
            self.assertIn(family, PLANT_FAMILIES)
            for seed in range(8):
                spec = drawn(family, seed)
                json.dumps(spec, allow_nan=False)
                values = signal(spec)
                self.assertEqual(values.shape, (LENGTH,))
                self.assertTrue(np.isfinite(values).all(), family)
                np.testing.assert_array_equal(values, signal(spec))

    def test_unknown_pattern_is_refused(self):
        with self.assertRaises(ValueError):
            sample_pattern(np.random.default_rng(0), LENGTH, "no_such_pattern")

    def test_daylight_rests_at_exactly_zero_and_is_never_negative(self):
        for seed in range(10):
            spec = drawn("daylight", seed)
            values = signal(spec)
            self.assertGreaterEqual(values.min(), 0.0)
            night = 1 - spec["day_fraction"]
            self.assertGreaterEqual((values == 0).mean(), night - 0.05)

    def test_counts_are_whole_and_not_negative(self):
        for seed in range(10):
            values = signal(drawn("counts", seed))
            np.testing.assert_array_equal(values, np.round(values))
            self.assertGreaterEqual(values.min(), 0.0)

    def test_random_walk_changes_are_unrelated_and_centred(self):
        rng = np.random.default_rng(0)
        for law in WALK_LAWS:
            steps = walk_steps(law, 0.5, rng, 200_000)
            self.assertLess(abs(steps.mean()), 0.03 * steps.std() + 0.01, law)
            related = np.corrcoef(steps[:-1], steps[1:])[0, 1]
            self.assertLess(abs(related), 0.02, law)

    def test_random_walk_ignores_the_demand(self):
        spec = sample_plant(np.random.default_rng(1), LENGTH, {"random_walk": 1.0}, 5)
        self.assertEqual(spec["load_coupling"], 0.0)
        clock = np.arange(-100, LENGTH)
        calm = simulate_plant(spec, np.zeros(len(clock)), clock)["pv"]
        rng = np.random.default_rng(2)
        busy = simulate_plant(spec, rng.normal(size=len(clock)) * 5, clock)["pv"]
        np.testing.assert_array_equal(calm, busy)

    def test_counts_and_daylight_keep_their_direction(self):
        for family in ("counts", "daylight"):
            for seed in range(20):
                spec = sample_plant(
                    np.random.default_rng(seed), LENGTH, {family: 1.0}, 5
                )
                self.assertGreater(spec["output_gain"], 0)

    def test_counter_staircase_only_moves_one_way(self):
        spec = drawn("staircase", 0, counter=True, noise=0.0, ramp=0.0)
        self.assertGreaterEqual(np.diff(signal(spec)).min(), 0.0)

    def test_coarse_held_value_takes_few_levels(self):
        spec = drawn("held_value", 0, levels=5, moves=0.0, drift=0.0, spikes=0.0)
        spec.update(tail=0.0, busy=0.0)
        self.assertLessEqual(len(np.unique(signal(spec))), 12)

    def test_plateau_duty_cycle_rests_at_zero_between_runs(self):
        spec = drawn("duty_cycle", 0, form="plateau", idle_noise=0.0, rise=0)
        values = signal(spec)
        self.assertGreater((values == 0).mean(), 0.05)
        self.assertGreater((values > 0.5).mean(), 0.05)

    def test_daily_load_repeats_with_its_period(self):
        spec = drawn("daily_load", 0, period=48.0, noise=1e-6, day_spread=1e-6)
        spec.update(rest_days=0, level_wander=0.0)
        values = signal(spec)
        related = np.corrcoef(values[:-48], values[48:])[0, 1]
        self.assertGreater(related, 0.99)

    def test_busy_stretches_make_large_changes_follow_large_changes(self):
        def clustering(busy):
            spec = drawn("held_value", 0, busy=busy, busy_tau=60.0)
            spec.update(memory=0.0, tail=0.0, moves=0.0, drift=0.0, spikes=0.0)
            spec.update(levels=0, load_gain=0.0)
            size = np.abs(np.diff(signal(spec)))
            return np.corrcoef(size[:-1], size[1:])[0, 1]

        # Successive changes of plain noise share a point, which alone relates
        # their sizes a little; busy stretches add clearly to that.
        self.assertGreater(clustering(1.2), clustering(0.0) + 0.15)


class CompanionTests(unittest.TestCase):
    def test_patterns_offer_what_drives_them_and_a_second_sensor(self):
        clock = np.arange(-100, LENGTH)
        demand = np.sin(clock / 50)
        for family in PATTERNS:
            spec = sample_plant(np.random.default_rng(4), LENGTH, {family: 1.0}, 5)
            channels = simulate_plant(spec, demand, clock)
            if family == "random_walk":
                # Nothing tells about a random walk, so nothing is offered.
                self.assertNotIn("companion", channels)
                self.assertEqual(spec["exports"], [])
                continue
            driver = channels["status" if family == "duty_cycle" else "driver"]
            self.assertEqual(driver.shape, channels["pv"].shape)
            self.assertTrue(np.isfinite(channels["companion"]).all(), family)
        states = simulate_plant(
            sample_plant(np.random.default_rng(4), LENGTH, {"duty_cycle": 1.0}, 5),
            demand,
            clock,
        )["status"]
        self.assertEqual(set(np.unique(states)), {0.0, 1.0})

    def test_second_sensor_follows_the_driver_and_not_the_future(self):
        clock = np.arange(-100, LENGTH)
        rng = np.random.default_rng(6)
        demand = np.sin(clock / 50) + 0.2 * rng.normal(size=len(clock))
        spec = sample_plant(np.random.default_rng(2), LENGTH, {"daily_load": 1.0}, 5)
        spec["companion"].update(noise=0.01, lag=2, tau=0.0)
        channels = simulate_plant(spec, demand, clock)
        related = np.corrcoef(channels["driver"][:-2], channels["companion"][2:])[0, 1]
        self.assertGreater(abs(related), 0.95)
        changed = demand.copy()
        changed[1200:] += 3
        again = simulate_plant(spec, changed, clock)
        np.testing.assert_array_equal(
            again["companion"][:1200], channels["companion"][:1200]
        )


class RecordingTests(unittest.TestCase):
    def setUp(self):
        rng = np.random.default_rng(0)
        self.clean = np.cumsum(rng.normal(size=1500)) + rng.normal(size=1500)

    def test_damping_smooths_without_moving_the_start(self):
        spec = {"kind": "damping", "tau": 5.0}
        damped = _apply_artifacts(self.clean, [spec], np.random.default_rng(1))
        self.assertAlmostEqual(damped[0], self.clean[0])
        self.assertLess(np.diff(damped).std(), 0.5 * np.diff(self.clean).std())
        related = np.corrcoef(damped, self.clean)[0, 1]
        self.assertGreater(related, 0.9)

    def test_resolution_and_damping_are_drawn_by_their_own_settings(self):
        columns = [{"name": "outcome", "discrete": False}]
        settings = {
            "next_index": 3,
            "redundant_sensor_probability": 0.0,
            "measurement_probability": 0.0,
            "artifact_weights": {"noise": 1.0},
            "missingness_probability": 0.0,
            "missingness_weights": {"mcar": 1.0},
        }
        rng = np.random.default_rng(0)
        self.assertEqual(sample_observation(rng, columns, settings)["artifacts"], {})
        settings.update(resolution_probability=1.0, damping_probability=1.0)
        drawn_kinds = [
            artifact["kind"]
            for artifact in sample_observation(rng, columns, settings)["artifacts"][
                "outcome"
            ]
        ]
        self.assertEqual(sorted(drawn_kinds), ["damping", "quantize"])

    def test_stored_resolution_limits_the_distinct_values(self):
        spec = {"kind": "quantize", "levels": 20.0, "phase": 0.3}
        stored = _apply_artifacts(self.clean, [spec], np.random.default_rng(1))
        self.assertLess(len(np.unique(stored)), 40)


if __name__ == "__main__":
    unittest.main()

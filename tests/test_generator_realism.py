import json
import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(
    0, str(Path(__file__).resolve().parents[1] / "src" / "01_data_generation")
)
from effect_mechanisms import evaluate_extra
from measurement_layer import (
    PROTECTED_TAIL,
    _apply_artifacts,
    apply_observation,
    sample_observation,
)
from plant_mechanisms import (
    LIFECYCLE_KINDS,
    PLANT_FAMILIES,
    lifecycle_path,
    sample_lifecycle,
    sample_plant,
    simulate_plant,
)
from scm_generator import OperationalSCMGenerator

REALISM = dict(
    full_length_probability=0.5,
    prehistory_probability=0.5,
    max_prehistory_rows=128,
    row_gap_probability=0.2,
    seasonal_table_probability=0.4,
    noise_profile_weights={"clean": 1, "low": 1, "moderate": 1, "noisy": 1},
    slow_dynamics_probability=0.25,
    transport_delay_probability=0.1,
    step_response_probability=0.5,
    lifecycle_trend_probability=0.5,
    calendar_gate_probability=0.3,
    plant_probability=0.5,
    startup_transient_probability=0.6,
    redundant_sensor_probability=0.3,
    measurement_probability=0.5,
    missingness_probability=0.5,
    nonseasonal_wander_probability=0.7,
    seasonal_wander_probability=0.25,
    effect_weights={"second_order": 2.0},
)


def frame_of(values, **columns):
    data = {"time_index": np.arange(len(values)), "outcome": values, **columns}
    return pd.DataFrame(data)


class DefaultsTests(unittest.TestCase):
    def test_new_behaviour_is_opt_in(self):
        generator = OperationalSCMGenerator("unused", seed=3, min_rows=60, max_rows=90)
        for trend_free in (True, False) * 5:
            model = generator.sample_model(trend_free=trend_free)
            for key in ("plant", "observation", "recorded_rows", "row_gaps"):
                self.assertNotIn(key, model)
            self.assertNotIn("wander", model["outcome"])
            frame = generator.simulate(model)
            self.assertEqual(len(frame), model["rows"])
            self.assertTrue(np.isfinite(frame.to_numpy()).all())


class EndToEndTests(unittest.TestCase):
    def test_tables_are_finite_replayable_and_serializable(self):
        generator = OperationalSCMGenerator(
            "unused", min_rows=120, max_rows=260, **REALISM
        )
        seen = set()
        for seed in range(60):
            generator.rng = np.random.default_rng(seed)
            model, frame = generator._sample_accepted_model(seed % 2 == 0)
            json.dumps(model, allow_nan=False)
            self.assertEqual(len(frame), model.get("recorded_rows", model["rows"]))
            np.testing.assert_array_equal(frame["time_index"], np.arange(len(frame)))
            values = frame.to_numpy(dtype=float)
            self.assertFalse(np.isinf(values).any())
            # Evaluation labels are always observed.
            self.assertTrue(np.isfinite(frame["outcome"].to_numpy()[-64:]).all())
            pd.testing.assert_frame_equal(generator.simulate(model), frame)
            seen.add("plant" in model)
            seen.add(("missing", bool(frame.isna().any().any())))
            seen.add(("prehistory", "recorded_rows" in model))
        self.assertEqual(
            seen,
            {True, False}
            | {("missing", flag) for flag in (True, False)}
            | {("prehistory", flag) for flag in (True, False)},
        )

    def test_full_length_point_mass_and_prehistory(self):
        generator = OperationalSCMGenerator(
            "unused",
            seed=5,
            min_rows=100,
            max_rows=300,
            full_length_probability=0.5,
            prehistory_probability=1.0,
            max_prehistory_rows=200,
        )
        lengths = []
        for _ in range(60):
            model = generator.sample_model()
            frame = generator.simulate(model)
            lengths.append(len(frame))
            if "recorded_rows" in model:
                self.assertEqual(len(frame), 300)
                self.assertGreaterEqual(model["rows"], 364)
                clean = generator._simulate_process(model)
                # The table is the TAIL of the longer simulated history.
                np.testing.assert_array_equal(
                    frame["outcome"], clean["outcome"].to_numpy()[-300:]
                )
        self.assertTrue(0.3 < np.mean(np.array(lengths) == 300) < 0.75)

    def test_coarse_copy_of_cut_history_replays(self):
        with tempfile.TemporaryDirectory() as folder:
            generator = OperationalSCMGenerator(
                folder,
                seed=9,
                min_rows=600,
                max_rows=600,
                full_length_probability=1.0,
                prehistory_probability=1.0,
                max_prehistory_rows=100,
                coarse_copy_probability=1.0,
                save_plot=False,
                save_dag=False,
            )
            model, frame = generator._sample_accepted_model(True)
            directory = Path(folder) / "table"
            generator._write_table(model, frame, directory)
            target = generator._write_coarse_copy(model, frame, directory)
            derived = json.loads((target / "scm.json").read_text())
            pd.testing.assert_frame_equal(
                generator.simulate(derived), pd.read_parquet(target / "data.parquet")
            )


class PriorTests(unittest.TestCase):
    def test_cycle_free_tables_and_noise_profiles(self):
        generator = OperationalSCMGenerator(
            "unused",
            seed=2,
            min_rows=100,
            max_rows=150,
            seasonal_table_probability=0.0,
            nonseasonal_covariate_season_probability=0.0,
            noise_profile_weights={"clean": 1.0},
            nonseasonal_wander_probability=1.0,
        )
        for _ in range(15):
            model = generator.sample_model(trend_free=True)
            self.assertEqual(model["outcome"]["seasons"], [])
            self.assertEqual(model["seasonal_lags"], [])
            self.assertEqual(model["outcome"]["white_noise_scale"], 0.0)
            self.assertIn("wander", model["outcome"])
            self.assertFalse(
                any(n.get("seasons") for n in model["nodes"] if n["kind"] == "process")
            )
        with self.assertRaises(ValueError):
            OperationalSCMGenerator("unused", noise_profile_weights={"silent": 1})

    def test_constant_outcomes_are_resampled(self):
        generator = OperationalSCMGenerator(
            "unused", seed=4, min_rows=80, max_rows=80, reject_constant_outcome=True
        )
        original, flat = generator._simulate_process, [True]

        def simulate(model):
            frame = original(model)
            if flat.pop() if flat else False:
                frame["outcome"] = 250.0
            return frame

        generator._simulate_process = simulate
        model, frame = generator._sample_accepted_model(True)
        self.assertGreaterEqual(model["quality"]["sampling_attempts"], 2)
        self.assertGreater(np.ptp(frame["outcome"]), 0)

    def test_wander_is_stationary_and_slow(self):
        spec = {"tau": 50.0, "scale": 2.0, "seed": 4}
        path = OperationalSCMGenerator._wander(spec, 60000)
        self.assertAlmostEqual(float(np.std(path)), 2.0, delta=0.25)
        self.assertGreater(np.corrcoef(path[:-1], path[1:])[0, 1], 0.97)
        np.testing.assert_array_equal(
            path[:100], OperationalSCMGenerator._wander(spec, 100)
        )

    def test_second_order_step_overshoots_then_settles(self):
        p = dict(k=1, width=1, threshold=1, floor=0.05, tau=4, limit=100, period=20)
        state = {}
        response = [
            evaluate_extra("second_order", 1.0, 0, {**p, "damping": 0.2}, state)
            for _ in range(200)
        ]
        # Analytic overshoot exp(-pi*zeta/sqrt(1-zeta^2)) = 0.527 at zeta 0.2.
        self.assertAlmostEqual(max(response), 1.527, places=2)
        self.assertAlmostEqual(response[-1], 1.0, places=6)
        state = {}
        damped = [
            evaluate_extra("second_order", 1.0, 0, {**p, "damping": 1.5}, state)
            for _ in range(200)
        ]
        self.assertTrue(np.all(np.diff(damped) >= 0))
        self.assertLessEqual(max(damped), 1.0 + 1e-9)

    def test_lifecycle_shapes(self):
        time = np.arange(-50, 1000)
        kinds = set()
        for seed in range(80):
            trend = sample_lifecycle(np.random.default_rng(seed), 1000, 10.0)
            json.dumps(trend, allow_nan=False)
            path = lifecycle_path(trend, time)
            kinds.add(trend["kind"])
            self.assertTrue(np.isfinite(path).all())
            np.testing.assert_array_equal(path[time < 0], 0)
            if trend["kind"] in ("saturating", "exponential_wear", "onset", "logistic"):
                self.assertTrue(np.all(np.diff(path[time >= 0]) >= -1e-9))
                self.assertAlmostEqual(path[-1], 10.0, delta=0.1)
            if trend["kind"] == "maintenance_sawtooth":
                # Service visibly restores the level at least once.
                self.assertLess(np.diff(path).min(), -0.3)
        self.assertEqual(kinds, set(LIFECYCLE_KINDS))
        generator = OperationalSCMGenerator(
            "unused", seed=1, min_rows=200, max_rows=200, lifecycle_trend_probability=1
        )
        model, frame = generator._sample_accepted_model(False)
        self.assertIn(model["outcome"]["trend"]["kind"], LIFECYCLE_KINDS)
        self.assertGreater(
            max(
                model["quality"]["trend_shape_r_squared"],
                model["quality"]["linear_trend_r_squared"],
            ),
            0.15,
        )

    def test_coast_down_and_inrush(self):
        generator = OperationalSCMGenerator(
            "unused",
            seed=6,
            min_rows=300,
            max_rows=300,
            switching_probability=1,
            process_episode_probability=0,
            startup_transient_probability=1,
        )
        for _ in range(40):
            model = generator.sample_model(trend_free=True)
            operation = model["outcome"]["operation"]
            if operation["mode"] == "shutdown" and operation["stop_tau"] > 10:
                break
        else:
            self.fail("no coast-down shutdown sampled")
        frame = generator.simulate(model)
        switch = next(n for n in model["nodes"] if n["kind"] == "switch")
        self.assertIn(operation["restart_shape"], ("linear", "exponential"))
        self.assertTrue(np.isfinite(frame["outcome"]).all())
        instant = json.loads(json.dumps(model))
        instant["outcome"]["operation"]["stop_tau"] = 0.0
        self.assertEqual(switch["kind"], "switch")
        # With a coast-down the stopped unit is NOT pinned to the flat value.
        stopped = generator.simulate(instant)["outcome"] == operation["flat_value"]
        if stopped.any():
            self.assertFalse(
                (frame["outcome"][stopped] == operation["flat_value"]).all()
            )


class PlantTests(unittest.TestCase):
    def test_every_family_is_bounded_causal_and_replayable(self):
        clock = np.arange(-200, 1500)
        rng = np.random.default_rng(11)
        demand = np.sin(clock / 40) + 0.2 * rng.normal(size=len(clock))
        for family in PLANT_FAMILIES:
            for seed in range(12):
                spec = sample_plant(np.random.default_rng(seed), 1500, {family: 1.0}, 5)
                json.dumps(spec, allow_nan=False)
                channels = simulate_plant(spec, demand, clock)
                again = simulate_plant(spec, demand, clock)
                for name, values in channels.items():
                    self.assertTrue(np.isfinite(values).all(), (family, name))
                    np.testing.assert_array_equal(values, again[name])
                self.assertIn(spec["outcome_channel"], channels)
                self.assertNotEqual(spec["outcome_channel"], "status")
                for export in spec["exports"]:
                    self.assertIn(export["channel"], channels)
                # A later change in demand cannot alter earlier output.
                changed = demand.copy()
                changed[1000:] += 5
                prefix = simulate_plant(spec, changed, clock)
                np.testing.assert_array_equal(
                    prefix[spec["outcome_channel"]][:1000],
                    channels[spec["outcome_channel"]][:1000],
                )

    def test_relay_cycles_inside_its_band(self):
        spec = sample_plant(np.random.default_rng(0), 3000, {"relay": 1.0}, 5)
        spec.update(
            on_time=20.0,
            off_time=60.0,
            leak=0.0,
            dead_time=0,
            band_jitter=0.0,
            wear=0.0,
            load_coupling=0.0,
            process_noise=0.0,
            min_on=1,
            min_off=1,
        )
        clock = np.arange(-100, 3000)
        channels = simulate_plant(spec, np.zeros(len(clock)), clock)
        pv, status = channels["pv"][100:], channels["status"][100:]
        self.assertEqual(set(np.unique(status)), {0.0, 1.0})
        self.assertTrue(-0.06 <= pv.min() and pv.max() <= 1.06)
        starts = np.flatnonzero(np.diff(status) > 0)
        # Fall 60 samples, rise 20: one cycle is about 80 samples.
        self.assertAlmostEqual(float(np.mean(np.diff(starts))), 80, delta=4)
        # The value falls while off and rises while on: a closed loop.
        self.assertTrue(np.all(np.diff(pv)[status[1:] == 0] < 0))
        self.assertTrue(np.all(channels["actuator"][100:][status == 0] == 0))

    def test_totalizer_counts_up_and_plant_columns_are_exported(self):
        spec = sample_plant(np.random.default_rng(3), 800, {"totalizer": 1.0}, 5)
        clock = np.arange(-50, 800)
        channels = simulate_plant(spec, np.sin(clock / 30), clock)
        self.assertTrue(np.all(np.diff(channels["pv"]) >= 0))
        generator = OperationalSCMGenerator(
            "unused", seed=8, min_rows=200, max_rows=200, plant_probability=1.0
        )
        families = set()
        for _ in range(30):
            model = generator.sample_model(trend_free=True)
            families.add(model["plant"]["family"])
            frame = generator.simulate(model)
            names = [e["name"] for e in model["plant"]["exports"]]
            self.assertEqual(list(frame)[len(frame.columns) - len(names) :], names)
            self.assertEqual(len(set(frame.columns)), len(frame.columns))
            self.assertEqual(model["univariate"], len(frame.columns) == 2)
        self.assertEqual(
            families,
            {
                name
                for name, weight in generator.plant_weights.items()
                if weight > 0 and name != "totalizer"
            },
        )


class RecordingTests(unittest.TestCase):
    def setUp(self):
        rng = np.random.default_rng(0)
        self.clean = np.sin(np.arange(2000) / 25) + 0.05 * rng.normal(size=2000)

    def artifact(self, **spec):
        return _apply_artifacts(self.clean, [spec], np.random.default_rng(1))

    def test_artifacts_do_what_they_say(self):
        coarse = self.artifact(kind="quantize", levels=8.0, phase=0.3)
        self.assertLessEqual(len(np.unique(coarse)), 12)
        clipped = self.artifact(kind="clip", side="upper", upper=0.8, lower=0.1)
        self.assertAlmostEqual(np.mean(clipped == clipped.max()), 0.2, delta=0.02)
        self.assertEqual(clipped.min(), self.clean.min())
        held = self.artifact(kind="hold", every=5)
        self.assertTrue(np.all(held.reshape(-1, 5) == held.reshape(-1, 5)[:, :1]))
        stuck = self.artifact(kind="stuck", segments=[{"start": 0.5, "length": 0.1}])
        runs = np.flatnonzero(np.diff(stuck) == 0)
        self.assertGreaterEqual(len(runs), 198)
        band = self.artifact(kind="deadband", fraction=0.05)
        low, high = np.percentile(self.clean, [1, 99])
        self.assertLess(np.mean(np.diff(band) != 0), 0.5)
        self.assertLessEqual(np.abs(band - self.clean).max(), 0.05 * (high - low))
        spikes = self.artifact(
            kind="spikes", rate=0.01, magnitude=10.0, direction="up", width=1
        )
        self.assertGreater(spikes.max(), self.clean.max() + 3)
        self.assertGreaterEqual((spikes - self.clean).min(), 0)
        drift = self.artifact(kind="drift", magnitude=1.0, resets=[0.5])
        bias = drift - self.clean
        self.assertAlmostEqual(bias[0], 0)
        self.assertLess(bias[1001], bias[999])  # Recalibrated at the reset.
        constant = _apply_artifacts(
            np.ones(50), [{"kind": "noise", "fraction": 0.05}], np.random.default_rng(1)
        )
        np.testing.assert_array_equal(constant, 1)

    def missing(self, spec, **columns):
        frame = frame_of(self.clean, **columns)
        return apply_observation(frame, {"missing": [spec]}, None, None, seed=7)

    def test_three_missingness_paradigms(self):
        # Completely at random: the rate is met and values do not matter.
        mcar = self.missing(
            {"paradigm": "mcar", "columns": ["outcome"], "rate": 0.1, "blocks": []}
        )
        gone = mcar["outcome"].isna().to_numpy()
        self.assertAlmostEqual(gone.mean(), 0.1, delta=0.025)
        high = self.clean > np.median(self.clean)
        self.assertAlmostEqual(gone[high].mean(), gone[~high].mean(), delta=0.04)
        # At random: decided by ANOTHER column, which itself stays observed.
        driver = np.cos(np.arange(2000) / 90)
        mar = self.missing(
            {
                "paradigm": "mar",
                "columns": ["outcome"],
                "driver": "covariate_01",
                "quantile": 0.7,
                "direction": "above",
                "high": 0.9,
                "low": 0.0,
                "clock_period": 24,
            },
            covariate_01=driver,
        )
        gone = mar["outcome"].isna().to_numpy()[:-PROTECTED_TAIL]
        risky = (driver > np.quantile(driver, 0.7))[:-PROTECTED_TAIL]
        self.assertFalse(mar["covariate_01"].isna().any())
        self.assertEqual(gone[~risky].sum(), 0)
        self.assertGreater(gone[risky].mean(), 0.4)
        # Structural: an exact schedule.
        periodic = self.missing(
            {
                "paradigm": "structural",
                "pattern": "periodic",
                "columns": ["outcome"],
                "period": 50,
                "length": 5,
                "shift": 0,
            }
        )
        gone = periodic["outcome"].isna().to_numpy()
        expected = np.arange(2000) % 50 < 5
        expected[-PROTECTED_TAIL:] = False
        np.testing.assert_array_equal(gone, expected)
        # Not at random: the largest readings are the ones that vanish.
        censored = self.missing(
            {
                "paradigm": "structural",
                "pattern": "censor",
                "columns": ["outcome"],
                "quantile": 0.9,
                "direction": "above",
            }
        )
        kept = censored["outcome"].dropna()
        self.assertLessEqual(kept[:-PROTECTED_TAIL].max(), np.quantile(self.clean, 0.9))

    def test_sampled_recording_layer(self):
        settings = {
            "next_index": 4,
            "redundant_sensor_probability": 1.0,
            "measurement_probability": 1.0,
            "artifact_weights": {"noise": 1.0, "quantize": 1.0, "stuck": 1.0},
            "missingness_probability": 1.0,
            "missingness_weights": {"mcar": 1, "mar": 1, "structural": 1},
        }
        columns = [
            {"name": "outcome", "discrete": False},
            {"name": "covariate_01", "discrete": True},
        ]
        switch = (np.arange(2000) // 100 % 2).astype(float)
        paradigms = set()
        for seed in range(40):
            spec = sample_observation(np.random.default_rng(seed), columns, settings)
            json.dumps(spec, allow_nan=False)
            self.assertNotIn("covariate_01", spec["artifacts"])
            paradigms.add(spec["missing"][0]["paradigm"])
            frame = frame_of(self.clean, covariate_01=switch)
            recorded = apply_observation(frame, spec, None, None, seed)
            pd.testing.assert_frame_equal(
                recorded, apply_observation(frame, spec, None, None, seed)
            )
            self.assertEqual(
                list(recorded)[:3], ["time_index", "outcome", "covariate_01"]
            )
            self.assertEqual(list(recorded)[3], "covariate_04")
            # Discrete levels stay exact wherever they are observed.
            levels = set(recorded["covariate_01"].dropna().unique())
            self.assertTrue(levels <= {0.0, 1.0})
            self.assertTrue(np.isfinite(recorded["outcome"].to_numpy()[-64:]).all())
            self.assertLessEqual(recorded["outcome"].isna().mean(), 0.35)
        self.assertEqual(paradigms, {"mcar", "mar", "structural"})

    def test_removed_rows_keep_length_and_create_a_jump(self):
        ramp = np.arange(1000, dtype=float)
        gaps = {"kind": "blocks", "lengths": [100], "positions": [0.5], "total": 100}
        recorded = apply_observation(frame_of(ramp), {}, 900, gaps, seed=1)
        self.assertEqual(len(recorded), 900)
        np.testing.assert_array_equal(recorded["time_index"], np.arange(900))
        steps = np.diff(recorded["outcome"])
        self.assertEqual(steps.max(), 101)
        self.assertEqual((steps == 1).sum(), 898)
        scattered = apply_observation(
            frame_of(ramp), {}, 950, {"kind": "scattered", "total": 50}, seed=1
        )
        self.assertEqual(len(scattered), 950)
        self.assertEqual(scattered["outcome"].iloc[-1], 999)


if __name__ == "__main__":
    unittest.main()

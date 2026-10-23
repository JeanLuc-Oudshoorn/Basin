import copy
import json
import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from data_generation.scm_generator import OperationalSCMGenerator


class GeneratorTests(unittest.TestCase):
    def test_fourier_families_energy_period_and_sampling_limit(self):
        season = {
            "period": 64.0,
            "phase": 0.0,
            "amplitude": 1.0,
            "modulation": 0.0,
            "modulation_period": 256.0,
        }
        time = np.arange(4096) * 64 / 4096
        spectra = []
        for family in OperationalSCMGenerator.DEFAULT_FOURIER_WEIGHTS:
            weights = {
                name: int(name == family)
                for name in OperationalSCMGenerator.DEFAULT_FOURIER_WEIGHTS
            }
            generator = OperationalSCMGenerator(
                "unused",
                seed=12,
                fourier_weights=weights,
                seasonal_shape_drift_probability=0,
            )
            spec = generator._sample_fourier(season, 512)
            self.assertEqual(spec["family"], family)
            curve = generator._seasonal([{**season, "fourier": spec}], time)
            self.assertAlmostEqual(float(curve.mean()), 0.0, places=12)
            self.assertAlmostEqual(float(np.mean(curve**2)), 0.5, places=12)
            np.testing.assert_allclose(
                curve,
                generator._seasonal([{**season, "fourier": spec}], time + 64),
                atol=1e-12,
            )
            spectra.append(curve)
            # A six-sample cycle cannot resolve sixteen separate harmonics.
            short = {**season, "period": 6.0, "frequency_depth": 0.35}
            spec = generator._sample_fourier(short, 128)
            self.assertLess(max(spec["orders"]) * 1.35 / 6, 0.5)
        # The template prior produces meaningfully different cycle shapes.
        self.assertTrue(
            all(
                np.std(spectra[i] - spectra[j]) > 0.05
                for i in range(len(spectra))
                for j in range(i)
            )
        )

    def test_fourier_evolution_prefix_and_known_coefficients(self):
        generator = OperationalSCMGenerator(
            "unused", seed=2, seasonal_shape_drift_probability=1
        )
        season = {
            "period": 64.0,
            "phase": 0.0,
            "amplitude": 1.0,
            "modulation": 0.0,
            "modulation_period": 256.0,
        }
        spec = {"orders": [1, 2], "start": {"cos": [0.3, 0.4], "sin": [0.5, 0.6]}}
        t = np.arange(200)
        phase = 2 * np.pi * t / 64
        expected = (
            0.3 * np.cos(phase)
            + 0.4 * np.cos(2 * phase)
            + 0.5 * np.sin(phase)
            + 0.6 * np.sin(2 * phase)
        )
        np.testing.assert_allclose(
            generator._seasonal([{**season, "fourier": spec}], t), expected, atol=1e-14
        )
        evolving = {**season, "fourier": generator._sample_fourier(season, 512)}
        restored = json.loads(json.dumps(evolving))
        long = generator._seasonal([restored], np.arange(-100, 512))
        short = generator._seasonal([evolving], np.arange(-100, 128))
        np.testing.assert_array_equal(short, long[:228])
        self.assertGreater(np.std(long[100:164] - long[484:548]), 0.01)
        # Older saved seasonal specifications retain their original equations.
        legacy = {**season, "harmonics": [{"order": 2, "weight": 0.4, "phase": 0.7}]}
        np.testing.assert_allclose(
            generator._seasonal([legacy], t),
            (np.sin(phase) + 0.4 * np.sin(2 * phase + 0.7)) / np.sqrt(1.16),
            atol=1e-14,
        )

    def test_lags_dag_and_hidden_covariates(self):
        generator = OperationalSCMGenerator("unused", min_rows=100, max_rows=100)
        for _ in range(12):
            model = generator.sample_model()
            frame = generator.simulate(model)
            available = {f"outcome_lag_{lag}" for lag in model["lags"]}
            for node in model["nodes"]:
                self.assertTrue(all(e["parent"] in available for e in node["edges"]))
                available.add(node["name"])
                self.assertEqual(node["name"] in frame, node["visible"])
            for lag in model["lags"]:
                if lag < len(frame):
                    np.testing.assert_array_equal(
                        frame[f"outcome_lag_{lag}"].to_numpy()[lag:],
                        frame["outcome"].to_numpy()[:-lag],
                    )
            hidden = copy.deepcopy(model)
            for node in hidden["nodes"]:
                node["visible"] = False
            hidden_frame = generator.simulate(hidden)
            pd.testing.assert_series_equal(frame["outcome"], hidden_frame["outcome"])
            self.assertEqual(len(hidden_frame.columns), 2 + len(model["lags"]))

    def test_saved_artifacts_and_replay(self):
        with tempfile.TemporaryDirectory() as directory:
            generator = OperationalSCMGenerator(directory, min_rows=20, max_rows=20)
            first = generator.generate(1)[0]
            model = json.loads((first / "scm.json").read_text())
            recorded = pd.read_parquet(first / "data.parquet")
            pd.testing.assert_frame_equal(recorded, generator.simulate(model))
            self.assertGreater((first / "series.png").stat().st_size, 1000)
            self.assertGreater((first / "dag.png").stat().st_size, 1000)
            second = generator.generate(1)[0]
            self.assertNotEqual(first, second)
            self.assertTrue((first / "data.parquet").exists())

    def test_reproducibility_bounds_and_options(self):
        with tempfile.TemporaryDirectory() as directory:
            kwargs = dict(
                seed=123,
                min_rows=2056,
                max_rows=2056,
                max_lag=1,
                hidden_probability=1,
                save_scm=False,
                save_plot=False,
                save_dag=False,
            )
            left = OperationalSCMGenerator(directory, **kwargs)
            right = OperationalSCMGenerator(directory, **kwargs)
            self.assertEqual(left.sample_model(), right.sample_model())
            path = left.generate(1)[0]
            frame = pd.read_parquet(path / "data.parquet")
            self.assertEqual(frame.shape, (2056, 3))
            self.assertTrue(np.isfinite(frame.to_numpy()).all())
            self.assertEqual({p.name for p in path.iterdir()}, {"data.parquet"})

    def test_noise_families_and_sampled_shapes(self):
        generator = OperationalSCMGenerator("unused")
        rng = np.random.default_rng(94)
        for family in generator.NOISES:
            with self.subTest(family=family):
                spec = generator._sample_noise(family)
                draws = generator._noise(rng, spec, 20000)
                self.assertTrue(np.isfinite(draws).all())
                self.assertGreater(np.std(draws), 0)
                self.assertNotEqual(spec, generator._sample_noise(family))
        # Test two clearly separated but overlapping Student-t components.
        spec = {
            "kind": "student_t_mixture",
            "location": 0,
            "scale": 1,
            "weights": [0.5, 0.5],
            "components": [
                {"kind": "student_t", "location": center, "scale": 0.6, "df": 7}
                for center in [-2, 2]
            ],
        }
        draws = generator._noise(rng, spec, 100000)
        self.assertGreater(np.mean(np.abs(draws - 2) < 0.5), 0.25)
        self.assertGreater(np.mean(np.abs(draws + 2) < 0.5), 0.25)
        self.assertLess(np.mean(np.abs(draws) < 0.5), 0.03)
        spec = generator._sample_noise("quantization")
        errors = generator._noise(rng, spec, 10000)
        self.assertLessEqual(
            np.max(np.abs(errors - spec["location"])), spec["scale"] * spec["step"] / 2
        )

    @staticmethod
    def _quiet_process(spec):
        # Remove unrelated mechanisms to make temporal assertions analytical.
        spec.update(
            edges=[],
            interaction=0,
            intercept=0,
            seasons=[],
            persistence=0,
            slope=0,
            slope_walk=0,
            walk_scale=0,
            event_parent=None,
            level_jump=0,
            shock_size=0,
            heteroskedasticity=0,
            quantization_step=0,
            noise={"kind": "normal", "location": 0, "scale": 0},
        )
        spec.update(white_noise_scale=0, outcome_ar_weights=[], event_lag=0)
        spec.pop("level_limit", None)
        spec.pop("operation", None)
        spec.pop("trend", None)
        spec.pop("ringing", None)
        spec.pop("volatility", None)
        spec.pop("multiplicative_scaling", None)

    def test_binary_event_reverses_structural_drift(self):
        generator = OperationalSCMGenerator(
            "unused",
            min_rows=100,
            max_rows=100,
            min_covariates=3,
            max_covariates=3,
            max_lag=1,
            switching_probability=0,
        )
        model = generator.sample_model()
        for spec in model["nodes"][1:] + [model["outcome"]]:
            self._quiet_process(spec)
        event = model["nodes"][0]
        event.update(visible=True, probability=0.3)
        outcome = model["outcome"]
        outcome.update(
            ar_weights=[0],
            baseline=0,
            scale=1,
            slope=0.2,
            event_parent=event["name"],
            event_mode="reverse",
        )
        frame = generator.simulate(model)
        differences = np.diff(frame["outcome"])
        np.testing.assert_allclose(np.abs(differences), 0.2, atol=1e-12)
        np.testing.assert_allclose(
            differences[1:],
            differences[:-1]
            * np.where(frame[event["name"]].to_numpy()[2:] == 1, -1, 1),
            atol=1e-12,
        )

    def test_delayed_causal_chain(self):
        generator = OperationalSCMGenerator(
            "unused",
            min_rows=64,
            max_rows=64,
            min_covariates=3,
            max_covariates=3,
            max_lag=5,
            switching_probability=0,
        )
        model = generator.sample_model()
        for spec in model["nodes"][1:] + [model["outcome"]]:
            self._quiet_process(spec)
        source, mediator = model["nodes"][1:]
        source.update(slope=0.1, visible=True)
        mediator["visible"] = True
        edge = {
            "parent": source["name"],
            "lag": 10,
            "weight": 0.7,
            "input_scale": 20,
            "input_offset": 0,
            "output_offset": 0,
            "program": [
                {"op": "affine", "a": 0, "b": 0, "gain": 1, "offset": 0, "limit": 10}
            ],
        }
        mediator["edges"] = [edge]
        model["outcome"].update(
            edges=[{**edge, "parent": mediator["name"], "lag": 20}],
            ar_weights=[0] * len(model["lags"]),
            scale=1,
            baseline=0,
        )
        frame = generator.simulate(model)
        for i in range(30, len(frame)):
            self.assertAlmostEqual(
                frame[mediator["name"]][i],
                generator._effect(edge, frame[source["name"]][i - 10]),
            )
            self.assertAlmostEqual(
                frame["outcome"][i],
                generator._effect(edge, frame[mediator["name"]][i - 20]),
            )

    def test_expression_operators_and_graph_topology(self):
        generator = OperationalSCMGenerator(
            "unused", min_covariates=8, max_covariates=8
        )
        operators, topologies, delayed, indirect = set(), set(), False, False
        for _ in range(50):
            model = generator.sample_model()
            topologies.add(model["topology"])
            graph = generator.graph_spec(model)
            pending = set(graph["vertices"])
            visited = set()
            while pending:
                ready = {
                    n
                    for n in pending
                    if all(
                        e["source"] in visited
                        for e in graph["edges"]
                        if e["target"] == n
                    )
                }
                self.assertTrue(ready)
                pending -= ready
                visited |= ready
            indirect |= len(model["outcome"]["edges"]) < len(model["nodes"]) - 1
            for spec in model["nodes"] + [model["outcome"]]:
                for edge in spec["edges"]:
                    delayed |= edge["lag"] > 0
                    operators.update(p["op"] for p in edge["program"])
                    for value in [-1000, 0, 1000]:
                        self.assertTrue(np.isfinite(generator._effect(edge, value)))
        self.assertEqual(operators, set(generator.OPERATORS))
        self.assertEqual(topologies, {"sparse", "fork", "diamond", "dense"})
        self.assertTrue(delayed and indirect)

    def test_legacy_replay(self):
        from data_generation.legacy_scm_generator import (
            OperationalSCMGenerator as Legacy,
        )

        legacy = Legacy("unused", min_rows=20, max_rows=20)
        model = legacy.sample_model()
        pd.testing.assert_frame_equal(
            legacy.simulate(model), OperationalSCMGenerator("unused").simulate(model)
        )

    def test_perlin_smoothness_replay_and_mixture_time_axis(self):
        generator = OperationalSCMGenerator("unused")
        spec = generator._sample_noise("perlin")
        spec.update(frequency=0.02, octaves=1, location=0, scale=1)
        short = generator._noise(np.random.default_rng(1), spec, 512)
        long = generator._noise(
            np.random.default_rng(99), json.loads(json.dumps(spec)), 1024
        )
        np.testing.assert_array_equal(short, long[:512])
        self.assertGreater(np.std(short), 0.02)
        self.assertGreater(np.corrcoef(short[:-1], short[1:])[0, 1], 0.98)
        changed = {**spec, "seed": spec["seed"] + 1}
        self.assertFalse(
            np.array_equal(
                short, generator._noise(np.random.default_rng(1), changed, 512)
            )
        )
        # Selecting samples from a mixture must not compress the Perlin clock.
        mixture = {
            "kind": "heterogeneous_mixture",
            "location": 0,
            "scale": 1,
            "weights": [0.5, 0.5],
            "components": [spec, {"kind": "normal", "location": 10, "scale": 0}],
        }
        labels = np.random.default_rng(12).choice(2, size=512, p=[0.5, 0.5])
        mixed = generator._noise(np.random.default_rng(12), mixture, 512)
        np.testing.assert_array_equal(mixed[labels == 0], short[labels == 0])
        np.testing.assert_array_equal(mixed[labels == 1], 10)

    def test_weighted_noise_selection_and_perlin_covariates(self):
        generator = OperationalSCMGenerator("unused", seed=114)
        counts = {family: 0 for family in generator.NOISES}
        for _ in range(5000):
            counts[generator._choose_noise(generator.NOISES)] += 1
        for preferred in ("perlin", "student_t", "pink"):
            self.assertGreater(counts[preferred], 650)
            self.assertLess(counts[preferred], 1000)
            self.assertGreater(
                counts[preferred],
                2
                * max(
                    count
                    for family, count in counts.items()
                    if family not in ("perlin", "student_t", "pink")
                ),
            )
        # Force Perlin everywhere to exercise outcome noise, exogenous starting
        # processes, hidden covariates and JSON replay through the simulator.
        weights = {name: float(name == "perlin") for name in generator.NOISES}
        perlin = OperationalSCMGenerator(
            "unused",
            noise_weights=weights,
            min_rows=100,
            max_rows=100,
            edge_probability=0,
            delayed_probability=0,
            switching_probability=0,
        )
        model = perlin.sample_model()
        for process in model["nodes"][1:] + [model["outcome"]]:
            self.assertEqual(process["noise"]["kind"], "perlin")
        self.assertFalse(model["nodes"][1]["edges"])
        frame = perlin.simulate(model)
        pd.testing.assert_frame_equal(
            frame, perlin.simulate(json.loads(json.dumps(model)))
        )
        for lag in model["lags"]:
            if lag < len(frame):
                np.testing.assert_array_equal(
                    frame[f"outcome_lag_{lag}"].to_numpy()[lag:],
                    frame["outcome"].to_numpy()[:-lag],
                )

    def test_invalid_noise_weights_and_perlin_priors(self):
        for weights in (
            {"bad": 1},
            {"perlin": -1},
            {"perlin": float("nan")},
            {family: 0 for family in OperationalSCMGenerator.NOISES},
        ):
            with self.assertRaises(ValueError):
                OperationalSCMGenerator("unused", noise_weights=weights)
        with self.assertRaises(ValueError):
            OperationalSCMGenerator("unused", prior_ranges={"perlin_octaves": (1.2, 6)})

    def test_pink_spectrum_replay_and_mixture_clock(self):
        from scipy.signal import welch

        generator = OperationalSCMGenerator("unused")
        spec = generator._sample_noise("pink")
        spec.update(exponent=1.0, filter_length=2048, location=0, scale=1)
        draws = generator._noise(np.random.default_rng(1), spec, 65536)
        frequencies, power = welch(draws, nperseg=2048)
        band = (frequencies > 0.005) & (frequencies < 0.1)
        slope = np.polyfit(np.log(frequencies[band]), np.log(power[band]), 1)[0]
        self.assertAlmostEqual(slope, -1, delta=0.15)
        short = generator._noise(
            np.random.default_rng(99), json.loads(json.dumps(spec)), 512
        )
        np.testing.assert_allclose(short, draws[:512], atol=1e-14)
        self.assertEqual(generator._noise(np.random.default_rng(1), spec, 0).size, 0)
        mixture = {
            "kind": "heterogeneous_mixture",
            "location": 0,
            "scale": 1,
            "weights": [0.5, 0.5],
            "components": [spec, {"kind": "normal", "location": 10, "scale": 0}],
        }
        labels = np.random.default_rng(12).choice(2, size=512, p=[0.5, 0.5])
        mixed = generator._noise(np.random.default_rng(12), mixture, 512)
        np.testing.assert_array_equal(mixed[labels == 0], short[labels == 0])
        weights = {name: float(name == "pink") for name in generator.NOISES}
        pink = OperationalSCMGenerator(
            "unused",
            noise_weights=weights,
            min_rows=30,
            max_rows=30,
            switching_probability=0,
        )
        model = pink.sample_model()
        self.assertTrue(all(n["noise"]["kind"] == "pink" for n in model["nodes"][1:]))
        pd.testing.assert_frame_equal(
            pink.simulate(model), pink.simulate(json.loads(json.dumps(model)))
        )

    def test_inverse_square_effect_and_shift(self):
        operator = {
            "op": "inverse_square",
            "a": 0,
            "b": 0,
            "gain": 1,
            "offset": 0,
            "limit": 10,
            "floor": 0.05,
        }
        evaluate = OperationalSCMGenerator._evaluate
        for value in (-4, -2, 2, 4):
            self.assertAlmostEqual(
                evaluate([operator], value), 10 * np.tanh((1 / value**2) / 10)
            )
        self.assertGreater(evaluate([operator], 2), evaluate([operator], 4))
        self.assertTrue(np.isfinite(evaluate([operator], 0)))
        self.assertAlmostEqual(
            evaluate([{**operator, "offset": -2}], 4), evaluate([operator], 2)
        )

    def test_v3_graph_and_trend_profile_constraints(self):
        generator = OperationalSCMGenerator("unused", seed=300)
        residual_count, lag_count = 0, 0
        for _ in range(100):
            model = generator.sample_model(trend_free=True)
            processes = [n for n in model["nodes"] if n["kind"] == "process"]
            self.assertGreaterEqual(
                len(model["outcome"]["edges"]), min(2, len(processes))
            )
            for node in processes + [model["outcome"]]:
                for key in (
                    "slope",
                    "slope_walk",
                    "walk_scale",
                    "event_slope",
                    "level_jump",
                ):
                    self.assertEqual(node[key], 0)
                for edge in node["edges"]:
                    if not edge["parent"].startswith("outcome_lag_"):
                        if node is model["outcome"]:
                            self.assertTrue(
                                0 <= edge["lag"] <= generator.max_covariate_lag
                            )
                        else:
                            self.assertTrue(
                                0 <= edge["lag"] <= generator.max_covariate_lag
                            )
            residual = model["outcome"]["ar_weights"]
            residual_count += np.count_nonzero(residual)
            lag_count += len(residual)
            for weight, outcome_weight in zip(
                residual, model["outcome"]["outcome_ar_weights"]
            ):
                self.assertTrue(not weight or outcome_weight > 0)
        self.assertTrue(0.03 < residual_count / lag_count < 0.15)

    def test_trend_free_batch_quota(self):
        with tempfile.TemporaryDirectory() as directory:
            generator = OperationalSCMGenerator(
                directory,
                seed=43,
                min_rows=100,
                max_rows=100,
                min_covariates=3,
                max_covariates=5,
                save_plot=False,
                save_dag=False,
            )
            paths = generator.generate(5)
            screened = 0
            for path in paths:
                model = json.loads((path / "scm.json").read_text())
                frame = pd.read_parquet(path / "data.parquet")
                if model["quality"]["screened_trend_free"]:
                    screened += 1
                    self.assertLessEqual(generator.trend_score(frame), 0.15)
                else:
                    self.assertGreaterEqual(generator.trend_strength(frame, model), 0.4)
            self.assertGreaterEqual(screened, 3)

    def test_permanent_shock_trend_and_required_trending_profile(self):
        generator = OperationalSCMGenerator(
            "unused",
            drift_probability=0,
            min_rows=100,
            max_rows=100,
            multiplicative_trend_probability=0,
        )
        kinds = set()
        for _ in range(40):
            model = generator.sample_model(trend_free=False)
            trend = model["outcome"]["trend"]
            kinds.add(trend["kind"])
            self.assertNotEqual(trend["initial_slope"], 0)
        self.assertEqual(kinds, {"linear", "accelerating", "decelerating", "broken"})
        time = np.arange(-20, 100)
        events = np.zeros(len(time))
        events[time == 40] = 1
        trend = {
            "kind": "broken",
            "initial_slope": 1.0,
            "event_time": 40,
            "event_parent": "event",
            "slope_multiplier": -2.0,
        }
        path = generator._trend_path(trend, time, events)
        np.testing.assert_array_equal(path[time < 0], 0)
        differences = np.diff(path)
        np.testing.assert_array_equal(differences[(time[1:] >= 0) & (time[1:] < 40)], 1)
        np.testing.assert_array_equal(differences[time[1:] >= 40], -2)
        # Without the event there is no break, even after the eligible time.
        unbroken = generator._trend_path(trend, time, np.zeros(len(time)))
        self.assertEqual(unbroken[-1], 100)

    def test_multiplicative_trend_feedback_and_log_linearity(self):
        generator = OperationalSCMGenerator(
            "unused",
            min_rows=100,
            max_rows=100,
            switching_probability=0,
            multiplicative_trend_probability=1,
        )
        model = generator.sample_model(trend_free=False)
        self.assertEqual(model["outcome"]["trend"]["kind"], "multiplicative")
        self.assertNotIn("trend", generator.sample_model(trend_free=True)["outcome"])
        trend = model["outcome"]["trend"].copy()
        for node in model["nodes"][1:] + [model["outcome"]]:
            self._quiet_process(node)
        outcome = model["outcome"]
        outcome.update(baseline=10, scale=2, ar_weights=[0] * len(model["lags"]))
        trend.update(initial_level=5)
        outcome["trend"] = trend
        for rate in (0.015, -0.015):
            trend["log_rate"] = rate
            frame = generator.simulate(model)
            expected = 10 * np.exp(rate * np.arange(1, 101))
            np.testing.assert_allclose(frame["outcome"], expected, rtol=1e-12)
            np.testing.assert_allclose(
                np.diff(np.log(frame["outcome"])), rate, atol=1e-12
            )
            self.assertAlmostEqual(generator.trend_strength(frame, model), 1)
            pd.testing.assert_frame_equal(
                frame, generator.simulate(json.loads(json.dumps(model)))
            )
            for lag in model["lags"]:
                if lag < len(frame):
                    np.testing.assert_array_equal(
                        frame[f"outcome_lag_{lag}"].to_numpy()[lag:],
                        frame["outcome"].to_numpy()[:-lag],
                    )

    def test_variance_growth_with_constant_mean(self):
        generator = OperationalSCMGenerator(
            "unused",
            min_rows=4000,
            max_rows=4000,
            min_covariates=3,
            max_covariates=3,
            switching_probability=0,
        )
        model = generator.sample_model(trend_free=True)
        for node in model["nodes"][1:] + [model["outcome"]]:
            self._quiet_process(node)
        outcome = model["outcome"]
        outcome.update(
            baseline=10,
            scale=1,
            ar_weights=[0] * len(model["lags"]),
            white_noise_scale=1,
            noise={"kind": "normal", "location": 4, "scale": 0},
        )
        reference = generator.simulate(model)["outcome"].to_numpy()
        outcome["volatility"] = {
            "mode": "growth",
            "duration": 4000,
            "log_change": np.log(4),
        }
        varying = generator.simulate(model)["outcome"].to_numpy()
        envelope = np.exp(np.log(4) * np.arange(4000) / 4000)
        # The saved distribution mean (4) is not scaled along with the noise.
        np.testing.assert_allclose(
            varying, 14 + (reference - 14) * envelope, atol=1e-12
        )
        self.assertGreater(np.std(varying[-1000:]), 2 * np.std(varying[:1000]))
        self.assertAlmostEqual(varying[:1000].mean(), 14, delta=0.2)
        self.assertAlmostEqual(varying[-1000:].mean(), 14, delta=0.4)

    def test_multiplicative_noise_and_seasonality_track_level(self):
        generator = OperationalSCMGenerator(
            "unused",
            min_rows=2000,
            max_rows=2000,
            min_covariates=3,
            max_covariates=3,
            switching_probability=0,
        )
        model = generator.sample_model(trend_free=True)
        for node in model["nodes"][1:] + [model["outcome"]]:
            self._quiet_process(node)
        outcome = model["outcome"]
        outcome.update(
            baseline=10,
            scale=1,
            ar_weights=[0] * len(model["lags"]),
            white_noise_scale=1,
            trend={
                "kind": "multiplicative",
                "initial_level": 10,
                "log_rate": np.log(6) / 2000,
                "event_parent": None,
            },
            seasons=[
                {
                    "period": 20,
                    "amplitude": 0.5,
                    "phase": 0,
                    "modulation": 0,
                    "modulation_period": 100,
                }
            ],
        )
        t = np.arange(2000)
        ratio = np.exp(np.log(6) * (t + 1) / 2000)
        seasonal = generator._seasonal(outcome["seasons"], t)
        reference_error = (
            generator.simulate(model)["outcome"].to_numpy() - 10 * ratio - seasonal
        )
        outcome["multiplicative_scaling"] = {
            "noise_elasticity": 1.2,
            "seasonal_elasticity": 1.1,
            "seasonal_base_scale": 2,
        }
        observed = generator.simulate(model)
        error = observed["outcome"].to_numpy() - 10 * ratio - seasonal * 2 * ratio**1.1
        np.testing.assert_allclose(error, reference_error * ratio**1.2, atol=1e-10)
        self.assertGreater(np.std(error[-400:]), 4 * np.std(error[:400]))
        pd.testing.assert_frame_equal(
            observed, generator.simulate(json.loads(json.dumps(model)))
        )

    def test_multiplicative_constant_variation_exceptions(self):
        # Force each independent branch and ensure secondary growth mechanisms
        # cannot silently undo the fixed-scale choice.
        for constant_noise, constant_seasons in ((0, 0), (1, 0), (0, 1), (1, 1)):
            generator = OperationalSCMGenerator(
                "unused",
                seed=84,
                multiplicative_trend_probability=1,
                variance_dynamics_probability=1,
                multiplicative_constant_noise_probability=constant_noise,
                multiplicative_constant_seasonality_probability=constant_seasons,
            )
            model = generator.sample_model(trend_free=False)
            outcome = model["outcome"]
            scaling = outcome["multiplicative_scaling"]
            self.assertEqual(scaling["noise_elasticity"] == 0, bool(constant_noise))
            self.assertEqual(
                scaling["seasonal_elasticity"] == 0, bool(constant_seasons)
            )
            if constant_noise:
                self.assertNotIn("volatility", outcome)
                self.assertEqual(outcome["heteroskedasticity"], 0)
            if constant_seasons:
                self.assertTrue(all(s["growth"] == 0 for s in outcome["seasons"]))
            time = np.arange(200)
            noise, seasons = generator._component_scales(outcome, time)
            if constant_noise:
                np.testing.assert_array_equal(noise, np.ones(len(time)))
            if constant_seasons:
                np.testing.assert_array_equal(
                    seasons, np.full(len(time), scaling["seasonal_base_scale"])
                )
            # Persisted equations alone reproduce the same envelopes.
            restored = json.loads(json.dumps(outcome))
            for original, replay in zip(
                (noise, seasons), generator._component_scales(restored, time)
            ):
                np.testing.assert_array_equal(original, replay)

    def test_recent_and_seasonal_lag_priors(self):
        generator = OperationalSCMGenerator("unused", seed=15)
        count_two, distant, multiple_seasons = 0, 0, 0
        short_counts, season_counts, short_support = [], [], set()
        for _ in range(400):
            lags, roles, seasonal = generator._sample_lags(1000)
            self.assertEqual(lags[0], 1)
            self.assertEqual(len(set(lags)), len(lags))
            self.assertLessEqual(max(lags), generator.max_lag)
            count_two += 2 in lags
            distant += any(
                lag >= 8 and role != "seasonal" for lag, role in zip(lags, roles)
            )
            multiple_seasons += len(seasonal) > 1
            short_counts.append(sum(2 <= lag <= 7 for lag in lags))
            season_counts.append(len(seasonal))
            short_support.update(lag for lag in lags if 2 <= lag <= 7)
            self.assertLessEqual(len(lags), generator.max_lag_count)
            weights = generator._lag_weights(lags, roles)
            recent_weights = [w for w, role in zip(weights, roles) if role == "recent"]
            self.assertEqual(recent_weights, sorted(recent_weights, reverse=True))
        self.assertTrue(40 < count_two < 200)
        self.assertTrue(50 < distant < 110)
        self.assertTrue(25 < multiple_seasons < 100)
        self.assertTrue(250 < short_counts.count(1) < 350)
        self.assertEqual(short_support, set(range(2, 8)))
        self.assertEqual(set(season_counts), {0, 1, 2, 3})
        for _ in range(25):
            model = generator.sample_model()
            periods = {s["period"] for s in model["outcome"]["seasons"]}
            self.assertTrue(set(model["seasonal_lags"]).issubset(periods))

    def test_covariate_delay_prior_and_temporal_validation(self):
        generator = OperationalSCMGenerator(
            "unused",
            seed=29,
            min_covariates=3,
            max_covariates=3,
            switching_probability=0,
        )
        draws = np.array([generator._sample_covariate_lag() for _ in range(10000)])
        self.assertTrue(0.08 < np.mean(draws == 0) < 0.12)
        self.assertTrue(0.64 < np.mean(draws == 1) < 0.69)
        for lag in range(1, 5):
            expected = 0.9 * (np.exp(-(lag - 1) / 0.75) - np.exp(-lag / 0.75))
            self.assertAlmostEqual(np.mean(draws == lag), expected, delta=0.015)
        self.assertTrue(np.all((0 <= draws) & (draws <= 24)))
        self.assertLess(np.mean(draws > 4), 0.01)
        model = generator.sample_model()
        graph = generator.graph_spec(model)
        for edge in graph["edges"]:
            self.assertEqual(graph["vertices"][edge["target"]]["time_offset"], 0)
            self.assertEqual(
                graph["vertices"][edge["source"]]["time_offset"], -edge["lag"]
            )
        # Reject malformed specs before either simulating or rendering them.
        for parent, lag in [
            ("covariate_02", 0),
            ("covariate_00", -1),
            ("covariate_00", 0.5),
            ("outcome_lag_0", 0),
            ("outcome_lag_5", -1),
            ("unknown", 2),
        ]:
            broken = copy.deepcopy(model)
            broken["nodes"][1]["edges"] = [{"parent": parent, "lag": lag, "weight": 1}]
            with self.subTest(parent=parent, lag=lag):
                with self.assertRaises(ValueError):
                    generator.simulate(broken)
                with self.assertRaises(ValueError):
                    generator.graph_spec(broken)
        # A later node's previous value is causal, even though its current value
        # is not yet available. Keep this valid feedback path supported.
        valid = copy.deepcopy(model)
        valid["nodes"][1]["edges"] = [generator._edge("covariate_02", 10)]
        self.assertTrue(np.isfinite(generator.simulate(valid).to_numpy()).all())

    def test_outcome_centered_dag_preserves_explicit_edges(self):
        generator = OperationalSCMGenerator("unused")
        model = generator.sample_model()
        full = generator.graph_spec(model)
        graph, positions, badges = generator._dag_view(model)
        np.testing.assert_array_equal(positions["outcome"], [0, 0])
        for edge in full["edges"]:
            if edge["target"] == "outcome":
                if edge["label"] in ("state", "dwell state"):
                    self.assertIn("outcome", badges)
                else:
                    self.assertTrue(graph.has_edge(edge["source"], "outcome"))
        for source, target, edge in graph.edges(data=True):
            self.assertEqual(
                graph.nodes[target]["time_offset"] - graph.nodes[source]["time_offset"],
                edge["lag"],
            )
        self.assertEqual(set(positions), set(graph))

    def test_graph_unrolls_delayed_mediation(self):
        generator = OperationalSCMGenerator(
            "unused", min_covariates=3, max_covariates=3, switching_probability=0
        )
        model = generator.sample_model()
        for node in model["nodes"][1:] + [model["outcome"]]:
            self._quiet_process(node)
        model["nodes"][2]["edges"] = [generator._edge("covariate_01", 2)]
        model["outcome"]["edges"] = [generator._edge("covariate_02", 1)]
        model["outcome"]["ar_weights"] = [0] * len(model["lags"])
        graph, _, _ = generator._dag_view(model)
        self.assertTrue(graph.has_edge("covariate_01[t-3]", "covariate_02[t-1]"))
        self.assertTrue(graph.has_edge("covariate_02[t-1]", "outcome"))
        self.assertNotIn("covariate_01", graph)
        self.assertNotIn("covariate_02", graph)

    def test_delayed_special_mechanisms_and_sampled_edge_prior(self):
        generator = OperationalSCMGenerator("unused", seed=19)
        lags = []
        for _ in range(50):
            model = generator.sample_model(trend_free=False)
            for node in model["nodes"] + [model["outcome"]]:
                lags.extend(
                    edge["lag"]
                    for edge in node["edges"]
                    if edge["parent"].startswith("covariate_")
                )
                if node.get("event_parent"):
                    lags.append(node["event_lag"])
                if node.get("ringing"):
                    lags.append(node["ringing"]["lag"])
                if node.get("operation"):
                    lags.append(node["operation"]["lag"])
            graph = generator.graph_spec(model)
            for edge in graph["edges"]:
                if edge["label"] == "event":
                    target = (
                        next(n for n in model["nodes"] if n["name"] == edge["target"])
                        if edge["target"] != "outcome"
                        else model["outcome"]
                    )
                    self.assertEqual(edge["lag"], target["event_lag"])
        self.assertTrue(0.06 < np.mean(np.array(lags) == 0) < 0.14)
        values = np.array([1.0, 0.0, 0.0, 0.0])
        np.testing.assert_array_equal(generator._delay_values(values, 2), [0, 0, 1, 0])
        np.testing.assert_array_equal(generator._delay_values(values, 8), [0, 0, 0, 0])

    def test_event_and_ringing_delays_reach_outcome_at_correct_time(self):
        generator = OperationalSCMGenerator(
            "unused",
            min_rows=40,
            max_rows=40,
            min_covariates=3,
            max_covariates=3,
            switching_probability=0,
        )
        model = generator.sample_model()
        for node in model["nodes"][1:] + [model["outcome"]]:
            self._quiet_process(node)
        event = model["nodes"][0]
        event.update(probability=0, forced_event_times=[10])
        outcome = model["outcome"]
        outcome.update(
            scale=1,
            baseline=0,
            ar_weights=[0] * len(model["lags"]),
            event_parent=event["name"],
            event_lag=3,
            shock_size=2,
            shock_decay=0,
        )
        expected = np.zeros(40)
        expected[13] = 2
        np.testing.assert_array_equal(generator.simulate(model)["outcome"], expected)
        outcome["ringing"] = {
            "parent": event["name"],
            "lag": 4,
            "decay": 0.1,
            "period": 20,
            "amplitude": 3,
            "phase": 0,
            "wave": "cosine",
        }
        since_event = np.arange(26)
        expected[14:] += (
            3 * np.exp(-0.1 * since_event) * np.cos(2 * np.pi * since_event / 20)
        )
        observed = generator.simulate(model)
        np.testing.assert_allclose(observed["outcome"], expected, atol=1e-12)
        pd.testing.assert_frame_equal(
            observed, generator.simulate(json.loads(json.dumps(model)))
        )

    def test_brown_noise_sources_replay_and_mixture_clock(self):
        from scipy.signal import welch

        generator = OperationalSCMGenerator("unused")
        spec = generator._sample_noise("red")
        self.assertEqual(spec["kind"], "brown")
        spec.update(location=0, scale=1, initial_value=0, step_scale=0.1)
        draws = generator._noise(np.random.default_rng(1), spec, 65536)
        increments = np.diff(draws)
        self.assertAlmostEqual(np.std(increments), 0.1, delta=0.002)
        self.assertLess(abs(np.corrcoef(increments[:-1], increments[1:])[0, 1]), 0.02)
        frequencies, power = welch(draws, nperseg=2048)
        band = (frequencies > 0.005) & (frequencies < 0.1)
        self.assertAlmostEqual(
            np.polyfit(np.log(frequencies[band]), np.log(power[band]), 1)[0],
            -2,
            delta=0.15,
        )
        short = generator._noise(
            np.random.default_rng(99), json.loads(json.dumps(spec)), 512
        )
        np.testing.assert_array_equal(short, draws[:512])
        mixture = {
            "kind": "heterogeneous_mixture",
            "location": 0,
            "scale": 1,
            "weights": [0.5, 0.5],
            "components": [spec, {"kind": "normal", "location": 10, "scale": 0}],
        }
        labels = np.random.default_rng(12).choice(2, size=512, p=[0.5, 0.5])
        mixed = generator._noise(np.random.default_rng(12), mixture, 512)
        np.testing.assert_array_equal(mixed[labels == 0], short[labels == 0])
        weights = {name: float(name == "brown") for name in generator.NOISES}
        brown = OperationalSCMGenerator(
            "unused",
            noise_weights=weights,
            switching_probability=0,
            edge_probability=0,
            delayed_probability=0,
        )
        model = brown.sample_model()
        self.assertFalse(model["nodes"][1]["edges"])
        self.assertEqual(model["nodes"][1]["noise"]["kind"], "brown")
        pd.testing.assert_frame_equal(
            brown.simulate(model), brown.simulate(json.loads(json.dumps(model)))
        )

    def test_damped_effects_and_causal_event_ringing(self):
        generator = OperationalSCMGenerator("unused")
        for wave in ("sine", "cosine"):
            instruction = {
                "op": "damped_" + wave,
                "a": 0,
                "b": 0,
                "gain": 1,
                "offset": 0,
                "limit": 10,
                "decay": 0.15,
                "frequency": 1.2,
                "phase": 0.4,
            }
            function = np.sin if wave == "sine" else np.cos
            for value in (-10, 0, 2, 10):
                expected = np.exp(-0.15 * abs(value)) * function(1.2 * value + 0.4)
                self.assertAlmostEqual(
                    generator._evaluate([instruction], value),
                    10 * np.tanh(expected / 10),
                )
            events = np.zeros(200)
            events[5] = 1
            spec = {
                "decay": 0.03,
                "period": 20,
                "amplitude": 2,
                "phase": 0.2,
                "wave": wave,
            }
            ring = generator._ringing(spec, events)
            elapsed = np.arange(195)
            np.testing.assert_array_equal(ring[:5], 0)
            np.testing.assert_allclose(
                ring[5:],
                2 * np.exp(-0.03 * elapsed) * function(2 * np.pi * elapsed / 20 + 0.2),
                atol=1e-14,
            )
            events[50] = 1
            combined = generator._ringing(spec, events)
            second = events.copy()
            second[5] = 0
            np.testing.assert_allclose(
                combined, ring + generator._ringing(spec, second), atol=1e-14
            )

    def test_evolving_seasonal_waveforms(self):
        generator = OperationalSCMGenerator("unused")
        season = {
            "period": 20,
            "amplitude": 1,
            "phase": 0,
            "modulation": 0,
            "modulation_period": 100,
            "growth": 0.01,
        }
        wave = generator._seasonal([season], np.array([5, 105]))
        self.assertAlmostEqual(wave[1] / wave[0], np.e)
        alternating = generator._seasonal(
            [{**season, "growth": 0, "alternation": 0.5}], np.array([5, 25])
        )
        self.assertGreater(alternating[0], 2 * alternating[1])
        time = np.arange(200)
        baseline = generator._seasonal([{**season, "growth": 0}], time)
        warped = generator._seasonal(
            [{**season, "growth": 0, "frequency_depth": 0.3, "frequency_period": 100}],
            time,
        )
        self.assertGreater(np.std(baseline - warped), 0.3)
        harmonic = generator._seasonal(
            [
                {
                    **season,
                    "growth": 0,
                    "harmonics": [{"order": 2, "weight": 0.7, "phase": 0.3}],
                }
            ],
            time,
        )
        self.assertGreater(np.std(harmonic - baseline), 0.2)

    def test_shutdown_restart_and_ramp_reset(self):
        generator = OperationalSCMGenerator(
            "unused",
            min_rows=60,
            max_rows=60,
            min_covariates=4,
            max_covariates=4,
            switching_probability=1,
            max_lag=5,
        )
        model = generator.sample_model()
        switch = next(n for n in model["nodes"] if n["kind"] == "switch")
        switch.update(visible=True, on_duration=[6, 6], off_duration=[4, 4])
        for node in [n for n in model["nodes"] if n["kind"] == "process"] + [
            model["outcome"]
        ]:
            self._quiet_process(node)
        outcome = model["outcome"]
        outcome.update(baseline=10, scale=1, ar_weights=[0] * len(model["lags"]))
        operation = {
            "parent": switch["name"],
            "mode": "shutdown",
            "flat_value": 0,
            "restart_rate": 0.25,
            "ramp_rate": 0.5,
        }
        outcome["operation"] = operation
        for mode in ("shutdown", "reset"):
            operation["mode"] = mode
            frame = generator.simulate(model)
            expected, recovery, ramp, previous_on = [], 1.0, 0.0, True
            for i in range(model["burn_in"] + model["rows"]):
                on = i % 10 < 6
                if mode == "shutdown":
                    recovery = min(1, recovery + 0.25) if on else 0
                    value = recovery * 10
                else:
                    if on != previous_on:
                        ramp = 0
                    ramp += 0.5 * (1 if on else -1)
                    value = 10 + ramp
                expected.append(value)
                previous_on = on
            np.testing.assert_allclose(frame["outcome"], expected[model["burn_in"] :])
            pd.testing.assert_frame_equal(
                frame, generator.simulate(json.loads(json.dumps(model)))
            )
            for lag in model["lags"]:
                np.testing.assert_array_equal(
                    frame[f"outcome_lag_{lag}"].to_numpy()[lag:],
                    frame["outcome"].to_numpy()[:-lag],
                )


if __name__ == "__main__":
    unittest.main()

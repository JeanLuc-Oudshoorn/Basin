import copy
import json
from pathlib import Path
import sys
import tempfile
import unittest

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from data_generation.scm_generator import OperationalSCMGenerator


class GeneratorTests(unittest.TestCase):
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
                    np.testing.assert_array_equal(frame[f"outcome_lag_{lag}"].to_numpy()[lag:],
                                                  frame["outcome"].to_numpy()[:-lag])
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
            kwargs = dict(seed=123, min_rows=2056, max_rows=2056,
                          max_lag=1, hidden_probability=1, save_scm=False, save_plot=False, save_dag=False)
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
        spec = {"kind": "student_t_mixture", "location": 0, "scale": 1,
                "weights": [0.5, 0.5], "components": [
                    {"kind": "student_t", "location": center, "scale": 0.6, "df": 7}
                    for center in [-2, 2]]}
        draws = generator._noise(rng, spec, 100000)
        self.assertGreater(np.mean(np.abs(draws - 2) < 0.5), 0.25)
        self.assertGreater(np.mean(np.abs(draws + 2) < 0.5), 0.25)
        self.assertLess(np.mean(np.abs(draws) < 0.5), 0.03)
        spec = generator._sample_noise("quantization")
        errors = generator._noise(rng, spec, 10000)
        self.assertLessEqual(np.max(np.abs(errors - spec["location"])), spec["scale"] * spec["step"] / 2)

    @staticmethod
    def _quiet_process(spec):
        # Remove unrelated mechanisms to make temporal assertions analytical.
        spec.update(edges=[], interaction=0, intercept=0, seasons=[], persistence=0,
                    slope=0, slope_walk=0, walk_scale=0, event_parent=None,
                    level_jump=0, shock_size=0, heteroskedasticity=0, quantization_step=0,
                    noise={"kind": "normal", "location": 0, "scale": 0})

    def test_binary_event_reverses_structural_drift(self):
        generator = OperationalSCMGenerator("unused", min_rows=100, max_rows=100,
                                            min_covariates=3, max_covariates=3, max_lag=1)
        model = generator.sample_model()
        for spec in model["nodes"][1:] + [model["outcome"]]:
            self._quiet_process(spec)
        event = model["nodes"][0]
        event.update(visible=True, probability=0.3)
        outcome = model["outcome"]
        outcome.update(ar_weights=[0], baseline=0, scale=1, slope=0.2,
                       event_parent=event["name"], event_mode="reverse")
        frame = generator.simulate(model)
        differences = np.diff(frame["outcome"])
        np.testing.assert_allclose(np.abs(differences), 0.2, atol=1e-12)
        np.testing.assert_allclose(differences[1:], differences[:-1] *
                                   np.where(frame[event["name"]].to_numpy()[2:] == 1, -1, 1), atol=1e-12)

    def test_delayed_causal_chain(self):
        generator = OperationalSCMGenerator("unused", min_rows=20, max_rows=20,
                                            min_covariates=3, max_covariates=3, max_lag=5)
        model = generator.sample_model()
        for spec in model["nodes"][1:] + [model["outcome"]]:
            self._quiet_process(spec)
        source, mediator = model["nodes"][1:]
        source.update(slope=0.1, visible=True)
        mediator["visible"] = True
        edge = {"parent": source["name"], "lag": 3, "weight": 0.7,
                "input_scale": 20, "input_offset": 0, "output_offset": 0,
                "program": [{"op": "affine", "a": 0, "b": 0, "gain": 1, "offset": 0, "limit": 10}]}
        mediator["edges"] = [edge]
        model["outcome"].update(edges=[{**edge, "parent": mediator["name"], "lag": 2}],
                                ar_weights=[0] * len(model["lags"]), scale=1, baseline=0)
        frame = generator.simulate(model)
        for i in range(5, len(frame)):
            self.assertAlmostEqual(frame[mediator["name"]][i], generator._effect(edge, frame[source["name"]][i - 3]))
            self.assertAlmostEqual(frame["outcome"][i], generator._effect(edge, frame[mediator["name"]][i - 2]))

    def test_expression_operators_and_graph_topology(self):
        generator = OperationalSCMGenerator("unused", min_covariates=8, max_covariates=8)
        operators, topologies, delayed, indirect = set(), set(), False, False
        for _ in range(50):
            model = generator.sample_model()
            topologies.add(model["topology"])
            graph = generator.graph_spec(model)
            pending = set(graph["vertices"])
            visited = set()
            while pending:
                ready = {n for n in pending if all(e["source"] in visited for e in graph["edges"] if e["target"] == n)}
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
        self.assertEqual(topologies, {"sparse", "chain", "diamond", "dense"})
        self.assertTrue(delayed and indirect)

    def test_legacy_replay(self):
        from data_generation.legacy_scm_generator import OperationalSCMGenerator as Legacy
        legacy = Legacy("unused", min_rows=20, max_rows=20)
        model = legacy.sample_model()
        pd.testing.assert_frame_equal(legacy.simulate(model), OperationalSCMGenerator("unused").simulate(model))

    def test_perlin_smoothness_replay_and_mixture_time_axis(self):
        generator = OperationalSCMGenerator("unused")
        spec = generator._sample_noise("perlin")
        spec.update(frequency=0.02, octaves=1, location=0, scale=1)
        short = generator._noise(np.random.default_rng(1), spec, 512)
        long = generator._noise(np.random.default_rng(99), json.loads(json.dumps(spec)), 1024)
        np.testing.assert_array_equal(short, long[:512])
        self.assertGreater(np.std(short), 0.02)
        self.assertGreater(np.corrcoef(short[:-1], short[1:])[0, 1], 0.98)
        changed = {**spec, "seed": spec["seed"] + 1}
        self.assertFalse(np.array_equal(short, generator._noise(np.random.default_rng(1), changed, 512)))
        # Selecting samples from a mixture must not compress the Perlin clock.
        mixture = {"kind": "heterogeneous_mixture", "location": 0, "scale": 1,
                   "weights": [0.5, 0.5], "components": [spec,
                       {"kind": "normal", "location": 10, "scale": 0}]}
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
            self.assertGreater(counts[preferred], 2 * max(count for family, count in counts.items()
                               if family not in ("perlin", "student_t", "pink")))
        # Force Perlin everywhere to exercise outcome noise, exogenous starting
        # processes, hidden covariates and JSON replay through the simulator.
        weights = {name: float(name == "perlin") for name in generator.NOISES}
        perlin = OperationalSCMGenerator("unused", noise_weights=weights,
                    min_rows=100, max_rows=100, edge_probability=0, delayed_probability=0)
        model = perlin.sample_model()
        for process in model["nodes"][1:] + [model["outcome"]]:
            self.assertEqual(process["noise"]["kind"], "perlin")
        self.assertFalse(model["nodes"][1]["edges"])
        frame = perlin.simulate(model)
        pd.testing.assert_frame_equal(frame, perlin.simulate(json.loads(json.dumps(model))))
        for lag in model["lags"]:
            if lag < len(frame):
                np.testing.assert_array_equal(frame[f"outcome_lag_{lag}"].to_numpy()[lag:], frame["outcome"].to_numpy()[:-lag])

    def test_invalid_noise_weights_and_perlin_priors(self):
        for weights in ({"bad": 1}, {"perlin": -1}, {"perlin": float("nan")},
                        {family: 0 for family in OperationalSCMGenerator.NOISES}):
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
        short = generator._noise(np.random.default_rng(99), json.loads(json.dumps(spec)), 512)
        np.testing.assert_allclose(short, draws[:512], atol=1e-14)
        self.assertEqual(generator._noise(np.random.default_rng(1), spec, 0).size, 0)
        mixture = {"kind": "heterogeneous_mixture", "location": 0, "scale": 1,
                   "weights": [0.5, 0.5], "components": [spec,
                       {"kind": "normal", "location": 10, "scale": 0}]}
        labels = np.random.default_rng(12).choice(2, size=512, p=[0.5, 0.5])
        mixed = generator._noise(np.random.default_rng(12), mixture, 512)
        np.testing.assert_array_equal(mixed[labels == 0], short[labels == 0])
        weights = {name: float(name == "pink") for name in generator.NOISES}
        pink = OperationalSCMGenerator("unused", noise_weights=weights, min_rows=30, max_rows=30)
        model = pink.sample_model()
        self.assertTrue(all(n["noise"]["kind"] == "pink" for n in model["nodes"][1:]))
        pd.testing.assert_frame_equal(pink.simulate(model), pink.simulate(json.loads(json.dumps(model))))

    def test_inverse_square_effect_and_shift(self):
        operator = {"op": "inverse_square", "a": 0, "b": 0,
                    "gain": 1, "offset": 0, "limit": 10, "floor": 0.05}
        evaluate = OperationalSCMGenerator._evaluate
        for value in (-4, -2, 2, 4):
            self.assertAlmostEqual(evaluate([operator], value), 10 * np.tanh((1 / value**2) / 10))
        self.assertGreater(evaluate([operator], 2), evaluate([operator], 4))
        self.assertTrue(np.isfinite(evaluate([operator], 0)))
        self.assertAlmostEqual(evaluate([{**operator, "offset": -2}], 4), evaluate([operator], 2))


if __name__ == "__main__":
    unittest.main()

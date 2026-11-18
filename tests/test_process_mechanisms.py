"""Recorded process dynamics, discrete sources and measurement-limit causality."""

import json
import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src/01_data_generation"))
from process_mechanisms import discrete_path, episode_envelope
from scm_generator import OperationalSCMGenerator


class ProcessTests(unittest.TestCase):
    def model(self):
        generator = OperationalSCMGenerator(
            "unused",
            seed=8,
            min_rows=40,
            max_rows=40,
            min_covariates=4,
            max_covariates=4,
            process_episode_probability=1,
            discrete_covariate_probability=0,
            formula_covariate_probability=0,
            sensor_bounds_probability=0,
        )
        model = generator.sample_model(trend_free=True)
        # These analytical fixtures exercise the legacy binary-response equation.
        # New sampled formula drivers have their own integration tests.
        model["nodes"][1] = {
            "name": "covariate_01",
            "kind": "switch",
            "edges": [],
            "visible": True,
            "profile": "single_episode",
            "start": 5,
            "stop": 20,
            "initial_on": False,
        }
        model.update(lags=[1], burn_in=5)
        for node in [n for n in model["nodes"] if n["kind"] == "process"] + [
            model["outcome"]
        ]:
            node.update(
                edges=[],
                interaction=0,
                intercept=0,
                seasons=[],
                persistence=0,
                slope=0,
                slope_walk=0,
                walk_scale=0,
                event_parent=None,
                shock_size=0,
                level_jump=0,
                heteroskedasticity=0,
                quantization_step=0,
                white_noise_scale=0,
                noise={"kind": "normal", "location": 0, "scale": 0},
                ar_weights=[0],
                outcome_ar_weights=[],
                visible=True,
            )
            for key in (
                "node_link",
                "parent_combination",
                "operation",
                "ringing",
                "volatility",
                "multiplicative_scaling",
                "trend",
                "sensor",
                "domain",
            ):
                node.pop(key, None)
        model["outcome"].update(baseline=50, scale=5)
        return generator, model

    @staticmethod
    def edge(parent, lag=0):
        return {
            "parent": parent,
            "lag": lag,
            "weight": 1.0,
            "input_scale": 1.0,
            "input_offset": 0.0,
            "output_offset": 0.0,
            "program": [
                {
                    "op": "affine",
                    "a": 0,
                    "b": 0,
                    "gain": 1.0,
                    "offset": 0.0,
                    "limit": 100.0,
                }
            ],
        }

    def test_envelope_rises_and_releases_causally(self):
        operation = {"rise_steps": 4, "fall_steps": 4, "ramp_shape": "smoothstep"}
        command = np.array([0] * 5 + [1] * 10 + [0] * 10)
        envelope = episode_envelope(operation, command)
        np.testing.assert_array_equal(envelope[:5], 0)
        self.assertTrue(np.all(np.diff(envelope[4:9]) > 0))
        self.assertTrue(np.all(np.diff(envelope[14:19]) < 0))
        np.testing.assert_array_equal(envelope[19:], 0)
        modified = command.copy()
        modified[15:] = 1
        np.testing.assert_array_equal(
            envelope[:15], episode_envelope(operation, modified)[:15]
        )

    def test_episode_has_idle_ends_and_a_measured_causal_history(self):
        generator, model = self.model()
        command = next(n for n in model["nodes"] if n["kind"] == "switch")
        command.update(start=5, stop=20, visible=True)
        operation = {
            "schema_version": 2,
            "mode": "episode",
            "parent": command["name"],
            "lag": 2,
            "idle_value": 0.0,
            "amplitude": 10.0,
            "rise_steps": 4,
            "fall_steps": 4,
            "ramp_shape": "smoothstep",
            "detail_fraction": 0.0,
            "detail_scale": 1.0,
            "idle_noise_scale": 0.0,
        }
        model["outcome"]["operation"] = operation
        predictor = next(n for n in model["nodes"] if n["kind"] == "process")
        edge = self.edge("outcome_lag_1")
        predictor["edges"] = [edge]
        frame = generator.simulate(model)
        np.testing.assert_array_equal(frame.outcome[:7], 0.0)
        np.testing.assert_array_equal(frame.outcome[26:], 0.0)
        self.assertEqual(frame.outcome.max(), 10.0)
        self.assertGreater(frame.outcome[8], frame.outcome[7])
        self.assertLess(frame.outcome[23], frame.outcome[22])
        np.testing.assert_allclose(
            frame[predictor["name"]][1:],
            [generator._effect(edge, (v - 50) / 5) for v in frame.outcome[:-1]],
            atol=1e-13,
        )
        model["outcome"]["sensor"] = {"floor": 0.0, "ceiling": 6.0}
        capped = generator.simulate(model)
        self.assertEqual(capped.outcome.max(), 6.0)
        np.testing.assert_allclose(
            capped[predictor["name"]][1:],
            [generator._effect(edge, (v - 50) / 5) for v in capped.outcome[:-1]],
            atol=1e-13,
        )
        pd.testing.assert_frame_equal(
            capped, generator.simulate(json.loads(json.dumps(model))), check_exact=True
        )
        self.assertGreater(generator.trend_strength(frame, model), 0.999)

    def test_episode_preserves_active_oscillations_and_idle_ends(self):
        generator, model = self.model()
        command = next(n for n in model["nodes"] if n["kind"] == "switch")
        command.update(start=5, stop=30)
        operation = {
            "schema_version": 2,
            "mode": "episode",
            "parent": command["name"],
            "lag": 1,
            "idle_value": 0.0,
            "amplitude": 10.0,
            "rise_steps": 3,
            "fall_steps": 3,
            "ramp_shape": "linear",
            "detail_fraction": 0.0,
            "detail_scale": 1.0,
            "idle_noise_scale": 0.0,
            "noise_fraction": 0.0,
            "seasonal_fraction": 0.3,
        }
        model["outcome"].update(
            operation=operation,
            seasons=[
                {
                    "period": 4.0,
                    "phase": 0.0,
                    "amplitude": 0.6,
                    "modulation": 0.0,
                    "modulation_period": 40.0,
                }
            ],
        )
        frame = generator.simulate(model)
        time = frame.time_index.to_numpy()
        envelope = episode_envelope(operation, (time >= 6) & (time < 31))
        expected = (
            10 * envelope * (1 + 0.3 * np.tanh(0.6 * np.sin(2 * np.pi * time / 4)))
        )
        np.testing.assert_allclose(frame.outcome, expected, atol=1e-12)
        self.assertGreater(frame.outcome[10:29].std(), 1.0)
        np.testing.assert_array_equal(frame.outcome[:6], 0.0)
        np.testing.assert_array_equal(frame.outcome[34:], 0.0)

    def test_univariate_hides_sources_without_changing_the_process(self):
        settings = dict(seed=17, min_rows=64, max_rows=64)
        visible_generator = OperationalSCMGenerator("unused", **settings)
        hidden_generator = OperationalSCMGenerator("unused", **settings)
        visible_model = visible_generator.sample_model(visible_covariate_count=100)
        hidden_model = hidden_generator.sample_model(visible_covariate_count=0)
        visible = visible_generator.simulate(visible_model)
        hidden = hidden_generator.simulate(hidden_model)
        self.assertTrue(visible_model["nodes"])
        self.assertTrue(all(n["visible"] for n in visible_model["nodes"]))
        self.assertTrue(hidden_model["univariate"])
        self.assertTrue(all(not n["visible"] for n in hidden_model["nodes"]))
        self.assertEqual(list(hidden), ["time_index", "outcome"])
        np.testing.assert_array_equal(visible.outcome, hidden.outcome)
        pd.testing.assert_frame_equal(
            hidden, hidden_generator.simulate(json.loads(json.dumps(hidden_model)))
        )

    def test_poisson_visibility_distribution_capping_and_uniform_subsets(self):
        generator = OperationalSCMGenerator("unused", seed=117)
        nodes = [{"name": str(i)} for i in range(8)]
        counts, selections = [], np.zeros(8)
        for _ in range(20000):
            sampled = generator._sample_visibility(nodes)
            count = sum(n["visible"] for n in nodes)
            self.assertEqual(count, min(sampled, len(nodes)))
            counts.append(count)
            selections += [n["visible"] for n in nodes]
        counts = np.asarray(counts)
        for count in range(5):
            expected = (
                np.exp(-1.3) * 1.3**count / float(np.prod(np.arange(1, count + 1)))
            )
            self.assertAlmostEqual(np.mean(counts == count), expected, delta=0.01)
        self.assertLess(selections.max() / selections.min(), 1.12)
        self.assertEqual(generator._sample_visibility(nodes, 100), 100)
        self.assertTrue(all(n["visible"] for n in nodes))
        self.assertEqual(generator._sample_visibility(nodes, 0), 0)
        self.assertFalse(any(n["visible"] for n in nodes))

    def test_observation_and_rare_mechanism_defaults(self):
        generator = OperationalSCMGenerator("unused")
        self.assertEqual(generator.process_episode_probability, 0.02)
        self.assertEqual(generator.sensor_bounds_probability, 0.01)
        self.assertEqual(generator.visible_covariate_lambda, 1.3)
        for value in (-0.1, float("inf"), float("nan"), True):
            with self.assertRaises(ValueError):
                OperationalSCMGenerator("unused", visible_covariate_lambda=value)
        for count in (-1, 1.3, True):
            with self.assertRaises(ValueError):
                generator.sample_model(visible_covariate_count=count)

    def test_sensor_caps_covariate_before_downstream_equations(self):
        generator, model = self.model()
        source, mediator = [n for n in model["nodes"] if n["kind"] == "process"]
        source.update(intercept=3.0, sensor={"floor": 0.0, "ceiling": 1.0})
        mediator["edges"] = [self.edge(source["name"], 1)]
        model["outcome"].update(
            baseline=0, scale=1, edges=[self.edge(mediator["name"])]
        )
        frame = generator.simulate(model)
        np.testing.assert_array_equal(frame[source["name"]], 1.0)
        expected = generator._effect(mediator["edges"][0], 1.0)
        np.testing.assert_array_equal(frame[mediator["name"]], expected)
        np.testing.assert_array_equal(
            frame.outcome, generator._effect(model["outcome"]["edges"][0], expected)
        )
        source["sensor"] = {"floor": 4.0, "ceiling": None}
        np.testing.assert_array_equal(generator.simulate(model)[source["name"]], 4.0)

    def test_markov_levels_persistence_and_replay(self):
        alternating = {
            "kind": "markov",
            "values": [0, 1],
            "transition": [[0, 1], [1, 0]],
            "initial_state": 0,
        }
        np.testing.assert_array_equal(
            discrete_path(alternating, np.random.default_rng(1), 8),
            [0, 1, 0, 1, 0, 1, 0, 1],
        )
        sticky = {**alternating, "transition": [[0.99, 0.01], [0.02, 0.98]]}
        path = discrete_path(sticky, np.random.default_rng(2), 10000)
        self.assertLess(np.mean(np.diff(path) != 0), 0.03)
        self.assertEqual(set(path), {0, 1})
        multi = {
            "kind": "markov",
            "values": [-2.0, 3.0, 9.0],
            "transition": [[0.9, 0.05, 0.05], [0.1, 0.8, 0.1], [0.05, 0.05, 0.9]],
            "initial_state": 1,
        }
        values = discrete_path(multi, np.random.default_rng(9), 1000)
        self.assertEqual(set(values), {-2, 3, 9})
        np.testing.assert_array_equal(
            values,
            discrete_path(
                json.loads(json.dumps(multi)), np.random.default_rng(9), 1000
            ),
        )
        bernoulli = discrete_path(
            {"kind": "bernoulli", "probability": 0.5}, np.random.default_rng(2), 10000
        )
        self.assertGreater(np.mean(np.diff(bernoulli) != 0), 0.45)

    def test_markov_states_follow_the_cumulative_transition_search(self):
        # The definition: one uniform draw per row, searched to the right in the
        # current state's cumulative transition row. Same draws, same states.
        rng = np.random.default_rng(11)
        for seed in range(40):
            count = int(rng.integers(2, 8))
            spec = {
                "kind": "markov",
                "values": rng.normal(size=count).tolist(),
                "transition": rng.dirichlet(np.full(count, 0.3), count).tolist(),
                "initial_state": int(rng.integers(count)),
            }
            cumulative = np.asarray(spec["transition"]).cumsum(axis=1)
            cumulative[:, -1] = 1.0
            state, expected = spec["initial_state"], [spec["values"][0]] * 500
            expected[0] = spec["values"][state]
            draws = np.random.default_rng(seed).random(499)
            for index, draw in enumerate(draws, 1):
                state = int(np.searchsorted(cumulative[state], draw, side="right"))
                expected[index] = spec["values"][state]
            path = discrete_path(spec, np.random.default_rng(seed), 500)
            self.assertEqual(path.tobytes(), np.asarray(expected).tobytes())

    def test_sampled_discrete_sources_and_hidden_causality(self):
        generator = OperationalSCMGenerator(
            "unused",
            seed=121,
            min_rows=100,
            max_rows=100,
            min_covariates=8,
            max_covariates=8,
            discrete_covariate_probability=1,
            process_episode_probability=0,
            switching_probability=0,
            sensor_bounds_probability=0,
        )
        families = set()
        for _ in range(12):
            model = generator.sample_model(trend_free=True)
            frame = generator.simulate(model)
            for node in model["nodes"]:
                if node["kind"] in ("bernoulli", "markov"):
                    families.add(node["family"])
                    node["visible"] = True
                    if node["kind"] == "markov":
                        np.testing.assert_allclose(
                            np.sum(node["transition"], axis=1), 1
                        )
            visible = generator.simulate(model)
            np.testing.assert_array_equal(frame.outcome, visible.outcome)
            for node in model["nodes"]:
                if node["kind"] in ("bernoulli", "markov"):
                    self.assertTrue(
                        set(visible[node["name"]]).issubset(node.get("values", [0, 1]))
                    )
            pd.testing.assert_frame_equal(
                visible, generator.simulate(json.loads(json.dumps(model)))
            )
        self.assertEqual(families, {"bernoulli", "binary_markov", "categorical_markov"})

    def test_long_dwell_and_single_episode_batches(self):
        generator = OperationalSCMGenerator("unused", seed=3)
        switch = generator._sample_switch(1000, force_episode=True)
        self.assertLess(switch["start"], switch["stop"])
        for _ in range(20):
            switch = generator._sample_switch(1000)
            if switch["profile"] == "long_dwell":
                self.assertGreater(switch["off_duration"][1], 100)
                break
        else:
            self.fail("Expected a long-dwell profile")
        with tempfile.TemporaryDirectory() as directory:
            generator = OperationalSCMGenerator(
                directory,
                seed=31,
                min_rows=100,
                max_rows=100,
                process_episode_probability=1,
                save_plot=False,
                save_dag=False,
            )
            for path in generator.generate(4):
                model = json.loads((path / "scm.json").read_text())
                frame = pd.read_parquet(path / "data.parquet")
                self.assertTrue(model["process_episode"])
                pd.testing.assert_frame_equal(frame, generator.simulate(model))


if __name__ == "__main__":
    unittest.main()

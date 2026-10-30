"""Shape, causal-memory and joint-parent invariants for the SCM effect prior."""

import copy
import json
import sys
import unittest
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(
    0, str(Path(__file__).resolve().parents[1] / "src" / "01_data_generation")
)
from effect_mechanisms import (
    TEMPORAL_OPERATORS,
    apply_node_link,
    combine_parents,
    evaluate_extra,
)
from scm_generator import OperationalSCMGenerator


class EffectTests(unittest.TestCase):
    def joint_generator(self, mode, **kwargs):
        return OperationalSCMGenerator(
            "unused",
            seed=51,
            parent_combination_weights={
                name: float(name == mode)
                for name in OperationalSCMGenerator.DEFAULT_PARENT_COMBINATION_WEIGHTS
            },
            **kwargs,
        )

    def test_positive_products_can_dominate_and_saddles_are_rare(self):
        generator = self.joint_generator("product")
        spec = generator._sample_parent_combination(2)
        spec.update(
            indices=[0, 1],
            positive_map="softplus",
            scale=1.0,
            strength=1.0,
            application="replace",
        )
        self.assertGreater(combine_parents([4, 4], spec), 8.0)
        self.assertGreater(combine_parents([-4, -4], spec), 0.0)
        for fixed in (-3.0, 0.0, 3.0):
            curve = [combine_parents([x, fixed], spec) for x in (-4.0, -1.0, 1.0, 4.0)]
            self.assertTrue(np.all(np.diff(curve) > 0))
            curve = [combine_parents([fixed, x], spec) for x in (-4.0, -1.0, 1.0, 4.0)]
            self.assertTrue(np.all(np.diff(curve) > 0))
        spec["application"] = "additive"
        self.assertGreater(combine_parents([4, 4], spec), 16.0)
        self.assertLessEqual(combine_parents([1e12, 1e12], spec), spec["numeric_limit"])
        self.assertGreater(
            generator.DEFAULT_PARENT_COMBINATION_WEIGHTS["product"],
            10 * generator.DEFAULT_PARENT_COMBINATION_WEIGHTS["saddle"],
        )

    def test_random_subsets_pair_prior_and_all_new_maps_replay(self):
        generator = self.joint_generator("gate")
        specs = [generator._sample_parent_combination(6) for _ in range(500)]
        self.assertTrue(0.80 < np.mean([len(s["indices"]) == 2 for s in specs]) < 0.90)
        self.assertGreater(len({tuple(s["indices"]) for s in specs}), 20)
        for mode in generator.DEFAULT_PARENT_COMBINATION_WEIGHTS:
            gen = self.joint_generator(mode)
            spec = gen._sample_parent_combination(6)
            actual = combine_parents([-3, -1, 0, 2, 4, 6], spec)
            self.assertTrue(np.isfinite(actual), mode)
            self.assertEqual(
                actual,
                combine_parents([-3, -1, 0, 2, 4, 6], json.loads(json.dumps(spec))),
            )
            self.assertEqual(len(set(spec["indices"])), len(spec["indices"]))
            if mode == "mlp":
                self.assertGreaterEqual(len(spec["indices"]), 3)
                self.assertEqual(np.asarray(spec["w"]).shape[1], len(spec["indices"]))
                zero = np.zeros(6)
                base = combine_parents(zero, spec)
                for index in spec["indices"]:
                    perturbed = zero.copy()
                    perturbed[index] = 0.3
                    self.assertNotAlmostEqual(
                        base, combine_parents(perturbed, spec), places=9
                    )

    def test_threshold_and_convex_mix_preserve_their_meanings(self):
        generator = self.joint_generator("threshold")
        spec = generator._sample_parent_combination(4)
        spec.update(indices=[1, 3], scale=1.0, strength=1.0, threshold=0.0)
        self.assertEqual(combine_parents([2, 5, 3, -1], spec), 5.0)
        self.assertEqual(combine_parents([2, 5, 3, 1], spec), 10.0)
        spec = self.joint_generator("convex_mix")._sample_parent_combination(4)
        spec.update(indices=[1, 3], weights=[0.2, 0.8])
        self.assertAlmostEqual(
            combine_parents([2, 5, 3, 10], spec), 2 + 3 + 0.2 * 5 + 0.8 * 10
        )
        self.assertAlmostEqual(sum(spec["weights"]), 1.0)

    def test_power_law_and_node_links(self):
        generator = self.joint_generator("log_sum")
        spec = generator._sample_parent_combination(2)
        spec.update(
            indices=[0, 1], powers=[1.0, 0.5], scale=1.0, positive_map="softplus"
        )
        link = generator._sample_node_link(force_exponential=True)
        link["gain"] = 2.0
        effects = [2.0, 3.0]
        positive = np.logaddexp(0.0, effects) + spec["floor"]
        result = apply_node_link(combine_parents(effects, spec), link)
        self.assertAlmostEqual(result, 2 * positive[0] * np.sqrt(positive[1]))
        for mode in generator.DEFAULT_NODE_LINK_WEIGHTS:
            link["mode"] = mode
            self.assertTrue(np.isfinite(apply_node_link(3.0, link)))
        with self.assertRaises(ValueError):
            self.joint_generator(
                "log_sum",
                node_link_weights={
                    name: float(name == "identity")
                    for name in generator.DEFAULT_NODE_LINK_WEIGHTS
                },
            )._sample_parent_combination(3)

    def generator_for(self, operator):
        return OperationalSCMGenerator(
            "unused",
            seed=17,
            composite_effect_probability=0,
            effect_weights={
                name: float(name == operator)
                for name in OperationalSCMGenerator.OPERATORS
            },
        )

    def test_all_operators_finite_serializable_and_state_isolated(self):
        for op in OperationalSCMGenerator.OPERATORS:
            with self.subTest(op=op):
                generator = self.generator_for(op)
                program = generator._expression()
                self.assertEqual(len(program), 1)
                self.assertEqual(program[0]["op"], op)
                persisted = json.dumps(program)
                state, other = {}, {}
                for value in [-1000, -3, 0, 1, 4, 1000] * 3:
                    actual = generator._evaluate(program, value, state)
                    self.assertTrue(np.isfinite(actual))
                    self.assertEqual(
                        actual, generator._evaluate(json.loads(persisted), value, other)
                    )
                self.assertEqual(json.dumps(program), persisted)

    def test_weight_priors_and_composite_frequency(self):
        generator = OperationalSCMGenerator("unused", seed=73)
        programs = [generator._expression() for _ in range(2000)]
        self.assertTrue(0.22 < np.mean([len(p) > 1 for p in programs]) < 0.28)
        weights = generator.DEFAULT_EFFECT_WEIGHTS
        self.assertGreater(weights["affine"], weights["hill"])
        self.assertGreater(weights["hill"], weights["tangent"])
        for keyword in ("effect_weights", "parent_combination_weights"):
            for bad in (
                {"unknown": 1},
                {"sum" if keyword.startswith("parent") else "hill": -1},
            ):
                with self.assertRaises(ValueError):
                    OperationalSCMGenerator("unused", **{keyword: bad})
        generator = OperationalSCMGenerator(
            "unused", max_effect_components=1, composite_effect_probability=1
        )
        self.assertEqual(len(generator._expression()), 1)

    def parameters(self, op):
        p = self.generator_for(op)._expression()[0]
        p.update(
            gain=1.0,
            offset=0.0,
            limit=100.0,
            k=1.0,
            width=1.0,
            threshold=0.5,
            center=0.0,
            shape=2.0,
            level=1.0,
            leak=0.0,
            tau=4.0,
            floor=0.05,
            rise_rate=0.2,
            fall_rate=0.1,
            steps=3,
        )
        return p

    def test_static_physical_shapes(self):
        x = np.linspace(-5, 5, 101)
        for op in (
            "hill",
            "michaelis_menten",
            "langmuir",
            "arrhenius",
            "antoine",
            "queue",
            "stefan_boltzmann",
        ):
            p = self.parameters(op)
            values = np.array([evaluate_extra(op, z, 1, p, {}) for z in x])
            self.assertTrue(np.all(np.diff(values) >= 0), op)
        p = self.parameters("gaussian_optimum")
        self.assertGreater(
            evaluate_extra("gaussian_optimum", 0, 0, p, {}),
            evaluate_extra("gaussian_optimum", 2, 0, p, {}),
        )
        p = self.parameters("deadband")
        self.assertEqual(evaluate_extra("deadband", 0.3, 0, p, {}), 0)
        self.assertAlmostEqual(evaluate_extra("deadband", 1, 0, p, {}), 0.5)

    def test_temporal_responses(self):
        p = self.parameters("delay")
        state = {}
        result = [evaluate_extra("delay", z, 0, p, state) for z in [1, 0, 0, 0, 0]]
        self.assertEqual(result, [0, 0, 0, 1, 0])
        state = {}
        response = [
            evaluate_extra("first_order_lag", 1, 0, p, state) for _ in range(40)
        ]
        self.assertTrue(np.all(np.diff(response) > 0))
        self.assertAlmostEqual(response[-1], 1.0, places=4)
        state = {}
        self.assertEqual(
            [evaluate_extra("hysteresis", z, 0, p, state) for z in [1, 0, -1, 0]],
            [1, 1, 0, 0],
        )
        state = {}
        values = [evaluate_extra("rate_limiter", z, 0, p, state) for z in [10, 10, -10]]
        np.testing.assert_allclose(values, [0.2, 0.4, 0.3])
        for op in TEMPORAL_OPERATORS:
            p, state = self.parameters(op), {}
            self.assertTrue(
                all(
                    np.isfinite(evaluate_extra(op, z, 0, p, state))
                    for z in [0, 1, 0, -1] * 30
                )
            )

    def test_joint_parent_maps_use_distinct_inputs(self):
        spec = {
            "mode": "gate",
            "scale": 1.0,
            "limit": 100.0,
            "floor": 0.05,
            "threshold": 0.0,
            "low_gain": 0.1,
            "high_gain": 2.0,
        }
        self.assertGreater(
            combine_parents([2, 10], spec), 100 * combine_parents([2, -10], spec)
        )
        # These specs deliberately use the original schema; preserve its signed
        # product and exact saturation arithmetic for already generated tables.
        for mode in (
            "sum",
            "product",
            "ratio",
            "min",
            "max",
            "soft_min",
            "gate",
            "regime_switch",
            "volatility_coupling",
            "swiglu",
        ):
            spec["mode"] = mode
            self.assertTrue(np.isfinite(combine_parents([2.0, 3.0, -1.0], spec)))
        spec["mode"] = "min"
        self.assertAlmostEqual(combine_parents([2, 3], spec), 100 * np.tanh(2 / 100))
        spec["mode"] = "product"
        self.assertGreater(combine_parents([2, 3], spec), 0)
        self.assertLess(combine_parents([2, -3], spec), 0)

    def test_joint_combiner_and_stateful_edge_are_used_by_simulator(self):
        generator = self.generator_for("first_order_lag")
        model = generator.sample_model(trend_free=True)
        # A single deterministic event source makes the causal arrival time
        # observable independently of seasonality, trend and innovation draws.
        event = model["nodes"][0]
        event.update(probability=0, forced_event_times=[10])
        outcome = model["outcome"]
        outcome.update(
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
            ar_weights=[0] * len(model["lags"]),
            outcome_ar_weights=[],
            baseline=0,
            scale=1,
        )
        for field in (
            "operation",
            "ringing",
            "volatility",
            "multiplicative_scaling",
            "trend",
            "parent_combination",
            "node_link",
        ):
            outcome.pop(field, None)
        p = self.parameters("first_order_lag")
        edge = {
            "parent": event["name"],
            "lag": 3,
            "weight": 1,
            "input_scale": 1,
            "input_offset": 0,
            "output_offset": 0,
            "program": [p],
        }
        outcome["edges"] = [edge]
        with_event = generator.simulate(model)
        without = copy.deepcopy(model)
        without["nodes"][0]["forced_event_times"] = []
        baseline = generator.simulate(without)
        np.testing.assert_array_equal(
            with_event["outcome"][:13], baseline["outcome"][:13]
        )
        self.assertGreater(with_event["outcome"][13], 0)
        self.assertGreater(with_event["outcome"][13], with_event["outcome"][14])
        # A second, distinct parent input gates the first in the actual simulator.
        outcome["edges"].append({**edge, "lag": 1, "output_offset": -5})
        spec = {
            "mode": "gate",
            "scale": 1,
            "limit": 20,
            "floor": 0.1,
            "threshold": 0,
            "low_gain": 0.1,
            "high_gain": 2,
        }
        outcome["parent_combination"] = spec
        gated = generator.simulate(model)
        outcome["parent_combination"] = {**spec, "mode": "sum"}
        additive = generator.simulate(model)
        self.assertGreater(np.max(abs(gated["outcome"] - additive["outcome"])), 1)
        outcome["node_link"] = {
            "mode": "saturation",
            "scale": 1.0,
            "input_offset": 0.0,
            "output_offset": 0.0,
            "gain": 1.0,
            "capacity": 2.0,
            "numeric_limit": 1e6,
        }
        linked = generator.simulate(model)
        np.testing.assert_allclose(
            linked["outcome"], 2 * np.tanh(additive["outcome"] / 2)
        )

    def test_simulation_causal_prefix_and_replay_with_temporal_edges(self):
        generator = self.generator_for("first_order_lag")
        model = generator.sample_model()
        original = json.dumps(model)
        frame = generator.simulate(model)
        pd.testing.assert_frame_equal(frame, generator.simulate(json.loads(original)))
        # Noise draw ordering across arrays is horizon-dependent, so inspect
        # the evaluator's causal prefix independently of RNG consumption.
        p = self.parameters("erlang")
        state, prefix_state = {}, {}
        inputs = np.sin(np.arange(100) / 10)
        whole = [evaluate_extra("erlang", z, 0, p, state) for z in inputs]
        prefix = [evaluate_extra("erlang", z, 0, p, prefix_state) for z in inputs[:50]]
        np.testing.assert_array_equal(whole[:50], prefix)
        self.assertEqual(json.dumps(model), original)

    def test_trig_requires_angle_context_in_sampled_edges(self):
        generator = self.generator_for("sine")
        self.assertEqual(generator._expression("angle")[0]["op"], "sine")
        with self.assertRaises(ValueError):
            generator._expression("real")
        generator = OperationalSCMGenerator(
            "unused", seed=77, angle_covariate_probability=0
        )
        model = generator.sample_model()
        for node in model["nodes"] + [model["outcome"]]:
            for edge in node["edges"]:
                self.assertTrue(
                    all(
                        p["op"] not in ("sine", "cosine", "tangent")
                        for p in edge["program"]
                    )
                )


if __name__ == "__main__":
    unittest.main()

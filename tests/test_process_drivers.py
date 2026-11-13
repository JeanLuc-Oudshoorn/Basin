"""Clock-driven exogenous sources and asymmetric cyber-physical machine programs."""

import copy
import json
import sys
import unittest
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src/01_data_generation"))
from process_mechanisms import formula_path, scheduled_path
from scm_generator import OperationalSCMGenerator


class ProcessDriverTests(unittest.TestCase):
    def test_setpoint_response_has_no_anticipation_and_handles_interruptions(self):
        spec = {
            "initial_level": 0,
            "transitions": [
                {"start": 5, "target": 1, "steps": 4, "shape": "linear"},
                {"start": 7, "target": 2, "steps": 2, "shape": "linear"},
                {"start": 10, "target": 0, "steps": 4, "shape": "linear"},
            ],
        }
        path = scheduled_path(spec, np.arange(20))
        np.testing.assert_array_equal(path[:5], 0)
        np.testing.assert_allclose(path[5:10], [0.25, 0.5, 1.25, 2, 2])
        np.testing.assert_allclose(path[10:14], [1.5, 1, 0.5, 0])
        changed = copy.deepcopy(spec)
        changed["transitions"][2]["target"] = 100
        np.testing.assert_array_equal(path[:10], scheduled_path(changed, np.arange(10)))
        np.testing.assert_array_equal(
            path, scheduled_path(json.loads(json.dumps(spec)), np.arange(20))
        )

    def test_sampled_program_diversity_and_idle_endpoints(self):
        g = OperationalSCMGenerator("unused", seed=27)
        fast_up = fast_down = uneven_idle = multilevel = priming = near_edge = 0
        shapes = set()
        for _ in range(300):
            driver = g._sample_process_driver("covariate_01", 512)
            first, last = driver["transitions"][0], driver["transitions"][-1]
            fast_up += last["steps"] > 4 * first["steps"]
            fast_down += first["steps"] > 4 * last["steps"]
            uneven_idle += max(driver["start"], driver["idle_after"]) > 4 * max(
                1, min(driver["start"], driver["idle_after"])
            )
            multilevel += len(driver["transitions"]) > 2
            priming += bool(driver["priming"])
            near_edge += min(driver["start"], driver["idle_after"]) < 5
            shapes.update(t["shape"] for t in driver["transitions"])
            values = formula_path(driver, np.arange(-50, 512))
            np.testing.assert_array_equal(values[:50], 0)
            self.assertAlmostEqual(values[-1], 0)
            self.assertGreater(values.max(), 0.2)
        for count in (fast_up, fast_down, uneven_idle, multilevel, priming, near_edge):
            self.assertGreater(count, 15)
        self.assertEqual(shapes, {"linear", "smoothstep", "cosine", "power", "release"})

    def test_driver_is_an_scm_parent_and_uses_measured_causal_history(self):
        g = OperationalSCMGenerator(
            "unused",
            seed=8,
            min_rows=100,
            max_rows=100,
            min_covariates=4,
            max_covariates=4,
            process_episode_probability=1,
            formula_covariate_probability=0,
            discrete_covariate_probability=0,
            sensor_bounds_probability=0,
        )
        model = g.sample_model(trend_free=True, visible_covariate_count=100)
        op = model["outcome"]["operation"]
        driver = next(n for n in model["nodes"] if n["name"] == op["parent"])
        self.assertEqual(driver["kind"], "formula")
        self.assertEqual(driver["edges"], [])
        self.assertEqual(op["activity_source"], "direct")
        op.update(
            detail_fraction=0, noise_fraction=0, seasonal_fraction=0, idle_noise_scale=0
        )
        model["outcome"].pop("sensor", None)
        frame = g.simulate(model)
        expected = op["idle_value"] + op["amplitude"] * formula_path(
            driver, np.arange(100) - op["lag"]
        )
        np.testing.assert_allclose(frame.outcome, expected, atol=1e-12)
        np.testing.assert_array_equal(
            frame[driver["name"]], formula_path(driver, np.arange(100))
        )
        self.assertGreater(g.trend_strength(frame, model), 0.99999)
        graph = g.graph_spec(model)
        self.assertTrue(
            any(
                e["target"] == "outcome" and e["label"] == "episode"
                for e in graph["edges"]
            )
        )
        pd.testing.assert_frame_equal(
            frame, g.simulate(json.loads(json.dumps(model))), check_exact=True
        )
        changed = copy.deepcopy(model)
        for node in changed["nodes"]:
            node["visible"] = False
        np.testing.assert_array_equal(frame.outcome, g.simulate(changed).outcome)

    def test_formula_sources_cover_families_with_optional_noise_and_exact_replay(self):
        families = set()
        for noise_free in (0, 1):
            g = OperationalSCMGenerator(
                "unused",
                seed=72,
                min_rows=100,
                max_rows=100,
                min_covariates=6,
                max_covariates=6,
                process_episode_probability=0,
                switching_probability=0,
                formula_covariate_probability=1,
                noise_free_covariate_probability=noise_free,
            )
            for _ in range(16):
                model = g.sample_model(trend_free=True, visible_covariate_count=100)
                frame = g.simulate(model)
                for node in model["nodes"]:
                    if node["kind"] != "formula":
                        continue
                    families.add(node["family"])
                    self.assertEqual(node["edges"], [])
                    self.assertEqual(node["noise"]["scale"] == 0, bool(noise_free))
                    pure = formula_path(node, np.arange(100))
                    if noise_free:
                        np.testing.assert_array_equal(frame[node["name"]], pure)
                    else:
                        self.assertGreater(np.std(frame[node["name"]] - pure), 0)
                pd.testing.assert_frame_equal(
                    frame, g.simulate(json.loads(json.dumps(model))), check_exact=True
                )
        self.assertEqual(
            families,
            {
                "ramp",
                "power",
                "sigmoid",
                "bump",
                "damped_wave",
                "chirp",
                "sawtooth",
                "triangle",
                "state_machine",
            },
        )


if __name__ == "__main__":
    unittest.main()

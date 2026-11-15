"""Invariants for physical templates and by-construction driver controls."""

import copy
import json
import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src/01_data_generation"))
from plant_mechanisms import _coupled_storage, _rotating_machine, sample_plant
from scm_generator import OperationalSCMGenerator


class SharedPhysicsTests(unittest.TestCase):
    def test_joint_inventory_balance_and_positive_stocks(self):
        for seed in range(10):
            rng = np.random.default_rng(seed)
            spec = sample_plant(rng, 700, {"coupled_storage": 1}, 5)
            drive, noise = rng.uniform(0.1, 1.9, 700), rng.normal(0, 0.1, 700)
            channels = _coupled_storage(spec, drive, noise, np.random.default_rng(3))
            a, b = channels["pv"], channels["inventory_b"]
            previous_a = np.r_[spec["initial_inventory"][0], a[:-1]]
            previous_b = np.r_[spec["initial_inventory"][1], b[:-1]]
            np.testing.assert_allclose(
                a - previous_a, channels["inflow"] - channels["transfer"], atol=1e-15
            )
            np.testing.assert_allclose(
                b - previous_b, channels["transfer"] - channels["outflow"], atol=1e-15
            )
            np.testing.assert_allclose(
                a + b - previous_a - previous_b,
                channels["inflow"] - channels["outflow"],
                atol=2e-15,
            )
            self.assertTrue((a >= 0).all() and (b >= 0).all())
            mode = channels["mode"]
            np.testing.assert_allclose(
                channels["transfer"],
                np.asarray(spec["transfer_by_mode"])[mode] * previous_a,
            )

    def test_speed_drives_continuous_phase_and_shared_mode(self):
        spec = sample_plant(np.random.default_rng(2), 700, {"rotating_machine": 1}, 5)
        drive, noise = np.ones(700), np.zeros(700)
        c = _rotating_machine(spec, drive, noise, np.random.default_rng(4))
        self.assertGreater(len(np.unique(c["mode"])), 1)
        np.testing.assert_allclose(
            np.diff(c["phase"]), 2 * np.pi * c["speed"][:-1], atol=1e-13
        )
        np.testing.assert_array_equal(
            c["load_torque"], np.asarray(spec["torque_by_mode"])[c["mode"]]
        )
        # A speed change affects subsequent phase, without phase resets.
        changed = drive.copy()
        changed[300:] = 1.8
        d = _rotating_machine(spec, changed, noise, np.random.default_rng(4))
        np.testing.assert_array_equal(c["pv"][:301], d["pv"][:301])
        self.assertGreater(np.max(np.abs(c["phase"][350:] - d["phase"][350:])), 0.1)

    def test_prefix_causality_for_both_templates(self):
        for family, fn in [
            ("coupled_storage", _coupled_storage),
            ("rotating_machine", _rotating_machine),
        ]:
            spec = sample_plant(np.random.default_rng(2), 700, {family: 1}, 5)
            drive = np.linspace(0.1, 1.9, 700)
            full = fn(spec, drive, np.zeros(700), np.random.default_rng(4))
            prefix = fn(spec, drive[:301], np.zeros(301), np.random.default_rng(4))
            for name in prefix:
                np.testing.assert_array_equal(full[name][:301], prefix[name])

    def test_saved_models_replay_and_shared_state_has_multiple_children(self):
        for family in ("coupled_storage", "rotating_machine"):
            g = OperationalSCMGenerator(
                "unused",
                seed=4,
                min_rows=120,
                max_rows=120,
                min_covariates=5,
                max_covariates=5,
                shared_state_probability=1,
                persistent_hidden_probability=1,
                max_dominant_drivers=2,
                plant_probability=1,
                plant_weights={family: 1},
            )
            m = g.sample_model(trend_free=True, visible_covariate_count=0)
            before = copy.deepcopy(m)
            frame = g.simulate(m)
            pd.testing.assert_frame_equal(
                frame, g.simulate(json.loads(json.dumps(m))), check_exact=True
            )
            self.assertEqual(before, m)
            self.assertEqual(m["plant"]["family"], family)
            mode = next(n for n in m["nodes"] if n.get("shared_mode"))
            children = [
                n
                for n in m["nodes"] + [m["outcome"]]
                if any(e["parent"] == mode["name"] for e in n["edges"])
            ]
            self.assertGreaterEqual(len(children), 2)
            policy = m["driver_policy"]
            self.assertLessEqual(len(policy["dominant_sources"]), 2)
            for name, gain in policy["outgoing_gains"].items():
                self.assertTrue(
                    gain == 1 if name in policy["dominant_sources"] else gain <= 0.15
                )

    def test_parallel_shared_templates_and_plots(self):
        with tempfile.TemporaryDirectory() as directory:
            options = dict(
                seed=41,
                min_rows=60,
                max_rows=60,
                min_covariates=4,
                max_covariates=4,
                shared_state_probability=1,
                max_dominant_drivers=2,
                persistent_hidden_probability=1,
                trend_free_fraction=1,
                plant_probability=1,
                plant_weights={"coupled_storage": 1, "rotating_machine": 1},
                save_plot=True,
                save_dag=True,
                sign_inversion_probability=0,
            )
            left = OperationalSCMGenerator(
                Path(directory) / "left",
                generation_workers=2,
                generation_batch_size=1,
                **options,
            )
            right = OperationalSCMGenerator(
                Path(directory) / "right",
                generation_workers=3,
                generation_batch_size=2,
                **options,
            )
            for a, b in zip(left.generate(4), right.generate(4), strict=True):
                pd.testing.assert_frame_equal(
                    pd.read_parquet(a / "data.parquet"),
                    pd.read_parquet(b / "data.parquet"),
                    check_exact=True,
                )
                self.assertEqual(
                    (a / "scm.json").read_bytes(), (b / "scm.json").read_bytes()
                )
                self.assertGreater((a / "dag.png").stat().st_size, 1000)
                self.assertEqual(
                    (a / "dag.png").read_bytes(), (b / "dag.png").read_bytes()
                )

    def test_physical_dags_show_shared_state_and_respect_time(self):
        import networkx as nx
        from plant_mechanisms import physical_dag_view

        for family in ("coupled_storage", "rotating_machine"):
            for seed in range(8):
                spec = sample_plant(np.random.default_rng(seed), 400, {family: 1}, 5)
                graph, positions, _ = physical_dag_view(spec)
                self.assertTrue(nx.is_directed_acyclic_graph(graph))
                self.assertIn("outcome", graph)
                np.testing.assert_array_equal(positions["outcome"], [0, 0])
                for source, target in graph.edges:
                    self.assertLessEqual(
                        graph.nodes[source]["time_offset"],
                        graph.nodes[target]["time_offset"],
                    )

    def test_persistent_hidden_noise_is_causal_and_reduces_fast_jitter(self):
        g = OperationalSCMGenerator("unused", seed=2, persistent_hidden_probability=1)
        node = {
            "name": "c",
            "kind": "process",
            "visible": False,
            "noise": {"kind": "normal", "scale": 1, "location": 0},
            "edges": [],
        }
        g._budget_drivers([node], {"edges": []}, 4)
        spec = node["noise"]
        self.assertIn("memory_tau", spec)
        path = g._noise(np.random.default_rng(8), spec, 5000)
        prefix = g._noise(np.random.default_rng(8), spec, 100)
        np.testing.assert_array_equal(path[:100], prefix)
        self.assertGreater(np.corrcoef(path[:-1], path[1:])[0, 1], 0.8)
        for value in (0, -1, True, 1.5):
            with self.assertRaises(ValueError):
                OperationalSCMGenerator("unused", max_dominant_drivers=value)


if __name__ == "__main__":
    unittest.main()

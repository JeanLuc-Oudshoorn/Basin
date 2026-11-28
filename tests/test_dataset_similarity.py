"""Stage 13: the descriptors, the distances and a run on small made-up groups."""

import json
import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src" / "13_dataset_similarity"))


def walk(rng, points):
    return np.cumsum(rng.normal(size=points))


def cycle(rng, points, period=24):
    return np.sin(2 * np.pi * np.arange(points) / period) + 0.2 * rng.normal(
        size=points
    )


class DescriptorTests(unittest.TestCase):
    def setUp(self):
        import series_features

        self.features = series_features
        self.rng = np.random.default_rng(0)

    def test_known_processes_get_the_numbers_they_should(self):
        describe = self.features.describe
        noise = describe(self.rng.normal(size=512))
        drift = describe(walk(self.rng, 512))
        wave = describe(cycle(self.rng, 512))
        self.assertEqual(set(noise), set(self.features.FEATURES + self.features.EXTRA))
        # White noise: no memory, a flat spectrum. A walk: the opposite.
        self.assertLess(abs(noise["acf_1"]), 0.15)
        self.assertGreater(drift["acf_1"], 0.9)
        self.assertAlmostEqual(noise["long_memory"], 0.5, delta=0.15)
        self.assertAlmostEqual(drift["long_memory"], 1.5, delta=0.2)
        self.assertGreater(noise["spectral_entropy"], 0.85)
        self.assertLess(drift["spectral_entropy"], 0.4)
        self.assertGreater(drift["power_slow"], 0.6)
        self.assertGreater(noise["noise_share"], 3 * drift["noise_share"])
        # A cycle is found with its length; noise and a walk have none.
        self.assertEqual(wave["season_period"], 24)
        self.assertGreater(wave["seasonal_strength"], 0.8)
        self.assertEqual(noise["season_period"], 0)
        self.assertEqual(noise["seasonal_strength"], 0)
        self.assertLess(wave["season_harmonics"], 0.1)

    def test_the_shape_of_a_cycle_and_its_longer_parent(self):
        time = np.arange(512)
        humps = np.maximum(0, np.sin(2 * np.pi * time / 48)) ** 2
        humps = humps + 0.5 * np.maximum(0, np.sin(4 * np.pi * time / 48))
        found = self.features.describe(humps + 0.01 * self.rng.normal(size=512))
        self.assertEqual(found["season_period"], 48)
        self.assertGreater(found["season_harmonics"], 0.3)

    def test_level_and_scale_do_not_matter(self):
        series = cycle(self.rng, 512) + 0.3 * walk(self.rng, 512)
        first = self.features.describe(series)
        second = self.features.describe(1000.0 + 50.0 * series)
        for name in self.features.FEATURES:
            self.assertAlmostEqual(first[name], second[name], places=6, msg=name)

    def test_sensor_behaviour(self):
        series = walk(self.rng, 512)
        coarse = self.features.describe(np.round(series))
        self.assertGreater(coarse["flat_share"], 0.2)
        self.assertLess(coarse["distinct_share"], 0.2)
        steps = self.features.describe(np.repeat(self.rng.normal(size=8), 64))
        self.assertGreater(steps["jump_share"], 0.9)
        self.assertLessEqual(steps["long_memory"], 2.5)
        gaps = np.where(np.arange(512) % 4 == 0, np.nan, series)
        self.assertAlmostEqual(self.features.describe(gaps)["missing_share"], 0.25)

    def test_a_window_without_a_shape_is_refused(self):
        self.assertIsNone(self.features.describe(np.full(512, 3.0)))
        mostly_missing = np.full(512, np.nan)
        mostly_missing[:100] = self.rng.normal(size=100)
        self.assertIsNone(self.features.describe(mostly_missing))
        table = self.features.describe_many([np.full(512, 3.0), walk(self.rng, 512)])
        self.assertTrue(np.isnan(table[0]).all())
        self.assertTrue(np.isfinite(table[1]).all())


class AnalysisTests(unittest.TestCase):
    def setUp(self):
        import series_features
        import similarity_analysis
        import similarity_data

        self.features = series_features
        self.analysis = similarity_analysis
        self.data = similarity_data

    def test_settings(self):
        options = self.data.settings({})
        self.assertEqual(options["window_points"], [512, 2048])
        changed = self.data.settings(
            {"dataset_similarity": {"map": {"perplexity": 10}}}
        )
        self.assertEqual(changed["map"]["perplexity"], 10)
        self.assertEqual(changed["map"]["per_group"], options["map"]["per_group"])
        for wrong in ({"windows": 1}, {"map": {"size": 1}}, {"window_points": [64]}):
            with self.assertRaises(ValueError):
                self.data.settings({"dataset_similarity": wrong})

    def test_every_group_and_family_counts_equally(self):
        rng = np.random.default_rng(1)
        count = len(self.features.FEATURES)
        table = np.concatenate(
            [rng.normal(0, 1, (1000, count)), rng.normal(4, 1, (10, count))]
        )
        groups = np.array(["large"] * 1000 + ["small"] * 10)
        scaled, weighted, _ = self.analysis.standardise(table, groups)
        # The centre lies midway between the groups, not at the large one.
        self.assertAlmostEqual(
            scaled[:1000, 0].mean(), -scaled[1000:, 0].mean(), delta=0.3
        )
        sizes = pd.Series(self.features.FAMILIES).value_counts()
        column = self.features.FEATURES.index("linear_trend")
        self.assertAlmostEqual(
            weighted[0, column] / scaled[0, column], 1 / np.sqrt(sizes["trend"])
        )

    def test_coverage_tells_the_same_data_from_other_data(self):
        rng = np.random.default_rng(2)
        points = np.concatenate(
            [
                rng.normal(0, 1, (2400, 6)),
                rng.normal(0, 1, (300, 6)),
                rng.normal(3, 1, (300, 6)),
            ]
        )
        role = np.array(["reference"] * 2000 + ["control"] * 400 + ["benchmark"] * 600)
        found = self.analysis.coverage(points, role, self.data.settings({}))
        same, other = found["covered"][2400:2700], found["covered"][2700:]
        self.assertGreater(found["covered"][2000:2400].mean(), 0.85)
        self.assertGreater(same.mean(), 0.85)
        self.assertLess(other.mean(), 0.2)
        self.assertGreater(
            np.median(found["ratio"][2700:]), 2 * np.median(found["ratio"][2400:2700])
        )
        self.assertTrue((found["nearest"][2000:] < 2000).all())
        self.assertTrue((found["nearest"][:2000] == -1).all())

    def test_chance_that_a_value_is_higher(self):
        higher = self.analysis.auc_higher(
            np.array([5.0, 6.0]), np.array([1.0, 2.0, 3.0])
        )
        self.assertEqual(higher, 1.0)
        self.assertEqual(self.analysis.auc_higher(np.ones(4), np.ones(7)), 0.5)

    def test_windows_of_one_recording_share_a_fold(self):
        frame = pd.DataFrame(
            {
                "group": ["ind_eval", "ind_eval", "operational"],
                "dataset": ["solar", "solar", "ucr"],
                "series": [
                    "solar-H_short|plant-1|0|power",
                    "solar-H_long|plant-1|7|power",
                    "a",
                ],
            }
        )
        keys = self.analysis.underlying(frame)
        self.assertEqual(keys[0], keys[1])
        self.assertNotEqual(keys[0], keys[2])


class RunTests(unittest.TestCase):
    """Collect made-up synthetic tables, describe them and analyse the result."""

    def test_synthetic_tables_from_a_folder_and_a_whole_analysis(self):
        import series_features
        import similarity_analysis
        import similarity_data

        rng = np.random.default_rng(3)
        with tempfile.TemporaryDirectory() as folder:
            folder = Path(folder)
            for number in range(3):
                table = folder / "generated" / f"table_{number}"
                table.mkdir(parents=True)
                pd.DataFrame(
                    {
                        "time_index": np.arange(700),
                        "outcome": walk(rng, 700),
                        "covariate_01": 0.0,
                    }
                ).to_parquet(table / "data.parquet")
            options = similarity_data.settings(
                {
                    "dataset_similarity": {
                        "max_points": 600,
                        "window_points": [128, 512],
                        "synthetic": {"source": str(folder / "generated")},
                        "map": {"perplexity": 5},
                    }
                }
            )
            rows = similarity_data.collect_synthetic({}, options, log=lambda _: None)
            self.assertEqual([len(r["values"]) for r in rows], [600] * 3)
            self.assertEqual([r["source_points"] for r in rows], [700] * 3)
            # Made-up groups: walks, and cycles that the walks do not cover.
            for number in range(240):
                rows.append(
                    similarity_data.row(
                        "synthetic", "synthetic", f"w{number}", walk(rng, 300)
                    )
                )
            for number in range(60):
                rows.append(
                    similarity_data.row(
                        "fakebench", "normal", f"f{number}", walk(rng, 300)
                    )
                )
                rows.append(
                    similarity_data.row(
                        "ind_eval",
                        "solar",
                        f"t|plant-{number % 9}|{number}|power",
                        cycle(rng, 300),
                    )
                )
            rows.append(
                similarity_data.row(
                    "operational", "ucr", "flat", np.ones(128), view=128
                )
            )
            index = similarity_data.save(rows, folder, options["max_points"])
            _, windows = similarity_data.load(folder)
            self.assertTrue(np.isnan(windows[3, :300]).all())
            np.testing.assert_allclose(windows[3, 300:], rows[3]["values"], rtol=1e-6)
            in_view = similarity_data.view_rows(index, 128)
            self.assertEqual(len(in_view), len(rows))
            self.assertEqual(len(similarity_data.view_rows(index, 512)), 3)
            table = series_features.describe_many(
                np.asarray(windows[in_view][:, -128:])
            )
            del windows  # Windows cannot remove a file that is still mapped.
            frame = pd.DataFrame(
                table, columns=list(series_features.FEATURES + series_features.EXTRA)
            )
            frame.insert(0, "row", in_view)
            frame.to_parquet(similarity_data.features_path(folder, 128), index=False)
            summary = similarity_analysis.analyse_view(
                folder, 128, options, lambda _: None
            )
            saved = json.loads(
                (folder / "analysis_128.json").read_text(encoding="utf-8")
            )
            self.assertEqual(saved["counts"]["operational"]["without_shape"], 1)
            covered = {
                r["group"]: r["covered"]
                for r in summary["coverage"]
                if r["dataset"] == "all"
            }
            # With so few synthetic series the neighbours are far apart, so
            # the limit is wide; the order of the groups is what must hold.
            self.assertGreater(covered["fakebench"], 0.7)
            self.assertLess(covered["ind_eval"], covered["fakebench"] - 0.3)
            ratio = {
                r["group"]: r["median_ratio"]
                for r in summary["coverage"]
                if r["dataset"] == "all"
            }
            self.assertGreater(ratio["ind_eval"], 1.3 * ratio["fakebench"])
            groups = summary["separation"]["groups"]
            self.assertGreater(groups["ind_eval"]["classifier_auc"], 0.95)
            self.assertGreater(
                groups["ind_eval"]["features"]["seasonal_strength"][
                    "above_synthetic_p99"
                ],
                0.8,
            )
            result = pd.read_parquet(folder / "analysis_128.parquet")
            self.assertEqual(len(result), len(rows) - 1)
            self.assertTrue(result["tsne3_3"].notna().all())
            self.assertIn("z_seasonal_strength", result)


if __name__ == "__main__":
    unittest.main()

import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(
    0, str(Path(__file__).resolve().parents[1] / "src" / "13_dataset_similarity")
)
import generator_realism
import series_features
from sklearn.metrics import roc_auc_score


def clusters(rng, sources, per_source, columns, spread=0.05):
    """Real series as a few tight clusters inside the range of the generated ones."""
    centres = rng.uniform(-1.5, 1.5, (sources, columns))
    data = np.concatenate(
        [centre + spread * rng.normal(size=(per_source, columns)) for centre in centres]
    )
    names = np.repeat([f"source_{i}" for i in range(sources)], per_source)
    return data, names


class ClassifierTests(unittest.TestCase):
    def test_same_distribution_cannot_be_told_apart_and_a_shifted_one_can(self):
        rng = np.random.default_rng(0)
        real = rng.normal(size=(400, 6))
        keys = np.array([f"k{i}" for i in range(1200)])
        positive = np.arange(1200) < 400
        for shift, check in ((0.0, self.assertLess), (2.0, self.assertGreater)):
            own = rng.normal(size=(800, 6)) + shift
            scores, gain = generator_realism.fit_scores(
                np.concatenate([real, own]), positive, keys, workers=2
            )
            self.assertEqual(gain.shape, (6,))
            away = abs(roc_auc_score(positive, scores) - 0.5)
            check(away, 0.1 if shift == 0 else 0.4)

    def test_unseen_sources_are_not_recognised_but_seen_ones_are(self):
        rng = np.random.default_rng(1)
        real, sources = clusters(rng, sources=10, per_source=60, columns=4)
        own = rng.uniform(-1.6, 1.6, (1500, 4))
        data = np.concatenate([real, own])
        positive = np.arange(len(data)) < len(real)
        own_keys = np.array([f"generated|{i}" for i in range(len(own))])
        # Every real window is its own recording: sources are seen in training.
        keys = np.concatenate([[f"r{i}" for i in range(len(real))], own_keys])
        scores, _ = generator_realism.fit_scores(data, positive, keys, workers=2)
        seen = roc_auc_score(positive, scores)
        unseen = generator_realism.unseen_source_auc(
            data, positive, keys, np.concatenate([sources, own_keys]), workers=2
        )
        self.assertGreater(seen, 0.95)
        self.assertLess(unseen, 0.8)
        self.assertLess(unseen, seen - 0.15)

    def test_one_source_alone_has_no_unseen_score(self):
        rng = np.random.default_rng(2)
        data = rng.normal(size=(300, 3))
        positive = np.arange(300) < 100
        keys = np.array([f"k{i}" for i in range(300)])
        sources = np.where(positive, "only", keys)
        self.assertIsNone(
            generator_realism.unseen_source_auc(data, positive, keys, sources)
        )


class BreadthTests(unittest.TestCase):
    def test_breadth_falls_when_part_of_the_range_is_no_longer_made(self):
        rng = np.random.default_rng(3)
        names = list(series_features.FEATURES)

        def sample(rows, kinds=(0.0, 4.0)):
            # Two kinds of series, far apart in every descriptor.
            centre = rng.choice(kinds, size=(rows, 1))
            values = centre + rng.normal(size=(rows, len(names)))
            return pd.DataFrame(values, columns=names)

        original = generator_realism.OUTPUT
        with tempfile.TemporaryDirectory() as folder:
            generator_realism.OUTPUT = Path(folder)
            try:
                self.assertIsNone(generator_realism.breadth(sample(50)))
                for name in generator_realism.BASELINES:
                    target = Path(folder) / "trials" / name
                    target.mkdir(parents=True)
                    sample(700).to_parquet(target / "features.parquet", index=False)
                same = generator_realism.breadth(sample(700))
                lost = generator_realism.breadth(sample(700, kinds=(0.0,)))
            finally:
                generator_realism.OUTPUT = original
        self.assertGreater(same, 0.88)
        # One of the two kinds is gone: about half of the range is lost.
        self.assertLess(lost, 0.65)


class SamplingTests(unittest.TestCase):
    def test_tables_are_drawn_with_the_runner_settings_and_described(self):
        jobs = [(seed, seed % 2 == 0) for seed in range(6)]
        results = generator_realism.sample_block(jobs)
        self.assertEqual(len(results), 6)
        for values, facts in results:
            self.assertIn("plant", facts)
            self.assertIn("noise_profile", facts)
            self.assertEqual(values is None, facts["rows"] < generator_realism.POINTS)
            if values is not None:
                self.assertEqual(values.shape, (generator_realism.POINTS,))
        again = generator_realism.sample_block(jobs)
        for (first, _), (second, _) in zip(results, again):
            if first is not None:
                np.testing.assert_array_equal(first, second)

    def test_columns_that_are_not_measurements_are_left_out(self):
        skipped = ("Unnamed: 0", "hour", "is_weekend", "load_lag_3", "DE_wind_capacity")
        kept = ("Temperature", "DE_load_actual", "Motor_current", "hum")
        for name in skipped:
            self.assertTrue(generator_realism.NOT_A_SIGNAL.search(name), name)
        for name in kept:
            self.assertFalse(generator_realism.NOT_A_SIGNAL.search(name), name)


if __name__ == "__main__":
    unittest.main()

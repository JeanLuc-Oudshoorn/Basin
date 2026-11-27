"""Annealing: sets drawn in a chosen mix, independent experiments, start from a checkpoint."""

import copy
import inspect
import json
import sys
import tempfile
import unittest
from dataclasses import asdict
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np
import pandas as pd
import torch

ROOT = Path(__file__).resolve().parents[1]
for stage in ("03_transforming_input_data", "05_model_training", "10_annealing"):
    sys.path.insert(0, str(ROOT / "src" / stage))
import anneal
import annealing_experiment as experiments
import annealing_set
import annealing_start
from table_categories import (
    category_thresholds,
    matching_categories,
    table_statistics,
)
from transform import load_config


def recipe(**changes):
    value = {
        "experiment": "trial",
        "base_config": "configs/config.json",
        "set": {
            "name": "small",
            "tables": 16,
            "seed": 3,
            "mix": {"standard": 0.5, "flat": 0.25, "smooth": 0.25},
        },
        "training": {"learning_rate": 2e-5},
    }
    value.update(changes)
    return value


class TemporaryRoot:
    """Run a test with all annealing folders inside a temporary directory."""

    def __enter__(self):
        self.directory = tempfile.TemporaryDirectory()
        root = Path(self.directory.name)
        (root / "configs").mkdir()
        base = json.loads((ROOT / "configs/config.json").read_text(encoding="utf-8"))
        (root / "configs/config.json").write_text(json.dumps(base), encoding="utf-8")
        self.patches = [
            patch.object(module, "ROOT", root)
            for module in (experiments, annealing_set, annealing_start, anneal)
        ]
        for item in self.patches:
            item.start()
        return root

    def __exit__(self, *error):
        for item in self.patches:
            item.stop()
        self.directory.cleanup()


class TableCategoryTests(unittest.TestCase):
    def test_statistics_count_repeats_and_skip_missing_values(self):
        values = [1.0, 1.0, 1.0, np.nan, 1.0, 2.0, 3.0, 3.0]
        statistics = table_statistics(values)
        self.assertEqual(statistics["rows"], 8)
        self.assertEqual(statistics["observed"], 7)
        # Steps between observed values: 0 0 0 1 1 0.
        self.assertAlmostEqual(statistics["flat_share"], 4 / 6)
        self.assertEqual(statistics["longest_flat_run"], 3)
        self.assertGreater(statistics["roughness"], 0)

    def test_constant_and_empty_series_have_no_roughness(self):
        self.assertTrue(np.isnan(table_statistics([5.0] * 10)["roughness"]))
        self.assertEqual(table_statistics([np.nan, np.nan])["observed"], 0)

    def test_categories_follow_their_thresholds(self):
        thresholds = category_thresholds()
        flat = table_statistics([0.0] * 30 + list(np.arange(30.0)))
        smooth = table_statistics(np.sin(np.linspace(0, 6, 2000)))
        rough = table_statistics(np.random.default_rng(0).normal(size=500))
        self.assertEqual(matching_categories(flat, thresholds), ["flat"])
        self.assertEqual(matching_categories(smooth, thresholds), ["smooth"])
        self.assertEqual(matching_categories(rough, thresholds), [])
        strict = category_thresholds({"flat": {"min_longest_run": 100}})
        self.assertEqual(matching_categories(flat, strict), [])

    def test_unknown_categories_and_thresholds_are_rejected(self):
        with self.assertRaisesRegex(ValueError, "Unknown annealing category"):
            category_thresholds({"spiky": {}})
        with self.assertRaisesRegex(ValueError, "Unknown flat thresholds"):
            category_thresholds({"flat": {"minimum": 1}})


class RecipeTests(unittest.TestCase):
    def test_defaults_are_filled_in(self):
        checked = experiments.validate_recipe(recipe())
        self.assertEqual(checked["start"]["weights"], "best")
        self.assertFalse(checked["start"]["allow_unfinished"])
        self.assertEqual(checked["set"]["categories"], category_thresholds())

    def test_invalid_recipes_are_rejected(self):
        cases = {
            "add up to 1": {"set": dict(recipe()["set"], mix={"standard": 0.5})},
            "Unknown category": {"set": dict(recipe()["set"], mix={"spiky": 1.0})},
            "set.tables": {"set": dict(recipe()["set"], tables=3)},
            "experiment must be": {"experiment": "../outside"},
            "without": {"training": {"resume": "results/model/training/x.pt"}},
            "Unknown recipe fields": {"sets": {}},
            "start.weights": {"start": {"weights": "average"}},
        }
        for message, change in cases.items():
            with self.subTest(message), self.assertRaisesRegex(ValueError, message):
                experiments.validate_recipe(recipe(**change))

    def test_quotas_are_whole_tables_that_add_up(self):
        counts = experiments.quotas(10, {"standard": 0.65, "flat": 0.25, "smooth": 0.1})
        self.assertEqual(sum(counts.values()), 10)
        self.assertEqual(counts, {"standard": 6, "flat": 3, "smooth": 1})


class ExperimentConfigTests(unittest.TestCase):
    def setUp(self):
        self.base = json.loads((ROOT / "configs/config.json").read_text("utf-8"))
        self.recipe = experiments.validate_recipe(recipe())

    def test_data_comes_from_the_set_and_output_goes_to_the_experiment(self):
        before = copy.deepcopy(self.base)
        config = experiments.experiment_config(self.recipe, self.base, "2026-11-21")
        self.assertEqual(self.base, before)
        paths = config["paths"]
        for name in ("raw", "processed", "patch_index"):
            self.assertTrue(paths[name].startswith("data/04_annealing/small/"))
        for name in ("training", "holdout_test", "forecasts", "fakebench", "logs"):
            self.assertTrue(paths[name].startswith("results/model/annealing/trial/"))
        # No output path of the experiment is one of the main run's.
        self.assertFalse(set(paths.values()) & set(self.base["paths"].values()))
        self.assertEqual(config["data"]["generated_dates"], ["2026-11-21"])

    def test_training_settings_merge_and_the_schedule_is_replaced(self):
        training = experiments.experiment_config(self.recipe, self.base, "2026-11-21")[
            "training"
        ]
        self.assertEqual(training["learning_rate"], 2e-5)  # recipe
        self.assertEqual(training["epochs"], 1)  # annealing default
        self.assertEqual(training["resume"], "auto")
        self.assertNotIn("warmup_steps", training["scheduler"])
        self.assertEqual(training["scheduler"]["min_learning_rate"], 0.0)
        # Inherited from the main run where annealing has no opinion.
        self.assertEqual(training["batch_size"], self.base["training"]["batch_size"])
        self.assertEqual(
            training["early_stopping"]["monitor"],
            self.base["training"]["early_stopping"]["monitor"],
        )

    def test_holdout_test_uses_the_main_runs_holdout_tables(self):
        config = experiments.experiment_config(self.recipe, self.base, "2026-11-21")
        self.assertEqual(
            config["holdout"]["split_checkpoint"],
            "results/model/training/latest_checkpoint.pt",
        )
        self.assertEqual(config["holdout"]["split_config"], "configs/config.json")
        self.assertEqual(
            config["annealing"]["start"]["checkpoint"],
            "results/model/training/latest_checkpoint.pt",
        )

    def test_derived_configuration_is_a_valid_pipeline_configuration(self):
        config = experiments.experiment_config(self.recipe, self.base, "2026-11-21")
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "config.json"
            path.write_text(json.dumps(config), encoding="utf-8")
            self.assertEqual(load_config(path)["annealing"]["experiment"], "trial")

    def test_an_experiment_is_created_once_and_never_overwritten(self):
        with TemporaryRoot():
            directory, path, created = experiments.prepare_experiment(
                self.recipe, self.base, "2026-11-21"
            )
            self.assertTrue(created)
            saved = path.read_bytes()
            self.assertEqual(
                experiments.prepare_experiment(self.recipe, self.base, "2026-11-21"),
                (directory, path, False),
            )
            changed = experiments.validate_recipe(
                recipe(training={"learning_rate": 5e-6})
            )
            with self.assertRaisesRegex(FileExistsError, "never overwritten"):
                experiments.prepare_experiment(changed, self.base, "2026-11-21")
            self.assertEqual(path.read_bytes(), saved)
            self.assertEqual(experiments.experiment_config_path("trial"), path)
            with self.assertRaisesRegex(FileNotFoundError, "No annealing experiment"):
                experiments.experiment_config_path("other")


class AnnealingSetTests(unittest.TestCase):
    def build(self, root, chosen, **options):
        base = json.loads((root / "configs/config.json").read_text("utf-8"))
        options.setdefault("workers", 1)
        options.setdefault("progress", lambda message: None)
        return annealing_set.build_set(chosen, base, **options), base

    def test_set_has_the_requested_mix_of_fresh_tables(self):
        chosen = experiments.validate_recipe(recipe())
        with TemporaryRoot() as root:
            summary, base = self.build(root, chosen)
            directory = experiments.set_directory("small")
            tables = pd.read_parquet(directory / "tables.parquet")
            self.assertEqual(
                tables.category.value_counts().to_dict(),
                {"standard": 8, "flat": 4, "smooth": 4},
            )
            # The standard part is the first draws, whatever they are.
            self.assertEqual(
                sorted(tables[tables.category == "standard"].candidate), list(range(8))
            )
            thresholds = chosen["set"]["categories"]
            for row in tables.itertuples():
                frame = pd.read_parquet(root / row.source)
                statistics = table_statistics(frame["outcome"])
                self.assertAlmostEqual(statistics["flat_share"], row.flat_share)
                if row.category != "standard":
                    self.assertIn(
                        row.category, matching_categories(statistics, thresholds)
                    )
            # Only kept tables remain on disk, each once.
            on_disk = sorted(
                path.relative_to(root).as_posix()
                for path in directory.glob("generated/*/*/data.parquet")
            )
            self.assertEqual(on_disk, sorted(tables.source))
            self.assertEqual(tables.sha256.nunique(), len(tables))
            self.assertEqual(summary["tables"], 16)
            self.assertGreater(summary["drawn_tables"], 16)
            self.assertFalse(summary["holdout_overlap_checked"])  # no main split here
            # A finished set of the same definition is reused, not rebuilt.
            again, _ = self.build(root, chosen)
            self.assertEqual(again, summary)
            # The same name with another definition is refused and left alone.
            other = experiments.validate_recipe(
                recipe(set=dict(recipe()["set"], seed=4))
            )
            with self.assertRaisesRegex(FileExistsError, "different definition"):
                self.build(root, other)
            self.assertEqual(
                sorted(pd.read_parquet(directory / "tables.parquet").source), on_disk
            )

    def test_tables_depend_on_the_seed_and_not_on_batching(self):
        tiny = dict(recipe()["set"], tables=10, mix={"standard": 0.6, "flat": 0.4})
        seeds = []
        for name, batch in (("first", 16), ("second", 3)):
            chosen = experiments.validate_recipe(recipe(set=dict(tiny, name=name)))
            with TemporaryRoot() as root:
                self.build(root, chosen, batch_size=batch)
                tables = pd.read_parquet(
                    experiments.set_directory(name) / "tables.parquet"
                )
                seeds.append(tables[["candidate", "seed", "category"]])
        pd.testing.assert_frame_equal(seeds[0], seeds[1])

    def test_tables_identical_to_holdout_tables_stop_the_build(self):
        tiny = dict(recipe()["set"], tables=10, mix={"standard": 1.0})
        chosen = experiments.validate_recipe(recipe(set=tiny))
        with TemporaryRoot() as root:
            self.build(root, chosen)
            tables = pd.read_parquet(
                experiments.set_directory("small") / "tables.parquet"
            )
            sealed = {tables.sha256[0], "another"}
            copy_of = experiments.validate_recipe(recipe(set=dict(tiny, name="again")))
            with (
                patch.object(annealing_set, "holdout_hashes", return_value=sealed),
                self.assertRaisesRegex(RuntimeError, "identical to holdout tables"),
            ):
                self.build(root, copy_of)
            directory = experiments.set_directory("again")
            self.assertFalse((directory / experiments.SET_FILE).exists())
            # The unfinished folder is neither reused nor removed.
            with self.assertRaisesRegex(FileExistsError, "interrupted"):
                self.build(root, copy_of)
            self.assertTrue(directory.is_dir())

    def test_a_category_that_never_occurs_fails_instead_of_running_forever(self):
        impossible = dict(
            recipe()["set"],
            tables=10,
            mix={"standard": 0.5, "flat": 0.5},
            categories={"flat": {"min_longest_run": 10**9}},
        )
        chosen = experiments.validate_recipe(recipe(set=impossible))
        with TemporaryRoot() as root:
            with self.assertRaisesRegex(RuntimeError, "still short"):
                self.build(root, chosen, max_candidates_per_table=2)


def checkpoint(path, *, stop_reason="epoch_limit", width=768):
    model = dict(json.loads((ROOT / "configs/config.json").read_text("utf-8"))["model"])
    model.update(width=width, quantiles=tuple(model["quantiles"]))
    # As training saves it: the settings after the model has normalised them.
    model = asdict(anneal.ForecastConfig(**model))
    torch.save(
        {
            "format": "rwkv_timed_checkpoint_v1",
            "identity": {"model": model},
            "state_dict": {"weight": torch.full((2, 3), 2.0, dtype=torch.bfloat16)},
            "best_state_dict": {"weight": torch.ones(2, 3, dtype=torch.bfloat16)},
            "step": 30,
            "best_step": 20,
            "epoch": 3,
            "stop_reason": stop_reason,
        },
        path,
    )
    return model


class StartTests(unittest.TestCase):
    def config(self, root, **start):
        base = json.loads((ROOT / "configs/config.json").read_text("utf-8"))
        chosen = experiments.validate_recipe(recipe(start=start))
        return experiments.experiment_config(chosen, base, "2026-11-21")

    def test_best_or_last_weights_are_read_and_described(self):
        with TemporaryRoot() as root:
            path = root / "results/model/training/latest_checkpoint.pt"
            path.parent.mkdir(parents=True)
            settings = checkpoint(path)
            weights, start = annealing_start.load_start_weights(
                self.config(root), settings
            )
            self.assertTrue(torch.equal(weights["weight"].float(), torch.ones(2, 3)))
            self.assertEqual((start["step"], start["checkpoint_step"]), (20, 30))
            self.assertEqual(
                start["weights_sha256"], annealing_start.weights_digest(weights)
            )
            weights, start = annealing_start.load_start_weights(
                self.config(root, weights="last"), settings
            )
            self.assertEqual(float(weights["weight"][0, 0]), 2.0)
            self.assertEqual(start["step"], 30)
            # The checkpoint file is no longer held open: it can be replaced.
            checkpoint(path)

    def test_unfinished_run_or_other_model_is_refused(self):
        with TemporaryRoot() as root:
            path = root / "results/model/training/latest_checkpoint.pt"
            path.parent.mkdir(parents=True)
            settings = checkpoint(path, stop_reason=None)
            with self.assertRaisesRegex(ValueError, "not the final checkpoint"):
                annealing_start.load_start_weights(self.config(root), settings)
            weights, start = annealing_start.load_start_weights(
                self.config(root, allow_unfinished=True), settings
            )
            self.assertIsNone(start["checkpoint_stop_reason"])
            checkpoint(path, width=384)
            with self.assertRaisesRegex(ValueError, r"another model.*'width'"):
                annealing_start.load_start_weights(self.config(root), settings)
            with self.assertRaisesRegex(FileNotFoundError, "start checkpoint"):
                annealing_start.load_start_weights(
                    self.config(root, checkpoint="results/missing.pt"), settings
                )

    def test_weights_are_in_place_before_the_optimizer_is_built(self):
        events = []
        model = SimpleNamespace(load_state_dict=lambda w: events.append(("load", w)))
        original = lambda model, options: events.append(("optimizer", options)) or "o"  # noqa: E731
        stage = SimpleNamespace(make_optimizer=original)
        applied = []
        with annealing_start.starting_from(stage, {"w": 1}, applied):
            self.assertEqual(stage.make_optimizer(model, {"lr": 1}), "o")
        self.assertEqual(events, [("load", {"w": 1}), ("optimizer", {"lr": 1})])
        self.assertEqual(applied, [True])
        self.assertIs(stage.make_optimizer, original)

    def test_train_still_builds_its_optimizer_where_annealing_hooks_in(self):
        # starting_from() relies on this order inside train(): the model is
        # built, the optimizer made through the module-level name, then fitted.
        source = inspect.getsource(anneal.training_stage.train)
        made = source.index("optimizer = make_optimizer(model, t)")
        self.assertLess(source.index("model = RWKVForecaster(settings)"), made)
        self.assertLess(made, source.index("fit_epochs("))


class AnnealTests(unittest.TestCase):
    def run_anneal(self, root, train, *, own_checkpoint=None, **start):
        base = json.loads((ROOT / "configs/config.json").read_text("utf-8"))
        chosen = experiments.validate_recipe(recipe(start=start))
        config = experiments.experiment_config(chosen, base, "2026-11-21")
        with (
            patch.object(anneal, "location", lambda c, name: root / c["paths"][name]),
            patch.object(anneal, "resume_path", return_value=own_checkpoint),
            patch.object(anneal.training_stage, "train", train),
        ):
            anneal.anneal(config)
        return config

    def test_first_start_takes_the_main_runs_weights_and_records_them(self):
        loaded = []
        model = SimpleNamespace(load_state_dict=loaded.append)

        def train(config):
            anneal.training_stage.make_optimizer(model, config["training"])

        with (
            TemporaryRoot() as root,
            patch.object(anneal.training_stage, "make_optimizer", lambda m, o: None),
        ):
            path = root / "results/model/training/latest_checkpoint.pt"
            path.parent.mkdir(parents=True)
            checkpoint(path)
            before = path.read_bytes()
            config = self.run_anneal(root, train)
            self.assertEqual(float(loaded[0]["weight"][0, 0]), 1.0)
            start = json.loads(
                (
                    root / config["paths"]["training"] / "annealing_start.json"
                ).read_text()
            )
            self.assertEqual(start["step"], 20)
            self.assertEqual(path.read_bytes(), before)  # main checkpoint untouched

    def test_restart_continues_the_experiments_own_checkpoint(self):
        calls = []
        with TemporaryRoot() as root:
            # No main checkpoint exists: it must not be needed.
            self.run_anneal(root, calls.append, own_checkpoint=Path("own.pt"))
        self.assertEqual(len(calls), 1)

    def test_training_that_skipped_the_start_weights_is_reported(self):
        with TemporaryRoot() as root:
            path = root / "results/model/training/latest_checkpoint.pt"
            path.parent.mkdir(parents=True)
            checkpoint(path)
            with self.assertRaisesRegex(RuntimeError, "without loading the start"):
                self.run_anneal(root, lambda config: None)

    def test_only_annealing_configurations_in_their_own_folder_are_run(self):
        base = json.loads((ROOT / "configs/config.json").read_text("utf-8"))
        with self.assertRaisesRegex(ValueError, "Not an annealing experiment"):
            anneal.anneal(base)
        with self.assertRaisesRegex(ValueError, "trains under"):
            anneal.anneal(dict(base, annealing={"experiment": "x"}))


if __name__ == "__main__":
    unittest.main()

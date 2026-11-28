"""Stage 9: FakeBench-TS headline, covariates for Basin, and the upload."""

import json
import sys
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock, patch

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src" / "09_fakebench"))
from test_holdout_result_upload import HAS_AZURE, FakeContainer

HAS_SUBMODULE = (ROOT / "benchmarks/fakebench-ts/src/fakebench_ts").is_dir()


@unittest.skipUnless(HAS_SUBMODULE, "the FakeBench-TS submodule is not checked out")
class FakebenchStageTests(unittest.TestCase):
    def setUp(self):
        import fakebench

        self.stage = fakebench

    def test_the_headline_is_rmse_against_naive_by_track(self):
        from fakebench_ts import metrics, models
        from fakebench_ts.dataset import Spec, generate_dataset

        dataset = generate_dataset(
            Spec(
                name="small",
                horizon=16,
                context_lengths=(64,),
                series_per_weight=20,
                families={"normal": 1, "skew_gamma": 1},
            )
        )
        forecasts = {
            name: models.run_forecaster(models.MODELS[name](), dataset)
            for name in ("mean", "naive")
        }
        results, _ = metrics.evaluate(dataset, forecasts, bootstrap=20)
        with tempfile.TemporaryDirectory() as temporary:
            output = Path(temporary)
            (output / "results.json").write_text(json.dumps(results))
            table = self.stage.headline(output)
        self.assertEqual(list(table), ["all", "skewed", "symmetric"])
        for models_of_track in table.values():
            # Best first; the reference row of the true distribution is left out.
            self.assertEqual(list(models_of_track), ["naive", "mean"])
            self.assertEqual(models_of_track["naive"], 1.0)
            self.assertGreater(models_of_track["mean"], 1.0)

    def test_basin_reads_the_history_of_every_covariate(self):
        levels = (0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9)
        model = Mock()
        model.eval.return_value = model
        model.config = SimpleNamespace(quantiles=levels, patch_size=16)
        model.parameters.return_value = [torch.zeros(3)]
        seen = []

        def forecasts(model, requests, batch_size, prediction_patches=1):
            seen.append((requests, batch_size, prediction_patches))
            return [
                r["context"][0, 0, -1] + torch.arange(9.0).expand(1, 32, 9)
                for r in requests
            ]

        saved = {"config": {}, "state_dict": {}, "training_summary": {"best_epoch": 3}}
        with (
            patch.object(self.stage.torch, "load", return_value=saved),
            patch.object(self.stage.RWKVForecaster, "load", return_value=model),
            patch.object(self.stage, "configure_variate_execution"),
            patch.object(self.stage, "configure_fast_inference"),
            patch.object(self.stage, "forecast_contexts", side_effect=forecasts),
        ):
            forecast = self.stage.basin(
                {"training": {}},
                checkpoint="model.pt",
                batch_size=4,
                quantiles=True,
                covariates=True,
            )
            self.assertTrue(forecast.uses_covariates)
            self.assertEqual(forecast.details["epoch"], 3)
            contexts = [np.arange(40.0), np.arange(40.0) + 100]
            past = np.ones((2, 3, 40), dtype=np.float32)
            past[:, 1:] = 7.0
            output = forecast(contexts, 20, past_covariates=past)
            self.assertEqual(output.shape, (2, 9, 20))
            np.testing.assert_allclose(output[1, :, 0], 139 + np.arange(9))
            requests, batch_size, patches = seen[-1]
            self.assertEqual((batch_size, patches), (4, 2))
            self.assertEqual(requests[0]["context"].shape, (1, 1, 40))
            covariates = requests[0]["past_covariates"]
            self.assertEqual(covariates.shape, (1, 3, 40))
            self.assertTrue(torch.isfinite(covariates).all())
            self.assertEqual(covariates[0, :, 0].tolist(), [1.0, 7.0, 7.0])
            # Without covariates.
            forecast(contexts, 20)
            self.assertIsNone(seen[-1][0][0]["past_covariates"])
            # Without quantiles the median comes back.
            median = self.stage.basin({"training": {}}, checkpoint="model.pt")
            self.assertFalse(hasattr(median, "uses_covariates"))
            np.testing.assert_allclose(median(contexts, 20)[0], np.full(20, 43.0))

    def test_earlier_forecasts_count_a_series_once(self):
        from fakebench_ts import models

        dataset = SimpleNamespace(sha256="data")
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            run = root / "20261127T000000Z_epoch3_step1"
            (run / "forecasts").mkdir(parents=True)
            (root / "trials").mkdir()
            forecasts = {
                "a": np.zeros(4, np.float32),
                "b": np.zeros(4, np.float32),
                "b+cov": np.zeros(4, np.float32),
            }
            path = run / "forecasts" / "chronos2.parquet"
            models.save_forecasts(forecasts, path, model="chronos2", dataset=dataset)
            config = {"paths": {"fakebench": str(root)}}
            # An unfinished run is not a source of forecasts.
            self.assertIsNone(
                self.stage.earlier_forecasts(config, "chronos2", dataset, 2)
            )
            (run / "results.json").write_text("{}")
            self.assertEqual(
                self.stage.earlier_forecasts(config, "chronos2", dataset, 2), path
            )
            self.assertIsNone(
                self.stage.earlier_forecasts(config, "chronos2", dataset, 3)
            )
            other = SimpleNamespace(sha256="other data")
            self.assertIsNone(
                self.stage.earlier_forecasts(config, "chronos2", other, 2)
            )

    @unittest.skipUnless(HAS_AZURE, "needs the azure-storage-blob package")
    def test_a_run_folder_is_copied_with_results_json_last(self):
        store = object.__new__(self.stage.FakebenchResultStore)
        store.container = FakeContainer()
        store.container_name, store.endpoint = "results", "https://account.example/"
        store.prefix, store.block_bytes, store.concurrency = (
            "results/model/fakebench",
            1024,
            3,
        )
        with tempfile.TemporaryDirectory() as temporary:
            folder = Path(temporary) / "20261127T000000Z_epoch3_step1"
            (folder / "forecasts").mkdir(parents=True)
            (folder / "forecasts/basin.parquet").write_bytes(b"f" * 3000)
            for name in ("results.json", "run.json", "notes.md", "leaderboard.md"):
                (folder / name).write_text(name)
            summary = self.stage.upload_run(store, folder)
            base = f"results/model/fakebench/{folder.name}"
            self.assertEqual(summary["files"], 5)
            self.assertEqual(store.container.order[-1], f"{base}/results.json")
            self.assertEqual(
                set(store.container.blobs),
                {
                    f"{base}/{name}"
                    for name in (
                        "forecasts/basin.parquet",
                        "results.json",
                        "run.json",
                        "notes.md",
                        "leaderboard.md",
                    )
                },
            )
            self.assertEqual(store.runs()[folder.name]["complete"], True)
            store.container.refuse.add(f"{base}/notes.md")
            (folder / "notes.md").write_text("changed")
            with self.assertRaisesRegex(RuntimeError, f"--upload {folder.name}"):
                self.stage.upload_run(store, folder)

    def test_the_upload_is_configured_and_named_after_the_results_folder(self):
        config = {"paths": {}, "fakebench": {"result_upload": None}}
        self.assertIsNone(self.stage.make_result_store(config))
        self.assertIsNone(self.stage.make_result_store({"paths": {}}))
        config["fakebench"]["result_upload"] = {"container": "checkpoints"}
        with patch.object(self.stage, "FakebenchResultStore") as store:
            self.stage.make_result_store(config)
        self.assertEqual(store.call_args.kwargs["prefix"], "results/model/fakebench")
        config["paths"]["fakebench"] = str(ROOT.parent / "results_elsewhere")
        with self.assertRaisesRegex(ValueError, "outside the project"):
            self.stage.make_result_store(config)


if __name__ == "__main__":
    unittest.main()

"""Preparation cannot accumulate projections; training still learns the encoder."""

import copy
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src/04_creating_embeddings"))
import embed as embedding_stage
from embed import ForecastConfig, PatchEncoder
from transform import file_digest, load_config, read_artifact, signature


class LazyEmbeddingTests(unittest.TestCase):
    def fixture(self, base, count=2):
        config = copy.deepcopy(load_config())
        config["paths"].update(
            processed=str(base / "processed"), embedded=str(base / "embedded")
        )
        config["model"].update(width=128, layers=2, context_length=64, max_patches=16)
        config["data"]["context_length"] = 64
        config["embedding"] = {"cpu_threads": 1, "progress_every": 1000}
        target = torch.arange(64, dtype=torch.float32)[None, None]
        records = [
            {
                "id": f"window_{i}",
                "source": "fixture.parquet",
                "split": "train",
                "batch": {
                    "context": target[..., :48],
                    "past_covariates": target[..., :48] * 2,
                    "horizon": 16,
                },
            }
            for i in range(count)
        ]
        path = base / "processed/windows.pt"
        path.parent.mkdir(parents=True, exist_ok=True)
        torch.save({"signature": signature(config), "records": records}, path)
        return config

    def test_stage_never_projects_and_cached_size_does_not_scale_with_windows(self):
        with tempfile.TemporaryDirectory() as directory:
            base = Path(directory)
            config = self.fixture(base)
            with patch.object(
                PatchEncoder,
                "encode_tokens",
                side_effect=AssertionError("Offline projection is forbidden"),
            ):
                embedding_stage.embed(config)
                first_size = (base / "embedded/embeddings.pt").stat().st_size
                self.fixture(base, count=1000)
                embedding_stage.embed(config)
            path = base / "embedded/embeddings.pt"
            self.assertLess(abs(path.stat().st_size - first_size), 1024)
            raw = torch.load(path, weights_only=True)
            self.assertNotIn("records", raw)
            self.assertEqual(raw["record_count"], 1000)
            self.assertEqual(
                len((base / "embedded/windows.jsonl").read_text().splitlines()), 1000
            )

    def test_lazy_features_are_not_cached_and_allow_training_gradients(self):
        with tempfile.TemporaryDirectory() as directory:
            base = Path(directory)
            config = self.fixture(base)
            embedding_stage.embed(config)
            artifact = read_artifact(base / "embedded/embeddings.pt", config)
            record = artifact["records"][0]
            prepared = record["prepared"]
            self.assertNotIn("prepared", record.keys())
            self.assertNotIn("initial_embeddings", record.keys())
            self.assertFalse(prepared["features"].is_inference())
            encoder = PatchEncoder(
                ForecastConfig(
                    **dict(
                        config["model"], quantiles=tuple(config["model"]["quantiles"])
                    )
                )
            )
            encoder.load_state_dict(artifact["encoder_state"])
            encoder.encode_tokens(prepared).square().mean().backward()
            self.assertGreater(float(encoder.embedding.skip.weight.grad.abs().sum()), 0)
            self.assertGreater(float(encoder.roles.weight.grad.abs().sum()), 0)
            torch.testing.assert_close(
                record["prepared"]["features"], prepared["features"]
            )

    def test_changed_processed_file_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            base = Path(directory)
            config = self.fixture(base)
            embedding_stage.embed(config)
            self.fixture(base, count=3)
            with self.assertRaisesRegex(ValueError, "Processed data changed"):
                read_artifact(base / "embedded/embeddings.pt", config)

    def test_failed_save_preserves_previous_index_and_restores_threads(self):
        with tempfile.TemporaryDirectory() as directory:
            base = Path(directory)
            config = self.fixture(base)
            embedding_stage.embed(config)
            path = base / "embedded/embeddings.pt"
            previous = file_digest(path)
            previous_threads = torch.get_num_threads()

            def fail(path, payload):
                path.write_bytes(b"interrupted write")
                raise OSError("synthetic disk failure")

            with patch.object(embedding_stage, "save_artifact", side_effect=fail):
                with self.assertRaisesRegex(OSError, "synthetic disk failure"):
                    embedding_stage.embed(config)
            self.assertEqual(file_digest(path), previous)
            self.assertFalse(list((base / "embedded").glob("*.tmp")))
            self.assertEqual(torch.get_num_threads(), previous_threads)


if __name__ == "__main__":
    unittest.main()

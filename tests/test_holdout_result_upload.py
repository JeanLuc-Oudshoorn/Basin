"""Holdout run folders are copied to the storage container completely and verifiably."""

import hashlib
import importlib.util
import os
import sys
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace

from loguru import logger

ROOT = Path(__file__).resolve().parents[1]
for stage in ("03_transforming_input_data", "05_model_training", "06_holdout_test"):
    sys.path.insert(0, str(ROOT / "src" / stage))
from result_upload import (
    HoldoutResultStore,
    make_result_store,
    run_name,
    upload_changed_run,
    upload_run,
)
from transform import load_config

try:
    HAS_AZURE = importlib.util.find_spec("azure.storage.blob") is not None
except ModuleNotFoundError:
    HAS_AZURE = False


class FakeBlob:
    def __init__(self, container, name):
        self.container, self.name = container, name
        self.staged = {}

    def _store(self, data, content_settings, metadata):
        if self.name in self.container.refuse:
            raise ConnectionError(f"no route to {self.name}")
        if self.name in self.container.corrupt:
            data = data[:-1]
        self.container.blobs[self.name] = SimpleNamespace(
            name=self.name,
            data=data,
            size=len(data),
            content_settings=content_settings,
            metadata=dict(metadata),
        )
        self.container.order.append(self.name)

    def upload_blob(self, data, *, length, overwrite, validate_content, **options):
        assert overwrite and validate_content and length == len(data)
        self._store(bytes(data), options["content_settings"], options["metadata"])

    def stage_block(self, block_id, data, *, length, validate_content):
        assert validate_content and length == len(data)
        self.staged[block_id] = bytes(data)

    def commit_block_list(self, blocks, *, content_settings, metadata):
        data = b"".join(self.staged[block.id] for block in blocks)
        self.container.block_uploads.append(self.name)
        self._store(data, content_settings, metadata)

    def get_blob_properties(self):
        return self.container.blobs[self.name]

    def download_blob(self, **_):
        data = self.container.blobs[self.name].data
        return SimpleNamespace(chunks=lambda: iter([data[:7], data[7:]]))


class FakeContainer:
    def __init__(self):
        self.blobs, self.order, self.block_uploads = {}, [], []
        self.refuse, self.corrupt = set(), set()

    def get_blob_client(self, name):
        return FakeBlob(self, name)

    def list_blobs(self, name_starts_with="", include=()):
        # As an account with a hierarchical namespace does: every folder is
        # listed too, as an empty blob whose metadata says what it is.
        folders = {
            name.rsplit("/", depth)[0]
            for name in self.blobs
            for depth in range(1, name.count("/") + 1)
        }
        entries = dict(self.blobs)
        for folder in folders:
            entries[folder] = SimpleNamespace(
                name=folder,
                size=0,
                content_settings=SimpleNamespace(content_md5=None),
                metadata={"hdi_isfolder": "true"},
            )
        return [
            SimpleNamespace(
                name=name,
                size=blob.size,
                content_settings=blob.content_settings,
                metadata=blob.metadata if "metadata" in include else None,
            )
            for name, blob in sorted(entries.items())
            if name.startswith(name_starts_with)
        ]


def setUpModule():
    logger.disable("result_upload")


def tearDownModule():
    logger.enable("result_upload")


def fake_store(prefix="results/model/holdout_test"):
    store = object.__new__(HoldoutResultStore)
    store.container = FakeContainer()
    store.container_name, store.endpoint = "results", "https://account.example/"
    store.prefix, store.block_bytes, store.concurrency = prefix, 1024, 3
    return store


def run_folder(base, name="20261123T000000Z_epoch2_step10"):
    folder = Path(base) / name
    (folder / "plots").mkdir(parents=True)
    for index in range(12):
        (folder / f"test_table_{index:03d}.parquet").write_bytes(os.urandom(200))
    (folder / "plots/first.html").write_text("<html>plot</html>")
    (folder / "empty.json").write_bytes(b"")
    (folder / "test_predictions.parquet").write_bytes(os.urandom(5000))  # blocks
    (folder / "test_metrics.json").write_text('{"table_count": 12}')
    return folder


def contents(folder):
    return {
        path.relative_to(folder).as_posix(): path.read_bytes()
        for path in Path(folder).rglob("*")
        if path.is_file()
    }


@unittest.skipUnless(HAS_AZURE, "needs the azure-storage-blob package")
class HoldoutResultUploadTests(unittest.TestCase):
    def test_a_run_folder_is_copied_whole_with_the_completion_mark_last(self):
        store = fake_store()
        with tempfile.TemporaryDirectory() as temporary:
            folder = run_folder(temporary)
            summary = upload_run(store, folder)
            base = f"results/model/holdout_test/{folder.name}"
            self.assertEqual(
                {name: blob.data for name, blob in store.container.blobs.items()},
                {f"{base}/{name}": data for name, data in contents(folder).items()},
            )
            self.assertEqual(store.container.order[-1], f"{base}/test_metrics.json")
            self.assertEqual(
                store.container.block_uploads, [f"{base}/test_predictions.parquet"]
            )
            self.assertEqual(summary["files"], 16)
            self.assertEqual(summary["uploaded_files"], 16)
            self.assertEqual(
                summary["destination"], f"https://account.example/results/{base}"
            )
            for name, blob in store.container.blobs.items():
                digest = hashlib.sha256(blob.data).hexdigest()
                self.assertEqual(blob.metadata["sha256"], digest, name)
                self.assertEqual(
                    bytes(blob.content_settings.content_md5),
                    hashlib.md5(blob.data, usedforsecurity=False).digest(),
                    name,
                )
            plot = store.container.blobs[f"{base}/plots/first.html"]
            self.assertEqual(plot.content_settings.content_type, "text/html")

    def test_sending_again_sends_only_what_changed_and_deletes_nothing(self):
        store = fake_store()
        with tempfile.TemporaryDirectory() as temporary:
            folder = run_folder(temporary)
            upload_run(store, folder)
            store.container.order.clear()
            self.assertEqual(upload_run(store, folder)["uploaded_files"], 0)
            self.assertEqual(store.container.order, [])
            # A regenerated plot and rewritten metrics; a file removed locally.
            (folder / "plots/first.html").write_text("<html>new plot</html>")
            (folder / "test_metrics.json").write_text('{"table_count": 12, "x": 1}')
            (folder / "test_table_003.parquet").unlink()
            summary = upload_run(store, folder)
            base = f"results/model/holdout_test/{folder.name}"
            self.assertEqual(
                store.container.order,
                [f"{base}/plots/first.html", f"{base}/test_metrics.json"],
            )
            self.assertEqual(summary["unchanged_files"], 13)
            self.assertIn(f"{base}/test_table_003.parquet", store.container.blobs)

    def test_a_copy_that_differs_or_fails_is_an_error_that_names_the_way_back(self):
        with tempfile.TemporaryDirectory() as temporary:
            folder = run_folder(temporary)
            before = contents(folder)
            base = f"results/model/holdout_test/{folder.name}"
            for attribute, reason in (
                ("corrupt", "missing or different"),
                ("refuse", "ConnectionError"),
            ):
                store = fake_store()
                getattr(store.container, attribute).add(
                    f"{base}/test_table_005.parquet"
                )
                with self.assertRaises(RuntimeError) as raised:
                    upload_run(store, folder)
                message = str(raised.exception)
                self.assertIn(reason, message)
                self.assertIn(f"results are complete in {folder}", message)
                self.assertIn(f"result_upload.py upload --run {folder.name}", message)
                # The run is not marked complete in the container.
                self.assertNotIn(f"{base}/test_metrics.json", store.container.blobs)
                self.assertEqual(contents(folder), before)

    def test_a_run_comes_back_from_the_container_as_it_was(self):
        store = fake_store()
        with tempfile.TemporaryDirectory() as temporary:
            folder = run_folder(temporary)
            other = run_folder(Path(temporary) / "elsewhere", "20261124T000000Z_epoch3")
            upload_run(store, folder)
            upload_run(store, other)
            self.assertEqual(
                store.runs(),
                {
                    folder.name: {
                        "files": 16,
                        "bytes": mock_size(folder),
                        "complete": True,
                    },
                    other.name: {
                        "files": 16,
                        "bytes": mock_size(other),
                        "complete": True,
                    },
                },
            )
            restored = Path(temporary) / "restored" / folder.name
            result = store.fetch(folder.name, restored)
            self.assertEqual(contents(restored), contents(folder))
            self.assertEqual(result["downloaded_files"], 16)
            # Only what is missing or different is fetched the next time.
            (restored / "test_table_001.parquet").write_bytes(b"damaged")
            (restored / "empty.json").unlink()
            self.assertEqual(store.fetch(folder.name, restored)["downloaded_files"], 2)
            self.assertEqual(contents(restored), contents(folder))
            with self.assertRaisesRegex(FileNotFoundError, "No blobs under"):
                store.fetch("20200101T000000Z_missing", restored)

    def test_results_saved_before_run_folders_keep_their_place(self):
        store = fake_store()
        with tempfile.TemporaryDirectory() as temporary:
            config = load_config()
            root = Path(temporary) / "holdout_test"
            config["paths"]["holdout_test"] = str(root)
            run = run_folder(root)
            (root / "test_metrics.json").write_text("{}")
            (root / "test_table_000.parquet").write_bytes(b"old")
            self.assertEqual(run_name(config, root), "")
            self.assertEqual(run_name(config, run), run.name)
            upload_run(store, root, run_name(config, root))
            # The root's own files only; the run folder below is a run of its own.
            self.assertEqual(
                sorted(store.container.blobs),
                [
                    "results/model/holdout_test/test_metrics.json",
                    "results/model/holdout_test/test_table_000.parquet",
                ],
            )
            upload_run(store, run)
            self.assertEqual(sorted(store.runs()), ["", run.name])
            self.assertEqual(store.runs()[""]["files"], 2)
            restored = Path(temporary) / "restored"
            self.assertEqual(store.fetch("", restored)["files"], 2)
            self.assertEqual(
                sorted(contents(restored)),
                ["test_metrics.json", "test_table_000.parquet"],
            )


def mock_size(folder):
    return sum(len(data) for data in contents(folder).values())


class HoldoutResultUploadSettingsTests(unittest.TestCase):
    def test_upload_is_off_unless_configured_and_settings_are_validated(self):
        config = load_config()
        configured = config["holdout"].pop("result_upload")
        self.assertIsNone(make_result_store(config))
        self.assertIsNone(upload_changed_run(config, ROOT))
        config["holdout"]["result_upload"] = {"enabled": False}
        self.assertIsNone(make_result_store(config))
        for bad in ("checkpoints", {"bucket": "x"}, {"container": ""}, {"prefix": "/"}):
            config["holdout"]["result_upload"] = bad
            with self.assertRaisesRegex(ValueError, "holdout.result_upload", msg=bad):
                make_result_store(config)
        # Results outside the project have no natural place in the container.
        config["holdout"]["result_upload"] = dict(configured, prefix=None)
        config["paths"]["holdout_test"] = str(ROOT.parent / "results_elsewhere")
        with self.assertRaisesRegex(ValueError, "result_upload.prefix"):
            make_result_store(config)

    @unittest.skipUnless(HAS_AZURE, "needs the azure-storage-blob package")
    def test_the_configured_store_mirrors_the_results_folder(self):
        from unittest.mock import patch

        config = load_config()
        environment = {
            "AZURE_ENDPOINT": "https://account.blob.core.windows.net/",
            "AZURE_STORAGE_KEY": "a2V5",
        }
        with (
            patch.dict(os.environ, environment),
            tempfile.TemporaryDirectory() as temporary,
        ):
            store = make_result_store(config, root=temporary)
            self.assertEqual(
                store.describe(),
                "https://account.blob.core.windows.net/checkpoints/"
                + Path(config["paths"]["holdout_test"]).as_posix(),
            )
            self.assertEqual(store.concurrency, 8)
            self.assertIn("holdout test has not started", store.if_unavailable)


@unittest.skipUnless(
    os.environ.get("BASIN_AZURE_LIVE_TEST") == "1",
    "set BASIN_AZURE_LIVE_TEST=1 to write to the real storage container",
)
class LiveHoldoutResultUploadTests(unittest.TestCase):
    def test_round_trip_through_the_real_container(self):
        options = {
            "container": "checkpoints",
            "prefix": None,
            "block_mib": 1,
            "concurrency": 4,
        }
        store = HoldoutResultStore(options, prefix="basin_live_test/holdout")
        store.preflight()
        with tempfile.TemporaryDirectory() as temporary:
            folder = run_folder(temporary)
            (folder / "test_predictions.parquet").write_bytes(os.urandom(3 * 2**20))
            try:
                summary = upload_run(store, folder)
                self.assertEqual(summary["uploaded_files"], 16)
                self.assertEqual(upload_run(store, folder)["uploaded_files"], 0)
                (folder / "plots/first.html").write_text("<html>new plot</html>")
                self.assertEqual(upload_run(store, folder)["uploaded_files"], 1)
                self.assertEqual(
                    store.runs()[folder.name],
                    {"files": 16, "bytes": mock_size(folder), "complete": True},
                )
                restored = Path(temporary) / "restored"
                self.assertEqual(
                    store.fetch(folder.name, restored)["downloaded_files"], 16
                )
                self.assertEqual(contents(restored), contents(folder))
            finally:
                # Files first, then the folders that held them, deepest first.
                entries = [
                    blob.name
                    for blob in store.container.list_blobs(
                        name_starts_with=store.prefix
                    )
                ]
                for name in sorted(entries, key=lambda n: n.count("/"), reverse=True):
                    store.container.delete_blob(name)
        self.assertEqual(
            list(store.container.list_blobs(name_starts_with=store.prefix)), []
        )


if __name__ == "__main__":
    unittest.main()

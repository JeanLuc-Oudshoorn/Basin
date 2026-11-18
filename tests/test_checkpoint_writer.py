"""Background checkpoint writes and cloud copies must never lose a checkpoint."""

import hashlib
import json
import sys
import tempfile
import threading
import time
import unittest
from pathlib import Path
from types import SimpleNamespace

import torch

ROOT = Path(__file__).resolve().parents[1]
for stage in (
    "03_transforming_input_data",
    "04_patch_indexing",
    "05_model_training",
):
    sys.path.insert(0, str(ROOT / "src" / stage))
from checkpoint_upload import (
    UploadSuperseded,
    make_checkpoint_store,
    read_environment_file,
    upload_options,
)
from checkpoint_writer import CheckpointWriter
from checkpoints import atomic_torch_save, host_copy, save_checkpoint
from training_log import training_run
from transform import load_config


class FakeUpload:
    """Receives the checkpoint's bytes while the file is written."""

    def __init__(self, store, path, cancel):
        self.store, self.path, self.cancel = store, Path(path), cancel
        self.data = bytearray()
        self.size = 0
        self.closed = False

    def write(self, data):
        self.store.started.set()
        if self.store.wait_for_cancel:
            if not self.cancel.wait(timeout=30):
                raise AssertionError("the upload was never asked to stop")
            raise UploadSuperseded(str(self.path))
        if self.store.fail:
            raise ConnectionError("storage unreachable")
        self.data += data
        self.size += len(data)

    def commit(self, digests, size, metadata=None):
        if self.store.fail_commit:
            raise ConnectionError("storage unreachable at commit")
        if size != self.size:
            raise AssertionError(f"commit of {size} bytes after {self.size} written")
        self.store.uploads.append(
            {
                "bytes": bytes(self.data),
                "digests": digests,
                "metadata": dict(metadata or {}),
            }
        )
        return {
            "blob": f"prefix/{self.path.name}",
            "bytes": size,
            "blocks": 1,
            "seconds": 0.0,
            "mib_per_second": 0.0,
            "sha256": digests["sha256"],
        }

    def close(self):
        self.closed = True
        self.store.closed += 1


class FakeStore:
    """Records uploads; can fail, or hang until it is told to give up."""

    def __init__(self, *, fail=False, wait_for_cancel=False, fail_commit=False):
        self.fail, self.wait_for_cancel = fail, wait_for_cancel
        self.fail_commit = fail_commit
        self.uploads = []
        self.closed = 0
        self.started = threading.Event()

    def describe(self):
        return "fake://checkpoints"

    def begin(self, path, *, cancel=None):
        return FakeUpload(self, path, cancel)

    def upload(self, path, *, digests=None, metadata=None, cancel=None):
        """From a file on disk: only when the file was written beforehand."""
        session = self.begin(path, cancel=cancel)
        session.write(Path(path).read_bytes())
        return session.commit(digests, Path(path).stat().st_size, metadata)


def training_state(seed=3):
    torch.manual_seed(seed)
    model = torch.nn.Linear(4, 3)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-2)
    model(torch.randn(5, 4)).sum().backward()
    optimizer.step()
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, 1)
    early = SimpleNamespace(best=0.5, best_epoch=1, bad_epochs=0)
    history = [
        {"epoch": 1, "step": 1, "loss": 0.25, "embedding_shapes": [[1, 1, 16, 8]]}
    ]
    return model, optimizer, scheduler, early, history


def save(path, state, *, writer=None, improved=True, loop_state=None):
    model, optimizer, scheduler, early, history = state
    save_checkpoint(
        path,
        model,
        optimizer,
        scheduler,
        {"identity": "fixture"},
        {"name": "constant"},
        early,
        history,
        [{"epoch": 0, "loss": 1.0}],
        10,
        2,
        1,
        path.with_name("best_weights.pt"),
        improved,
        None,
        loop_state=loop_state,
        writer=writer,
    )


def assert_equal_payload(case, left, right, ignore=()):
    case.assertEqual(set(left), set(right))
    for key in left:
        if key in ignore:
            continue
        a, b = left[key], right[key]
        if key in ("state_dict", "best_state_dict"):
            case.assertEqual(set(a), set(b))
            for name in a:
                case.assertTrue(torch.equal(a[name], b[name]), name)
        elif key == "optimizer_state_dict":
            case.assertEqual(a["param_groups"], b["param_groups"])
            for index in a["state"]:
                for name, value in a["state"][index].items():
                    case.assertTrue(torch.equal(value, b["state"][index][name]))
        elif key == "rng_state":
            case.assertTrue(torch.equal(a["torch"], b["torch"]))
            case.assertEqual(a["python"], b["python"])
        else:
            case.assertEqual(a, b, key)


class CheckpointWriterTests(unittest.TestCase):
    def setUp(self):
        # Whether the test machine happens to be short of memory must not
        # decide which write route these tests exercise.
        import checkpoints

        original = checkpoints.available_host_memory
        checkpoints.available_host_memory = lambda: None
        self.addCleanup(setattr, checkpoints, "available_host_memory", original)

    def test_host_copy_shares_nothing_with_the_live_state(self):
        live = {
            "weights": {"w": torch.ones(3)},
            "history": [{"step": 1, "shapes": [[1, 2]]}],
            "limits": {(1, 16): 4},
            "rng": (3, (1, 2, 3), None),
        }
        copied = host_copy(live)
        live["weights"]["w"].add_(1)
        live["history"][0]["shapes"][0].append(9)
        live["history"].append({"step": 2})
        live["limits"][(1, 16)] = 1
        self.assertTrue(torch.equal(copied["weights"]["w"], torch.ones(3)))
        self.assertEqual(copied["history"], [{"step": 1, "shapes": [[1, 2]]}])
        self.assertEqual(copied["limits"], {(1, 16): 4})
        self.assertEqual(copied["rng"], (3, (1, 2, 3), None))

    def test_background_checkpoint_equals_the_blocking_one_despite_later_training(
        self,
    ):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            (root / "a").mkdir()
            (root / "b").mkdir()
            torch.manual_seed(11)
            save(root / "a" / "latest_checkpoint.pt", training_state())
            torch.manual_seed(11)
            state = training_state()
            release = threading.Event()
            original = torch.save

            def slow_save(*args, **kwargs):
                release.wait(timeout=30)  # training moves on before the file is written
                return original(*args, **kwargs)

            writer = CheckpointWriter()
            torch.save = slow_save
            try:
                save(root / "b" / "latest_checkpoint.pt", state, writer=writer)
                self.assertTrue(writer.busy)
                model, optimizer, _, early, history = state
                with torch.no_grad():
                    model.weight.add_(5)  # the next updates
                optimizer.state[model.weight]["exp_avg"].add_(7)
                history.append({"epoch": 1, "step": 2, "loss": 0.1})
                early.best = 0.0
                release.set()
                writer.finish()
            finally:
                torch.save = original
            self.assertFalse(writer.busy)
            blocking = torch.load(
                root / "a" / "latest_checkpoint.pt", weights_only=True
            )
            background = torch.load(
                root / "b" / "latest_checkpoint.pt", weights_only=True
            )
            assert_equal_payload(self, blocking, background)

    def test_failed_write_stops_training_and_keeps_the_previous_checkpoint(self):
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "latest_checkpoint.pt"
            writer = CheckpointWriter()
            save(path, training_state(), writer=writer)
            writer.finish()
            first = path.read_bytes()
            original = torch.save

            def broken_save(*args, **kwargs):
                raise OSError("disk full")

            torch.save = broken_save
            try:
                save(path, training_state(5), writer=writer)
                with self.assertRaisesRegex(RuntimeError, "failed in the background"):
                    writer.finish()
            finally:
                torch.save = original
            self.assertEqual(path.read_bytes(), first)
            self.assertEqual(list(Path(temporary).glob(".*tmp")), [])
            writer.finish()  # the failure is reported once

    def test_upload_receives_the_written_file_with_its_digests(self):
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "latest_checkpoint.pt"
            store = FakeStore()
            writer = CheckpointWriter(store)
            save(
                path,
                training_state(),
                writer=writer,
                loop_state={"epoch": 2, "step": 7},
            )
            writer.finish(require_upload=True)
            self.assertEqual((writer.uploads, writer.upload_failures), (1, 0))
            upload = store.uploads[0]
            self.assertEqual(upload["bytes"], path.read_bytes())
            self.assertEqual(
                upload["digests"]["sha256"],
                hashlib.sha256(path.read_bytes()).hexdigest(),
            )
            self.assertEqual(
                upload["digests"]["md5"],
                hashlib.md5(path.read_bytes(), usedforsecurity=False).digest(),
            )
            self.assertEqual(upload["metadata"]["step"], 7)
            self.assertEqual(writer.last_upload["blob"], "prefix/latest_checkpoint.pt")
            self.assertEqual(store.closed, 1)

    def test_failed_upload_keeps_training_unless_it_is_the_required_first_one(self):
        for failure in ({"fail": True}, {"fail_commit": True}):  # mid-way, at the end
            with tempfile.TemporaryDirectory() as temporary:
                path = Path(temporary) / "latest_checkpoint.pt"
                store = FakeStore(**failure)
                writer = CheckpointWriter(store)
                save(path, training_state(), writer=writer)
                writer.finish()  # later checkpoints: logged, not raised
                self.assertEqual(writer.upload_failures, 1)
                # The file is complete although its upload broke off.
                reference = Path(temporary) / "reference.pt"
                torch.manual_seed(0)
                save(reference, training_state())
                assert_equal_payload(
                    self,
                    torch.load(path, weights_only=True),
                    torch.load(reference, weights_only=True),
                    ignore=("rng_state",),
                )
                save(path, training_state(), writer=writer)
                with self.assertRaisesRegex(RuntimeError, "could not be uploaded"):
                    writer.finish(require_upload=True)
                self.assertEqual(writer.upload_failures, 2)
                self.assertEqual(store.uploads, [])
                self.assertEqual(store.closed, 2)

    def test_unreachable_store_at_the_start_of_an_upload_still_writes_the_file(self):
        class Unreachable(FakeStore):
            def begin(self, path, *, cancel=None):
                raise ConnectionError("no route to the storage account")

        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "latest_checkpoint.pt"
            writer = CheckpointWriter(Unreachable())
            save(path, training_state(), writer=writer)
            writer.finish()
            self.assertEqual(writer.upload_failures, 1)
            torch.load(path, weights_only=True)

    def test_next_checkpoint_abandons_an_unfinished_upload_not_the_file(self):
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "latest_checkpoint.pt"
            slow = FakeStore(wait_for_cancel=True)
            writer = CheckpointWriter(slow)
            save(path, training_state(), writer=writer)
            self.assertTrue(slow.started.wait(timeout=30))  # the upload hangs
            started = time.perf_counter()
            writer.finish(abandon_upload=True)  # the next checkpoint is due
            self.assertLess(time.perf_counter() - started, 20)
            self.assertEqual(writer.uploads_abandoned, 1)
            self.assertEqual(writer.upload_failures, 0)
            first = path.read_bytes()  # complete on disk all the same
            torch.load(path, weights_only=True)
            writer.store = fast = FakeStore()
            save(path, training_state(9), writer=writer)
            writer.finish(require_upload=True)
            self.assertEqual(len(fast.uploads), 1)
            self.assertEqual(fast.uploads[0]["bytes"], path.read_bytes())
            self.assertNotEqual(first, path.read_bytes())

    def test_short_host_memory_falls_back_to_a_blocking_write_and_still_uploads(
        self,
    ):
        import checkpoints

        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "latest_checkpoint.pt"
            store = FakeStore()
            writer = CheckpointWriter(store)
            original = checkpoints.available_host_memory
            checkpoints.available_host_memory = lambda: 1024  # next to nothing
            try:
                save(path, training_state(), writer=writer)
                # Written before the call returned, not by the background thread.
                self.assertTrue(path.is_file())
                torch.load(path, weights_only=True)
            finally:
                checkpoints.available_host_memory = original
            writer.finish(require_upload=True)
            self.assertEqual(store.uploads[0]["bytes"], path.read_bytes())
            self.assertEqual(
                store.uploads[0]["digests"]["sha256"],
                hashlib.sha256(path.read_bytes()).hexdigest(),
            )

    def test_writer_thread_reports_to_the_training_journal(self):
        with tempfile.TemporaryDirectory() as temporary:
            config = {"paths": {"logs": temporary}}
            with training_run(config, "train") as journal:
                writer = CheckpointWriter(FakeStore())
                save(
                    Path(temporary) / "latest_checkpoint.pt",
                    training_state(),
                    writer=writer,
                    loop_state={"epoch": 1, "step": 3},
                )
                writer.finish()
                journal_path = journal.path
            events = [
                json.loads(line)["event"]
                for line in journal_path.read_text(encoding="utf-8").splitlines()
            ]
            self.assertIn("timed_training_checkpoint_saved", events)
            self.assertIn("checkpoint_uploaded", events)

    def test_digests_describe_the_file_on_disk(self):
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "weights.pt"
            digests = atomic_torch_save({"w": torch.arange(1000.0)}, path, digests=True)
            data = path.read_bytes()
            self.assertEqual(digests["sha256"], hashlib.sha256(data).hexdigest())
            self.assertEqual(
                digests["md5"], hashlib.md5(data, usedforsecurity=False).digest()
            )
            self.assertIsNone(atomic_torch_save({"w": torch.zeros(1)}, path))
            # An upload fed while the file is written receives the same bytes;
            # one that breaks off does not disturb the file.
            received = bytearray()

            class Sink:
                def write(self, data):
                    received.extend(data)

            class Broken:
                def write(self, data):
                    raise ConnectionError("network gone")

            digests = atomic_torch_save(
                {"w": torch.arange(1000.0)}, path, mirror=Sink()
            )
            self.assertEqual(bytes(received), path.read_bytes())
            self.assertIsNone(digests["mirror_error"])
            digests = atomic_torch_save(
                {"w": torch.arange(1000.0)}, path, mirror=Broken()
            )
            self.assertIsInstance(digests["mirror_error"], ConnectionError)
            self.assertEqual(
                digests["sha256"], hashlib.sha256(path.read_bytes()).hexdigest()
            )
            self.assertTrue(
                torch.equal(
                    torch.load(path, weights_only=True)["w"], torch.arange(1000.0)
                )
            )
            self.assertEqual(list(Path(temporary).glob(".*tmp")), [])


class CheckpointUploadSettingsTests(unittest.TestCase):
    def test_upload_is_off_unless_configured_and_settings_are_validated(self):
        config = load_config("configs/legacy_500m.json")
        config["training"].pop("checkpoint_upload", None)
        self.assertIsNone(upload_options(config["training"]))
        self.assertIsNone(make_checkpoint_store(config))
        self.assertIsNone(upload_options({"checkpoint_upload": {"enabled": False}}))
        self.assertEqual(
            upload_options({"checkpoint_upload": {}}),
            {
                "container": "checkpoints",
                "prefix": None,
                "block_mib": 16,
                "concurrency": 4,
            },
        )
        for bad in (
            "checkpoints",
            {"bucket": "x"},
            {"container": ""},
            {"block_mib": 0},
            {"concurrency": True},
            {"prefix": "/"},
            {"enabled": "yes"},
        ):
            with self.assertRaises(ValueError, msg=repr(bad)):
                upload_options({"checkpoint_upload": bad})

    def test_environment_file_is_read_without_overriding_the_environment(self):
        import os

        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / ".env"
            path.write_text(
                '# comment\nBASIN_TEST_A=from file\nBASIN_TEST_B="quoted value"\n'
                "export BASIN_TEST_C='c'\nnot a setting\n",
                encoding="utf-8",
            )
            os.environ["BASIN_TEST_A"] = "from environment"
            try:
                read_environment_file(path)
                self.assertEqual(os.environ["BASIN_TEST_A"], "from environment")
                self.assertEqual(os.environ["BASIN_TEST_B"], "quoted value")
                self.assertEqual(os.environ["BASIN_TEST_C"], "c")
            finally:
                for name in ("BASIN_TEST_A", "BASIN_TEST_B", "BASIN_TEST_C"):
                    os.environ.pop(name, None)
            read_environment_file(Path(temporary) / "missing.env")  # no error


@unittest.skipUnless(
    __import__("os").environ.get("BASIN_AZURE_LIVE_TEST") == "1",
    "set BASIN_AZURE_LIVE_TEST=1 to write to the real storage container",
)
class LiveAzureUploadTests(unittest.TestCase):
    def test_round_trip_replacement_and_abandoned_upload(self):
        from checkpoint_upload import AzureCheckpointStore, file_digests

        options = {
            "container": "checkpoints",
            "prefix": None,
            "block_mib": 1,
            "concurrency": 4,
        }
        store = AzureCheckpointStore(options, prefix="basin_live_test")
        store.preflight()
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "latest_checkpoint.pt"
            name = store.blob_name(path)
            try:
                torch.manual_seed(1)
                torch.save({"w": torch.randn(700_000)}, path)  # several blocks
                first = store.upload(path, metadata={"step": 1})
                self.assertGreater(first["blocks"], 2)
                back = Path(temporary) / "back.pt"
                result = store.download(name, back)
                self.assertTrue(result["sha256_verified"])
                self.assertEqual(back.read_bytes(), path.read_bytes())
                # An abandoned upload leaves the committed blob as it was.
                torch.save({"w": torch.randn(900_000)}, path)
                cancel = threading.Event()
                cancel.set()
                with self.assertRaises(UploadSuperseded):
                    store.upload(path, cancel=cancel)
                self.assertEqual(store.download(name, back)["sha256"], first["sha256"])
                # The next complete upload replaces it.
                second = store.upload(path, metadata={"step": 2})
                self.assertEqual(second["sha256"], file_digests(path)["sha256"])
                self.assertEqual(store.download(name, back)["sha256"], second["sha256"])
                listed = {blob["name"]: blob for blob in store.blobs()}
                self.assertEqual(listed[name]["metadata"]["step"], "2")
                self.assertEqual(listed[name]["bytes"], path.stat().st_size)
            finally:
                store.container.delete_blob(name)
        self.assertNotIn(name, {blob["name"] for blob in store.blobs()})


if __name__ == "__main__":
    unittest.main()

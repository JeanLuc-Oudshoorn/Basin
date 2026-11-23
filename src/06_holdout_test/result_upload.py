"""Optional copy of every holdout run folder in Azure Blob Storage.

The run folder on the local disk stays the one every later step reads. This
module adds a second copy in a storage container, so that a lost or damaged
disk does not cost the results. Settings (`holdout.result_upload`):

    {"container": "checkpoints", "prefix": null, "block_mib": 16, "concurrency": 8}

`prefix` defaults to the holdout results folder (for example
`results/model/holdout_test`), so a file of a run is the blob
`<prefix>/<run folder>/<file>`. The account is the one training checkpoints go
to: `AZURE_ENDPOINT` and `AZURE_STORAGE_KEY`, read from the environment or from
the `.env` file in the project root (see `checkpoint_upload.py`).

What is checked, and when:

* Before the holdout test starts, a small blob is written, read back and
  deleted. Wrong keys, a missing container or no network stop the run there,
  not after an hour of forecasting.
* When the run is complete, every file of its folder is sent. The service
  verifies an MD5 of every request.
* `test_metrics.json` is sent last, as it is written last on the local disk: a
  run folder in the container that has this file is complete.
* Then the container is listed, and the size and MD5 of every blob are compared
  with the local file.
* Sending a folder again only sends the files that are missing or differ, so
  plots made later (`plotting.py`, `error_distribution.py`) cost one small
  upload. Nothing is ever deleted from the container.

Only the stage's command line uploads. Code that calls `holdout_test.run`
itself (tests, verification scripts) gets no upload unless it passes a store.

Command line:

    python src/06_holdout_test/result_upload.py check
    python src/06_holdout_test/result_upload.py list
    python src/06_holdout_test/result_upload.py upload [--run NAME | --all]
    python src/06_holdout_test/result_upload.py download --run NAME
"""
# ruff: noqa: E402

import argparse
import hashlib
import mimetypes
import sys
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
from pathlib import Path
from time import perf_counter

from loguru import logger

ROOT = Path(__file__).resolve().parents[2]
for stage in ("03_transforming_input_data", "05_model_training", "06_holdout_test"):
    if str(ROOT / "src" / stage) not in sys.path:
        sys.path.insert(0, str(ROOT / "src" / stage))
from checkpoint_upload import AzureCheckpointStore, file_digests, upload_options
from holdout_runs import COMPLETED, holdout_directory, holdout_runs, latest_holdout_run


def first_line(error):
    return f"{type(error).__name__}: {(str(error).splitlines() or [''])[0]}"


class HoldoutResultStore(AzureCheckpointStore):
    setting = "holdout.result_upload"
    label = "Holdout result upload"
    if_unavailable = (
        "The holdout test has not started. Fix the storage settings, or remove "
        "holdout.result_upload to keep the results on this disk only."
    )

    def base(self, name):
        """Blob prefix of a run folder; `name` is empty for the results root."""
        return f"{self.prefix}/{name}" if name else self.prefix

    def files(self, base):
        """The blobs below `base/`, without the entries that stand for folders.

        A storage account with a hierarchical namespace lists every folder as
        an empty blob of its own, marked in its metadata.
        """
        for blob in self.container.list_blobs(
            name_starts_with=base + "/", include=["metadata"]
        ):
            if (blob.metadata or {}).get("hdi_isfolder") != "true":
                yield blob

    def stored(self, base):
        """{blob name: (bytes, MD5)} of every file below `base/` in the container."""
        return {
            blob.name: (blob.size, bytes(blob.content_settings.content_md5 or b""))
            for blob in self.files(base)
        }

    def _put(self, name, data, digests):
        """One small file in one request; the service checks the request's MD5."""
        from azure.storage.blob import ContentSettings

        self.container.get_blob_client(name).upload_blob(
            data,
            length=len(data),
            overwrite=True,
            validate_content=True,
            content_settings=ContentSettings(
                content_type=mimetypes.guess_type(name)[0]
                or "application/octet-stream",
                content_md5=bytearray(digests["md5"]),
            ),
            metadata={
                "sha256": digests["sha256"],
                "bytes": str(len(data)),
                "uploaded_at_utc": datetime.now(timezone.utc).isoformat(),
            },
        )

    def sync(self, folder, name):
        """Make `<prefix>/<name>/` hold every file of `folder`, and verify it.

        Files the container already holds with the same size and MD5 are not
        sent again. Raises if anything is missing or different afterwards.
        """
        folder, base = Path(folder), self.base(name)
        started = perf_counter()
        # The results root (no name) has the run folders below it; they are
        # copied one by one under their own names, not as part of the root.
        entries = folder.rglob("*") if name else folder.iterdir()
        paths = sorted(path for path in entries if path.is_file())
        if not paths:
            raise FileNotFoundError(f"No files to upload in {folder}")
        stored = self.stored(base)
        marker = folder / COMPLETED
        sizes = {path: path.stat().st_size for path in paths}
        small = [p for p in paths if p != marker and sizes[p] <= self.block_bytes]
        large = [p for p in paths if p != marker and sizes[p] > self.block_bytes]

        def copy(path):
            blob = f"{base}/{path.relative_to(folder).as_posix()}"
            size = sizes[path]
            if size > self.block_bytes:
                # Streamed in blocks on a few threads, as checkpoints are.
                data, digests = None, file_digests(path)
            else:
                data = path.read_bytes()
                size = len(data)
                digests = {
                    "sha256": hashlib.sha256(data).hexdigest(),
                    "md5": hashlib.md5(data, usedforsecurity=False).digest(),
                }
            sent = stored.get(blob) != (size, digests["md5"])
            if sent and data is None:
                self.upload(path, name=blob, digests=digests)
            elif sent:
                self._put(blob, data, digests)
            return blob, (size, digests["md5"]), sent

        expected, uploaded, uploaded_bytes = {}, 0, 0
        reported = started

        def record(result):
            nonlocal uploaded, uploaded_bytes, reported
            blob, value, sent = result
            expected[blob] = value
            uploaded += sent
            uploaded_bytes += value[0] * sent
            if perf_counter() - reported >= 30:
                reported = perf_counter()
                logger.info(
                    "HOLDOUT upload {:,}/{:,} files | sent {:,} ({:.0f} MiB) | elapsed {:.0f}s",
                    len(expected),
                    len(paths),
                    uploaded,
                    uploaded_bytes / 2**20,
                    reported - started,
                )

        with ThreadPoolExecutor(
            max_workers=self.concurrency, thread_name_prefix="holdout-upload"
        ) as pool:
            for result in pool.map(copy, small):
                record(result)
        # One at a time: each large file already uses every upload thread.
        for path in large:
            record(copy(path))

        def verify():
            held = self.stored(base)
            wrong = sorted(b for b, value in expected.items() if held.get(b) != value)
            if wrong:
                raise RuntimeError(
                    f"{len(wrong)} of {len(expected)} files are missing or different "
                    f"in the container after the upload, for example {wrong[:3]}"
                )

        verify()
        if marker in sizes:
            # Only a copy that was found complete gets the mark of a complete run.
            record(copy(marker))
            verify()
        seconds = perf_counter() - started
        return {
            "destination": f"{self.describe_container()}/{base}",
            "files": len(expected),
            "bytes": sum(size for size, _ in expected.values()),
            "uploaded_files": uploaded,
            "uploaded_bytes": uploaded_bytes,
            "unchanged_files": len(expected) - uploaded,
            "verified": "size and MD5 of every file compared with the container listing",
            "seconds": seconds,
        }

    def describe_container(self):
        return f"{self.endpoint.rstrip('/')}/{self.container_name}"

    def runs(self):
        """Run folders in the container: files, bytes, and whether complete."""
        found = {}
        for blob in self.files(self.prefix):
            run, _, file = blob.name[len(self.prefix) + 1 :].partition("/")
            if not file:
                run, file = "", run  # results saved before runs had folders
            if file.startswith(".upload_check_"):
                continue
            entry = found.setdefault(run, {"files": 0, "bytes": 0, "complete": False})
            entry["files"] += 1
            entry["bytes"] += blob.size
            entry["complete"] |= file == COMPLETED
        return dict(sorted(found.items()))

    def fetch(self, name, folder):
        """Bring a run folder back; files that are already identical are kept."""
        folder, base = Path(folder), self.base(name)
        stored = self.stored(base)
        if not name:
            # The results root: its own files only, not the run folders below.
            stored = {
                blob: value
                for blob, value in stored.items()
                if "/" not in blob[len(base) + 1 :]
                and not blob[len(base) + 1 :].startswith(".upload_check_")
            }
        if not stored:
            raise FileNotFoundError(
                f"No blobs under {self.describe_container()}/{base}/"
            )
        marker = f"{base}/{COMPLETED}"

        def restore(blob):
            relative = Path(blob[len(base) + 1 :])
            if relative.is_absolute() or ".." in relative.parts:
                raise ValueError(f"Blob name leaves the run folder: {blob}")
            destination = folder / relative
            size, md5 = stored[blob]
            if (
                destination.is_file()
                and destination.stat().st_size == size
                and file_digests(destination)["md5"] == md5
            ):
                return False
            destination.parent.mkdir(parents=True, exist_ok=True)
            self.download(blob, destination)  # verifies the recorded SHA-256
            return True

        with ThreadPoolExecutor(
            max_workers=self.concurrency, thread_name_prefix="holdout-download"
        ) as pool:
            fetched = sum(pool.map(restore, sorted(set(stored) - {marker})))
        # As on upload: the file that marks a complete run arrives last.
        if marker in stored:
            fetched += restore(marker)
        return {
            "folder": str(folder),
            "files": len(stored),
            "downloaded_files": fetched,
            "unchanged_files": len(stored) - fetched,
        }


def make_result_store(config, *, root=ROOT):
    """The configured store, or None when `holdout.result_upload` is off."""
    options = upload_options(
        config.get("holdout", {}), key="result_upload", name="holdout"
    )
    if options is None:
        return None
    directory = holdout_directory(config)
    if directory.is_relative_to(ROOT):
        prefix = directory.relative_to(ROOT).as_posix()
    elif options["prefix"] is None:
        raise ValueError(
            f"Holdout results are kept outside the project ({directory}); "
            "set holdout.result_upload.prefix to name their place in the container"
        )
    else:
        prefix = options["prefix"]
    return HoldoutResultStore(options, prefix=prefix, root=root)


def upload_run(store, directory, name=None):
    """Copy one run folder to the container; `name` defaults to the folder's.

    A failure here loses nothing: the error says where the results are and how
    to send them again.
    """
    directory = Path(directory)
    name = directory.name if name is None else name
    logger.info(
        "HOLDOUT upload of {} to {}/{}",
        directory,
        store.describe_container(),
        store.base(name),
    )
    try:
        summary = store.sync(directory, name)
    except Exception as error:
        raise RuntimeError(
            f"The holdout results are complete in {directory}; only their copy "
            f"to {store.describe_container()}/{store.base(name)} failed "
            f"({first_line(error)}). Send them again with: "
            "python src/06_holdout_test/result_upload.py upload"
            + (f" --run {name}" if name else " --all")
        ) from error
    logger.success(
        "Holdout results copied to {}: {:,} files, {:.0f} MiB ({:,} sent, {:,} already "
        "there), verified, {:.0f}s",
        summary["destination"],
        summary["files"],
        summary["bytes"] / 2**20,
        summary["uploaded_files"],
        summary["unchanged_files"],
        summary["seconds"],
    )
    return summary


def run_name(config, directory):
    """Name of a run in the container; empty for results in the holdout root."""
    directory = Path(directory)
    return "" if directory == holdout_directory(config) else directory.name


def upload_changed_run(config, directory):
    """Bring the container's copy of a run up to date, if copies are kept."""
    store = make_result_store(config)
    if store is None:
        return None
    return upload_run(store, directory, run_name(config, directory))


def main():
    from transform import load_config

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=("check", "list", "upload", "download"))
    parser.add_argument("--config", default=str(ROOT / "configs/config.json"))
    parser.add_argument(
        "--run",
        help="holdout run folder name; upload defaults to the most recent completed run",
    )
    parser.add_argument("--all", action="store_true", help="upload every completed run")
    args = parser.parse_args()
    config = load_config(args.config)
    store = make_result_store(config)
    if store is None:
        raise SystemExit("holdout.result_upload is not configured")
    if args.action == "check":
        store.preflight()
    elif args.action == "list":
        for name, entry in store.runs().items():
            print(
                f"{name or '(results root)'}  files={entry['files']:,}  "
                f"MiB={entry['bytes'] / 2**20:,.1f}  "
                f"{'complete' if entry['complete'] else 'INCOMPLETE'}"
            )
    elif args.action == "upload":
        if args.all and args.run:
            parser.error("--all and --run exclude each other")
        directories = (
            holdout_runs(config) if args.all else [latest_holdout_run(config, args.run)]
        )
        for directory in directories:
            upload_run(store, directory, run_name(config, directory))
    else:
        if not args.run:
            parser.error("--run is required for download")
        print(store.fetch(args.run, holdout_directory(config) / args.run))


if __name__ == "__main__":
    main()

"""Optional copy of every training checkpoint in Azure Blob Storage.

The checkpoint on the local disk stays the one training resumes from. This
module adds a second copy in a storage container, so that a lost or damaged
disk does not cost the run. Settings (`training.checkpoint_upload`):

    {"container": "checkpoints", "prefix": null, "block_mib": 16, "concurrency": 4}

`prefix` defaults to the training output folder (for example
`results/model/training`), so the blob is `<prefix>/latest_checkpoint.pt`.
The account is read from two environment variables, or from the `.env` file in
the project root when they are not set:

    AZURE_ENDPOINT      https://<account>.blob.core.windows.net/
    AZURE_STORAGE_KEY   the account key

What is checked, and when:

* Before training starts, `preflight` writes, reads back and deletes a small
  blob. Wrong keys, a missing container or no network stop the run there.
* Every block of an upload carries an MD5 that the service verifies.
* A blob replaces the previous one only when all blocks have arrived (the
  block list is committed in one step). An interrupted upload leaves the
  previous copy untouched.
* After the commit, the stored size and whole-file MD5 are read back and
  compared with the local file.
* The upload is fed the checkpoint's bytes while they are written to disk. It
  never reads the file back, and it cannot hold the local write up for long:
  a stalled network gives up the upload after two minutes, and the next
  checkpoint abandons an upload that is still running.

Command line (inside the training environment):

    python src/05_model_training/checkpoint_upload.py check
    python src/05_model_training/checkpoint_upload.py list
    python src/05_model_training/checkpoint_upload.py download --to restored.pt
"""

import argparse
import base64
import hashlib
import os
import sys
from collections import deque
from concurrent.futures import ThreadPoolExecutor, wait
from datetime import datetime, timezone
from pathlib import Path
from time import perf_counter
from uuid import uuid4

from loguru import logger

ROOT = Path(__file__).resolve().parents[2]
ENDPOINT_VARIABLE = "AZURE_ENDPOINT"
KEY_VARIABLE = "AZURE_STORAGE_KEY"
OPTIONS = {"container", "prefix", "block_mib", "concurrency", "enabled"}


class UploadSuperseded(Exception):
    """A newer checkpoint is waiting; the unfinished upload was abandoned."""


def read_environment_file(path):
    """`KEY=value` lines of a .env file; existing environment variables win."""
    path = Path(path)
    if not path.is_file():
        return
    for line in path.read_text(encoding="utf-8-sig").splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        name, _, value = line.partition("=")
        name = name.strip().removeprefix("export ").strip()
        value = value.strip()
        if len(value) >= 2 and value[0] == value[-1] and value[0] in "\"'":
            value = value[1:-1]
        os.environ.setdefault(name, value)


def upload_options(training):
    """Validated `training.checkpoint_upload`; None when uploads are off."""
    options = training.get("checkpoint_upload")
    if options is None:
        return None
    if not isinstance(options, dict):
        raise ValueError("training.checkpoint_upload must be an object or null")
    unknown = set(options) - OPTIONS
    if unknown:
        raise ValueError(
            f"Unknown training.checkpoint_upload fields: {sorted(unknown)}"
        )
    if not isinstance(options.get("enabled", True), bool):
        raise ValueError("training.checkpoint_upload.enabled must be boolean")
    if not options.get("enabled", True):
        return None
    container = options.get("container", "checkpoints")
    prefix = options.get("prefix")
    if not isinstance(container, str) or not container.strip():
        raise ValueError("training.checkpoint_upload.container must be a name")
    if prefix is not None and (not isinstance(prefix, str) or not prefix.strip("/")):
        raise ValueError("training.checkpoint_upload.prefix must be a path or null")
    result = {"container": container, "prefix": prefix}
    for name, default, maximum in (("block_mib", 16, 4000), ("concurrency", 4, 32)):
        value = options.get(name, default)
        if (
            isinstance(value, bool)
            or not isinstance(value, int)
            or not (1 <= value <= maximum)
        ):
            raise ValueError(
                f"training.checkpoint_upload.{name} must be an integer from 1 to {maximum}"
            )
        result[name] = value
    return result


def file_digests(path, chunk=8 * 2**20):
    sha256, md5 = hashlib.sha256(), hashlib.md5(usedforsecurity=False)
    with open(path, "rb") as stream:
        while data := stream.read(chunk):
            sha256.update(data)
            md5.update(data)
    return {"sha256": sha256.hexdigest(), "md5": md5.digest()}


class BlobUpload:
    """One checkpoint on its way to the container, block by block.

    `write` takes the bytes in file order and uploads them in blocks on a few
    threads; `commit` makes them the blob, in one step, and verifies it. Until
    then the container still holds the previous checkpoint.
    """

    # A block that makes no progress for this long means the network is gone.
    # Give up the upload rather than hold the local checkpoint back any longer.
    STALL_SECONDS = 120

    def __init__(self, store, name, cancel):
        self.store, self.name, self.cancel = store, name, cancel
        self.blob = store.container.get_blob_client(name)
        self.pool = ThreadPoolExecutor(
            max_workers=store.concurrency, thread_name_prefix="checkpoint-upload"
        )
        self.prefix = uuid4().hex[:12]  # block ids of one upload never mix
        self.blocks, self.pending = [], deque()
        self.buffer = bytearray()
        self.size = 0
        self.started = perf_counter()

    def _check(self):
        if self.cancel is not None and self.cancel.is_set():
            raise UploadSuperseded(self.name)

    def _settle(self, limit):
        """Wait until at most `limit` blocks are in flight; surface their errors."""
        while len(self.pending) > limit:
            future = self.pending[0]
            deadline = perf_counter() + self.STALL_SECONDS
            while not future.done():
                self._check()
                if perf_counter() > deadline:
                    raise TimeoutError(
                        f"No upload progress for {self.STALL_SECONDS} s; giving up on {self.name}"
                    )
                wait([future], timeout=0.5)
            self.pending.popleft().result()

    def _stage(self, data):
        self._settle(self.store.concurrency - 1)
        block_id = base64.b64encode(
            f"{self.prefix}-{len(self.blocks):08d}".encode()
        ).decode()
        self.blocks.append(block_id)
        self.pending.append(
            self.pool.submit(
                self.blob.stage_block,
                block_id,
                data,
                length=len(data),
                validate_content=True,
            )
        )

    def write(self, data):
        self._check()
        self.buffer += data
        self.size += len(data)
        size = self.store.block_bytes
        while len(self.buffer) >= size:
            self._stage(bytes(self.buffer[:size]))
            del self.buffer[:size]

    def commit(self, digests, size, metadata=None):
        """Make the uploaded blocks the blob and check what the container holds."""
        from azure.storage.blob import BlobBlock, ContentSettings

        if self.size != size:
            raise RuntimeError(
                f"Upload of {self.name} received {self.size} bytes, the file has {size}"
            )
        if self.buffer:
            self._stage(bytes(self.buffer))
            self.buffer.clear()
        self._settle(0)
        self._check()
        tags = {
            "sha256": digests["sha256"],
            "bytes": str(size),
            "uploaded_at_utc": datetime.now(timezone.utc).isoformat(),
            **{str(k): str(v) for k, v in (metadata or {}).items()},
        }
        self.blob.commit_block_list(
            [BlobBlock(block_id=block_id) for block_id in self.blocks],
            content_settings=ContentSettings(
                content_type="application/octet-stream",
                content_md5=bytearray(digests["md5"]),
            ),
            metadata=tags,
        )
        properties = self.blob.get_blob_properties()
        stored_md5 = properties.content_settings.content_md5
        if properties.size != size or bytes(stored_md5 or b"") != digests["md5"]:
            raise RuntimeError(
                f"Uploaded checkpoint {self.name} does not match the local file: "
                f"size {properties.size} vs {size}"
            )
        seconds = perf_counter() - self.started
        return {
            "blob": self.name,
            "bytes": size,
            "blocks": len(self.blocks),
            "seconds": seconds,
            "mib_per_second": size / 2**20 / max(seconds, 1e-9),
            "sha256": digests["sha256"],
        }

    def close(self):
        """Release the upload threads; blocks that were not committed expire."""
        for future in self.pending:
            future.cancel()
        self.pending.clear()
        self.buffer.clear()
        self.pool.shutdown(wait=False, cancel_futures=True)


class AzureCheckpointStore:
    def __init__(self, options, *, prefix, root=ROOT):
        try:
            from azure.storage.blob import BlobServiceClient
        except ImportError as error:
            raise RuntimeError(
                "training.checkpoint_upload needs the azure-storage-blob package: "
                "pip install 'azure-storage-blob>=12.19,<13' in the training environment"
            ) from error
        read_environment_file(Path(root) / ".env")
        endpoint, key = os.environ.get(ENDPOINT_VARIABLE), os.environ.get(KEY_VARIABLE)
        if not endpoint or not key:
            raise RuntimeError(
                f"training.checkpoint_upload needs {ENDPOINT_VARIABLE} and "
                f"{KEY_VARIABLE} in the environment or in {Path(root) / '.env'}"
            )
        self.container_name = options["container"]
        self.prefix = (options["prefix"] or prefix).strip("/")
        self.block_bytes = options["block_mib"] * 2**20
        self.concurrency = options["concurrency"]
        self.endpoint = endpoint
        service = BlobServiceClient(
            account_url=endpoint,
            credential=key,
            retry_total=8,
            retry_backoff_factor=1.0,
            connection_timeout=20,
            read_timeout=300,
        )
        self.container = service.get_container_client(self.container_name)

    def describe(self):
        return f"{self.endpoint.rstrip('/')}/{self.container_name}/{self.prefix}"

    def blob_name(self, path):
        return f"{self.prefix}/{Path(path).name}"

    def preflight(self):
        """Write, read back and delete a small blob; raise if any step fails."""
        started = perf_counter()
        name = f"{self.prefix}/.upload_check_{uuid4().hex}"
        payload = os.urandom(256 * 1024)
        try:
            if not self.container.exists():
                raise RuntimeError(
                    f"Container {self.container_name!r} does not exist in the storage account"
                )
            blob = self.container.get_blob_client(name)
            blob.upload_blob(payload, overwrite=True, validate_content=True)
            returned = blob.download_blob().readall()
            blob.delete_blob()
        except Exception as error:
            raise RuntimeError(
                f"Checkpoint upload check failed for {self.describe()}: "
                f"{type(error).__name__}: {str(error).splitlines()[0] if str(error) else ''}. "
                "Training has not started. Fix the storage settings, or remove "
                "training.checkpoint_upload to train without a cloud copy."
            ) from error
        if returned != payload:
            raise RuntimeError(
                f"Checkpoint upload check for {self.describe()} read back different bytes"
            )
        logger.info(
            "Checkpoint upload check passed: wrote, read back and deleted a test "
            "blob in {} ({:.1f}s)",
            self.describe(),
            perf_counter() - started,
        )

    def begin(self, path, *, cancel=None):
        """Start an upload that is fed the file's bytes while they are written."""
        return BlobUpload(self, self.blob_name(path), cancel)

    def upload(self, path, *, digests=None, metadata=None, cancel=None):
        """Replace `<prefix>/<file name>` with a file on disk; verify the result.

        Training does not use this for its checkpoints: reading a multi-GB file
        back from a busy hard disk can take longer than writing it did (5 MB/s
        was measured here). The trainer feeds `begin()` instead.
        """
        path = Path(path)
        digests = digests or file_digests(path)
        session = self.begin(path, cancel=cancel)
        try:
            with path.open("rb") as stream:
                while data := stream.read(self.block_bytes):
                    session.write(data)
            return session.commit(digests, path.stat().st_size, metadata)
        finally:
            session.close()

    def blobs(self):
        return [
            {
                "name": blob.name,
                "bytes": blob.size,
                "modified_utc": blob.last_modified.isoformat(),
                "metadata": blob.metadata or {},
            }
            for blob in self.container.list_blobs(
                name_starts_with=self.prefix + "/", include=["metadata"]
            )
        ]

    def download(self, name, destination):
        """Fetch a blob next to its destination, verify its SHA-256, then move it."""
        destination = Path(destination)
        blob = self.container.get_blob_client(name)
        expected = (blob.get_blob_properties().metadata or {}).get("sha256")
        temporary = destination.with_name(f".{destination.name}.{uuid4().hex}.tmp")
        sha256 = hashlib.sha256()
        try:
            with temporary.open("wb") as stream:
                for chunk in blob.download_blob(
                    max_concurrency=self.concurrency
                ).chunks():
                    sha256.update(chunk)
                    stream.write(chunk)
                stream.flush()
                os.fsync(stream.fileno())
            if expected is not None and sha256.hexdigest() != expected:
                raise RuntimeError(
                    f"Downloaded {name} does not match the SHA-256 recorded at upload"
                )
            os.replace(temporary, destination)
        finally:
            temporary.unlink(missing_ok=True)
        return {
            "blob": name,
            "bytes": destination.stat().st_size,
            "sha256": sha256.hexdigest(),
            "sha256_verified": expected is not None,
        }


def make_checkpoint_store(config, *, root=ROOT):
    """The configured store, or None when `training.checkpoint_upload` is off."""
    options = upload_options(config["training"])
    if options is None:
        return None
    return AzureCheckpointStore(
        options, prefix=Path(config["paths"]["training"]).as_posix(), root=root
    )


def main():
    for directory in (ROOT, ROOT / "src/03_transforming_input_data"):
        sys.path.insert(0, str(directory))
    from transform import load_config

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=("check", "list", "download"))
    parser.add_argument("--config", default="configs/config.json")
    parser.add_argument("--blob", default=None, help="default: latest_checkpoint.pt")
    parser.add_argument("--to", default=None, help="download destination")
    args = parser.parse_args()
    config = load_config(args.config)
    store = make_checkpoint_store(config)
    if store is None:
        raise SystemExit("training.checkpoint_upload is not configured")
    if args.action == "check":
        store.preflight()
    elif args.action == "list":
        for blob in store.blobs():
            print(blob)
    else:
        name = args.blob or f"{store.prefix}/latest_checkpoint.pt"
        if not args.to:
            parser.error("--to is required for download")
        print(store.download(name, ROOT / args.to))


if __name__ == "__main__":
    main()

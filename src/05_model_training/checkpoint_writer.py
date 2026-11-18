"""Write checkpoints, and their optional cloud copies, without stopping training.

The trainer copies the state it wants to keep into host memory (a few seconds)
and hands a `write` function to `CheckpointWriter.submit`. One background
thread then writes the file. When a store is configured, the same bytes are
uploaded while they are written, so the file is never read back from disk.
Training continues meanwhile.

Rules that keep this safe:

* One job at a time. A new checkpoint waits for the previous file to be
  complete, so two writers never touch the same path and at most one copy of
  the training state waits in host memory.
* The file comes first. An upload that fails, stalls or is abandoned never
  stops the file from being completed.
* A failed local write is fatal: it is raised in the training thread at the
  next `submit`, `finish` or `raise_if_failed`.
* A failed upload is not fatal once training is under way, because the local
  checkpoint exists; it is logged as an error and the next checkpoint tries
  again. `finish(require_upload=True)` turns it into an error, which the
  trainer uses for the first checkpoint of a run.
* An upload still running when the next checkpoint is due is abandoned (the
  previous cloud copy stays in place) rather than delaying training.
"""

import contextvars
import threading
from time import perf_counter

from loguru import logger
from training_log import emit_training_event


class CheckpointWriter:
    def __init__(self, store=None):
        self.store = store
        self.uploads = 0
        self.upload_failures = 0
        self.uploads_abandoned = 0
        self.last_upload = None
        self._thread = None
        self._write_error = None
        self._upload_error = None
        self._abandon_upload = threading.Event()

    def submit(self, write, path):
        """Run `write(upload)` in the background.

        `write` writes the checkpoint to `path`, feeding `upload` (None without
        a store) the same bytes, and returns a dict with `path`, `bytes`,
        `digests` and `metadata`.
        """
        self.finish(abandon_upload=True)
        context = contextvars.copy_context()  # keeps the training journal active
        self._thread = threading.Thread(
            target=context.run,
            args=(self._run, write, path),
            name="checkpoint-writer",
            daemon=False,
        )
        self._thread.start()

    def _run(self, write, path):
        from checkpoint_upload import UploadSuperseded

        self._upload_error = None
        session = error = None
        if self.store is not None:
            try:
                session = self.store.begin(path, cancel=self._abandon_upload)
            except Exception as failure:
                error = failure
        try:
            try:
                result = write(session)
            except BaseException as failure:  # reported in the training thread
                self._write_error = failure
                return
            if self.store is None:
                return
            upload = None
            digests = result.get("digests") or {}
            error = error or digests.get("mirror_error")
            if error is None:
                try:
                    if session.size:
                        upload = session.commit(
                            digests, result["bytes"], result.get("metadata")
                        )
                    else:
                        # The file was written before this job started (host
                        # memory was short): upload it from disk.
                        upload = self.store.upload(
                            result["path"],
                            digests=digests,
                            metadata=result.get("metadata"),
                            cancel=self._abandon_upload,
                        )
                except Exception as failure:
                    error = failure
        finally:
            if session is not None:
                session.close()
        if isinstance(error, UploadSuperseded):
            self.uploads_abandoned += 1
            logger.warning(
                "Checkpoint upload abandoned: the next checkpoint was due before "
                "it finished. The previous cloud copy is unchanged and the local "
                "checkpoint is complete. Abandoned so far: {}",
                self.uploads_abandoned,
            )
            emit_training_event(
                "checkpoint_upload_abandoned", abandoned=self.uploads_abandoned
            )
        elif error is not None:
            self.upload_failures += 1
            self._upload_error = error
            logger.error(
                "CHECKPOINT UPLOAD FAILED ({} so far): {}: {}. The local checkpoint {} "
                "is complete; training continues and the next checkpoint will be "
                "uploaded again.",
                self.upload_failures,
                type(error).__name__,
                str(error).splitlines()[0] if str(error) else "",
                result["path"],
            )
            emit_training_event(
                "checkpoint_upload_failed",
                error_type=type(error).__name__,
                error=str(error),
                upload_failures=self.upload_failures,
            )
        else:
            self.uploads += 1
            self.last_upload = upload
            logger.info(
                "CHECKPOINT UPLOADED blob={} bytes={:,} seconds={:.1f} ({:.1f} MiB/s) "
                "size and MD5 verified in the container",
                upload["blob"],
                upload["bytes"],
                upload["seconds"],
                upload["mib_per_second"],
            )
            emit_training_event("checkpoint_uploaded", **upload, uploads=self.uploads)

    @property
    def busy(self):
        return self._thread is not None and self._thread.is_alive()

    def raise_if_failed(self):
        """Raise a background write failure in the calling (training) thread."""
        if self._write_error is not None:
            error, self._write_error = self._write_error, None
            raise RuntimeError(
                "Writing the training checkpoint failed in the background"
            ) from error

    def finish(self, *, require_upload=False, abandon_upload=False):
        """Wait until nothing is being written or uploaded.

        `abandon_upload`: the next checkpoint is due, so an upload still in
        progress gives way (the file itself is always completed).
        `require_upload`: raise if the upload of the finished job failed.
        """
        thread = self._thread
        if thread is not None:
            if abandon_upload and thread.is_alive():
                self._abandon_upload.set()
                started = perf_counter()
                thread.join()
                waited = perf_counter() - started
                if waited > 5:
                    logger.warning(
                        "Training waited {:.0f}s for the previous checkpoint to "
                        "finish writing before taking the next one",
                        waited,
                    )
            else:
                thread.join()
            self._thread = None
            self._abandon_upload.clear()
        self.raise_if_failed()
        if require_upload and self.store is not None and self._upload_error is not None:
            error, self._upload_error = self._upload_error, None
            raise RuntimeError(
                "The first checkpoint of this run could not be uploaded to "
                f"{self.store.describe()}. The local checkpoint is complete. Fix "
                "the storage settings or remove training.checkpoint_upload, then "
                "start again; training resumes from the local checkpoint."
            ) from error

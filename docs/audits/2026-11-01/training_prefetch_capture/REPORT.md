# CUDA graph capture versus background memory pinning

## Root cause established by the second failure

The failing run is `logs/20261101T121708_290800Z_train_cca41b20/`.
At 13:26:09.929 the main thread recovered from a graph-capture failure. At
13:26:10.177 optimizer update 5 completed. The next iteration then retrieved
the prefetch worker's failed future and stopped. Its exception came from
`Tensor.pin_memory()` in `performance.prepare_update`, running on the
`patch-preparation` background thread.

Pinning CPU memory invokes CUDA host-memory allocation/registration. It was
running concurrently with `make_graphed_callables` on the training thread.
CUDA capture restrictions apply across threads in the same process. This race
explains both the invalidated capture on the main thread and the forbidden CUDA
operation in the worker. The earlier fallback recovered only the main thread;
it could not repair a future that already contained the worker exception.

The previous synthetic probe prepared its batches synchronously. It therefore
missed the interaction with the actual trainer's concurrent prefetching. This
audit supersedes the earlier report's statement that the trigger was unknown.

Reference: [PyTorch 2.12 CUDA graph constraints](https://docs.pytorch.org/docs/2.12/notes/cuda.html#constraints)
prohibit unrelated CUDA work on other threads while capture is underway.

## Fix

`performance.prepared_updates` now submits only CPU preparation to its worker.
After consuming the completed future, the training thread pins the prepared
tensors before yielding the update to the training loop. Graph capture occurs
later on that same thread, so these operations cannot overlap.

CPU tokenization, packing, and preparation of the following update still overlap
GPU training. Both prefetch and graph replay stay enabled. Pinned transfers stay
enabled, and at most one future update is prefetched. Pinning itself now runs
synchronously on the consumer; this removes its overlap with GPU computation.
No model architecture, loss, batch membership, gradient accumulation, checkpoint,
or validation settings change. Direct `prepare_update(pin_memory=True)` remains
supported and uses the same pinning helper. The capture fallback remains available
for other recoverable capture errors.

## Verification

A regression test intercepts memory pinning and verifies that every pin occurs
on the consuming thread while batch packing occurs on the worker. It compares
features, labels, label aliases and point counts with synchronous preparation.
The old implementation would fail this test because it pins on the
worker. All ten training-performance tests passed, including existing gradient,
batching, validation and prefetch parity checks.

The full-model probe now uses `prepared_updates(prefetch=True, pin_memory=True)`.
With `--real-data`, it loads the existing processed/indexed corpus and uses the
first 40 batch groups from the same epoch-1 split, seed and length-bucketing logic
as `train.py`, with the configured 307M model, microbatch limits, compiled
attention, graph capture and optimizer. It also warms inference before training.
This is a bounded diagnostic, not a full validation sweep or a training epoch;
its model weights are discarded and production checkpoints are not overwritten.

Run from the activated WSL environment:

```bash
MAX_JOBS=1 TORCHINDUCTOR_COMPILE_THREADS=1 python scripts/check_training_graph_shapes.py \
  --graphs --real-data \
  --output docs/audits/2026-11-01/training_prefetch_capture/real_307m.json
```

The full-model run passed on the RTX 5070, PyTorch 2.12.0+cu130:

| Check | Result |
|---|---:|
| Parameters | 307,307,808 |
| Completed optimizer updates | 40 |
| Actual training tables processed | 320 |
| Successful graph captures | 643 |
| Graph replays | 1,354 |
| Capture failures | 0 |
| Graph replay disabled by fallback | No |
| Peak PyTorch allocated GPU memory | 6.762 GiB |

The first five shapes match the failed production run, including the previously
failing `[4,4,16,1024]` profile at update 5. The probe completed many subsequent
profiles and 38 cache evictions. Memory-budget fallbacks still operated normally
(20 layer/profile decisions), and replay remained active through update 40.
`real_307m.json` contains every update's metrics. The probe reseeds auxiliary
forecast-lead sampling for reproducibility; it is an execution check, not an
exact reproduction of production losses. Its timings exclude preparation and
are not a new full-epoch speed benchmark.

All ten training-performance tests and all applicable pre-commit hooks passed.
`verification.json` records the source hashes and log directory. Logs and the
configuration snapshot are saved under `logs/20261101T132200Z_training_prefetch_capture_fix/`.

## Restart

Run `src/05_model_training/train.py` normally. No upstream pipeline stage needs
to be repeated, and no configuration change is required. The interrupted run
did not finish its first training epoch, so it has no epoch-1 checkpoint to resume
from; the normal epoch-0 baseline and epoch-1 training will run again.

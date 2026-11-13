# Training graph-capture failure and recovery

Follow-up: the second production traceback established a race between background
memory pinning and graph capture. The fallback below was insufficient to prevent
the worker's exception. See the [root-cause fix and concurrent-prefetch tests](../training_prefetch_capture/REPORT.md).

## Observed failure

Source: `logs/20261101T113713_910754Z_train_3a26db41/events.log`.
The 307,307,808-parameter model completed epoch-0 validation on all 116,703
training tables. Validation took 377.920 seconds (6 minutes 18 seconds), with
normalized pinball loss 0.5743151218597056. Four optimizer updates then completed.
The next update failed while preparing a CUDA graph for cross-variable attention
with shape `[4, 4, 16, 1024]`: four tables, four variables, sixteen padded patches,
and embedding width 1024.

The final exception was `torch.AcceleratorError`, reporting
`cudaErrorStreamCaptureInvalidated`. The preceding exception originated in an
Inductor-generated Triton kernel launch inside `make_graphed_callables`' forward
capture. This was not an allocator out-of-memory exception, a bad table, a
nonfinite training loss, or a validation failure. The SM/autotuning notice did not
stop training. The stream-mismatch warning is relevant to capture compatibility,
but the log does not establish it as the underlying cause.

A bounded synthetic run of the same 307M model exercised validation before
training and the recorded shape sequence: `(B,V,T)` of `(4,1,48)`, `(4,2,80)`,
`(4,3,48)`, `(4,3,48)`, `(4,4,16)`, followed by repeated profiles and the longest
context. All ten updates completed, with 119 successful captures. Thus the
original failure is intermittent; the exact underlying PyTorch/driver trigger
has not been isolated. This report does not claim that the failing kernel itself
has been repaired. `before.json` records that investigation run.

## Change

`src/05_model_training/training_graphs.py` now treats recoverable capture failures
as a failure of the optional graph optimization. Capture still happens before
the real microbatch forward/backward and before its optimizer update.

When capture fails, the code restores the caller's CUDA stream and synchronizes
the device to check that CUDA remains usable. Stream restoration is necessary
because PyTorch 2.12's graph context can raise in `capture_end` before exiting its
stream context. After a successful health check, Basin clears the graph cache,
disables capture for the remainder of this training run, logs a warning, and
executes the current microbatch using the existing compiled attention path.
It does not skip the microbatch or update, reset model weights, change gradient
accumulation, or change the training objective. Compiler/capture warmup continues
to preserve RNG state.

The graph summary includes `capture_failures` and `disabled_reason`, which flow
into the existing training metrics and JSONL logs. Capture is not repeatedly
retried on subsequent microbatches. A new process or an explicitly rebuilt graph
manager can try capture again. The default configuration continues to enable
training graph replay; `training_cuda_graphs: false` remains the manual override.

Error classification uses the first exception line: PyTorch appends generic
device-assert debugging advice to capture errors, which is not evidence that an
assertion occurred. Illegal-memory-access and device-assert errors are not
suppressed. Unrelated runtime errors propagate, and failure of the CUDA health
check also propagates rather than continuing in a damaged CUDA context.

Model parameters, checkpoint identity, data artifacts, microbatch limits, full
epoch validation, checkpointing and inference graph replay remain unchanged.
Fallback gives up training replay's additional speedup for the rest of the run;
the existing compiled attention optimization remains active.

## Verification

The CUDA regression deliberately attempts a forbidden device synchronization
inside a real capture, reproducing `cudaErrorStreamCaptureInvalidated` after an
earlier layer has already been captured. It checks continuation through multiple
accumulated AdamW updates, exact loss/gradient/weight equality against compiled
training without replay, RNG preservation, stream restoration, and no repeated
capture attempts. Existing graph tests cover normal replay, checkpoint
recomputation, profile eviction and optimizer/model reload.

The bounded full-model probe is `scripts/check_training_graph_shapes.py`. It uses
synthetic tables and production preparation, optimizer and update code; it writes
audit artifacts only. `--inject-capture-failure` invalidates capture specifically
at `[4,4,16,1024]`. It does not run a full epoch or replace production checkpoints.

Results on the RTX 5070 with PyTorch 2.12.0+cu130:

- Four graph regression tests passed, including two real CUDA integration tests
  (55.639 seconds for the final test run).
- Seven epoch-training regression tests passed, including accumulation,
  validation cadence, scheduler/early stopping and epoch checkpoint behavior.
- The full 307M model completed ten updates with validation-shaped inference
  before training. A real capture invalidation was injected at update 5 with
  shape `[4,4,16,1024]`. That update completed and all five later updates also
  completed. Exactly one capture failure was recorded, no subsequent captures
  were attempted, and the cache was empty after recovery. Peak PyTorch allocation
  was 6.827 GiB, below the existing 8 GiB cap. See `recovered_307m.json`.
- All applicable pre-commit hooks passed, including Ruff and Bandit.

The full-model probe establishes continued execution and finite metrics across
the shape transitions; exact gradient/weight/RNG equality is checked by the small
CUDA regression rather than claimed for every BF16 full-model profile. These
checks do not constitute a full training epoch or repair an unhealthy CUDA
context. `verification.json` records results and source hashes. Logs are saved in
`logs/20261101T121452Z_training_graph_capture_fix/`.

## Restarting this interrupted run

Run `src/05_model_training/train.py` normally. No transform or embedding rerun is
needed. This run failed before its first completed training epoch, so its four
updates were not saved by the epoch checkpoint policy. With no completed epoch
checkpoint in `results/model/training`, training starts again from epoch 1 and
computes epoch-0 validation again. Existing completed-epoch checkpoints, when
present and compatible, still resume through the normal `resume: "auto"` policy.

The initial training ETA includes first-use compilation/capture overhead and is
not a reliable estimate of warmed-up epoch throughput. This change does not
establish a new full-epoch runtime.

Reference: [PyTorch CUDA graph documentation](https://docs.pytorch.org/docs/main/notes/cuda.html#cuda-graphs)
describes capture restrictions and the requirement for static graph execution.

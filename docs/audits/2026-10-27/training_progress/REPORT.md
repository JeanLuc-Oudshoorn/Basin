# Epoch progress, expected tokenization, and GPU batching

## Counts for the current data and configuration

The active run's `training_plan.json` records **97,045 training windows**,
microbatch size **1**, accumulation **8**, and **12,131 optimizer updates per
epoch** (`ceil(97045 / 8)`). The final update processes the five remaining windows.
The final-token validation set contains 97,045 tables and is evaluated after the
training pass. Whole test tables remain excluded. There is no fixed number of
epochs, because three non-improving validation epochs determine stopping.

The previous logs contained total updates only at epoch start. Subsequent update
records exposed the global update count without the denominator. The trainer now
logs an INFO progress line after every optimizer update, including epoch number,
current and total epoch updates, percentage, global update, windows consumed,
loss and estimated remaining training-pass time. Detailed input IDs, shapes and
source paths remain available at DEBUG and in JSONL.

The JSONL update event and loss-history row now contain `epoch_update`,
`epoch_total_updates`, `epoch_windows_completed`, `epoch_total_windows`,
`epoch_progress_percent`, `epoch_elapsed_seconds`, `epoch_updates_per_second`,
`epoch_eta_seconds`, and an explicit ETA scope. The plan event includes the
updates-per-epoch denominator. Counters reset at epoch boundaries while the
global update count continues increasing. Validation remains epoch-based.

ETA uses elapsed time per completed update. It is approximate, especially early
in a pass or with varied context/variable counts. It excludes the following full
validation sweep, checkpoint/forecast work and all future epochs. It is not an
estimate of when convergence will occur. Full validation now logs table progress
and its own ETA every 1,000 tables and on completion.

## Viewing a worker that is already running

```powershell
uv run python scripts/watch_training_progress.py --follow
```

Without `--follow`, print a single snapshot. Optional `--journal PATH` selects a
particular run; otherwise the newest training JSONL under `paths.logs` is used.
The watcher reads at most a bounded 2 MiB tail and the small training plan. It
imports no Torch, loads no tensors, uses no CUDA and changes no training state.
Stopping the watcher does not stop training. It handles incomplete lines while
the journal is being written and can derive epoch progress from older journals
that lack the new progress fields. It labels terminal errors/completion instead
of continuing to display an active ETA for a stopped worker.

The tool was exercised on the user's actual journal. One snapshot showed epoch
1, update **323/12,131 (2.7%)**, with an estimated **04:47:43** left in the training
pass. This was a point-in-time estimate, not a benchmark or a promised duration.
The run subsequently ended with `KeyboardInterrupt` at update **389/12,131
(3.2%)**; the watcher correctly reports that terminal state. No training worker
was stopped or restarted by this change. New inline progress logs apply when a
worker next starts; the watcher supports existing workers immediately.

## What the token and embedding messages mean

Training still needs to normalize each current window, divide it into patches,
and construct masks and positions. It then projects those features using the
model's current learned weights and backpropagates through that projection.
These per-microbatch `tokenize_next` calls are expected. They do not mean that
the removed up-front whole-corpus uniqueness audit has returned.

`Next-patch tokens=110 padded=112` means 110 input patch positions in the current
window, padded to a kernel-compatible length of 112. It is not optimizer update
110 or an index into the entire dataset. Real patches contain 16 observations;
padding is masked. For a combined batch, the shorter examples can also have
masked time padding up to the longest example's patch count.

`embedding_shape[B,V,T,D]=[1,5,64,1024]` describes a tensor, not four embedding
values:

| Axis | Meaning in the example |
|---|---|
| B = 1 | One dataset window in the current microbatch |
| V = 5 | Five variables: the configured outcome plus four historical covariates |
| T = 64 | Kernel-padded patch positions; the real count may be smaller |
| D = 1024 | Learned feature coordinates for each variable's patch |

There is one 1,024-dimensional embedding for each variable at each real patch
position. The last axis is not a forecast horizon or a count of raw observations.
With BF16, this particular input embedding tensor occupies about 0.625 MiB.
Overall VRAM additionally holds the 447.5M model's parameters, gradients, AdamW
state and layer activations. New microbatch logs name the axes and include both
the maximum real patch count and the padded kernel length. The same scalar loss
is reused for logging instead of performing a second device-to-host loss read.

## GPU utilization and next tuning experiment

The current accumulation loop runs eight microbatches **sequentially**. It gives
one optimizer update an effective batch of eight windows, but each forward still
has B=1. Accumulation alone does not increase the parallel work of an individual
GPU forward. Small launches, CPU dispatch/transfers, irregular sequence sizes,
the recurrent kernel and checkpoint recomputation can all affect throughput.
The reported 40% activity and 5.2 GB VRAM use alone do not identify which dominates.
No profiler trace or competing training benchmark was launched on the user's GPU.

The next bounded configuration experiment is:

```json
"batch_size": 2,
"gradient_accumulation_steps": 4
```

This retains a nominal eight windows per update while allowing two compatible
windows to execute together. It is a suggestion for the next run, not a measured
speedup or a guarantee of fitting the largest contexts/variable sets. The existing
batcher supports it; counts can be smaller at bucket/remainder boundaries. Compare
windows or observed-label points per second and peak VRAM, not utilization alone.
The active run's hyperparameters were not changed. A 5.2 GB observation does not
prove that every larger ragged batch will fit.

## Verification

Tests cover percentages, rate/ETA arithmetic, epoch-local reset, complete and
remainder updates, finite startup behavior, partial JSONL lines, older journals,
validation ETA and terminal interrupt/completion handling. Existing training and
accumulation tests remain in place. Complete test output is saved under
`logs/20261027_training_progress_tests.log`; static checks cover the changed
modules and watcher script.

Final result: **130 tests, OK, six CUDA-dependent skips**. Ruff lint and format
checks and Bandit passed on changed modules and the watcher. The watcher was
also run directly against the real training journal, including its subsequent
terminal `KeyboardInterrupt` event. No GPU benchmark was run concurrently with
the user's training session.

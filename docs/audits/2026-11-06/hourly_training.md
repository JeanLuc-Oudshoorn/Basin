# Hourly full-table training implementation audit

User request: support million-table runs with hourly recoverable checkpoints,
hourly Basin-only performance checks, configurable run duration, and full tables
available for gradient training. Implemented in the ordinary pipeline/config.

## Protocol decisions

- Default `data.validation_policy`: `sampled_training_origins`.
- Full source tables feed causal next-patch training, including the former
  final four-token validation tail. Labels remain masked for missing/future
  positions. Covariates remain historical and time matched.
- Preserve the user's configured whole-table holdout fraction (currently 0.03
  in the working configuration); do not silently restore an earlier 10% value.
- Select 25,000 fixed training sources that each support five complete
  four-token forecast requests. This requires at least 160 points with 16-point
  patches. Full-table training also admits shorter sources down to its 48-point
  minimum; these are excluded only from the five-origin monitoring panel.
- Use first eligible, nearest 25/50/75% and final eligible origins, preserving
  two initial context tokens and four full future tokens. Up to 125,000
  requests per sweep, with actual counts explicit for smaller corpora.
- Weights remain 80/16/3/1. Validation reports weighted normalized pinball,
  RMSE, MAE and MASE. All comparisons are Basin-only, without foundation model
  downloads. The panel is explicitly in sample, not generalization validation.

## Runtime and persistence

`fit_epochs` dispatches sampled-origin descriptors to `timed_training.fit_timed`.
Legacy reserved-tail records retain the previous epoch path.

`checkpoint_interval_seconds` and `validation_interval_seconds` default to
3600 active-training seconds. Active time includes per-update preparation and
execution, excluding housekeeping and validation. `max_run_hours` is a separate
per-invocation elapsed-loop soft limit, including loading resumable state and
validation/checkpoint overhead, excluding preceding model/data initialization.
It defaults to null. Finish an ongoing safe unit of work before exiting, so a
long check/write can exceed this time. Early stopping and explicit update/epoch
limits still apply. Hourly scheduler/early-stopping patience is labeled in
validation checks, through supported `patience_checks` keys.

Checkpoint format `rwkv_timed_checkpoint_v1` extends the existing atomic save:

- Current and best validated weights, complete optimizer and scheduler states.
- Torch CPU/CUDA, Python and NumPy RNG states.
- Current epoch, next batch-group offset, completed epoch count and deterministic
  order identity. Accumulation boundaries are respected.
- Active-clock/check cadence state, pending validation flag and processed totals.
- Unfinished epoch loss/mass/window counters and cumulative embedding gradient count.
- Recent bounded update history and full aggregate validation history.

At validation time save before evaluation, then save again after best-selection,
plateau scheduling and early-stopping updates. If interrupted during evaluation,
the saved pending flag reruns that same panel before any new training update.
Validation restores all RNG states so it does not change stochastic training.
At a planned time limit, save the final current optimizer state before returning.
Changing only the per-invocation duration does not invalidate resume identity.
Corpus/order, validation selection/cadence, microbatch/batch settings, model,
normalization policy and environment assumptions are checked or documented.

The latest optimizer trajectory remains in `latest_checkpoint.pt`. The separate
`model.pt` exports the best positively trained validated weights; a short run
before the first trained validation exports current weights rather than the
untrained baseline. Exported epoch/step provenance is explicit and consumed by
holdout evaluation. No existing production checkpoint was deleted or converted.

## Memory and logging

Validation source selection uses a top-k heap and generates origins only for
chosen sources. Descriptors contain no context tensors. Read each source once
per validation sweep, generate its five windows, and retain tokenization only
in bounded compatible batches. Transformation stores metadata-only evaluation
entries instead of duplicating tail/context tensors for every raw table; embedding
preparation supports those entries. Exact token inventory streams these windows
when explicitly requested; normal training keeps its cached/deferred policy.

Only the latest 1,000 update records are kept in resumable memory/history JSON.
The timestamped training JSONL keeps the full stream. Cohort arrays remain in
split artifacts instead of being duplicated as a giant initial JSONL event.
Console logs include epoch update counts and save/validation/exit countdowns.
Each validation check writes per-origin metrics to Parquet and aggregate metrics
to `validation_history.json`. Temporary data/checkpoints continue to use the
configured project/output locations on E.

## Verification

`tests/test_timed_training.py` uses a deterministic clock and a dropout-enabled
forecaster with AdamW and accumulation. It compares interrupted and uninterrupted
runs, checking exact final weights, AdamW moments/group options, scheduler state,
Torch/Python/NumPy RNG state, cursor and training losses. It also verifies
pending-check recovery, active-vs-wall clocks and duration expiry during a check.

`tests/test_rolling_training_validation.py` exercises transformation, embedding
index preparation, source splitting, exact inventory, streaming validation and
two real optimizer updates through the normal trainer dispatcher. It verifies
full labels including final observations, short-source support, panel separation
from the holdout, five full origins, source-order-independent selection and
serial/batched metric agreement. Its CPU temporal block is the existing test
stand-in; the real patch embedding, variate attention, multi-lead head, loss,
optimizer and training loop execute. Native CUDA kernel code was not changed.

The available GPU was occupied by an existing workload (approximately 6.3 GiB
in use, 25% utilization when checked), so no competing full-model GPU test was
launched. Final verification: **279 tests completed successfully, with 12
CUDA-only skips**. All pre-commit syntax/JSON/whitespace/Ruff/Bandit checks
passed. A controlled clock exercises hourly boundaries without waiting an hour;
this was not a million-table memory/throughput stress test. Raw training artifact
storage and corpus metadata still grow with the number of tables, even though
validation contexts and update history are bounded.

See [the operational guide](../../model_training/HOURLY_TRAINING.md) for settings,
clock semantics, commands and migration from reserved-tail artifacts.

# Automatic training JSONL journals

The reference is `logs/rwkv7_experiment_20260816_222400_pid17610.jsonl`.
Its event envelope is retained, with time-series metric semantics. The reference
file is preserved.

## Automatic behavior

Every invocation of the two public training entry points creates a new file under
`paths.logs`, normally `logs/`, without requiring a flag:

```text
rwkv7_train_YYYYMMDD_HHMMSS_microsecondsZ_pidPID_random.jsonl
rwkv7_capacity_test_YYYYMMDD_HHMMSS_microsecondsZ_pidPID_random.jsonl
```

Both CLI entry points and direct `train(config)` / `capacity_test.run(config)`
calls are covered. The journal is opened before model or data loading, recording
missing artifacts and other startup failures. Invalid CLI arguments or an unreadable
configuration prevent entry into a training run; no loaded configuration exists
yet to choose a log directory in that case. Standalone optimizer unit tests and
third-party backend diagnostics do not start experiment journals.

Existing Loguru logs, CSV/JSON metrics and checkpoints remain. Each CLI run's
Loguru directory also contains `training_jsonl.json`, pointing to its journal.
Exclusive creation, microseconds, PID and a random suffix prevent overwrites.

## Event stream

| Event | Contents |
|---|---|
| `run_start` | Input config, runtime, loss definition and identity |
| `worker_launch` | Launcher is handing off to WSL |
| `worker_start` | Actual worker runtime under the same identity |
| `model_initialized` | Measured parameters, architecture, device and dtype |
| `training_plan` | Resolved update budget, rate plan and dataset counts |
| `training_update` | Capacity update, batch sources, loss, gradient and rate |
| `metrics` | Batch training metrics or exhaustive capacity metrics |
| `training_complete` | Completed steps, target points and checkpoint path |
| `run_complete` | Successful terminal record and final/best validation summary |
| `run_error` | Failed/interrupted terminal record and exception details |

Each record has event, experiment, training mode, run ID, UTC/Unix timestamps,
elapsed seconds, global step and run step. Global/run steps both count updates
within this invocation, including capacity runs initialized from a checkpoint.
No unavailable cumulative training history is invented; the start record identifies
this scope.

Each line is one UTF-8 JSON object, flushed immediately. Serialization uses
`allow_nan=False`; nonfinite diagnostic values become JSON null. Caught failures
write `run_error` and re-raise the exception. Hard OS termination or power loss
cannot guarantee a terminal record; already flushed progress remains available.

## Metric mappings

Regular training's `train_loss` is the update batch's normalized pinball loss,
identified by `train_loss_scope="update_batch"`. That entry point does not perform
validation, so validation loss and best-validation fields are null.

Capacity `metrics` follows exhaustive evaluation: training loss is the full training
set (`train_loss_scope="full_training_set"`); `batch_train_loss` separately records
the update loss. Validation loss covers every final-patch validation target.
The initial sweep is step zero. RMSE, normalized RMSE, table counts, actual and next
learning rates, and cumulative observed target points are included. Best validation
loss/step track strict improvements; final loss remains the actual final value.

Perplexity is null: exponentiating continuous pinball loss would not yield language
model perplexity. Power/energy fields preserve the reference's names but are null,
with `training_energy_backend="not_monitored"`. Unsupported metrics are explained
in the start record; no unmeasured energy values are invented.

CUDA memory is measured from the active PyTorch allocator, as text and exact bytes:
allocated, reserved and peak. Process RSS and system-used memory come from the new
psutil dependency, installed in native and WSL environments and recorded in
pyproject.toml / uv.lock. WSL measurements describe the worker process/VM. Failed
telemetry produces null fields and an error message, rather than failing training.

The example's out-of-sample events are not fabricated. Neither training entry point
automatically runs separate unseen-dataset evaluation. Capacity final patches are
monitored validation. The independent forecasting stage retains its existing
metrics and plots.

## Windows/WSL handoff

The launcher creates the journal once and forwards its identity through
BASIN_TRAINING_JSONL, explicitly listed in WSLENV. The worker reconstructs the same
repository-relative log location and appends to that file. The launcher does not
duplicate the worker's terminal record. All writers use append mode, preventing
launcher offsets from overwriting worker events. A successful process exit without
shared-journal completion is an error.

The first real smoke check exposed missing WSLENV forwarding, which created separate
files. This was fixed and a missing-completion guard added. Those diagnostic files
are retained; final evidence identifies the later verified single-file runs.
Microsoft documents [WSLENV forwarding](https://learn.microsoft.com/en-us/windows/wsl/filesystems#share-environment-variables-between-windows-and-wsl-with-wslenv).

## Verification

Five tests cover unique filenames, timestamps, strict JSON, flushing, best/final
loss tracking, nested identity, failures, direct Python entry points, cross-process
append behavior, and rejection of missing worker completion. Actual three-update
301.8M CUDA jobs verify both CLI paths. The capacity smoke evaluates all 82 train
and 82 validation tables at baseline and after every update.

```powershell
uv run python src/05_model_training/train.py --config configs/jsonl_regular_smoke.json
uv run python src/05_model_training/capacity_test.py --config configs/jsonl_capacity_smoke.json
```

Smoke artifacts use results/jsonl_smoke/, preserving earlier experiments.
Verified journal names, metric checks, hashes and test results are in evidence.json.

## Completed smoke evidence

Both real 301.8M-parameter CUDA runs passed, with one shared journal each:

- [Regular training journal](../../../../logs/rwkv7_train_20261025_120332_242373Z_pid31620_45720515.jsonl):
  three updates and three metric records.
- [Capacity journal](../../../../logs/rwkv7_capacity_test_20261025_120453_511737Z_pid35188_b92c19a8.jsonl):
  three updates and four exhaustive metric records, including the initial baseline.
  All 82 training and 82 validation tables are represented at every sweep.

Each file has exactly one run-start and one run-complete record, and a worker-start
record under the same identity. Every logged loss and learning rate matched the
persisted training/evaluation history exactly. Best validation loss and step,
final loss, update counters, observed target-point totals, positive RSS and CUDA
allocation measurements were independently checked.

Five journal tests and two pipeline regression tests passed. Ruff, formatting,
Bandit and uv lock checks passed. The measured filenames and hashes are preserved
in [evidence.json](evidence.json).

# Full epoch validation, early stopping and gradient accumulation

This is the default behavior of the existing `src/05_model_training/train.py`.
It supersedes the earlier per-update validation protocol. The earlier proposal
to stop after 100 updates or sample validation tables was superseded by the user's
request to validate the full set once per epoch and stop after three bad epochs.
No new pipeline entry point or alternative training branch is required.

## Current configuration

The main `configs/config.json` now contains:

```json
{
  "training": {
    "epochs": null,
    "steps": null,
    "batch_size": 1,
    "gradient_accumulation_steps": 8,
    "early_stopping": {
      "patience_epochs": 3,
      "min_delta": 0.0,
      "monitor": "loss"
    },
    "scheduler": {
      "name": "reduce_on_plateau",
      "factor": 0.5,
      "patience_epochs": 1,
      "min_learning_rate": 0.00001
    },
    "checkpoint_every": 1000
  }
}
```

The optimizer remains AdamW with the configured initial learning rate and weight
decay. Architecture, input selection, target/covariate roles, patch size, context
limit, learned embeddings and 10% dataset holdout are unchanged. Training-only
settings do not invalidate the prepared index, so transformation and indexing do
not need to be repeated for this change.

## Meaning of an epoch and an update

One epoch traverses **every eligible training window exactly once**. Whole test
datasets remain excluded from both training and training-time validation. Long
tables can have several windows; an epoch covers all of those windows. The
order is reproducibly reshuffled each epoch using the run seed and epoch number.

`batch_size` is the maximum number of windows in a microbatch. Windows are
bucketed by exact variable-role counts; variable meanings remain local to their
table. Time is padded, labels are masked, and variate attention never crosses
table boundaries. The existing capacity diagnostic and regular trainer now share
the tested batch packing implementation in `batching.py`.

The default uses one window per microbatch to preserve the GPU memory footprint
of the existing model. Eight microbatches contribute to one optimizer update,
normally giving an effective batch of eight windows. `batch_size=2` can combine
two compatible windows in one GPU forward; with eight accumulation steps that
normally yields 16 windows per update. Actual counts can be smaller at bucket
boundaries or at the end of an epoch and are logged explicitly.

For microbatch mean losses `L_i` and observed label counts `n_i`, backpropagate
`L_i * n_i / sum(n_i)` and add the gradients. This matches a point-weighted mean
over the whole accumulation group, even with different window lengths and masked
labels. It does not give a short or mostly missing window the weight of a long,
fully observed window. Zero gradients once before the group, backpropagate each
microbatch, clip the accumulated gradient once, then call AdamW once. No optimizer
step occurs between microbatches. The last incomplete group is also updated, with
its own actual total label count; it is neither dropped nor divided by eight.

Features remain generated lazily per window. Only the current accumulation group's
CPU features are retained, and only one microbatch's computation graph is live
during backward. The large corpus-sized embedding cache is not reintroduced.

## Validation and stopping

There is **no validation sweep before training or after individual updates** in
the default run. Once the epoch's final update finishes, evaluate every non-test
table's reserved final token, using its actual preceding context capped at 2,048
observations. Validation runs without gradients and restores training mode.

Monitor `loss` by default: observed-point-weighted normalized pinball loss over
all configured quantiles and all validation tables. Also record pooled RMSE.
`monitor="rmse"` is supported if explicitly selected; both metrics are minimized.
The full validation set is stable, with no random table sampling.

An improvement is `metric < best_metric - min_delta`. With the default zero
delta, any strict decrease qualifies. The first completed epoch establishes a
baseline checkpoint. An improvement resets the bad-epoch counter. Three subsequent
consecutive epochs without improvement terminate training. For example, if epoch
2 is best and epochs 3–5 fail to improve, stop after epoch 5 and restore epoch 2.
If the model continues improving, it can continue across arbitrarily many epochs.

Both `epochs` and `steps` are null in the default config, so neither imposes a
training limit. Explicit finite values remain available for bounded experiments.
If an explicit step limit ends a partial epoch, perform one final validation
sweep, mark the epoch as partial and select its best checkpoint; do not report
it as a completed epoch. This debug-limit behavior does not affect the unlimited
default. `steps` always counts optimizer updates, not microbatches.

## Learning rate and checkpoints

A cosine schedule requires a known endpoint; it is no longer the default for an
unlimited run. `ReduceLROnPlateau` instead consumes the monitored full validation
metric **once per epoch**. Its configured patience of one permits one bad epoch
and reduces the learning rate on the second bad epoch, leaving another epoch at
the lower rate before three-epoch early stopping. It halves the rate, down to the
configured floor. An improvement resets its plateau count. The controller and
early stopping use the same absolute improvement threshold.

The existing constant and warmup/cosine schedules remain available for explicitly
bounded runs. Their clocks count optimizer updates, not microbatch forwards.
Scheduler snapshots distinguish update counts from validated epoch counts. The
infinity sentinel fields in PyTorch's plateau state are omitted from JSON; a fresh
controller reconstructs them when loading the saved state. A strict JSON and state
restoration test covers this behavior.

Save `best_weights.pt` atomically whenever validation improves. Periodic
`latest_weights.pt` snapshots remain configurable; the default interval is now
1,000 updates to avoid writing a roughly 900 MB model every 25 updates. At the
end of training, restore the best weights, write the normal `model.pt` with
selection metadata, then perform the existing rolling test comparison against
Prophet and Holt-Winters. Test data never drives checkpoint choice, learning rate
or stopping. These are weights checkpoints, not an exact AdamW-resume mechanism;
the scheduler metadata describes the completed training run, while selection
metadata identifies the restored model.

## Logs and audit outputs

Automatic timestamped training JSONL files remain under `paths.logs`.
`metrics` events record optimizer updates and contain no new validation value.
`epoch_end` events contain full validation loss, RMSE, table count, epoch,
optimizer-update count and bad-epoch count. `checkpoint_selected` and
`training_complete` identify the selected epoch and update separately from the
last optimizer update. The final dataset holdout comparison still runs once.

Training-directory outputs include:

- `training_plan.json`: limits, batch size, accumulation, updates per epoch,
  full validation policy and scheduler.
- `loss_history.json` / `.csv`: accumulated update losses, epoch numbers,
  observed labels, actual microbatch/window counts, input IDs, sources, shapes,
  gradient norms, learning rates and durations. Validation values appear only
  on an epoch's final update row; JSONL's separate epoch events are authoritative.
- `validation_history.json`: aggregate results per epoch.
- `validation_epoch_NNNNN.parquet`: complete per-table validation results for
  that epoch. These are separated from the history summary to avoid retaining
  every table from every past epoch in RAM.
- `early_stopping.json`: stop reason, completed epochs/updates, best epoch/update,
  monitored metric, bad-epoch count and confirmation that best weights were restored.
- `learning_rate_scheduler.json`, best/latest model snapshots and normal final
  training/test artifacts.

## Verification

CPU tests compare accumulated gradients against a combined batch with unequal
label counts, check remainder updates and complete per-epoch coverage, verify
deterministic shuffling, prove that validation runs only once per epoch, exercise
three-epoch patience and resets, test explicit partial-epoch limits, compare
restored weights exactly against the best epoch, and check plateau reductions
and scheduler state restoration.

The full-size GPU smoke test uses the normal 447,482,144-parameter trainer with
ten synthetic tables, nine in training/validation and one reserved for testing.
It performs eight optimizer updates across four epochs: each epoch has an
eight-window accumulation group followed by a one-window remainder. There are
four full validation sweeps over nine tables, rather than eight per-update sweeps.
For this stopping-control test only, `min_delta=1e9` deliberately prevents later
epochs from qualifying as improvements; it guarantees a baseline followed by
three bad epochs. This is not evidence that actual validation accuracy plateaued.
The production value is zero.

The smoke test stops after epoch 4, restores epoch 1/update 2, reduces the rate
to 0.00005, and performs the final rolling dataset holdout comparison. Reloading
the selected checkpoint reproduces the test metrics exactly. Measured results
are retained under `gpu_smoke/` and in the run's JSONL. Admission is disabled only
for the synthetic fixture; production admission settings are unchanged.

Run the normal training command without rerunning earlier stages:

```powershell
uv run python src/05_model_training/train.py --config configs/config.json
```

To repeat the bounded GPU control test:

```bash
source scripts/activate_rwkv7.sh
python scripts/verify_training_protocol.py --exercise-early-stopping --audit-dir docs/audits/2026-10-27/epoch_training/gpu_smoke
```

Recorded verification: **121 regression tests, OK, six CUDA-dependent skips**,
with complete output in `logs/20261027_epoch_training_tests.log`. The full-size
GPU control test passed; output is in `logs/20261027_epoch_training_gpu.log`.
JSONL event order and counts were checked: eight update events with no validation
loss, four epoch-end validation events, best-checkpoint selection, then final
holdout evaluation and successful completion. The run journal is
`logs/rwkv7_train_20261027_191047_098985Z_pid365_08bf7870.jsonl`.
The current prepared index still matches the main model/data configuration.
See [verification.json](verification.json) for measured counts, selection and
checkpoint replay checks. Ruff lint/format checks and Bandit passed for changed
training code and the smoke helper.

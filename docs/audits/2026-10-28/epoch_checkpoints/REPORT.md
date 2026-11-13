# Shuffling, full validation, epoch recovery and early stopping

## Confirmed data semantics

Shuffling rearranges independent **training windows/tables and batches**. It does
not permute timesteps, patch contents or variable columns within a window.
Outcome values and covariates retain their time alignment. Length bucketing only
chooses which compatible windows share a batch; it does not sort observations
inside a series. Larger tables may contribute several windows, whose processing
order can be shuffled while each individual window remains chronological.

The 128-table figure in the throughput report was an isolated timing benchmark.
Normal `train.py` calls `fit_epochs`, whose default validation function receives
the entire `validation_records` collection after every epoch. Each admitted
non-test table contributes its final held-out token (16 observations at the
current patch size). Every observed target value in every such token contributes
to the aggregate loss/RMSE; missing labels and padding are excluded. Historical
observations are context rather than additional validation labels. Validation
batch size 16 controls parallelism, not the number of tables evaluated.

On the current processed cohort, this means 97,045 validation tables. The 10,783
whole-table test datasets remain sealed until final training/checkpoint selection,
then receive the separate expanding-context evaluation against Prophet and
Holt-Winters. This change does not sample validation or move test tables into it.

## New defaults

```json
{
  "training": {
    "checkpoint_every": 0,
    "resume": "auto",
    "early_stopping": {
      "patience_epochs": 1,
      "min_delta": 0.0,
      "monitor": "loss"
    }
  }
}
```

The first epoch establishes the best validation loss. Every subsequent epoch
must improve it strictly; equal or higher loss stops the run at that epoch's end.
For example, losses `0.50, 0.45, 0.46` stop after epoch 3 and select epoch 2.
Losses `0.50, 0.50` stop after epoch 2 and select epoch 1. The monitored loss is
the same full-validation normalized pinball loss as before. `min_delta` remains
configurable. The code fallback also now uses patience 1 when omitted.

The learning-rate scheduler still observes each full validation result. With
early-stopping patience 1, training does not continue for another epoch to test
a lower learning rate after a plateau. The existing ReduceLROnPlateau patience
1 normally requires two bad epochs to reduce the learning rate, so the new early
stopping rule can end the run before that reduction. This follows the requested
first-non-improving-epoch stopping rule; scheduler settings were not changed.

## Recovery checkpoint versus activation checkpointing

Activation checkpointing recomputes activations to reduce GPU memory. It does not
save training progress to disk. Previous `latest_weights.pt`/`best_weights.pt`
snapshots contained weights only and could not restore optimizer/scheduler
history or exact continuation. They cannot be retroactively made complete.

After each **completed epoch and full validation**, the new trainer writes:

`results/447m_model/training/latest_checkpoint.pt`

The location follows `paths.training`. It is the latest completed epoch, which
can differ from the epoch with the best validation score. Its contents include:

- Current model and learned embedding weights.
- AdamW moments, optimizer parameter groups and step counters.
- Learning-rate scheduler state and schedule description.
- Completed epoch/global update, processed-window and label counters.
- Best validation score/epoch/update and non-improvement counter.
- Training and validation histories.
- PyTorch CPU/CUDA, Python and NumPy random-generator states.
- Best model weights, so recovery does not depend on an external best file that
  could have been replaced during an interrupted checkpoint operation.
- Identity information binding the checkpoint to the model, data windows,
  validation split, source checksums and optimization settings.

When the current epoch is best, tensor-storage deduplication avoids writing a
second copy of the same weights inside the checkpoint. Full optimizer recovery
makes these artifacts larger than weight-only files: roughly 2.5 GiB of raw BF16
model/moment tensors for this 447.5M model, plus metadata; an older distinct best
model adds approximately 0.83 GiB. CPU/GPU optimizer implementations can affect
the precise file size. Only the latest full checkpoint is retained. During a new
write, both the old checkpoint and temporary replacement occupy disk space.
Files are written beside the destination on E:, without staging them on C:.

The write flushes and fsyncs a temporary file, then atomically replaces the
destination. A failed write leaves the prior full checkpoint usable. Best-weight
snapshots are also written atomically. Text/JSONL logs record checkpoint paths,
epoch/update, file size and save/load time.

If training or validation fails during epoch 4, restarting `train.py` loads the
checkpoint after epoch 3 and repeats epoch 4. Partial-epoch gradients are not
retained. A failure before the first complete epoch has no epoch checkpoint to
resume. Explicit debug step limits that terminate an incomplete epoch do not
replace the last complete-epoch checkpoint. `checkpoint_every` remains available
for optional weight-only update snapshots; it defaults to 0 because epoch
checkpoints now provide the normal recovery mechanism.

## Restart behavior

Run the normal command again:

```powershell
uv run python src/05_model_training/train.py
```

`training.resume: "auto"` loads `paths.training/latest_checkpoint.pt` when present,
or starts from the normal initialization when absent. `training.resume: false`
disables loading for a fresh experiment. A string containing a specific checkpoint
path loads that file; relative paths resolve from the repository root. An explicit
missing path, invalid checkpoint or incompatible training setup raises an error
instead of silently resetting training.

Restoring a full checkpoint reconstructs the matching best-weight snapshot before
continuing. An already-completed or early-stopped checkpoint does not silently
train additional epochs on retry; it restores the best weights so final export
and test evaluation can be retried. Epoch/step limits can be extended for constant
or plateau scheduling. A changed warmup/cosine budget is rejected. Early-stopping
patience may be changed deliberately; its best score and accumulated bad-epoch
count are retained and evaluated against the new patience.

Changing the corpus/model, optimizer policy, batch/accumulation/shuffle settings,
holdout split, monitored metric or minimum improvement is rejected for exact
continuation. Use a new experiment (`resume: false`, preferably with a new output
directory) for those changes. Console verbosity, validation parallelism, prefetch
and the activation checkpoint budget can be adjusted without changing the
statistical continuation identity. CUDA RNG restoration requires the same CUDA
device count. Histories retain prior runs; every launch gets its own timestamped
JSONL journal, with an explicit `checkpoint_resumed` event.

`best_weights.pt` remains the selection snapshot; final `model.pt` remains the
forecasting artifact containing the selected model. The new full checkpoint is
for training continuation and is not the forecasting file.

## Verification and audit trail

`tests/test_epoch_checkpoints.py` deliberately interrupts epoch 2 after an update,
resumes from epoch 1, and compares with uninterrupted training through epoch 3.
The test includes stochastic dropout and consumes Python, NumPy and PyTorch RNGs.
Loss histories, selected weights and post-run random samples match exactly on CPU.
It also verifies optimizer/scheduler recovery, best-weight independence from an
external overwritten snapshot, rejection of incompatible optimization, atomic
write failure preservation and no extra updates when restarting a stopped run.

The actual CUDA RWKV backend is exercised by `scripts/verify_epoch_resume.py`.
A reduced 869,024-parameter model trains, is interrupted during epoch 2 and
resumes in a newly constructed model/optimizer. Its final selected weights match
the uninterrupted run within BF16 tolerance; both complete four updates/two
epochs. AdamW moments are restored to CUDA, optimizer steps reach four, and the
scheduler completes two validation epochs. See [gpu_verification.json](gpu_verification.json).
The interrupted smoke run's failure log is intentional.

The full Windows regression suite ran **140 tests: 134 passed and six CUDA-only
tests skipped**. Ruff formatting/lint and Bandit checks passed for the changed
training modules. Full-suite output is saved in
`logs/20261028_epoch_checkpoints_full_tests.log`.

Detailed logs are under `logs/20261028_epoch_resume_gpu_verification.log` and
`logs/20261028_checkpoint_recovery_tests.log`. The verification writes only its
isolated audit directory; it does not modify production model checkpoints.

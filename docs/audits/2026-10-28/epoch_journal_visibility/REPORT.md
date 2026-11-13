# Live epoch metrics and journal durability

The reported `training_jsonl.json` is a pointer written by `run_context` under
each ordinary run-log directory. Its `path` identifies the actual timestamped
`.jsonl` journal directly under `logs/`. It was never intended to contain metrics.
Windows launcher and WSL worker point to the same journal.

Inspection of the active run
`rwkv7_train_20261028_191123_881636Z_pid11292_362703f9` found a growing journal
of approximately 78 MB during epoch 2, with per-update metrics being flushed.
Epoch 1 had already completed full validation on 97,045 tables:

- Training normalized pinball loss: 0.24540895392027304.
- Validation normalized pinball loss: 0.14027899091869356.
- Validation RMSE in original data units, pooled across tables: 25498.35637063438.
- Global update: 12,161.

These results are also in
`results/447m_model/training/validation_history.json`.
The resumable `latest_checkpoint.pt` existed (2,701,259,357 bytes).

## Changes

`training_log.TrainingLog.emit` continues to flush every JSONL event. It now also
calls `os.fsync` after `epoch_end`, `epoch_checkpoint_saved`, `training_complete`,
`run_complete` and `run_error`. This requests OS-level persistence at important
boundaries without adding a disk-sync call to every optimizer update. Epoch
metrics are emitted before writing the resumable checkpoint; checkpoint success
is a separate event, so the log does not imply that an unfinished save succeeded.

New pointer files explicitly describe their purpose and include the journal's
filename, a relative path and the `epoch_end` event name. Existing pointer keys
remain compatible. The two pointer files for the active launcher/worker were
also annotated; their journal and model files were not changed.

The running worker already flushes each event and records epoch metrics and
checkpoints. Its loaded Python code is not replaced mid-run. The added fsync
behavior applies to future workers; no restart is needed to read current metrics.

## Verification

`tests/test_training_log.py` checks that epoch metrics are readable while the
run remains open, epoch and checkpoint events invoke fsync, and ordinary update
metrics avoid that extra sync. Existing tests cover direct training calls,
nested contexts, errors and a worker appending to the launcher's journal.
Output: `logs/20261028_epoch_journal_tests.log`. Ruff formatting/lint passed.

To inspect the current epoch summaries in PowerShell:

```powershell
Get-Content results/447m_model/training/validation_history.json
```

To inspect the live journal, open the timestamped `.jsonl` file specified by
the pointer's `journal_filename` under `logs/`. Most per-update rows have
`validation_loss=null` because full validation runs only at epoch end. Look for
`event="epoch_end"` to find each completed epoch's validation metrics.

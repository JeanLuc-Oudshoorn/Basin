# Overnight training interruption: November 12, 2026

Read-only investigation. Training was not restarted, and production checkpoints,
configuration and logs were not changed. No GPU work was launched.

## Cause supported by the system logs

Windows System event 1074 at **November 11, 23:29:08 CET** records
`MoUsoCoreWorker.exe` requesting a planned restart for an operating-system
service pack. The final training JSONL metric is at **23:29:09 CET**. The
training event log also ends during an ordinary microbatch at that time;
its status remains `running`, without a normal completion or failure record.

A further event at 23:32:32 records TrustedInstaller requesting a planned
operating-system upgrade restart. These events strongly identify Windows
Update as the interruption mechanism, rather than model early stopping.
Windows later records a StartMenuExperienceHost-initiated power-off at
November 12, 06:05:03; that is separate from the initial training interruption.
The logs do not establish who initiated that later power-off.

Raw local timestamps and messages are preserved in
[windows_shutdown_events.json](windows_shutdown_events.json).

## Last executed work versus recoverable work

The final complete JSONL metric reports:

- Epoch 2, global update **799,651**.
- One complete epoch plus **815,598 / 5,339,817 tables (15.27%)** of epoch 2.
- About **53 active training hours** across the run.
- Finite loss 0.626528 and gradient norm 0.136719 for that particular update.
- Zero OOM retries for that update; live VRAM 7.54 GiB, reserved 9.68 GiB,
  peak 9.26 GiB. These values do not diagnose every earlier update.

`results/model/training/latest_checkpoint.pt` was read successfully using CPU
memory mapping. It was written **November 11, 22:31:17 CET**, and contains:

- Epoch 2, global update **789,251**, batch cursor **95,555 / 693,696**.
- One completed epoch and **735,491** tables processed in epoch 2.
- **6,075,308** table presentations cumulatively; epoch 2 revisits training tables.
- **52.0016 active training hours**.
- Model, optimizer state, FP32 master weights, FP32 moments, RNG/cursor state
  and validation history; learning rate **0.00005**.
- Best validation step **789,251**, zero unsuccessful validation checks since
  that best, no pending validation and no saved stop reason.

The difference is **10,400 updates / 80,107 table presentations / approximately
one active training hour** that will have to be repeated on resume. The latest
successful checkpoint also corresponds to the latest best validation result.
Reading the serialized checkpoint does not constitute a full GPU resume test.

Metadata: [checkpoint_summary.json](checkpoint_summary.json).

## Was the model still improving?

Yes. Metrics below average the five origins within each table, then take the
median across the same 25,000 reserved validation tables. Original mean metrics
remain unchanged and continue to include extreme cases.

| Check | Active hours | Mean loss | Mean MASE | Table median loss | Table median MASE |
| --- | --- | --- | --- | --- | --- |
| 5 | 20 | 1.727377 | 15.414036 | 0.233567 | 1.739124 |
| 6 | 24 | 1.729597 | 15.468687 | 0.234457 | 1.751336 |
| 7 | 28 | 1.704223 | 15.172654 | 0.205654 | 1.530941 |
| 8 | 32 | 1.700020 | 15.103548 | 0.201204 | 1.483276 |
| 9 | 36 | 1.697573 | 15.036591 | 0.199478 | 1.464704 |
| 10 | 40 | 1.695616 | 15.042680 | 0.197090 | 1.439456 |
| 11 | 44 | 1.696417 | 15.030325 | 0.197985 | 1.442999 |
| 12 | 48 | 1.693294 | 15.011646 | 0.194885 | 1.416653 |
| 13 | 52 | 1.692246 | 14.972116 | 0.193871 | 1.410062 |

Checks 5 to 13 show approximately **17.0% lower table median loss** and
**18.9% lower table median MASE**. Even the final interval, check 12 to 13,
improves both medians. Progress is gradual and not monotonic at every check,
but the earlier plateau did not persist throughout this continuation.

The latest full validation completed **November 11, 22:29:23 CET**. Its RMSE
is 22,092.42 and MAE 104.93; raw-unit averages still fluctuate substantially.
No completed validation exists for the final unsaved hour, so its quality
cannot be inferred from isolated training losses.

The November 10 startup log records `FP32MasterAdamW`, migration from the older
parameter-dtype optimizer, and FP32 update precision. The checkpoint confirms
the saved FP32 masters/moments. The learning rate also halved at check 6.
Both factors changed around the improvement, so these observations alone
cannot separate the benefits of precision correction, LR reduction and more
training data.

All 14 validation summaries are in [validation_progress.json](validation_progress.json).
They were generated with `scripts/analyze_validation_plateau.py`'s
`validation_summary` function, without rerunning model predictions.

## Recovery

The current configuration specifies `training.resume="auto"`. Running the
normal training entry point will select the saved checkpoint, restore its
optimizer and cursor, and repeat the unsaved hour:

```powershell
uv run python .\src\05_model_training\train.py
```

The investigation did not change Windows Update settings or restart training.

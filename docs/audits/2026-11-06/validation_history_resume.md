# Validation history and within-epoch progress

Hourly training already stored an ordered `validation_history` in the resumable
checkpoint, wrote the cumulative `validation_history.json`, and resumed at a
deterministically reconstructed batch cursor rather than restarting an epoch.

This update adds unique timestamped per-check aggregate JSON snapshots alongside
the per-origin Parquet files. Their names include check number, optimizer step,
UTC time and an ID, so retries cannot replace an earlier result. The cumulative
history is atomically replaced after appending each check and remains the plotting
index for the resumed trajectory. Completed attempts not committed to a checkpoint
before a crash remain available individually, while the resumed trajectory reruns
that pending validation as before. Intentional fresh runs have separate snapshots
but reset their trajectory's cumulative history.

Rows now include completion UTC time, active training seconds, batch cursor,
epoch windows and epoch supervised target points. Checkpoints' `loop_state` now
also contains `epoch_target_points_processed`, incremented at completed optimizer
updates and reset at epoch rollover. The already existing epoch window count,
cursor, epoch loss, cumulative point/window counts, optimizer/scheduler states and
RNG states continue to be restored together. Target points count supervised valid
labels, including repeated leads/windows; they do not count unique raw rows.

An older hourly checkpoint without the new diagnostic point counter can still
resume exactly from its existing cursor. That counter stays unknown (`null`)
for the remainder of that epoch and starts counting from zero at the next epoch.

Verification: six timed-training tests pass, including a stop within an epoch
with gradient accumulation followed by exact model, optimizer and RNG comparisons
against uninterrupted training. A new regression test pauses after check one,
resumes through checks two and three, verifies ordered steps/times and aggregate
losses, and verifies original snapshot bytes are unchanged. Per-epoch target point
counts are checked both before and after resume. Pre-commit checks pass. No
production training, checkpoint replacement or GPU experiment was run.

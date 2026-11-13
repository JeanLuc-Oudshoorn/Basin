# Learning-rate scheduling (2026-10-25)

## What changed

AdamW was already present in both training entry points. AdamW adapts parameter
updates using moment estimates; it does not by itself lower its configured global
learning rate as training progresses. The prior capacity experiment manually used
two constant rates in two separate runs. A shared configurable scheduler now
controls that global rate automatically.

`src/05_model_training/learning_rate.py` builds a PyTorch `LambdaLR` around the
actual AdamW optimizer. Both `train.py` and `capacity_test.py` use it. For the FP32
master-weight capacity optimizer, it wraps `MasterAdamW.optimizer`, so it changes
the rate of the actual FP32 master updates. It is not attached to a separate or
unused optimizer.

## Selected policy

The default is `warmup_cosine`: linear warmup followed by smooth cosine decay to a
configured minimum. This is a **step-based schedule**, not a validation-triggered
plateau detector. It continuously reduces the rate after warmup without waiting
for a manually selected plateau or a second process. It does not inspect training
or validation targets to decide when to reduce the rate.

```json
"scheduler": {
  "name": "warmup_cosine",
  "warmup_fraction": 0.05,
  "min_learning_rate": 0.00001
}
```

The peak is `training.learning_rate`. Regular default training uses peak `1e-4`
and minimum `1e-5`. The new 20-epoch capacity configuration uses peak `3e-4` and
minimum `3e-5`. Warmup covers the first 5% of planned updates, rounded up; the
last optimizer update uses exactly the configured minimum.

![Configured 20-epoch schedule](planned_schedule.png)

For the current admitted dataset, the 20-epoch capacity plan has **300 updates**:
15 warmup updates, peak `3e-4` at update 15, and minimum `3e-5` at update 300.
Its first warmup update is `2e-5`. The minimum is the **decay floor**, not a lower
bound during warmup; early warmup rates can be below it.

An explicit `warmup_steps` integer may replace `warmup_fraction`; specifying both
is an error. Unknown names/fields, nonfinite or out-of-range rates, and invalid
warmup lengths fail before the first update. An explicit warmup length must leave
at least one decay update. Fractional warmup is clipped to leave a decay update
for short runs. A one-update run uses the peak, since both endpoints cannot occur
in one update. No warmup starts at zero, so the first update is not wasted.

Configs without `training.scheduler`, including the archived capacity/refinement
configs, retain their original **constant** rate. Explicit `{"name":"constant"}`
is also supported. The previous experiment's configuration and result artifacts
have not been rewritten to imply a scheduler was used historically.

## Update order and logs

The scheduler is initialized before the first update to set its initial rate.
Each loop captures that rate, calls `optimizer.step()`, then `scheduler.step()` to
prepare the next update. This avoids skipping the first scheduled value. The
schedule is indexed by optimizer updates, not table reads, token positions,
epochs, or validation sweeps. Full validation in the capacity test never advances
the scheduler.

Regular loss history now includes `learning_rate` and `next_learning_rate`.
Capacity evaluation history contains the same fields, alongside full training and
validation losses after every update. The initial baseline has no used rate
(`null`) and records the planned first rate as `next_learning_rate`. Capacity
learning curves now have a third panel showing the rate actually used.

Both entry points save `learning_rate_scheduler.json` with resolved plan, completed
update count, next rates and scheduler state. The final model checkpoint embeds
this snapshot too. This provides an audit trail; it does **not** add exact training
resume, because existing checkpoints still omit Adam moments and FP32 master
weights. The unit tests separately verify that scheduler plus optimizer state can
be restored consistently when both states are provided.

## Running

Normal pipeline training now reads the scheduler from `configs/config.json`:

```powershell
uv run python src/05_model_training/train.py
```

For a complete scheduled capacity experiment with full training and all 82
validation tables evaluated after every update:

```powershell
uv run python src/05_model_training/capacity_test.py --config configs/capacity_scheduled.json
```

This starts from the original one-epoch validated checkpoint and writes separate
results under `results/capacity_300m_scheduled/training/`. It is a new experiment,
not an exact resume of the previous two-phase run. A full new 20-epoch accuracy
experiment has **not** been run as part of this scheduler implementation.

The capacity entry point now also honors an optional `training.steps` cap within
the configured epoch budget. Its schedule is resolved against that actual cap,
so short smoke runs still exercise both endpoints. It reports planned epochs,
completed epochs and fractional epoch progress separately; partial epochs are not
reported as fully completed.

## Verification

Four scheduler unit tests verify rates actually used by AdamW, warmup peak,
monotonic cosine decay, exact floor on the last update, post-budget clamping,
no-warmup and one-update runs, legacy constant-rate behavior, state restoration
and invalid settings. A capacity unit test additionally checks actual FP32 master
weight changes against the sum of scheduled rates.

Two isolated CUDA smoke configurations exercise the real 301.8M model:

- `configs/scheduler_smoke.json`: 8 regular optimizer updates, 2 warmup updates,
  peak `1e-4`, decay floor `1e-5`.
- `configs/capacity_scheduler_smoke.json`: 12 FP32-master optimizer updates,
  3 warmup updates, peak `3e-4`, decay floor `3e-5`, with exhaustive train and
  validation evaluation at the baseline and after every update.

Smoke outputs use `results/scheduler_smoke/`; previous checkpoints are preserved.
Short smoke curves test numerical execution and scheduler integration, not an
accuracy improvement claim. Scheduling does not remove the input ambiguity found
in the earlier capacity experiment.

Final measured smoke rates, coverage checks and file hashes are recorded in
`evidence.json`. Full logs are `logs/scheduler_regular_smoke.log`,
`logs/scheduler_capacity_smoke.log`, `logs/scheduler_cpu_tests.log` and
`logs/scheduler_capacity_cuda_tests.log`.

## Primary references

- [PyTorch AdamW](https://docs.pytorch.org/docs/stable/generated/torch.optim.AdamW.html)
- [PyTorch LambdaLR](https://docs.pytorch.org/docs/stable/generated/torch.optim.lr_scheduler.LambdaLR.html)
- [PyTorch optimizer/scheduler update ordering](https://docs.pytorch.org/docs/stable/optim.html#how-to-adjust-learning-rate)
- [PyTorch cosine annealing](https://docs.pytorch.org/docs/stable/generated/torch.optim.lr_scheduler.CosineAnnealingLR.html)

## Completed CUDA smoke evidence

Both 301.8M model runs passed. The regular run performed **8 updates**, using
`5e-5` initially, peak `1e-4` on update 2, and exactly `1e-5` on update 8. The
FP32-master capacity run performed **12 updates**, using `1e-4` initially, peak
`3e-4` on update 3, and exactly `3e-5` on update 12. Every decay segment was
monotonic.

The capacity smoke completed **13 exhaustive sweeps**, including its baseline,
each covering all 82 training and all 82 validation tables. Its persisted
table-level metrics have **2,132 rows**, with no source omitted from any sweep.
It correctly reports 0 completed epochs and 0.8 epoch progress from the explicit
12-update cap, rather than claiming a complete epoch.

Scheduler state was verified inside both saved model checkpoints and the JSON
snapshots. **4 scheduler tests passed on CPU and 4 capacity tests passed on CUDA**;
the native capacity test run passed 3 and skipped the CUDA-only case. Ruff,
formatting, Bandit and uv lock checks passed.

![Capacity smoke loss and actual learning rates](smoke_learning_curves.png)

See [evidence.json](evidence.json) for all actual rates and source/artifact hashes.
The smoke loss curves are deliberately not used to claim that this schedule
improves accuracy; the 20-epoch scheduled configuration is prepared for that
separate controlled experiment.

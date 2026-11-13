# Optimizer gradient-staging OOM, 2026-11-11

## Failure and cause

The 09:11:41 failure was inside `FP32MasterAdamW.step`, at
`parameter.grad.detach().float()`. Forward/backward and gradient clipping had
already completed; AdamW had not yet been called for this update. The failure
was therefore outside the activation-OOM microbatch-retry mechanism.

The previous master optimizer converted every model gradient to FP32 before
updating anything. For 500,056,352 parameters, the upper bound for those temporary
copies is **1.863 GiB**. Master weights and Adam moments were already resident.
The failure message reports **9.36 GiB allocated**, about **910 MiB cached but
unallocated**, and a process allocator limit of **10.26 GiB**. Cached free space
is not necessarily available in a suitable block for a new request. The allocation
layout/fragmentation explanation is consistent with the reported 16-MiB failure;
the historical allocator snapshot needed to prove the precise layout was not
captured. Fresh-process success does not establish long-run allocation safety.

The impossible `17179869184.00 GiB` process-usage value is a reporting anomaly;
it cannot represent actual usage on this 11.94-GiB device. The other reported
allocator values and the traceback identify the actionable issue.

For allocator behavior, see [PyTorch 2.12 CUDA memory management](https://docs.pytorch.org/docs/2.12/notes/cuda.html#memory-management).

## Change

[`master_optimizer.py`](../../src/05_model_training/master_optimizer.py) now
converts and updates bounded buckets, default **64 MiB** of FP32 gradients.
A parameter larger than the budget is handled alone; this checkpoint's largest
individual gradient is only 16 MiB. Staging is cleared after each bucket,
including when an exception occurs.

AdamW updates are independent across parameters. Each parameter with a gradient
appears in exactly one bucket and receives one update/step increment. Parameters
without gradients receive no decay or step increment. The public parameter
groups remain intact, and clipping/gradient accumulation still happens once at
the logical-update boundary. The scheduler advances once per logical update.

Master weights and first/second moments remain FP32. The checkpoint format and
optimizer identity are unchanged, so existing FP32 checkpoints load directly.
No precision downgrade or optimizer reset is involved. Any optimizer OOM remains
fatal: an earlier bucket may already have updated weights, so a blind in-process
retry or emergency save could be incorrect. Restart from a completed checkpoint.

This bounds temporary allocation; it does not remove the persistent memory cost
of FP32 moments or promise larger activation microbatches or faster throughput.
No allocator environment variables, memory limits, learning rates, batch sizes,
or training settings were changed.

## Verification

- **30 tests passed** across master optimizer, activation OOM recovery, epoch
  checkpoints and timed training. New tests compare bounded updates and moments
  exactly against FP32 AdamW, exercise missing gradients and oversized tensors,
  verify cleanup without retry after a partial bucket failure, and serialize/
  restore master state with a different staging budget.
- An isolated CUDA test loaded the actual saved checkpoint, allocated synthetic
  gradients for all 1,110 parameter tensors, and held an additional **1 GiB** of
  device memory to add pressure. At the same **10.26-GiB allocator limit**, the
  update succeeded in **37 buckets**, with a maximum staged **60.08 MiB** and
  peak allocation **8.51 GiB including the extra 1 GiB**. Representative AdamW
  counters advanced exactly once, and no master gradients remained afterward.
- This is an optimizer memory-pressure test, not a replay of ten hours of
  fragmented allocations or a complete training benchmark. CPU and CUDA tests
  used separate in-memory state and temporary test artifacts. Production
  checkpoint size and modification time remained unchanged.

Regression log: `results/optimizer-bounded-staging-regressions.log`.

## Resume point and command

The complete recovery checkpoint is
`results/model/training/latest_checkpoint.pt`, saved at **08:40:52**:

- Global step: **722,917**.
- Epoch: **2**; next batch offset: **29,221**.
- Optimizer: `FP32MasterAdamW`, FP32 masters present.
- Saved learning rate: **0.00005**.

The last successful journal update was **728,895**. Thus **5,978 completed
updates** after the last checkpoint will be replayed, approximately the last
half-hour of wall time. The failed update was the following one. The run's
exception path did not overwrite the checkpoint with failure-time state.

From the repository root in PowerShell:

```powershell
.\.venv\Scripts\python.exe .\src\05_model_training\train.py --config .\configs\config.json
```

`training.resume="auto"` restores that completed checkpoint, including weights,
moments, FP32 masters, scheduler, RNG and batch cursor. No data regeneration or
manual checkpoint conversion is required. The full training run was not started
by this investigation.

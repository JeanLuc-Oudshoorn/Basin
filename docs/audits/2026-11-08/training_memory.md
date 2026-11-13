# November 8 training activation OOM investigation

## Evidence and cause

Production log `logs/20261108T150439_410728Z_train_a502735b/events.log`
shows 200 successful updates followed by a forward-pass allocation failure in
the third-party RWKV kernel's `s = torch.empty(B,H,T//CHUNK_LEN,N,N, ...)` saved
state buffer. The caller is the **uncheckpointed** block branch of
`RWKVForecaster.forward_next_encoded`. It is not a dataset transfer/allocation.

At failure PyTorch had 7.78 GiB live and 220.73 MiB reserved but unused, against
its configured 8 GiB allocator limit. The device still reported 2.39 GiB free.
This indicates the allocator cap plus live activations, rather than evidence
of a growing corpus-sized GPU cache or a fragmentation-only problem. The
impossible `17179869184.00 GiB` process-memory value in the exception is not a
credible physical usage measurement; use the live/reserved/free counters.

The saved architecture report contains **500,056,352 parameters and 28 layers**.
This investigation preserves that user-selected architecture. It is larger
than the earlier 397M/22-layer configuration. AdamW weights, gradients and
moment estimates leave less space for shape-dependent activations.

The previous microbatch limits were 8 tables / 1536 variable-patch tokens.
`checkpoint_token_threshold=2048` permits retaining all 28 layers whenever
`B×V×T <= 2048`. Since almost every microbatch satisfies the 1536 token cap,
activation checkpointing was bypassed for many nontrivial shapes despite
`gradient_checkpointing=true`. Variable lengths/covariate counts explain why
small early batches could work before a later batch exceeded the budget.

## Changes

- The requested default allocator ceiling is 11 GiB, replacing 8 GiB and the
  old 80% device-capacity clamp. It is bounded by free VRAM plus this allocator's
  current reservations, minus 0.5 GiB of external headroom. Startup logs report
  the actual chosen budget. External allocations after startup can still change
  available memory; a PyTorch ceiling alone cannot prohibit Windows paging.
- Logical batch size remains eight, with one optimizer update after all its
  gradients. Physical size is at most eight; the 1536 variable-patch-token cap
  further lowers the size on long multivariate tables.
- Activation checkpoint threshold is 1024, retaining fewer layer intermediates
  at larger shapes. No change to architecture, objective, coverage or inference.
- Activation OOM recovery retries the entire logical update after discarding
  partial gradients and restoring RNG. It halves physical size, remembers the
  safe limit per shape, and finally tries all-layer checkpointing if one table
  cannot fit. It never retries an optimizer-step OOM or invalidated graph capture.
- Hourly checkpoints persist learned physical limits and all-layer fallback.
  Resume tolerates changes in physical size/token caps, with a warning; logical
  batch/data/model/optimization identities are still enforced. Physical layout
  changes can affect rounding and stochastic RNG assignment, so they are not
  claimed to give bitwise-identical continuation.
- INFO progress includes allocated/reserved/peak VRAM, physical shapes and retries.
  `resume="auto"` replaces the invalid boolean `true` in the current config.

Dataset payloads were already bounded CPU/shard reads, not resident on CUDA.
Current plus one future update are prepared on CPU; CUDA holds only the current
physical microbatch. No retransform/reembedding is needed for these runtime
changes. The production baseline checkpoint was saved at step zero after
validation; resuming it preserves that baseline and avoids repeating it, but
cannot recover the subsequent unsaved ~200 updates.

## Verification

`scripts/verify_training_memory.py` exercises the actual current model, real
third-party forward/backward kernels, compiled variate attention and AdamW.
It uses bounded synthetic inputs spanning 256–2048 observations and 0–5
covariates, bypassing million-source startup and baseline validation. Outputs
are isolated under a timestamped `training_memory_stress` log and
`training_memory_stress.json` beside this report; production weights/checkpoints
are never written. Its optional old-settings arm uses the old 8 GiB/2048
checkpoint-threshold settings before switching to the current settings.

CPU regressions cover an OOM on the second microbatch *after* the first backward
has completed: the retry matches a reference small-microbatch AdamW update,
including loss, label coverage and optimizer moments, with only one optimizer
step and restored RNG. They also verify that single-table unrecoverable OOMs
raise without stepping, optimizer OOMs are never retried, shape limits are
reused, allocator headroom is respected, and hourly resume accepts safer
physical limits and restores the learned memory policy.

### Measured results

Two completed mixed-shape GPU runs used the actual 500M model:

| Run | Optimizer updates | Training-loop seconds | Peak live GiB | Peak reserved GiB | Further OOM retries |
| --- | ---: | ---: | ---: | ---: | ---: |
| Four-table maximum, after forced recovery | 600 | 298.78 | 6.69 | 7.62 | 0 |
| Final eight-table maximum, normal budget | 120 | 58.63 | 6.68 | 7.02 | 0 |

These times include cold-shape compilation and are not a paired throughput
comparison. Both runs alternate 256/1, 1024/6, 2048/6, 2048/1, 512/3 and 1536/5
observation/variable cases. Every logical update contains eight tables. No
production corpus/checkpoint is loaded or modified. The first run lasted
360.82 seconds including initialization, reproduction and recovery checks.

The old-policy arm **reproduced an 8 GiB OOM on its second update**, with AdamW
moments now resident. Deliberately lowering the new-policy budget to 5.5 GiB
then exercised real CUDA recovery: four tables failed, two failed, one succeeded;
the logical update still covered all eight tables. The 600-update run retained
that learned one-table limit for its 1024/6 profile. This is why that profile's
timing in that run should not be treated as the normal-policy throughput.

The final eight-table run uses no artificially retained recovery limit.
Excluding each case's first call, 256/1, 2048/1 and 512/3 cases averaged
139.6/140.3/138.5 ms per update, versus 255.7/251.4/259.5 ms with the four-table
maximum. These indicative separate-run timings justified retaining eight as
the maximum rather than imposing an unnecessary permanent split on small
inputs. The longest multivariate case still used four two-table microbatches.

The requested 11 GiB startup budget was reduced to approximately 10.2 GiB
because of desktop/external allocations plus the 0.5 GiB headroom. No test
needed that full allowance. See raw records in
[600-update recovery stress](training_memory_stress.json) and
[120-update final-settings stress](training_memory_microbatch8.json).

The complete CPU suite finished 324 tests with no failures and 13 skips;
subsequent focused runs passed six OOM tests and nine hourly-resume tests,
including additional four-lead weighted-gradient and all-layer-fallback cases.
Pre-commit checks cover the edited code, config and documentation. No training
data rebuild, architectural change or production checkpoint overwrite occurred.

The bounded GPU run is a stress check, not proof over every table in the corpus.
Runtime recovery is retained because table lengths and covariate counts vary.

## References

- [PyTorch CUDA memory management](https://docs.pytorch.org/docs/2.12/notes/cuda.html#memory-management)
- [PyTorch activation checkpointing](https://docs.pytorch.org/docs/2.12/checkpoint.html)

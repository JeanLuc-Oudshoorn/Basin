# Dedicated VRAM budget and physical microbatching

## Observed issue

The user reported an approximately 1.5-hour epoch ETA on the 12 GB RTX 5070.
The existing run journal
`logs/rwkv7_train_20261028_184759_859429Z_pid5800_ff0bdf92.jsonl`
recorded 16,322 MiB reserved by PyTorch at update 1,269, versus 8,832.6 MiB
peak live allocations and 12,227 MiB device capacity. `nvidia-smi` initially
showed 11,837 MiB used. This is consistent with the reported shared-memory
spillover; it does not directly measure paging traffic. Cached reservations,
as well as live activations, need bounding. The run processed 10,117 windows
in 546.17 seconds (18.52 windows/second). Its remaining-training ETA was
4,687.85 seconds, before full validation and checkpoint writing.

That worker subsequently recorded `KeyboardInterrupt` at 18:59:38 UTC.
The investigation did not terminate it. Benchmarks started only after checking
that no training worker remained and GPU use had returned to idle.

## Implementation

- `performance.prepare_update` first packs each existing optimizer batch on CPU,
  then slices its batch axis. It preserves table order, original padding,
  normalization, labels and masks. It never splits the sequence axis.
- Defaults are `microbatch_size=4` and `microbatch_token_limit=1536`. Physical
  batch size is the minimum of the table limit and
  `max(1, floor(token_limit / (variables * padded_patches)))`.
- `batch_size=8` and `gradient_accumulation_steps=1` still define optimizer
  groups. A group of eight long, eight-variable tables therefore becomes eight
  single-table forwards/backwards followed by one AdamW update. Shorter groups
  can run four tables per forward. Incomplete groups retain their existing size.
- Gradients use the original weighting by finite target points, and clipping,
  AdamW and the learning-rate scheduler run once per original optimizer group.
  Floating-point summation order can change; bitwise identical GPU trajectories
  are not promised. No architecture, forecasting horizon or validation subset
  was changed.
- Before model allocation, `train.py` calls `configure_cuda_memory` with
  `cuda_memory_limit_gib=8.0`. PyTorch's caching allocator receives a budget of
  `min(configured GiB, 80% of dedicated device capacity)`. This covers reserved
  cache as well as live tensors. It does not cap desktop, CUDA context or external
  library memory. Allocation errors are allowed to surface; the cap is not an
  automatic retry system.
- Either physical microbatch limit can be `null` to disable it. A single table
  is always permitted even if it exceeds the token limit; this limit does not
  shorten context or guarantee every possible future input fits. The memory
  limit can also be `null` to restore the unrestricted allocator.
- Existing full end-of-epoch validation, early stopping and epoch checkpoint
  behavior remain. These execution controls do not change checkpoint identity.
  Recovery still requires a completed epoch checkpoint; an interrupted first
  epoch cannot be recovered from this change alone.
- Console startup messages and the JSONL training plan record the controls.
  Per-update `microbatches` now counts actual physical forward/backward passes.
  Existing per-update loss, progress, ETA and CUDA memory telemetry remain.

## Experiments and verification

All GPU experiments use the actual 447,482,144-parameter model, BF16, fused
AdamW, compiled variate attention and the existing 2,048-token selective
activation-checkpoint policy. They reuse the bounded real-data sample at
`../training_speed/sample.pt`; no generated tables or production weights were
modified. Logs and JSONL journals are under `logs/`.

`micro4_tokens1536.json` completed a 91-update sweep with peak live allocation
7.06 GiB and 19.62 windows/second. This initial sweep includes compilation of
additional shapes, so it is not the steady-state throughput estimate.

`micro8_tokens2048.json` tested a less restrictive physical batch/token budget
under the same 8 GiB allocator cap. It failed during warmup with an allocation
error at 7.66 GiB live allocation plus 331 MiB unused reservation. It was rejected
as a default. The error's enormous non-PyTorch memory figure is an invalid driver
accounting value and is not used as a measurement.

`micro4_sustained.json` completed 91 warmup updates covering the sample's shapes,
then 300 measured optimizer updates: **24.62 windows/second**, **7.04 GiB peak
live allocation**, and **8.00 GiB maximum reservation** across all 391 updates.
This confirms that the allocator cap holds as shapes change; it is not a
full-corpus or multi-epoch stress test. A concurrent `nvidia-smi` spot check
showed 8,881 MiB total device use, leaving about 3.3 GiB dedicated headroom.

The measured rate is about 33% higher than the interrupted run's cumulative
18.52 windows/second, but those runs have different timing scopes and workloads.
It should not be interpreted as a controlled 33% full-epoch improvement.

`micro4_worst.json` completed ten measured updates on repeated largest-shape
tables, each logical batch `[8, 8, 128, 1024]` executed as eight
`[1, 8, 128, 1024]` microbatches. Results: 9.83 windows/second, 5.80 GiB peak
live allocation and 5.82 GiB peak reservation. Smaller microbatches can trade
some peak-shape throughput for memory headroom; preventing spillover is the goal.

Reproduce the sustained experiment from the activated WSL environment:

```bash
python scripts/benchmark_training_throughput.py \
  --case micro4_sustained --batch-size 8 --accumulation 1 --fused --bucket \
  --checkpoint-threshold 2048 --production-compile \
  --microbatch-size 4 --microbatch-token-limit 1536 --memory-limit-gib 8 \
  --warmup-sweep --updates 300 \
  --sample docs/audits/2026-10-28/training_speed/sample.pt \
  --output-dir docs/audits/2026-10-28/vram_budget
```

The CPU regression suite ran 145 tests: 138 passed, seven CUDA-only tests
skipped. New checks compare split versus unsplit weighted loss, updated weights,
padding, target coverage and optimizer-group checkpoint compatibility.
Output: `logs/20261028_vram_regression.log`.

The actual CUDA backend also passed `scripts/verify_epoch_resume.py
--compile-variate --microbatch-size 1` on a small two-layer model. It exercised
split physical batches through `fit_epochs`, full validation, epoch checkpoint
writing, intentional interruption and recovery. Resumed parameters matched the
uninterrupted reference and AdamW state remained on CUDA. Evidence:
`resume/gpu_verification.json` and `logs/20261028_vram_gpu_resume.log`.
Ruff formatting/lint checks passed. Bandit reported only the existing test
assertions in the standalone recovery verifier, with no medium/high findings.

Short benchmarks measure CPU preparation plus synchronized forward/backward and
optimizer execution. They exclude corpus loading, full-epoch validation and
checkpoint I/O; repeated sampled windows do not represent a complete epoch.
Earlier speed estimates also did not expose the longer run's allocator growth.
An end-to-end epoch speedup must be measured in the next real run.

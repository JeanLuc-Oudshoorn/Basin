# Training throughput investigation — 2026-10-28

Subsequent update: default early-stopping patience is now one epoch, and full
resumable checkpoints are written after each epoch. See the
[checkpoint audit](../epoch_checkpoints/REPORT.md). The measurements below precede
that update and exclude epoch-end checkpoint I/O.

## Outcome and scope

The normal `src/05_model_training/train.py` now uses faster execution defaults in
`configs/config.json`. The architecture remains 447,482,144 parameters, 25 layers,
width 1,024, 16 observations per patch and a 2,048-observation context limit.
Embedding projections remain trainable. No production checkpoints or processed
artifacts were replaced by these experiments. The existing transform/embed output
can be used directly with the next training launch.

On the RTX 5070 12 GB, both configurations processed the same 512 sampled real
training windows, after two warmup updates. Each measured sweep visits every
sample once, including incomplete batches. Measurements synchronize CUDA at
update boundaries and include feature preparation, copies, forward, backward,
gradient clipping, logging and the optimizer. Initialization, warmup, profiler
collection and validation are excluded from training throughput.

| Measurement | Reference execution | New defaults |
| --- | ---: | ---: |
| Windows per second | 5.525 | 24.210 |
| Relative throughput | 1.00× | 4.38× |
| Allocated VRAM peak in sweep | 3.63 GiB | 8.22 GiB |
| Projected 97,045-window training pass | 292.7 minutes | 66.8 minutes |
| Validation of the same 128 tables, fixed weights | 3.352 seconds | 1.613 seconds |

The time projection is **not a measured full epoch**. Full validation, checkpoint
writes and startup add time. Desktop activity and dataset shape distribution also
affect performance. The small sample has proportionally more incomplete role/length
buckets than the full corpus. This comparison is a throughput test, not an accuracy
comparison: shuffled batching and BF16 optimizer arithmetic can change the training
trajectory. The reference sweep uses the new metrics-transfer code but the old
batching/checkpoint/optimizer settings, making it a conservative comparison. The
initial pre-optimization profiler run independently measured 5.465 windows/second.

Evidence: [comparison.json](comparison.json), [reference_sweep.json](reference_sweep.json),
[final_sweep.json](final_sweep.json), and
[final_sweep_validation.json](final_sweep_validation.json).

## Bottleneck evidence

The original configuration performed eight separate one-window forward/backward
passes for each optimizer update. Gradient accumulation did not increase the
amount of work in each individual matrix multiplication. Every RWKV and variate
mixing layer was checkpointed, causing extra forward computation during backward.
AdamW used individual tensor updates with `foreach=False` and no fused optimizer.

The original profile contains approximately 37,672 CUDA runtime kernel launches,
12,664 driver launches and 12,416 `aten::mm` calls for one eight-window update.
CUDA launch/dispatch and many small operations consume substantial CPU time.
The RWKV backend already uses its custom CUDA kernels; this was not an accidental
CPU implementation of the recurrent model. See [baseline_profile.txt](baseline_profile.txt).

The prior real-corpus CPU benchmark prepared approximately 5,500 windows/second;
tokenization was far faster than the 5.5-window/second training loop. Tokenization
still occurs per accessed batch because it constructs causal features, while the
learned embedding projection runs on GPU with gradients. Corpus-wide caching of
embedding vectors is unnecessary and would reintroduce the prior memory problem.

A read-only `nvidia-smi` sample during the optimized sweep showed 92% utilization,
9,359 MiB device memory and 182.4 W. This is a spot measurement, not an average or
a guarantee of continuously full utilization. Throughput is the comparison metric.

## Implementation changes

1. **Parallel batches:** up to eight windows per forward, instead of one. Variable
   roles/counts must match; tables remain separate on the batch axis. Covariate
   names do not have to match across tables. No attention crosses table boundaries.
2. **Length buckets:** shuffle within equal kernel-padded patch-length buckets and
   shuffle batch order. This preserves randomness while avoiding large amounts of
   padding between very short and long series. No records or incomplete batches
   are dropped. Batch counts are recalculated for progress and epoch scheduling.
3. **Fused CUDA AdamW:** update tensors through the fused implementation. CPU runs
   use ordinary AdamW. Learning rate, weight decay and scheduler settings are retained.
4. **Selective checkpointing:** spend an activation budget on layers whose
   intermediate values can be kept, checkpoint the remaining layers. For a batch
   shape `[B,V,T,D]`, the number of uncheckpointed layers is
   `min(layers, floor(checkpoint_token_threshold * layers / (B*V*T)))`.
   At threshold 2,048, sufficiently small batches avoid recomputation entirely;
   larger batches recompute only the layers needed to stay within the tested budget.
   This is a shape-based budget, not an exact memory allocator or universal OOM guard.
5. **Bounded prefetch:** one CPU thread prepares the following optimizer update
   while the current one trains. At most current and next updates are prepared;
   nothing is cached for the entire corpus. Pinned host tensors and nonblocking
   transfers reduce copy stalls. The prefetch executor is closed on normal exit,
   explicit step caps, interruption and failure.
6. **Fewer synchronizations:** accumulate detached loss statistics on GPU, then
   transfer metrics once after the accumulation group. Check finite losses and
   gradient norms before calling the optimizer. CPU label counts avoid redundant
   GPU validity checks in the normal training path. Gradient clipping uses foreach.
   Labels already transferred as part of the prepared batch are reused.
7. **Batched full validation:** group compatible validation tables, gather each
   table's own last real patch before the output head, and compute per-table losses
   and squared errors together. Right padding never changes the forecast origin.
   Every validation table is evaluated, with point-weighted aggregate metrics.
8. **Less console traffic:** INFO progress reaches the console and Windows/WSL
   launcher; detailed per-window and per-microbatch DEBUG logs remain in each run's
   `logs/.../events.log`. Timestamped training JSONL output remains automatic.

These changes are in `batching.py`, `performance.py`, `epoch_training.py`,
`training_protocol.py`, and `train.py`, all under `src/05_model_training`.

## Default configuration and controls

```json
{
  "logging": {"level": "INFO"},
  "training": {
    "batch_size": 8,
    "gradient_accumulation_steps": 1,
    "validation_batch_size": 16,
    "fused_optimizer": true,
    "bucket_by_length": true,
    "prefetch": true,
    "checkpoint_token_threshold": 2048
  }
}
```

The nominal effective batch remains eight windows per update. Incomplete buckets
can produce smaller updates; their labels are still weighted by observed point
count. Accumulation is still available: for example, batch size 4 with accumulation
2 also gives up to eight windows per update. All new execution settings live under
`training`, so changing them does not invalidate the input artifact signature.

`model.gradient_checkpointing` must remain true to permit selective checkpointing.
Threshold 0 checkpoints every layer, recovering the previous memory-saving mode.
Setting model checkpointing false disables it entirely, irrespective of threshold;
that was unsuitable for large batches on this GPU. The activation budget was
benchmarked for this 1,024-wide, 25-layer architecture and should be retuned for a
different model size or GPU.

Full validation still runs only at the end of each epoch. Training stops after
three epochs without improvement; the best weights are restored before the sealed
10% dataset holdout is evaluated. No validation sampling or holdout reuse was
introduced. Progress logs and JSONL additionally expose epoch windows/second.

## Memory stress tests and numerical behavior

The largest observed training window has eight variables and 127 real input
patches, padded to 128. Eight copies of this window with selective checkpointing
completed updates at 8.30 GiB peak allocated memory. See
[selective_worst.json](selective_worst.json). This artificial repeated-window test
is a memory/throughput stress test, not a learning experiment.

An earlier attempt to disable all checkpointing with this eight-window batch
filled device memory (11,907 MiB shown by `nvidia-smi`) and stalled. Only that
benchmark process was terminated. It was excluded from throughput results and
was not selected as the default. Two such windows without checkpointing fit at
7.20 GiB, providing evidence for the selective activation budget.

CPU tests verify equivalent predictions/losses and parameter updates within
float32 tolerances. BF16 GPU batched and singleton outputs are not bitwise equal:
different GEMM shapes and execution paths change rounding. On the fixed-weight
128-table final sample, pinball losses were 0.2374623 and 0.2374675; pooled RMSE
was 179.4588 and 179.6456 (approximately 0.10% difference). Validation still uses
the same labels, causal normalization, quantiles and point weighting. These metrics
describe a short random-initialization benchmark and are not model quality claims.

## Reproduction and verification

Normal training uses the existing command:

```powershell
uv run python src/05_model_training/train.py
```

No transform/embed rerun is required. A worker already running continues with the
code/configuration it loaded at startup; new settings apply on the next launch.

From the configured WSL shell at the repository root:

```bash
source scripts/activate_rwkv7.sh
python scripts/benchmark_training_throughput.py --case reference_sweep --batch-size 1 --accumulation 8 --sweep
python scripts/benchmark_training_throughput.py --case final_sweep --batch-size 8 --accumulation 1 --fused --bucket --checkpoint-threshold 2048 --sweep --validation
python scripts/benchmark_training_throughput.py --case selective_worst --batch-size 8 --accumulation 1 --fused --bucket --checkpoint-threshold 2048 --worst-case --updates 4
python scripts/verify_training_speed.py
```

The throughput script creates an isolated reproducible `sample.pt` from admitted
training/validation tables on first use, then reuses that snapshot for matched
comparisons. Remove that benchmark sample to resample after changing the corpus.
Benchmark weights never overwrite production model files. The full-size throughput
test uses the actual 447.5M model; the end-to-end smoke uses a reduced 128-wide,
two-layer model to exercise two whole epochs, prefetch, accumulation, full validation,
best-checkpoint restoration, embedding gradients and automatic training journals.

CPU regression result: **136 tests run: 130 passed and six CUDA-only skips**.
New regression cases cover mixed-length validation, causal training gradient
equivalence, selective checkpointing gradients, complete bucket coverage,
prefetch parity and rejecting nonfinite updates before optimizer mutation.
The CUDA end-to-end smoke passed two complete epochs (22 updates, 64 window visits),
with fused AdamW, nonzero embedding gradients and best-checkpoint restoration.
The next-patch and forecaster suites also passed in WSL CUDA: five and six tests
respectively. Ruff and Bandit checks pass. GPU smoke results are recorded under
`gpu_smoke/verification.json`; detailed verification logs are under
`logs/20261028_training_speed_gpu_verification.log`. Benchmark launches now also
create timestamped text logs and automatic JSONL journals under `logs/`.

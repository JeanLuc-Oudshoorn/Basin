# Training CUDA graph integration: 307M complete-update benchmark

For the current implementation and selected defaults, see the
[capture-inclusive schedule benchmark](../training_graph_schedule/REPORT.md).
The [earlier remeasurement](../training_graphs_post_fixes/REPORT.md)
confirms warm short-context gains but also measures the substantial capture
overhead on changing real batch shapes. The warm results below are historical
and must not be interpreted as a speedup of the current shuffled training epoch.

2026-10-31. Implemented in the ordinary `train.py` path and enabled in the default
configuration. The implementation graphs **forward and backward of eligible
cross-variable attention layers**, retaining the existing RWKV kernels and optimizer.

## Measured update time and VRAM

RTX 5070 (12 GiB), PyTorch 2.12.0+cu130, BF16, four CPU threads. One 307,307,808
parameter model is resident. Every weight update contains eight independent tables.
Each variant starts from the same weights and seed, performs eight AdamW updates,
and reports the median of the final six. Timings include pinned input transfers,
embedding, RWKV blocks, both forecast losses, backward, point-weighted gradient
accumulation, clipping, metrics synchronization and fused AdamW. Compilation/capture
on the first updates is excluded from warm timing and retained in `cold_seconds`.
Data preparation, validation, checkpoint I/O and epoch logging are excluded.

| Configured microbatch | Variables | Patches | Existing ms/update | Graph ms/update | Time reduction | Peak allocated GiB | Peak reserved GiB |
|---|---:|---:|---:|---:|---:|---:|---:|
| 1 (1 actual) | 4 | 32 | 532.7 | 455.8 | 14.4% | 3.81 | 4.33 |
| 4 (4 actual) | 4 | 64 | 153.0 | 144.4 | 5.6% | 5.97 | 6.72 |
| 8 (8 actual) | 4 | 32 | 86.2 | 75.7 | 12.1% | 5.42 | 6.32 |
| 4 (2 actual) | 6 | 128 | 361.8 | 361.5 | 0.1% | 6.76 | 7.79 |

One patch is 16 observations. Thus 32/64/128 patches mean 512/1024/2048 input
observations. The training example also has one final label patch. The long case
uses physical microbatches of two because the existing 1536 variable-token limit
overrides the configured maximum of four: `2 Ã— 6 Ã— 128 = 1536`.

The default-like four-table case improved about 6%; the eight-table short-context
case improved about 12%. The one-table case improved about 14%. The long-context
case was essentially unchanged: heavier GPU computation leaves less CPU launch
overhead to eliminate. These timings are not a measured full-epoch speedup.

No allocator OOM occurred. All graph variants stayed below
6.8 GiB allocated and 7.8 GiB reserved by PyTorch, within the unchanged 8 GiB allocator
cap. CUDA libraries, desktop/display use and other applications are outside those
allocator figures. Windows shared GPU-memory counters were not recorded, so the
claim here is sufficient allocator headroom, not direct proof of zero OS offload.
This is not a guarantee for every possible covariate count.
The smaller model has sufficient headroom in these tested cases. Increasing the
microbatch consumes more VRAM; reducing it or the graph cache budget reduces use.

## Defaults and runtime behavior

`configs/config.json` now has:

```json
{
  "training_cuda_graphs": true,
  "training_graph_cache_shapes": 2,
  "training_graph_memory_gib": 1.5
}
```

The existing `microbatch_size: 4`, `microbatch_token_limit: 1536`, gradient
accumulation, learning rate, epoch validation and checkpoint policy are retained.
Set `training_cuda_graphs` to false to use the previous compiled training path.
These are runtime execution options; they do not invalidate processed artifacts
or change model parameter/checkpoint names, and can be changed on resume.

Since 2026-11-01, recoverable capture failures restore the CUDA stream, check GPU
health, and automatically disable training replay for the rest of the run while
continuing with compiled attention. The failure and fallback are logged; the
current microbatch is still trained. See the
[capture-recovery audit](../../audits/2026-11-01/training_graph_capture/REPORT.md)
for the observed failure, regression checks, and limits of this fallback.

Training prefetch now performs CPU preparation in its worker and pins memory on
the consuming training thread before capture. This prevents CUDA host-memory
allocation from racing with capture across threads; prefetch and replay remain
enabled. See the [prefetch/capture root-cause audit](../../audits/2026-11-01/training_prefetch_capture/REPORT.md).

`training_graphs.py` captures outside the model forward, before entering activation
checkpoint contexts. Each graph shares the original layer parameters. Post-accumulate
hooks copy retained parameter gradients only when that backward used a graph,
preventing the next replay from overwriting accumulated gradients. Compiler/capture
warmup runs inside an RNG-preservation context so auxiliary lead sampling is not
advanced by warmup.

At most two input-shape profiles are retained and persistent graph allocations
have a separate 1.5 GiB budget. Older profiles are retired as length buckets change;
retirement occurs after the previous microbatch's backward, when outputs are no
longer needed. Captures that would exceed the budget are skipped. Wider-than-eight
variable attention and layers requiring activation checkpoint recomputation use
the previous path. Memory estimates are checked before capture and actual live
allocation growth is checked afterward. This bounds graph-cache growth; ordinary
model activations still depend on batch size, length and covariates.

Capture events are logged at DEBUG. Epoch-end INFO, JSONL and validation-history
records include captures, replays, evictions, fallback counts and cache size.
Real epoch savings depend on shape reuse and capture amortization. The benchmark
uses homogeneous shapes, so its warm numbers do not include cache churn on a
heterogeneous corpus. Optimizer and scheduler operations are not captured, so
learning-rate changes remain ordinary Python/optimizer operations.

## Correctness checks and precision limits

Real RWKV CUDA integration checks compare accumulated gradients and AdamW updates
through multiple microbatches, missing values, partial sequences, a checkpointed
layer, evaluation, optimizer/model state reload and rebuilding the graph cache.
They also check shape-profile eviction and unchanged state-dict names. The small
integration model passes exact equality; see `verification.json` for final status.

For the full 307M default-like and eight-table cases, first-update gradient and
weight hashes matched exactly and all eight training losses matched. The long case
also matched exactly. The one-table case was not bitwise identical: gradient relative
L2 difference was 0.706%, post-update weight relative L2 difference was 0.0134%,
and maximum relative loss difference over eight updates was 0.0647%. The precise
kernel-level cause was not isolated; BF16 execution can differ under capture.
We do not claim bitwise-identical training for all shapes.

Full-model acceptance checks required first-update gradient relative L2 error
below 2%, weight relative L2 error below 0.1%, and per-update relative loss error
below 1%; actual differences above are substantially smaller. Raw losses, hashes
and measured differences are retained in `experiment.json`. This short benchmark
does not establish convergence or long-run accuracy equivalence.

## Files

- [Raw results](experiment.json), [summary](summary.csv), [environment](environment.json).
- `scripts/benchmark_training_graphs.py` reproduces complete-update measurements.
- `tests/test_training_graphs.py` checks GPU integration and runtime options.
- `src/05_model_training/training_graphs.py` owns the graph cache and gradient protection.
- Detailed run logs are under `logs/*_training_graphs_integration_*`.
- [Earlier isolated-module experiment](../training_graphs/README.md).

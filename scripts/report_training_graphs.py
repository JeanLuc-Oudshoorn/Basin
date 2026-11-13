"""Summarize complete training update measurements and memory usage."""

import json
import statistics
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
DIRECTORY = ROOT / "docs/time_benchmark/training_graphs_integration"


def main():
    rows = json.loads((DIRECTORY / "experiment.json").read_text())
    lines, summaries = [], []
    for baseline, graph in zip(rows[::2], rows[1::2]):
        a = statistics.median(baseline["warm_times_ms"])
        b = statistics.median(graph["warm_times_ms"])
        saving = (1 - b / a) * 100
        size = graph["physical_microbatch_sizes"][0]
        lines.append(
            f"| {graph['microbatch']} ({size} actual) | {graph['variables']} | {graph['tokens']} | {a:.1f} | {b:.1f} | {saving:.1f}% | {graph['peak_allocated_gib']:.2f} | {graph['peak_reserved_gib']:.2f} |"
        )
        summaries.append(
            dict(
                microbatch=graph["microbatch"],
                actual_microbatch=size,
                variables=graph["variables"],
                patches=graph["tokens"],
                baseline_ms=a,
                graphs_ms=b,
                time_reduction_percent=saving,
                allocated_gib=graph["peak_allocated_gib"],
                reserved_gib=graph["peak_reserved_gib"],
            )
        )
    pd.DataFrame(summaries).to_csv(DIRECTORY / "summary.csv", index=False)
    text = "".join(
        (
            """# Training CUDA graph integration: 307M complete-update benchmark

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
""",
            "\n".join(lines),
            """

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
""",
        )
    )
    (DIRECTORY / "REPORT.md").write_text(text, encoding="utf-8", newline="\n")


if __name__ == "__main__":
    main()

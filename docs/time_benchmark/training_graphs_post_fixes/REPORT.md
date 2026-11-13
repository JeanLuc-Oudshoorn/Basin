# Training graph replay after the memory and prefetch fixes

Historical measurements before the capture-admission and scheduling fix. See the
[completed follow-up benchmark and selected defaults](../training_graph_schedule/REPORT.md).
Training now defaults to larger physical microbatches with replay disabled;
replay remains available with bounded profiles and capture admission.

2026-11-01. Measured current execution; no production settings or weights changed.

## Repeated shapes: warm full-update latency

| Input observations | Variables | Maximum microbatch | Compiled reference ms | Replay ms | Time saved | Paired range |
|---:|---:|---:|---:|---:|---:|---:|
| 256 | 4 | 4 | 138.92 | 116.77 | 15.9% | 8.8% to 16.9% |
| 512 | 4 | 4 | 135.07 | 110.19 | 18.4% | 15.0% to 19.8% |
| 1024 | 4 | 4 | 134.15 | 132.12 | 1.5% | 0.7% to 16.0% |
| 512 | 4 | 1 | 497.41 | 431.49 | 13.3% | 13.0% to 13.6% |
| 2048 | 6 | 4 | 310.90 | 302.30 | 2.8% | 0.7% to 3.5% |

Positive time saved means replay reduced latency; negative means slower.

## Actual changing shapes: capture and eviction included

Across the first 24 real epoch-1 batch groups, mean latency per update was 149.44 ms without replay and 589.79 ms with replay. Replay took 3.95 times as long per update.

This comparison includes every update, including new graph captures, synchronized profile retirement and cache churn. Common attention compilation was warmed for both arms. Each trial starts with an empty graph cache. It is not a full-epoch measurement.

## Method and limits

- 307,307,808 parameters; NVIDIA GeForce RTX 5070; PyTorch 2.12.0+cu130, CUDA 13.0, BF16.
- One GPU model resident; 4 CPU threads; eight tables per update.
- Identical initial model state and batch sequence, fresh AdamW state and the same RNG seed for each arm.
- Compiled attention, prefetch, token-budget microbatch splitting, gradient accumulation, clipping and optimizer are enabled in both arms.
- Host timing includes waiting for preparation, main-thread pinning, transfer, forward, both losses, backward, clipping, metric synchronization and completed AdamW update.
- Per-update GPU synchronization bounds the measurement. Validation, checkpoint I/O and ordinary epoch/progress logging are excluded.
- Synthetic repeated-shape tables have precomputed CPU token features. Real mixed-shape tables use the actual lazy tokenizer and normal prefetch path.
- Repeated-shape trials discard 5 warmup updates and time 24 more; 3 paired repetitions alternate arm order.
- Repeated-shape table reports median trial medians. Mixed-shape result reports median trial total divided by update count, retaining costly captures.
- Graph cache budget 1.5 GiB; allocator cap 8.0 GiB. Configured microbatch four becomes two in the six-variable 128-patch case because of the 1536-token limit.
- First-update and all-update timings, physical shapes, losses, graph counters, memory peaks and paired variation are retained in results.json and summary.json.
- Estimated first-capture break-even counts use first-update excess latency divided by median warm savings; they exclude later shape changes and are approximate.
- BF16 capture can change rounding; loss agreement is measured rather than assumed. This benchmark does not establish bitwise gradient equality.

Detailed logs: `/mnt/e/PyCharm/PyCharm Community Edition 2022.2.3/Projects/Basin/logs/20261101T155833_571088Z_training_graph_speed_post_fixes_dbfdfb97`.

## Interpretation

Replay still reduces warm latency for short, repeated profiles. With the default four-table microbatch, the 256- and 512-observation cases saved approximately 16% and 18% of update time. The 1024-observation effect was small and variable; the longest control saved only about 3%.

The current shuffled mixed-profile pipeline did not receive an overall speedup: replay was 3.95x slower in this bounded test. Each replay-enabled mixed trial recorded 388 captures, 825 replays and 22 profile evictions in just 24 updates. Most profiles are replaced before their capture cost is amortized. Thus the warm gains must not be presented as an epoch-level improvement for the current default cache policy.

Warm replay timings apply only while the same batch profile stays cached. Changing shapes can require capture again, and safety synchronization adds cost when a profile is evicted. A warm speedup alone is insufficient to claim faster shuffled-corpus training.

No production configuration was changed. These results identify profile reuse and capture amortization as the remaining performance limitation, rather than establishing which individual safety change contributed what fraction of the overhead.

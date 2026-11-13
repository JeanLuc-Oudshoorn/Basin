# Live training throughput investigation

Follow-up: the idle-GPU experiments are now complete. See the
[selected policy, raw results and safety checks](../../time_benchmark/TRAINING_THROUGHPUT_2026-11-08.md).
The findings below describe the original live run before those experiments.

This review leaves the production training process running. No competing CUDA
benchmark was launched, no default hyperparameter changed, and no production
checkpoint was overwritten. Candidate profiles have only been prepared in a
dry run; their speedups and memory requirements have **not** been measured.

## What the memory readings mean

`allocated=3.8 GiB` is sampled after backward and the optimizer update, when most
activations have been freed. It is not the maximum needed during the update.
The live journal sample in [live_throughput.json](live_throughput.json) contains
7,196 metric events (steps 620–7,815). Median update peak is 5.58 GiB, p90 is
6.67 GiB and maximum is 6.72 GiB. This is below the approximately 10.2 GiB
allocator budget selected from an 11 GiB request after desktop/headroom checks.
`reserved` additionally includes reusable allocator cache. Neither post-update
live memory nor GPU utilization percentage alone measures achievable throughput.

The active model is 500,056,352 parameters / 28 layers. The training pass has
693,696 optimizer updates, averaging approximately 7.70 tables per update in
the sampled prefix. Roughly 0.204 seconds per update projects to approximately
39 hours for the remaining training work. This does not include future hourly
validation and checkpoint pauses. It does not establish that the implementation
is optimal, but the epoch covers millions of tables, not the former 110k corpus.

## Evidence for optimization targets

| Physical microbatches per update | Updates in consecutive-event sample | Mean active milliseconds | Share of measured active time |
| --- | ---: | ---: | ---: |
| 1 | 5,145 | 151.5 | 53.0% |
| 2 | 1,770 | 315.5 | 38.0% |
| 3 | 234 | 449.7 | 7.2% |
| 4 | 46 | 571.2 | 1.8% |

28.5% of metric events report multiple physical microbatches, accounting for
approximately 47% of active time in consecutive-event pairs. These comparisons
are **confounded by table length and variable count**: the larger workloads both
split more and cost more. They do not prove that simply merging them saves the
entire difference. No OOM retries occurred in this sample.

The current token cap is 1536 variable-patch tokens. Eight tables with two
variables and 112 patches are split into six plus two tables. Eight tables with
six variables and 128 patches are split into four groups of two. At the same
time, the checkpoint threshold 1024 checkpoints ten of 28 layers when a physical
batch reaches 1536 variable-patch tokens. Those layers are recomputed during
backward, trading compute for lower activation memory.

There is therefore a plausible route to use more dedicated VRAM: merge more of
the **same eight-table logical update**, and/or retain more activations rather
than recomputing them. Merely accumulating additional batches does not make each
GPU execution larger. Increasing the *logical* batch size changes the optimizer
trajectory and is a separate experiment; the proposed profiles preserve it.

Approximately 7.9 seconds of the 1477-second sampled wall interval fell outside
the active preparation/model-update clock (~0.5%). This bounds only bookkeeping
**between** updates, not all logging or data overhead. Per-table DEBUG logging,
CPU normalization, pinning and waiting for the prefetch worker are inside active
time. Existing journals cannot separate those components reliably. Individual
NVIDIA-SMI samples also cannot distinguish idle time from inefficient small
kernels; 70% reported GPU utilization is not a promise of a particular speedup.

## Concrete preparation

`timed_training.py` now records separate `preparation_wait_seconds` and
`optimizer_update_seconds`, plus embedding shapes, in its JSONL metrics. Every
100 updates an INFO line reports the means over the last 100 updates. These are
host-observed durations: preparation wait includes consumer-thread pinning, and
model/optimizer time includes the existing synchronization and kernel submission
boundaries. They are not a CUDA-kernel profiler. The logging-only change takes
effect on the next launch; the running process is not modified.

The read-only analyzer can be rerun without restarting or using CUDA:

```powershell
uv run python .\scripts\analyze_live_training.py
```

Controlled candidate profiles are in
`configs/training_throughput_tuning.json`:

| Profile | Physical table maximum | Variable-patch token cap | Checkpoint threshold |
| --- | ---: | ---: | ---: |
| current | 8 | 1536 | 1024 |
| less recomputation | 8 | 1536 | 1536 |
| larger microbatches | 8 | 3072 | 1536 |
| larger activation budget | 8 | 3072 | 2048 |
| repeated current baseline | 8 | 1536 | 1024 |

`scripts/tune_training_throughput.py --dry-run` prepares separate configs and
commands under a timestamped `docs/time_benchmark/training_tuning/` directory.
It never edits `configs/config.json`. Actual runs must use the activated WSL
CUDA environment and refuse to start while `train.py` is running. Each candidate
uses one sequential isolated full-model process, 180 eight-table optimizer
updates, six synthetic shape cases, the same seed and loss, and the same
dedicated-memory/OOM safeguards. Summaries discard the first three calls for
each shape and retain timings, physical batch counts, retries and peak memory.
The baseline is repeated to help identify thermal/runtime drift. These are
equal-weight anchor cases, not a claim of a corpus-wide speedup. Case matching
and representative real-data timing are required before projecting an epoch ETA.

The user was asked to pause after a saved checkpoint for an uncontended GPU
comparison. Without that pause, claiming a measured 2× speedup would be
unsupported. A faster candidate should only become the default after it passes
the stress cases and retains the intended loss/gradient and resume behavior.

## References

- [PyTorch CUDA allocator semantics](https://docs.pytorch.org/docs/2.12/notes/cuda.html#memory-management)
- [Activation checkpointing trades recomputation for memory](https://docs.pytorch.org/docs/2.12/checkpoint.html)

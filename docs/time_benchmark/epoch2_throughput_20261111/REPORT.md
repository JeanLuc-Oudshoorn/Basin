# Epoch-2 throughput investigation, 2026-11-11

## Result

There is a real throughput cost from FP32 master weights/moments, and the epoch-1 ETA concealed part of it by retaining the faster BF16 history. There was also an additional slowdown early in epoch 2, but a subsequent bounded sample returned close to late-epoch-1 FP32 throughput. The evidence does not show a permanent doubling at the epoch boundary.

This investigation read existing journals, configuration and source code. It did not change the running program or its files, load model weights, launch CUDA computations, attach a profiler, or change GPU settings. One read-only `nvidia-smi` snapshot was taken. The new analysis script and this report are separate from training.

## 1. Epoch 1 mixes two execution regimes

The run resumed with FP32 masters at step **465,162**, retaining **82,800.708 seconds (23 hours)** of accumulated active training time from the BF16 portion. Epoch 1 has 693,696 updates, so **67.06% of its updates had already happened before the repair**.

The ETA in `fit_timed` uses:

```text
epoch_eta = epoch_active_seconds / epoch_tables_completed
            * epoch_tables_remaining
```

It does not restart its timing baseline when optimizer precision changes. It restarts at the next epoch. The displayed percent is based on batch/update progress; the ETA uses completed tables. Their denominators are related but not identical.

Reconstructing the user's examples from journal records:

| Global step | Epoch | Active epoch time | Completed tables | Remaining ETA |
| --- | ---: | ---: | ---: | ---: |
| 672,900 | 1 | 147,393.582 s | 5,179,917 | 4,549.925 s = 1:15:50 |
| 693,600 | 1 | 154,309.686 s | 5,339,078 | 21.359 s |
| 693,696 | 1 | 154,332.421 s | 5,339,817 | 0 |
| 703,426 | 2 | 4,072.682 s | 74,893 | 286,306.612 s = 79:31:47 |

The completed first epoch took **42.87 active hours**, but that is a mixture of old and new precision. The post-repair portion alone took **71,531.713 seconds for 228,534 updates**, averaging **313.0 ms/update**. At that average and the same updates per epoch, a complete epoch would be approximately **60.31 active hours**. The apparent ~40-hour pace was not the actual post-repair-only pace.

These are active-training times. Validation, checkpoint writing and other work outside the measured update interval extend wall-clock time.

## 2. FP32 storage causes more physical microbatches

The logical batch remains eight tables. The backbone still runs in BF16. FP32 master parameters and Adam moments increase live allocation after updates from about **3.82 GiB to 7.55 GiB**. The startup log records a **10.26-GiB allocator budget**, accounting for other device use and configured headroom.

There are explicit activation OOM recoveries after resume. For example, at 2026-11-10 06:04:14 and 06:04:17 the `[variables=2, padded_patches=128]` shape shrinks from physical batch 8 to 4, then 2. These learned shape-specific limits remain cached and survive resumes; later updates can therefore have `oom_retries=0` while still being split into four or eight physical passes.

| Matched logical workload | BF16 physical passes | Late FP32 physical passes | BF16 model/optimizer segment | Late FP32 segment |
| --- | ---: | ---: | ---: | ---: |
| 8 tables × 1 variable × 64 patches | 1 | 1 | 126.1 ms | 172.8 ms |
| 8 tables × 2 variables × 128 patches | 1 | 4 | 216.1 ms | 552.6 ms |
| 8 tables × 3 variables × 96 patches | 1 | 4 | 233.4 ms | 531.1 ms |
| 8 tables × 4 variables × 128 patches | 2 | 8 | 462.5 ms | 1,021.1 ms |

Samples: BF16 steps 420,000–424,999; late FP32 steps 688,697–693,696. These are workload-bucket matches rather than identical data or a controlled replay. Across 32 common profiles with at least 20 observations in each sample, a common-count-weighted comparison shows approximately **1.71×** longer model/optimizer segments.

The arithmetic precision repair therefore has two costs: FP32 optimizer state traffic/conversion/copying, and less remaining activation memory, which forces more serial forward/backward microbatches. The average physical-pass count increased from **1.038 to 2.039** in these samples. Some larger shapes increased more strongly.

`optimizer_update_seconds` is unfortunately named: it measures the entire accumulated model/backward/clipping/optimizer segment, not AdamW alone. Existing journals cannot isolate how many milliseconds are specifically spent inside `optimizer.step()`.

## 3. The first part of epoch 2 was temporarily slower again

| Sample | Updates | Mean preparation wait | Mean model/optimizer segment | Mean physical passes |
| --- | ---: | ---: | ---: | ---: |
| BF16 epoch 1 | 5,000 | 3.0 ms | 173.4 ms | 1.038 |
| FP32 late epoch 1 | 5,000 | 3.7 ms | 306.6 ms | 2.039 |
| FP32 first 9,730 updates of epoch 2 | 9,730 | 13.4 ms | 405.2 ms | 2.052 |
| FP32 later epoch-2 sample, steps 703,427–704,500 | 1,074 | 2.0 ms | 312.8 ms | 2.003 |

The first epoch-2 sample extrapolates to approximately **80.66 active hours**, consistent with the displayed ETA. The later sample extrapolates to approximately **60.66 active hours**, close to the post-repair epoch-1 pace. These are descriptive extrapolations, not whole-epoch completion guarantees.

Live allocation and physical split counts were essentially unchanged between late epoch 1 and early epoch 2, and the sampled epoch-2 ranges contain no new OOM retries. Across 39 matched workload profiles, early epoch 2 was still about **25% slower** than a late epoch-1 sample, so changing tensor-shape proportions alone does not explain that interval. In the later epoch-2 sample, matched profiles are only about **1.8% slower** than the epoch-1-end sample.

Preparation stalls of up to ~17 seconds occurred early in epoch 2, but measured preparation wait accounts for only **3.19%** of its active update time. It is a secondary effect, not sufficient evidence that storage starvation caused the whole slowdown. The remaining temporary increase is inside the combined model/optimizer/CPU-dispatch segment. Historical clock, utilization, power, scheduling and competing-load traces were not recorded, so its exact cause cannot be established read-only from these logs. It would be speculation to attribute it conclusively to temperature, a specific application, or a compiler event.

## 4. GPU utilization

Smaller physical batches and more per-parameter conversion/copy operations are consistent with lower useful GPU occupancy or more CPU-dispatch gaps. The FP32 optimizer loops over about 1,110 parameter tensors to stage gradients and copy updated master values back. These are reasons to benchmark the implementation later; the current log does not separately time those operations.

The single hardware snapshot showed **79% GPU utilization**, **2,895-MHz SM clock**, **58°C**, and about **9,963 MiB** device memory used. This agrees with the user's current utilization observation, but cannot reconstruct earlier utilization or prove absence of past throttling. Driver/device memory and PyTorch allocator measurements are different accounting views.

## 5. What to change later, without interrupting this run

1. **Correct ETA presentation:** show a recent-window throughput estimate alongside epoch-to-date speed, explicitly distinguish active training from wall time, and restart the throughput regime when an execution-policy migration occurs. Avoid projecting old BF16 averages into an FP32-only epoch.
2. **Benchmark activation checkpointing versus physical batch size:** more aggressive recomputation may free enough memory to restore larger physical batches and offset its own cost. Test on a separate, stopped-run benchmark with representative variable/length profiles; do not assume that either maximal checkpointing or larger batches must win.
3. **Profile and batch/fuse optimizer staging/copying:** preserve FP32 masters/moments while reducing Python/kernel-launch overhead. The earlier full-size optimizer smoke test established functionality and peak optimizer memory, not end-to-end training throughput. A representative forward/backward/optimizer benchmark is needed.
4. **Do not clear OOM limits or raise the VRAM budget in the live process:** the limits were triggered by real allocation failures. The larger physical batches need a measured memory reduction first.
5. Record GPU clocks/utilization and separate transfer, forward/backward, optimizer and preparation time in a future diagnostic run. No profiler or continuous telemetry process was attached here.

Reverting to direct BF16 parameter updates would restore the numerical defect. The useful target is recovering throughput while retaining correct update accumulation.

## Evidence and reproduction

[Evidence JSON](evidence.json) contains bounded per-range statistics, shape groups, slow updates and timing endpoints. The [analysis script](../../../scripts/diagnose_epoch_throughput.py) binary-seeks journal ranges instead of scanning the full active journal, freezes each file's read endpoint, validates step continuity and uses only CPU parsing. It writes only this diagnostic folder.

```powershell
.\.venv\Scripts\python.exe scripts/diagnose_epoch_throughput.py
```

Relevant code: [ETA and timing](../../../src/05_model_training/timed_training.py), [OOM recovery and physical splitting](../../../src/05_model_training/epoch_training.py), [FP32 optimizer](../../../src/05_model_training/master_optimizer.py), [prefetch](../../../src/05_model_training/performance.py).

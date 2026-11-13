# Matched time-series forecast latency benchmark

Measured on NVIDIA GeForce RTX 5070, PyTorch 2.12.0+cu130, CUDA 13.0.
The initial full matrix used one WSL process; confirmations used isolated
processes in the same environment on the same GPU.
The original holdout comparison used **BF16 RWKV versus FP32 baselines**;
Timestral also enabled its default mirrored forecast averaging. Those numbers
compare usable default pipelines, but are not a controlled precision/single-pass comparison.

| Model | Parameters | Weight precision in matched GPU tests |
| --- | ---: | --- |
| RWKV, 25 layers, width 1024 | 447,482,144 | BF16 |
| RWKV, 17 layers, width 1024 | 307,292,448 | BF16 |
| Chronos-2 | 119,477,664 | BF16 |
| Timestral | 209,312,256 | BF16 |

Provider/API references: [Chronos](https://github.com/amazon-science/chronos-forecasting),
[Timestral](https://github.com/mistralai/timestral). Exact checkpoint revisions,
installed runtime versions and input/output contracts are saved in the environment JSON files.

## Main findings: confirmed anchors

- The paired depth confirmation measured **29.4% lower median
  latency**, approximately **1.42x faster**, for 307M versus 447M across five anchors.
- In the matching isolated-provider GPU anchors, the reduced model beat both
  Chronos and single-pass Timestral in **4/5** cases.
  These are the short-horizon cases; it remains much slower on the long horizon.
  The following table gives measured milliseconds, not estimates from parameter count.

| context | horizon | covariates | rwkv_447m | rwkv_307m | chronos | timestral | timestral_default |
| --- | --- | --- | --- | --- | --- | --- | --- |
| 32 | 16 | 3 | 24.10 | 16.80 | 18.67 | 22.90 | 45.73 |
| 512 | 16 | 0 | 23.08 | 16.52 | 17.92 | 20.87 | 40.79 |
| 512 | 16 | 3 | 24.33 | 17.17 | 18.99 | 25.29 | 47.19 |
| 512 | 256 | 3 | 393.24 | 278.77 | 19.28 | 26.83 | 53.99 |
| 2048 | 16 | 3 | 24.15 | 16.94 | 18.73 | 23.54 | 46.93 |

Full and reduced RWKV alternate order on each repeat in one process. External
providers are then measured one at a time with the exact same five input payloads,
GPU, BF16 weights, nine output quantiles, four CPU threads and CUDA math flags.
Both runs cap PyTorch allocation at 4 GiB. They are separate processes/times,
so they do not eliminate clock/thermal/scheduler variation. Each cell has three
warmups and 20 timed requests. This uses little memory and avoids co-resident
baseline models during the depth comparison.

## Full initial matrix and its limits

- Reducing RWKV from **447,482,144** to **307,292,448**
  parameters reduced per-case median latency by a median **13.7%**
  (1.13x geometric-mean speedup).
- The reduced model was faster than Chronos in **0/32** cases,
  faster than single-pass Timestral in **12/32**,
  and faster than both in **0/32** in that initial sequence.
  This differs from the isolated short-horizon confirmation above. The first
  matrix measured the reduced model in a later pass, after profiling and mixed
  provider traffic. It therefore should not be used to claim a stable universal
  ranking. Raw results are retained rather than discarded or blended with the
  confirmation. The paired confirmation is the stronger estimate of depth-only scaling.
- Horizon matters: RWKV currently predicts 16 points per evaluation and reprocesses
  its history for every new patch. A 256-point request requires 16 evaluations.
  The external models can produce much larger forecast blocks per evaluation.
- The reduced model is a **17-layer depth ablation** of the existing model, not
  a newly trained 300M checkpoint. Its speed is measured directly; its forecast
  accuracy is not evidence for the accuracy of a properly trained smaller model.

## Matched request contract

One target, contexts 32/128/512/2048, horizons 1/16/64/256, zero or three past
covariates: **32 identical scenarios per model**. Three warmups for each scenario
precede 20 timed requests. All use the same RTX GPU, CPU threads=4,
CPU float32 input tensors and nine CPU float32 output quantiles (0.1 through 0.9).
All main-comparison model weights and eligible model matrix operations use BF16.
The provider-specific normalization and numerical stabilization remain unchanged;
this does not mean every operation uses the same dtype. No actual future covariates
or future outcomes are supplied, and cross-table learning is disabled.

`timestral` disables symmetric averaging; `timestral_default` keeps the original
two-pass reflected-context average. Both retain the library's clipping behavior.
Timestral internally emits 21 quantiles and the adapter selects the requested nine;
RWKV internally emits nine. Different internal heads/block sizes/normalization
are architectural/API differences, not identical FLOP workloads. Disabling Timestral
augmentation can affect accuracy: this is a latency study, not a new accuracy ranking.

Every timed call starts with historical payloads already in host RAM and ends
with CPU quantiles. It includes model preprocessing/tokenization, transfers and
adapter execution. CUDA is synchronized before and after. Model loading, input
generation, logging and result writes are excluded. Model order rotates per repeat
in the primary matrix. Reduced-depth and FP32 passes are later in the same process,
so thermal/clock drift cannot be perfectly eliminated. The GPU also drives Windows
display workloads; no exclusive compute mode was claimed. Warmup rows remain
available separately; compilation is never silently included in steady-state averages.

Main model configurations remain production defaults apart from matched precision,
Timestral's explicitly labelled averaging switch and the clearly labelled depth ablation.
The multi-patch RWKV adapter uses the existing optimized holdout patch path repeatedly,
feeding back medians and marking future covariates unknown. It does not implement
recurrent-state caching, a new forecast head or architecture changes. It measures
the current efficient patch adapter extended to longer requests, not an idealized
stateful RWKV engine or the separate public GPU-tokenizing `forecast` API.

## Median GPU latency by scenario

### Context 32, one target + three past covariates

All entries are median milliseconds; 20 measured requests per cell.

| horizon | rwkv_447m | rwkv_307m | chronos | timestral | timestral_default |
| --- | --- | --- | --- | --- | --- |
| 1 | 25.06 | 21.61 | 19.01 | 22.64 | 45.04 |
| 16 | 24.97 | 20.13 | 19.17 | 22.80 | 45.54 |
| 64 | 98.05 | 82.04 | 19.24 | 24.97 | 49.39 |
| 256 | 394.87 | 327.35 | 19.41 | 28.05 | 53.95 |

### Context 512, one target + three past covariates

All entries are median milliseconds; 20 measured requests per cell.

| horizon | rwkv_447m | rwkv_307m | chronos | timestral | timestral_default |
| --- | --- | --- | --- | --- | --- |
| 1 | 27.09 | 21.66 | 20.47 | 24.51 | 49.65 |
| 16 | 25.84 | 22.34 | 19.92 | 24.50 | 47.29 |
| 64 | 121.95 | 92.79 | 23.13 | 32.29 | 61.99 |
| 256 | 456.90 | 455.09 | 22.34 | 31.73 | 61.72 |

### Context 2048, one target + three past covariates

All entries are median milliseconds; 20 measured requests per cell.

| horizon | rwkv_447m | rwkv_307m | chronos | timestral | timestral_default |
| --- | --- | --- | --- | --- | --- |
| 1 | 27.61 | 25.94 | 21.23 | 26.27 | 51.45 |
| 16 | 26.94 | 24.79 | 20.89 | 25.40 | 50.94 |
| 64 | 104.63 | 90.36 | 20.66 | 28.81 | 57.07 |
| 256 | 410.07 | 409.05 | 20.17 | 28.57 | 56.51 |

## Slowest RWKV holdout request

The recorded holdout contains 1076 scored RWKV requests.
Median was **22.739 ms**, p99 **26.531 ms**,
maximum **294.405 ms** (12.9x median).
Only **1** scored request exceeded ten times the median.
The maximum was table index 6, origin
32, context 32,
covariates 3. This exact shape was already used
and warmed in table 0, so it is incorrect to assert this specific maximum was
necessarily first-use shape compilation. The old journal has no GC/compiler/CPU
profiling telemetry, so its exact historical cause cannot be proven after the fact.

The benchmark replays that exact real input, instruments Python garbage collection,
wall and process CPU time, compiled-graph counts and memory. It also contrasts
normal GC, temporarily disabled GC, explicit full GC inside a request, and resetting
the Python graph cache while retaining disk binaries. Forced collection is a
diagnostic, not representative traffic. Disabled GC is an experiment, not a production
recommendation. GC scans the Python object heap regardless of whether predictions
are otherwise on GPU.

| phase | requests | median_ms | max_ms | gc_ms | gc_events | new_graphs |
| --- | --- | --- | --- | --- | --- | --- |
| diagnostic_forced_gc | 5 | 182.58 | 235.59 | 822.10 | 5 | 0 |
| diagnostic_gc_disabled | 100 | 24.81 | 40.34 | 0.00 | 0 | 0 |
| diagnostic_gc_enabled | 100 | 24.88 | 30.15 | 0.88 | 5 | 0 |
| diagnostic_graph_cache_reset | 3 | 26.67 | 1590.45 | 12.81 | 117 | 1 |
| diagnostic_warmup | 5 | 24.88 | 27.07 | 0.00 | 0 | 0 |

Interpret these as causal tests of mechanisms, not proof of which event happened
in the older run. `tail_replays.csv` exposes whether each new slow call actually
overlapped collection or compilation. `rwkv_warm_trace.json.gz` is a compressed Chrome/Perfetto
trace of the warmed request, with an operator summary in `rwkv_warm_profile.txt`.
See the detailed interpretation appended after reviewing those recordings.

### Interpretation of the tail investigation

The exact request did **not** reproduce its historical 294 ms delay in 100 normal
replays: median about 24.9 ms, maximum about 30.1 ms, with no new compiled graphs.
Temporarily disabling GC did not improve its median and still had a roughly
40 ms maximum. Explicitly forcing a full generation-2 collection produced
roughly 178–236 ms requests, with 151–209 ms directly attributed to GC callbacks.
Resetting the graph cache produced a roughly 1.59-second first call and one
new compiled graph, then returned to normal latency. These demonstrate two
mechanisms for large pauses; neither proves what happened to the historical
maximum. That run did not record the necessary telemetry. OS scheduling and
shared display-GPU activity also remain possible causes.

The warmed profiler trace recorded **1172 GPU kernels**,
with about **4.34 ms** of non-overlapping
kernel activity spread across **39.85 ms**
from first to last kernel. The profile itself adds overhead, so these are not
unprofiled benchmark latencies. They do indicate substantial host submission/idle
gaps among many small kernels. Parameter count alone therefore does not determine
request latency, and fewer layers do not guarantee proportional speedup.

The practical next experiments would be shape-specific prewarming and CUDA-graph
capture of the repeated inference region, following
[PyTorch CUDA guidance](https://docs.pytorch.org/docs/main/notes/cuda.html).
Those optimizations were **not** applied to the production model in this task.
Blindly disabling GC would not resolve all tails and was not made a default.

## Cold overhead and original precision

Maximum per-case warmup durations and compile/GC counters (not scored latency):

| model | precision | max_ms | compiled_graphs | gc_ms |
| --- | --- | --- | --- | --- |
| chronos | bf16 | 32.55 | 0 | 0.75 |
| chronos | fp32 | 77.79 | 0 | 0.16 |
| rwkv_307m | bf16 | 615.78 | 1 | 5.13 |
| rwkv_447m | bf16 | 10322.44 | 2 | 220.42 |
| timestral | bf16 | 132.96 | 0 | 0.00 |
| timestral | fp32 | 90.77 | 0 | 0.00 |
| timestral_default | bf16 | 76.31 | 0 | 6.27 |
| timestral_default | fp32 | 145.48 | 0 | 0.00 |

`summary.csv` also contains FP32 Chronos and Timestral single/default comparisons
for all scenarios. TF32 allowed in the runtime: True.
The BF16 main matrix is the controlled precision comparison; FP32 rows reproduce
the external providers' existing parameter precision and should not be mixed with it.

## CPU and limits

The vendored RWKV backend requires CUDA with BF16. A CPU benchmark of all three
would require an alternate RWKV kernel implementation. None was substituted,
so no three-model CPU ranking is claimed. Matching precision and request contracts
improves fairness; it cannot make different architectures, output heads, trained
quality or decoding algorithms literally identical. These measurements answer
which current adapter is faster for the specified workload on this machine.

Supplementary FP32 CPU measurements of Chronos and Timestral on
**AMD Ryzen 5 7600X3D 6-Core Processor**, four threads, five measured requests per cell:

| context | horizon | covariates | chronos | timestral | timestral_default |
| --- | --- | --- | --- | --- | --- |
| 128 | 16 | 0 | 31.89 | 41.09 | 82.67 |
| 128 | 16 | 3 | 61.36 | 88.30 | 189.09 |
| 128 | 256 | 0 | 49.46 | 58.19 | 106.79 |
| 128 | 256 | 3 | 102.73 | 165.11 | 344.39 |
| 2048 | 16 | 0 | 137.82 | 140.19 | 267.41 |
| 2048 | 16 | 3 | 441.97 | 892.08 | 1729.30 |
| 2048 | 256 | 0 | 126.19 | 141.83 | 254.49 |
| 2048 | 256 | 3 | 502.25 | 898.40 | 1900.84 |

Raw CPU results are
in `cpu_summary.csv` and `cpu_environment.json`. They do not constitute a CPU
comparison against RWKV. The CPU pass uses the same WSL environment and four
threads, after all GPU benchmarks have stopped.

## Interruption and resource controls

The complete initial matrix and diagnostics finished successfully. A second,
fully interleaved confirmation was interrupted by the user while loading external
weights because PyCharm froze; it had not saved any measured cases. Its partial
`paired/` directory is not used in these results. The cause of the IDE freeze
was not established from the saved logs.

The resumed depth confirmation loads **only one RWKV checkpoint**, sharing its
weights between the full and reduced views, caps PyTorch GPU allocations at
4 GiB, limits compiler/build workers to one and writes live data under `.cache`,
which the project excludes from PyCharm indexing. It measures five matched
anchor cases with full/reduced request order alternated on each repeat. Results
and process/VRAM peaks are in `depth_confirmation_summary.csv` and
`depth_confirmation_environment.json`. External-provider comparisons remain the
completed main matrix; confirmation measurements are not mixed into its win counts.
The provider-only anchor confirmation adds matched external timings without
loading them alongside RWKV. Peak process RSS stayed around 3.4–3.6 GiB for
these resumed GPU runs, with RWKV's allocator peak below 1 GiB. Those are
process/allocator measurements, not total Windows GPU/RAM consumption.

## Reproduce and inspect

Settings: `configs/time_benchmark.json`. From the repository in WSL:

```bash
source scripts/activate_rwkv7.sh
python -u scripts/time_benchmark.py
python -u scripts/time_benchmark_confirm.py
python -u scripts/time_benchmark_baseline_confirm.py
python -u scripts/time_benchmark_cpu.py
python scripts/report_time_benchmark.py
```

- [Interactive latency chart](2026-10-31/latency.html)
- [Interactive confirmed anchors](2026-10-31/confirmed_latency.html)
- [Full scenario statistics](2026-10-31/summary.csv)
- [Raw request timings and diagnostics](2026-10-31/requests.csv)
- [Environment/model/config metadata](2026-10-31/environment.json)
- [Exact outlier replays](2026-10-31/tail_replays.csv)
- [Historical worst requests](2026-10-31/historical_worst_requests.csv)
- [Size reduction and win counts](2026-10-31/comparison.json)
- [Paired size confirmation](2026-10-31/depth_confirmation_summary.csv)
- [Matched external GPU anchors](2026-10-31/baseline_confirmation_summary.csv)
- [Supplementary CPU timings](2026-10-31/cpu_summary.csv)

The compact interactive HTML charts load Plotly from its CDN and require network
access. Raw CSV/JSON statistics and this Markdown report work offline. The bulky
profiler trace is compressed to avoid IDE indexing of large JSON event streams.

No training checkpoint, production holdout adapter, configuration or scoring output
was modified by the benchmark. Raw timings and warmups are preserved; no tail
observations were removed from scored summaries.

## Verification

The initial matrix contains **5120 scored GPU requests**
plus separately retained warmups/diagnostics, across 32 scenarios. The resumed
checks add 200 scored RWKV requests and 300 scored external GPU anchor requests;
the CPU supplement adds 120 scored forecasts. The full repository suite passed:
**187 tests, 180 passed, 7 CUDA-dependent skips** in the Windows CPU test runtime.
Actual WSL GPU and CPU executions are recorded separately above. Applicable
repository lint, formatting and security hooks passed on the benchmark changes.
Peak recorded initial-matrix PyTorch allocation was
**1.86 GiB**, reserved
**2.13 GiB**; these do not include all driver/display allocations.
There is no evidence of allocator-cap overflow in these recordings; this does
not retrospectively establish the memory behavior of the old holdout outlier.

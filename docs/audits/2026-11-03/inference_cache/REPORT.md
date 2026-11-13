# Inference cache coverage regression

Date: 2026-11-03. Checkpoint: 394,914,080 parameters, best epoch 6,
step 85,746, SHA-256
`b5811ff4ea3a83688a8515e9df62638a8a9019e70474848e6db245e7f5b1e208`.

## Cause

Graph replay was enabled in the user's completed holdout, but only the first
eight profiles could be cached. There were 85 actual profiles across 1,270
forecast requests. Profiles distinguish variable counts, padded context lengths
and requested output patch counts. The additional horizons and changed cohort
changed which eight profiles arrived first. These profiles initially covered
only two variable counts, leaving common later cases uncached. More model
layers did not disable replay.

The original log proves replay enabled and records all eight captures:
`logs/20261103T170354_564971Z_holdout_test_2f206829/events.log`.
Joining those keys to the saved request latencies gives:

| Original execution | Requests | Median latency |
| --- | ---: | ---: |
| Cached profiles | 390 (30.7%) | 7.01 ms |
| Uncached profiles | 880 (69.3%) | 23.31 ms |

When the cache filled, `FastInference.quantiles` explicitly returned to the
third-party training-style forward path. That is why the overall median looked
like ordinary non-replay inference even though the configuration said replay
was enabled. The one-token median alone was 22.65 ms, so longer forecasts
cannot explain the latency regression by themselves.

## Runtime changes

- Default profile limit increases from 8 to 256, with a configurable tracked
  cache budget of 1.5 GiB, separate from total process VRAM. Existing global
  allocator controls remain. No model or training weights are changed.
- Cache admission measures allocator/device growth and static input storage,
  including setup/warmup growth that may persist outside the outer graph pool.
  Estimates reserve headroom before capture; an over-budget new graph is retired
  safely after synchronization. Compiler/driver memory can outlive a graph, so
  this is not a cap on all CUDA process memory. The overall allocator limit is
  still required. Established profiles are not continually evicted/recaptured.
- Uncached profiles retain the inference-only forward implementation instead
  of selecting the training-style implementation. Matched forecast verification
  includes these new fallback executions.
- Holdout uses request metadata to admit profiles expected at least eight times
  (`holdout.graph_min_profile_requests`). It counts 36 recurring profiles in
  this sample, covering 1,136 of 1,270 scored model passes (89.4%). Rare shapes
  use forward-only execution without paying capture costs. Planning uses table
  lengths, variable roles, existing origins and seeded horizons, not future
  values. No extra warmup forecasts are introduced and the timing boundaries
  remain unchanged. Ordinary API forecasts, with no advance request plan,
  continue capturing on first use within count and memory limits.
- Final metrics/progress expose replay coverage, captures/time, cache profiles,
  tracked bytes, fallback counts/reasons and the admission plan. Initial model
  provenance records runtime settings through initial execution statistics.
  Counters include warmup model passes; latency summaries exclude warmup.
- Profile-capture logs move to DEBUG. Count/memory saturation warns once per
  reason; intentional low-reuse fallbacks do not generate warning spam.

Relevant settings:

```json
"training": {
  "inference_cuda_graphs": true,
  "inference_graph_cache_size": 256,
  "inference_graph_memory_gib": 1.5
},
"holdout": {
  "graph_min_profile_requests": 8
}
```

Set the holdout threshold to 1 to capture every seen profile that fits. This
can help a long-running workload but increases cold tails on a small sample.
These settings also apply in `configs/holdout_447m_current.json`.

## Experiments

### Controlled RWKV-only replay

`benchmark.json` and `requests.parquet` repeat all 1,270 saved requests on the
same trained checkpoint. The original implementation is frozen as
`baseline_fast_inference.py`. Each engine runs a first pass and a second warm
pass, on one GPU without competitor model execution between requests.

| Warm pass | Median | p95 | Total request seconds |
| --- | ---: | ---: | ---: |
| Original eight-profile implementation | 21.50 ms | 24.01 ms | 22.41 s |
| Larger-cache prototype | 7.28 ms | 12.08 ms | 10.06 s |

All forecasts match the saved median forecasts exactly. This prototype
captured all 85 profiles. Its first-pass median was 7.23 ms but p95 84.24 ms,
showing why warm medians alone are insufficient. First-use code/compiler work
also occurred earlier in the original-engine phase; do not use those cold
totals as a fair independent cold-start comparison. The prototype preceded the
final conservative setup-memory accounting and frequency admission policy.

### Real four-model holdout repetitions

These run the actual holdout entrypoint with all four models resident on CUDA
where applicable, CPU naive, identical checkpoint/data/origins/horizons and the
existing timing protocol. They preserve the original production directory and
write independent audit outputs.

| Holdout execution | RWKV median | RWKV p95 |
| --- | ---: | ---: |
| User's original eight-profile run | 22.76 ms | 26.37 ms |
| Capture every profile (`paired_holdout`) | 7.61 ms | 93.36 ms |
| Admit recurring profiles (`paired_holdout_admission`) | 8.04 ms | 26.12 ms |
| Final bounded-memory implementation (`paired_holdout_final`) | **7.99 ms** | **25.08 ms** |

The second experiment reproduces the cold-tail cost: 85 captures in a small
sample are over 5% of requests. The admission experiment captures 36 profiles,
retains about 89% replay coverage and avoids rare-profile capture overhead.
Its median is about 2.8 times faster than the original, while p95 is essentially
unchanged. Repeated measurements can vary with clocking and host contention;
these are measured request timings, not guaranteed service-level latencies.

The final conservative-memory-accounting rerun is written separately to
`paired_holdout_final`; its metrics and `verification.json` are the authoritative
final implementation results. No statement here changes the accuracy scores
or excludes cold requests after the normal initial warmup. Rare multi-token
shapes can still be slower than the common cached one-token shapes.

The final run retained 32 of the 36 admitted profiles within the 1.5-GiB
tracked cache budget. Replay covered 1,078 of 1,271 model passes, including the
initial warmup (84.8%). There were 134 intentional low-reuse fallbacks and 59
memory-admission fallbacks, no profile-count-limit fallbacks. The tracked
setup/cache growth was 1,609,733,488 bytes, below the 1,610,612,736-byte budget.
Memory accounting includes setup-owned buffers; earlier prototype retained-byte
statistics excluded that setup growth and should not be interpreted as the
full cache footprint. Free device memory after final evaluation was about
6.9 GiB of 11.9 GiB total, with all models resident; this is a final snapshot,
not peak memory during capture.

In the same final run, Chronos median/p95 was 18.52/21.69 ms and Timestral
43.89/50.33 ms. Basin's median improved about 2.85 times relative to the user's
original run. It had the fastest foundation-model median, but Chronos retained
the lower p95. Overall end-to-end benchmark time includes all competitors,
startup, plotting and writing; it does not improve by that same factor.

Every experiment compares the forecast values with the original saved
holdout. All three completed full paired experiments matched all **22,276 scored
values for every model exactly**. Their identities match the checkpoint hash
above. Final verification is machine-readable in each output directory.

## Accuracy interpretation

The runtime fix does not change forecast accuracy. The original leaderboard
MASE of 2.227977 for Basin now pools one-, two-, three- and four-token requests.
Older reported headline scores came from one-token-only tests with different
selected sources and admission boundaries; they are not matched comparisons.

On this run's one-token requests only:

| Method | MASE |
| --- | ---: |
| Basin 395M | 1.690954 |
| Chronos-2 | 1.560317 |
| Timestral | 1.470941 |
| Naive | 2.646143 |

This remains a real accuracy gap. Restricting to identical saved one-token
requests shared with the earlier archived-447M evaluation gives 13,903 scored
points, 971 origins and 230 tables. Historical actuals and MASE scales match
exactly:

| Method | Matched-subset MASE | RMSE |
| --- | ---: | ---: |
| Basin 395M | 1.589983 | 45.1073 |
| Archived Basin 447M | 1.500930 | 39.7186 |
| Chronos-2 | 1.455774 | 37.2893 |
| Timestral | 1.394023 | 34.9446 |
| Naive | 2.461296 | 100.3436 |

The new model is about 5.9% worse in MASE than the archived model on this
intersection. This is descriptive evidence on an already inspected subset,
not a fresh unbiased evaluation. It does not isolate capacity, objective or
data effects. No training or accuracy-tuning experiment was run in this task.
The earlier stopping/scheduler issue also remains relevant to training policy;
the runtime fix does not address it.

## Artifacts and reproduction

Verification pinned the original 2% sample in `reference/`, independently of
the live configuration (which was changed to 25% during diagnosis). An initial
attempt with the live 25% setting was terminated before forecasting, then
restarted with the frozen selection. The current 25% user setting is preserved.

The native suite completed 258 tests, with 12 CUDA-only tests skipped there.
All eight fast-inference tests passed in the WSL CUDA runtime, including three
CUDA cases covering graph/fallback equivalence, current-stream recurrence and
state restoration after capture warmup. Final full-checkpoint verification
also exercised both low-reuse and memory-admission fallbacks against the saved
forecasts. Code checks passed (AST, Ruff, formatting and Bandit).

- `saved_run_evidence.json`: original replay-key coverage and matched accuracy.
- `benchmark.json`, `requests.parquet`: controlled cache comparison.
- `paired_holdout/`: all-profile prototype full holdout and verification.
- `paired_holdout_admission/`: frequency-admission full holdout and verification.
- `paired_holdout_final/`: final memory-accounting full holdout and verification.

From the activated WSL environment:

```bash
python scripts/report_inference_cache_regression.py
python scripts/benchmark_inference_cache.py
python scripts/verify_inference_cache_holdout.py
```

The verifier writes only the final audit directory. To regenerate the normal
holdout outputs using the new defaults:

```powershell
uv run python .\src\06_holdout_test\holdout_test.py
```

No model retraining, transformation or embedding preparation is needed.
The original `results/model/holdout_test` metrics remain untouched by diagnosis;
they retain their old latency values until the normal stage is rerun.

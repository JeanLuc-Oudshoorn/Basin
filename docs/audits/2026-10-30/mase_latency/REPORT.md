# Holdout MASE and request latency

## Request and scope

Make MASE the leading holdout accuracy metric, and compare the actual time
between a prediction request and its CPU result for RWKV-7, Chronos-2,
Timestral and configured classical baselines. Model architecture, weights,
precision, saved train/test split, origin sampling and training validation
remain unchanged. Changes are directly in the holdout stage.

## Accuracy definition

For historical target values `y[0:T]`, seasonality `m` and scored observations
`a[i]`, define `s = mean(abs(y[t] - y[t-m]))` over historical pairs for which
both values are finite. The point's scaled absolute error is
`abs(prediction[i] - a[i]) / s`; MASE is its arithmetic mean.

The default `holdout.mase_seasonality=1` uses the nonseasonal naive denominator.
Patch size 16 is not assumed to be the seasonal period. Positive integer `m`
can be configured globally when the compared tables share an appropriate period.
The historical context is capped identically for all models and the scale, and
recomputed at every forecast origin. Future labels never enter the denominator.
Missing pairs are dropped without interpolation or bridging time gaps.

Zero scales, insufficient history and histories without valid pairs yield an
undefined scale, stored as null in parquet. Such observations are excluded from
MASE only, with valid and excluded counts reported for each method. An entirely
undefined MASE is JSON `null`. RMSE and MAE still cover every scored finite
actual. There is no epsilon or silent fallback to a different denominator.

Pooled MASE weights each valid scored observation equally. Per-origin and
per-table MASE are also retained. This preserves the previous pooled metric
weighting; it is not an equal-table macro score. `leaderboard` sorts on MASE
ascending with undefined values last. MASE < 1 compares to the historical
one-step naive scaling error, not directly to the held-out multi-step naive
forecast. The latter has its own measured MASE.

Primary reference: [Hyndman and Koehler, Another look at measures of forecast
accuracy](https://robjhyndman.com/publications/another-look-at-measures-of-forecast-accuracy/).
See also the [forecast package's accuracy implementation](https://github.com/robjhyndman/forecast/blob/master/R/errors.R)
for seasonal/nonseasonal naive scaling.

## Timing definition and fairness

The old loop batched RWKV and Chronos origins but called Timestral separately.
Amortized batch wall time would therefore not be a comparable request latency.
The new default directly times one origin at a time for every model: one table,
all targets and covariates, one full patch. The final partial patch still asks
every model for 16 predictions but scores only the available actuals. Model
order rotates across tables and origins, with the order saved in the raw log.

Start boundary: request history is already in host memory, with shared
dataframe/tensor construction complete. Stop boundary: the adapter has returned
its CPU forecast. Includes model-specific preprocessing, embedding/tokenization,
normalization, host/device transfers, model execution and output conversion.
Prophet includes forecasting regressors, fitting targets and prediction. Each
origin gets new classical fits; warmup does not reuse a fitted Prophet model.
Excludes weights/downloads/loading, reading and validating source files, common
payload preparation, calculating metrics, plotting and result serialization.
These are local function requests; no network or HTTP serving overhead is modeled.

CUDA devices are synchronized before the timer starts and after the adapter
returns; CPU perf_counter measures wall-clock duration, not asynchronous kernel
submission time. The timer synchronizes the device actually holding that model.
See [PyTorch CUDA semantics](https://docs.pytorch.org/docs/main/notes/cuda.html)
and [torch.cuda.synchronize](https://docs.pytorch.org/docs/main/generated/torch.cuda.synchronize.html).

One warmup request per model on the first origin is retained separately by
default; subsequent origins with new shapes may still incur compilation cost.
This cost remains in scored latency rather than being silently filtered out.
Setting warmup_requests=0 includes initial cold work in scored measurements.
The repeat count is configurable, with only the first scored result used for
accuracy and every timing repeat retained. Fixed request shapes, targets,
covariates and horizons are shared between providers, with no cross-table
learning or future covariates. Precision and quantile output workloads retain
their actual adapter defaults; this is a comparison of the current pipelines,
not a controlled experiment with identical parameter count or output heads.

Reports contain model parameter count, dtype/device, library/hardware metadata,
loading durations separate from inference, mean/median/p95 and range, context
length buckets, plus paired RWKV/baseline latency ratios on identical requests.
Raw logs permit more detailed grouping by covariates and context length. Model
weights are resident together, as in the existing holdout runner; timings are
serial, not concurrent throughput or isolated-process benchmarks. Other running
GPU processes affect the results and should be absent for a performance run.

## Files and recovery

- `configs/config.json`: MASE lag, warmup and repetitions; the obsolete holdout
  inference batch size is removed because requests now consistently have size 1.
- `src/06_holdout_test/holdout.py`: historical scaling, leading MASE, serial
  measured requests, raw result persistence, ranked summary.
- `src/06_holdout_test/latency.py`: synchronized timer, warmup/repetition policy,
  summaries, contextual buckets and paired ratios.
- `src/06_holdout_test/benchmarks.py`: optional single-provider dispatch,
  separately recorded external-model load time; the batched public adapter API
  remains available to existing callers.
- `src/06_holdout_test/holdout_test.py`: separate RWKV load duration/provenance.
- `tests/test_holdout_metrics_latency.py`: arithmetic, coverage, no future-scale
  leakage, full/partial patch requests, adapters, synchronization, repeats and
  serializable metrics. Protocol fixtures explicitly disable warmup where they
  assert call counts and chronological context order.

Raw timing JSONL appends only new rows and flushes/closes after each completed
table. Final parquet is written once; the whole accumulated timing history is
not rewritten after every table. Progress/ETA remains INFO every 100 tables
and completion; timing summaries refresh at that same interval. Per-table
forecasts and progress JSON continue updating after each completed table.

Existing completed metrics cannot acquire historical latency retroactively.
An active process keeps previously imported Python code. New metrics become
available on the next holdout invocation; they do not require retraining.

## Verification

`python -m unittest discover -s tests`: **173 tests, 166 passed, 7 skipped**
(CUDA-dependent cases on the Windows CPU runtime), 28.349 seconds. New tests
cover numerical nonseasonal and seasonal MASE, missing/constant histories,
undefined counts, invalid configuration, denominator leakage, partial tails,
single-origin adapter calls, retained warmup/repeated timing, explicit CUDA
synchronization, percentile summaries, matched-origin ratios and context buckets.

All applicable pre-commit checks passed for the changed files: Python AST,
JSON, conflict/private-key checks, whitespace, EOF, line endings, Ruff lint and
formatting, and Bandit. TOML/YAML checks had no matching changed files.

[cpu_smoke.json](cpu_smoke.json) records a separate real CPU smoke using Prophet
with a forecast covariate and naive carry-forward: 64 historical observations,
16 future observations, one warmup and one measured request per method. Both
returned finite full-patch predictions; historical MASE scaling and raw timings
were recorded. On this small synthetic case Prophet took about 2.065 seconds
and naive about 0.0475 milliseconds for the measured request. These single
observations confirm instrumentation only, not a general latency estimate.

A user holdout process was already running in WSL with GPU activity during this task.
It was left untouched; no competing full-size GPU benchmark was started.
Therefore this change does not claim a measured RWKV/Chronos/Timestral speed
ranking. The next idle-GPU holdout run produces that comparison automatically.

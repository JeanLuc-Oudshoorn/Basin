# Holdout test stage

Run from the repository root:

```powershell
uv run python src/06_holdout_test/holdout_test.py
```

This automatically loads `paths.training/model.pt`, or `latest_checkpoint.pt`
in that directory if the recovery checkpoint is newer or no export exists.
The recovery checkpoint uses its saved `best_state_dict` (best validation weights),
with the original `dataset_split.json` beside it. Its model/data signature must
match the configuration. Recovery loading reads that split JSON, which can be
large; it does not load training windows or resume the optimizer.

Startup now streams the large split into a compact, reusable cache under
`training/.holdout_split_cache`. Training/validation overlap checks use sorted
128-bit source fingerprints instead of retaining millions of path strings.
Changing the split invalidates the cache. A fingerprint collision can only
reject an independent source; it cannot permit an actual overlap.

Raw augmentation-family checks are cached under `.cache/holdout_lineage`, with
invalidation when scanned directories change. Header reads use eight workers
and cache schema/row counts under `.cache/holdout_schemas`, keyed by the saved
source SHA256 and column policy. Actual source-file SHA256 checks still run
before evaluating each table. Startup progress is logged during cold reads.

Recovery provenance fingerprints the selected model tensors only, excluding
optimizer state. `model_used.json` records `evaluated_weights_sha256` and an
explicit scope; `checkpoint_sha256` is null for this recovery mode. Full
exported model files still receive a complete-file SHA256. Checksums are reused
only while artifact size and timestamps are unchanged. This avoids rereading
an 8 GB optimizer checkpoint just to identify a 1 GB inference model.

Checkpoint loading constructs the module on the meta device before restoring
all saved tensors, avoiding redundant random CPU initialization. Architecture,
weights, sampling, graph admission, forecast timing boundaries and scores are
unchanged. The first use of an uncached corpus still needs a streaming split
pass and a directory scan; subsequent launches reuse those results. These
changes affect new invocations, not a process already running older code.
Set `holdout.checkpoint` to a repository-relative or absolute checkpoint path
to pin a particular export or recovery checkpoint.

It replays the test split
stored in that model and verifies the checkpoint and selected source checksums. It does
not train or draw a new split. `train.py` now stops after saving its artifacts.

`data.generated_dates` selects data when preparing/training a model; changing it
does not replace a checkpoint's saved holdout sources. An old `model.pt` can
therefore still refer to deleted date folders. The automatic recovery selection
allows evaluation of the current run before training finishes, using its own
saved holdout sources and checksums. `model_used.json` records the chosen
checkpoint, weight field, best step, and recovery split path/checksum.

To evaluate the archived 447M weights on the **current model's holdout data**:

```powershell
uv run python .\src\06_holdout_test\holdout_test.py --config .\configs\holdout_447m_current.json
```

This configuration loads weights from `results/447m_model/training/model.pt`
and takes the saved holdout split from `results/model/training/model.pt`, using
`holdout.split_checkpoint`. It enables inference CUDA graph replay, samples 2%
with seed 7, and compares against naive, Chronos-2 and Timestral. Set
`holdout.sample_percent` in this configuration to change coverage. Matching the
percentage and seed in the normal configuration selects the same tables.

Results go to `results/447m_model/holdout_test/current_holdout`, including
`test_metrics.json`, interactive plots and `model_used.json`. Existing default
model results and the archived model's original holdout results are preserved.
Both checkpoint paths and hashes are recorded. The evaluated model's saved
column policy is used, and any overlap between the selected split's full
holdout and the evaluated model's recorded training sources is rejected before
sampling. Source checksums from the split checkpoint are verified as usual.
No old raw datasets, patch-index artifacts or retraining are required. Omit
`holdout.split_checkpoint` to retain the normal model-owned split behavior.

When the split checkpoint comes from a run with another data configuration, as
for an annealing experiment tested on the main run's holdout tables, name that
run's configuration in `holdout.split_config`. The split checkpoint is then
checked against it instead of against the configuration being run. Annealing
experiments set both options themselves; see
[Annealing](../../docs/annealing/ANNEALING.md).

Startup reads the checkpoint once using memory mapping, samples its saved test
sources, and reads only the selected Parquet schemas. It uses the checkpoint's
original target/covariate policy, independently of today's data admission filter.
It no longer loads `patch_index.pt` or `windows.pt`, hashes the training corpus,
or materializes records for every training table. Both those upstream artifacts
can be absent when running the holdout test. Weights are passed directly to the
model loader without rereading the checkpoint. Startup phase durations are logged;
model construction, GPU transfer and benchmark model initialization still take time.
See the [startup and screening audit](../../docs/audits/2026-10-30/bolt_screening_startup/REPORT.md).

Each run writes into its own new folder under the holdout directory, named
after the start time (UTC) and the evaluated weights, for example
`results/model/holdout_test/20261120T172808Z_epoch1_step1457590`. A run never
writes into an existing folder and nothing is cleared, so the results after
epoch 1, epoch 2 and epoch 3 stay side by side. A run that stops early leaves
its folder without `test_metrics.json`; tools that look for "the latest run"
skip it. Old run folders are removed by hand when no longer needed (one run
of 7,500 tables is about 0.6 GB).

The holdout directory is `results/model/holdout_test` by default, configured
by `paths.holdout_test`. A run folder contains metrics JSON, prediction parquet files,
interactive Plotly HTML plots, live progress/ETA and `model_used.json` recording the model's
checksum and selected epoch. `plotting.py` and `error_distribution.py` work on
the most recent completed run, or on the folder given with `--run <name>`. Ordinary logs and a timestamped JSONL journal are
written under `logs/`, including when Windows routes execution to WSL.

Each test table contributes up to five forecasts at the configured
origins, comparing RWKV, Prophet and last-observation carry-forward. All protocol
settings remain under `holdout` in `configs/config.json`.
Every method forecasts four tokens (64 timesteps) at every selected origin.
The four `model.horizon_loss_weights` coefficients (80%/16%/3%/1%) weight the
consecutive token errors. Every origin now reserves four complete future tokens.
The final patch-aligned origin is `floor((N-64)/16)*16`. Tables need at least
96 points (32 context + 64 future); shorter saved holdout tables are skipped and
counted explicitly. All 64 predictions are plotted and scored against real
actuals; masking is needed only for genuine missing data.
See the [full short-horizon protocol](../../docs/forecasting/SHORT_HORIZON_FORECASTING.md)
for boundary handling, per-horizon metrics and interpretation. This replaces
the previous sampled-horizon protocol. Legacy `horizon_probabilities` and
`horizon_seed` settings are ignored by evaluation.
`holdout.benchmark_models` selects comparisons (RWKV is always included):
`["prophet", "naive", "chronos"]` by default; add `"timestral"` or `"timeh2o"`
to include them.
`holdout.sample_percent` defaults to 100; set it to e.g. 1 for a seeded 1% sample
of the saved holdout tables. `sample_selection.json` records the selected sources.
All models use those same sources, origins and historical context.

The leading accuracy metric is **MASE** (`primary_metric="mase"`). For each
target and origin, the denominator is the mean absolute lag-`m` difference in
the historical target context, with `holdout.mase_seasonality=1` by default.
MASE is the mean absolute forecast error divided by that scale. Only finite
historical pairs enter the scale; future observations never enter it. The
history uses the same context cap as the models. Configure a different `m`
only when a common seasonal period is appropriate for the evaluated tables.
The default does not assume that every generated table has period 16.

`pooled.<model>.mase` averages scaled absolute errors using the configured lead
weights, with equal total weight per origin and target. The scale is
saved as `mase_scale` in the prediction parquet files, with per-origin and
per-table MASE in separate `origin_metrics.json` and `table_metrics.json` files.
`test_metrics.json` retains aggregates and detail-file links only.
Constant or insufficient histories have
undefined MASE: the JSON score is `null` when no valid points remain, with
`mase_observations` and `mase_undefined_observations` reporting coverage. No
epsilon or replacement denominator is used. RMSE, MAE, normalized RMSE and
RWKV pinball loss remain secondary metrics. MASE below 1 beats the historical
one-step naive MAE used for scaling; it does not necessarily beat the naive
baseline's out-of-sample, 16-step patch forecast.

The additional `table_summary.models.<model>.<metric>` reports `mean`, `median`,
`p10` and `p90` across **per-table** scores, with each table weighted equally.
This covers MASE, RMSE, MAE, normalized RMSE and RWKV normalized pinball.
Undefined/nonfinite scores are omitted and explicitly counted as
`undefined_tables`, alongside `valid_tables`. Percentiles use linear
interpolation. A mean of table RMSEs is a different statistic from pooled RMSE;
pooled RMSE takes the square root after weighted squared-error aggregation.
The leaderboard exposes the table MASE mean, median, p10 and p90 directly,
alongside pooled weighted MASE and `median_absolute_scaled_error` across points.

Every completed run also writes `error_distributions.html`: interactive,
overlaid Gaussian kernel densities with distinct colors and transparent fills
for all selected models. Each table contributes one score. By default, KDE is
estimated in `log(1 + error)` space to show both ordinary errors and extreme
tails without dropping or clipping cases. The axis label states this transform;
hover displays the original error. Areas under each model's density equal one
(approximately on the finite plotting grid). Density values are in transformed
space, not density per raw-error unit. Constant samples appear as labeled
markers rather than an invented smooth density. Raw JSON metrics and summary
percentiles always stay on the original metric scale.

Configure `plot.error_distribution` with `enabled` (default true), `axis`
(`"log1p"` or `"linear"`) and `grid_points` (default 512, minimum 32).
To add these summaries and plots to an already completed run, without rerunning
any forecasts or loading models:

```powershell
uv run python .\src\06_holdout_test\error_distribution.py
uv run python .\src\06_holdout_test\error_distribution.py --config .\configs\holdout_447m_current.json
```

This moves embedded detail lists into separate JSON files, updates aggregate
fields and writes error and latency distribution plots. Stored forecasts and pooled scores remain
unchanged.

Every model is also measured for **end-to-end local request latency**. A
request contains one table at one origin, all its targets and past covariates,
and asks for four patches (64 timesteps). Every model uses that same horizon, including
with four complete future tokens available for scoring. Requests run
serially with batch size 1, and their model order rotates across origins.
This replaces the old holdout `inference_batch_size` setting; training and
validation batching are unaffected.

Only a sample of the tables is timed this way: `holdout.latency.tables=500`
(about 2,500 requests per model; `null` times every table). The sample is a
seeded random choice (`holdout.latency.sample_seed`, default the global seed),
listed in `latency_selection.json`, and is forecast first. Every other table is
forecast in one call per model for all its origins, which is several times
faster and is not timed. Forecasts and scores of both kinds of table enter the
accuracy results in the same way. One-off costs (RWKV's graph captures) weigh
more in the mean of a few thousand requests than in a fully timed run; compare
medians.

A forecast made in a shared call should equal the one a request on its own
gives. It does up to rounding, checked on 197 tables forecast both ways on the
GPU (2026-11-23): single points differ on average by 0.4% of the history's
standard deviation for RWKV (BF16) and by less than 0.02% for Chronos-2,
Timestral and TimeH2O, in neither direction more often; the table MASE
percentiles of every model agree to three or four digits.

TimesFM 3.0 was part of the comparison for one run (epoch 2, 2026-11-23) and
was then taken out. It pads the histories of one call to the longest, and that
padding moved its forecasts by up to 0.3 MASE units, so it could not share
calls. That run's TimesFM latency (median 332 ms) is mostly not the model's
own: the `timesfm` package runs Python's garbage collection after every call.
In the holdout process that took 156 ms per call in a 200-table check, and more
as a run collects results; the same request in a process of its own takes 47 to
123 ms.

Timing starts with the historical CPU request payload already constructed,
includes adapter preprocessing/tokenization, device transfers and forecasting,
and ends with the forecast available on CPU. GPU requests synchronize their
device before and after timing. Prophet timing includes all covariate and
target fits needed for the request. Downloads, model loading, parquet reads,
shared dataframe/tensor payload construction, scoring, plotting and file writes
are excluded. Model loading is reported separately. These are the existing
model configurations: RWKV emits its configured quantiles, Chronos requests a
median and Timestral and TimeH2O return their default forecast distribution. Parameter count,
parameter dtype and device are reported rather than assuming equal model sizes
or precision. No architecture or model precision was changed for this benchmark.

`holdout.latency.warmup_requests=1` performs one unscored request per model at
the first origin; `repeats=1` times each subsequent scored request once. Increase
`repeats` for repeated measurements (only the first scored output enters accuracy
metrics). All warmup durations are retained separately. Warmup occurs only at
the first origin, so later shape-specific compilation is included in measured
latency. Set warmup to 0 to include initial cold-start work in scored latency.

The outputs include:

- `test_metrics.json`: MASE-ranked `leaderboard`, `latency` mean/median/p95/min/max
  milliseconds, context-length buckets, and `paired_latency` ratios. A paired
  RWKV/baseline ratio above 1 means RWKV took longer on matching requests.
- `request_latencies.parquet`: all individual warmup and scored requests, with
  source, origin, context length, target/covariate counts, order and repeat.
- `request_latencies.jsonl`: the same raw timings appended after each completed
  table, preserving completed-table measurements if the run is interrupted.
- `test_metrics_progress.json`: progress after every table; latency summaries
  refresh at the INFO progress interval (100 tables by default) and completion.

Run on an otherwise idle GPU to compare latency; concurrent GPU workloads can
distort timings. This measures serial local request latency, not serving
throughput under batching or concurrent traffic. New metrics apply on the next
holdout invocation; an already-running Python process keeps its imported code.
See the [MASE and latency audit](../../docs/audits/2026-10-30/mase_latency/REPORT.md)
for definitions, references and verification.

Prophet now uses historical covariates by default (`prophet.use_covariates=true`).
It forecasts each nonconstant covariate with Prophet, then uses those forecasts
as regressors for its outcome prediction (`prophet.covariate_forecast="prophet"`).
No observed future covariates are supplied. Set `covariate_forecast="naive"`
to carry covariates forward for a faster comparison, or `use_covariates=false`
to reproduce the univariate baseline. Extra fits increase evaluation time.

Holdout comparison plots use a separate reproducible random sample of the
evaluated tables: `plot.max_plots=10`, `plot.sample_seed=7`. Each plot overlays
actuals and every model selected in `holdout.benchmark_models`, plus RWKV.
If fewer tables are evaluated, all are plotted; set `max_plots=0` to disable.
`plot_selection.json` lists the chosen sources and their HTML filenames.
Open the HTML files in a browser to zoom, pan, hover over predictions and toggle
models using the legend. Plotly is embedded, so the files work offline.

Convert an existing completed run's saved random examples without rerunning any
models:

```powershell
uv run python src/06_holdout_test/plotting.py
```

This reuses the saved selection and parquet predictions, updates plot filenames
in the manifests, and leaves forecast values and scores unchanged. Original PNG
files are retained. Future holdout runs generate HTML automatically.

Chronos-2's public checkpoint is 119.5M parameters; no public 300M–500M Chronos-2
checkpoint was available when implemented. Timestral-209M is available without
authentication and is an optional dependency:

```powershell
uv sync --inexact --extra validation --extra benchmarks
```

That command only fills the Windows environment. On Windows the stage itself runs
in the WSL environment, which is managed with pip, so install the packages there
too (versions as in the `benchmarks` extra of `pyproject.toml`):

```bash
source scripts/activate_rwkv7.sh
python -m pip install timestral h2ofm==0.43.1
```

Both weights are downloaded anonymously into `.cache/huggingface` on first use.

TimeH2O-v1 (`"timeh2o"`, 108.8M parameters, `h2ofm` package in the same
`benchmarks` extra) is different in three ways:

- Its weights are gated and under H2O's non-commercial model licence. Accept
  the conditions once on <https://huggingface.co/h2oai/timeh2o-v1>, then put a
  Hugging Face access token in the project's `.env` file as `HF_TOKEN=...` (or
  set `HF_TOKEN`, or run `hf auth login`, where the stage runs).
- This checkpoint does not use past covariates, so it forecasts from the
  historical targets alone, each target on its own. The other models also
  receive the past covariates.
- It runs with the package defaults, which average the forecasts of up to four
  patch sizes (one pass each), so its latency covers those passes.

Chronos is a regular project dependency. Model IDs, pinned revisions and runtime
parameter counts are recorded in the metrics report. See the
[foundation benchmark audit](../../docs/audits/2026-10-29/foundation_benchmarks/REPORT.md).
Console progress is reported every 100 tables and at the final table, including
percentage, elapsed time and ETA (`holdout.progress_every_tables`). Per-origin
metrics and timing use DEBUG; Prophet/Stan fit chatter is suppressed while
warnings and errors remain visible. The live progress file still updates after
every table.

Inference graph coverage is recorded under `rwkv_inference` in final metrics
and the live progress file. Inspect `replay_coverage`, `fallback_requests`,
`fallback_reasons`, `cached_profiles` and `capture_seconds` when diagnosing
latency. The default allows 256 profiles with a tracked 1.5-GiB graph-cache
budget; it previously allowed only eight, which could silently leave most
variable/context/horizon combinations uncached. Uncached requests now retain
the forward-only inference implementation. The timing protocol still includes
first-use capture costs after the initial warmup, so cold p95 can exceed warm
p95. Individual capture logs use DEBUG, and saturation emits a warning once
per reason. See the [cache regression audit](../../docs/audits/2026-11-03/inference_cache/REPORT.md).

`holdout.graph_min_profile_requests=8` admits only recurring profiles for graph
capture. Planning uses source metadata and the existing deterministic requests;
it neither makes extra forecasts nor changes the request latency boundary.
`graph_profile_plan` records estimated coverage. Rare shapes use forward-only
execution and appear as `low_reuse` fallbacks. Set the threshold to 1 to capture
every encountered profile that fits the cache limits. This avoids reporting
good warm medians while spending excessive time capturing rare shapes in a
small holdout sample.

An epoch recovery checkpoint (`latest_checkpoint.pt`) serves training resume
and can supply its saved best validation weights for testing before export.
Testing selects it when newer than `model.pt`, unless `holdout.checkpoint` pins
a file. Results from earlier evaluations stay in their
previous directories; rerunning this stage replaces its fixed output filenames.

See [the stage separation audit](../../docs/audits/2026-10-29/separate_holdout_stage/REPORT.md).

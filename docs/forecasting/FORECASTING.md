Current holdout evaluation uses up to five origins per table: first eligible,
nearest 25%, 50%, and 75% row positions, and the final eligible origin. It compares
RWKV, Prophet and last-observation carry-forward. Short tables use each eligible
origin once. Training ends after exporting the model. Run
`uv run python src/06_holdout_test/holdout_test.py` to test that model separately.
Outputs are under `results/next_patch_300m_validated/holdout_test`
(`paths.holdout_test`), including table-level progress/ETA and model provenance.
The latest exported model and its stored test split are selected automatically.
General forecasting moved to `src/07_forecasting`. See the
[stage separation audit](../audits/2026-10-29/separate_holdout_stage/REPORT.md).

Comparisons are now configurable using `holdout.benchmark_models`: default
`["prophet", "naive", "chronos"]`, with `"timestral"` also available.
RWKV always runs. `holdout.sample_percent` selects a reproducible percentage
of the saved holdout tables (`sample_seed` controls selection). Results include
the selection manifest and provider model IDs, revisions and parameter counts.
See the [foundation benchmark audit](../audits/2026-10-29/foundation_benchmarks/REPORT.md).

Prophet's holdout model now uses historical covariates by default. It predicts
future regressor values from historical covariates with separate Prophet fits,
then forecasts the outcome with those regressors. Config:
`prophet.use_covariates=true`, `prophet.covariate_forecast="prophet"`.
Use `"naive"` for faster carry-forward regressor forecasts or disable covariates
to reproduce the old baseline. Actual future covariates remain excluded.

Watch progress of an already-running worker, without interrupting it:
`uv run python scripts/watch_training_progress.py --follow`.
New workers also log the current/total update count, percentage and training-pass
ETA after each optimizer update. See the [progress and tensor-shape audit](../audits/2026-10-27/training_progress/REPORT.md).

Training startup no longer performs an automatic exact unique-token scan.
It reuses a matching inventory report when present, and otherwise proceeds
directly toward model initialization. See the [startup performance audit](../audits/2026-10-27/training_startup/REPORT.md).

Normal training now runs complete epochs, processes up to eight compatible
windows together per optimizer update, and validates the full non-test validation set
before training (epoch 0) and at each epoch end. Epoch 0 is a diagnostic baseline,
excluded from the scheduler and early-stopping counters. There is no default epoch limit: one epoch without validation-loss
improvement stops the run. The best weights are restored for the final test.
After every complete epoch and full validation pass, the trainer atomically saves
`latest_checkpoint.pt` with model, optimizer, scheduler, random-state and progress
information. Running `train.py` again automatically resumes that checkpoint;
set `training.resume` to `false` to start a new run. See the
[checkpoint and validation audit](../audits/2026-10-28/epoch_checkpoints/REPORT.md).
The default execution now uses length buckets, fused AdamW, bounded CPU prefetch,
selective activation checkpointing and validation batches of 16. Gradient
accumulation remains configurable (default 1 with batch size 8). Detailed DEBUG
logs still go to disk; the console shows INFO progress. These execution settings
do not require regenerating processed data or the embedding index.
See the [training throughput audit](../audits/2026-10-28/training_speed/REPORT.md)
for the measured speedup, memory limits, configuration and reproduction commands.
Compiled cross-variable attention is enabled by default, including when
`training.compile_variate` is omitted from a custom configuration. It applies to
training with up to eight variables, measured about 17% faster in the latest
bounded optimizer-update comparison. The first batches may pause for compilation;
the cache defaults to `.cache/torchinductor` in the repository. Set the option to
`false` for eager training execution. `training.compile_variate_inference` separately
enables the same compiled attention during validation/forecasting (default true).
Both retain eager attention for wider tables; checkpoint names are unchanged. See the
[follow-up execution audit](../audits/2026-10-28/training_speed_round2/REPORT.md)
for experiments, numerical checks, memory use and restart verification.
See the [epoch training audit](../audits/2026-10-27/epoch_training/REPORT.md)
for configuration, accumulation weighting, learning-rate scheduling and tests.

`embed.py` now prepares a lightweight patch index instead of caching random
embedding vectors for the whole corpus. Raw tensors are memory mapped; features
are constructed per accessed window. The learned patch projection runs inside
the model and is optimized jointly during training. See the
[embedding memory investigation](../audits/2026-10-27/embedding_memory/REPORT.md)
for the failed run, Windows memory evidence, replacement artifact format, and tests.

The training JSONL utility lives in `src/05_model_training/training_log.py`.
It runs automatically through `train.py`, for both CLI execution and direct
`train(config)` calls; it is not a separate script to run. Timestamped journals
are saved under `paths.logs`. See the [module-location audit](../audits/2026-10-26/training_log_location.md).

Select generated date folders through paths.raw and data.generated_dates in
configs/config.json. See the [date-selection audit](../audits/2026-10-25/generated_dates/REPORT.md)
for single/multiple-date examples and the required stage order.

Training now supports configurable warmup and cosine learning-rate decay. See the
[learning-rate scheduler audit](../audits/2026-10-25/learning_rate_scheduler/REPORT.md) for settings, logs and scheduled capacity commands.

For the full-dataset, per-update capacity diagnostic, see the
[capacity experiment report](../audits/2026-10-24/capacity_300m/REPORT.md).

# Current default: 447.5M causal next-patch model

`configs/config.json` selects 25 layers of width 1,024, patch size 16, and a
2,048-observation context cap. The measured size is 447,482,144 parameters.
The normal `train.py` reserves 10% of admitted datasets for testing, with a
minimum table length of 48 observations. On the other 90%, it trains earlier
next-patch positions and validates every table's last token at the end of each epoch.
After the final update it saves the selected model. The separate holdout stage
evaluates the sealed test tables at up to five sampled origins, starting with
two context tokens, against Prophet and last-observation carry-forward.
Date folders are selected by `data.generated_dates` (currently `2026-10-25`).
Short histories use masked padding; long forecast histories use their final 2,048
points. Normalization is causal per prefix. The train and forecast commands below
remain unchanged.

See the [default training and holdout audit](../audits/2026-10-25/default_holdout/REPORT.md)
for rounding, isolation, rolling origins, parameter dimensions, untied projections,
logs, test artifacts, and the full-size GPU smoke test. Changing the model settings
requires rerunning transform and embed before training. Existing output path names
containing `300m` are retained for compatibility; the checkpoint and architecture
report record the actual model size.

Read the [complete experiment and audit report](../audits/2026-10-24/next_patch_300m/REPORT.md)
for the objective, artifact layout, dimensions, coverage, baseline settings,
100-table metrics and six plots from the original unscreened experiment. Current
validated outputs use `results/next_patch_300m_validated/` and numbered data
subdirectories named `next_patch_300m_validated`. Read the
[validation and embedding audit](../audits/2026-10-24/learnability_gate/REPORT.md)
for the current five-stage commands and admission policy. Prophet is available through
`uv sync --extra validation`; install it in the configured WSL runtime as well.

The remainder of this guide documents the retained **legacy masked-horizon mode**,
selected with `--config configs/smoke_masked_horizon.json`. Its fixed-window split
and full-context scaling do not apply to next-patch training.

# Inspectable forecasting pipeline

Four Python files implement independently executable stages, without Kedro.
Training and forecasting have separate files in their respective folders. All
pipeline settings live in `configs/config.json`; relative paths resolve from the
repository root, including when launched from PyCharm.

| Stage | Script | Output |
|---|---|---|
| Transform | `src/03_transforming_input_data/transform.py` | `data/02_processed/windows.pt`, manifest and per-window Parquet tables |
| Embed | `src/04_creating_embeddings/embed.py` | `data/03_embedded/embeddings.pt` and manifest |
| Train | `src/05_model_training/train.py` | `results/pipeline/training/model.pt`, loss CSV/JSON |
| Forecast | `src/07_forecasting/forecast.py` | `results/pipeline/forecasts/`: comparison PNGs, forecast CSVs, metrics JSON |

`src/07_forecasting/README.md` points to the forecast command. There is no longer
a `basin` Python package.

## Run stage by stage

```powershell
uv sync --extra data
uv run python src/03_transforming_input_data/transform.py --config configs/smoke_masked_horizon.json
uv run python src/04_creating_embeddings/embed.py --config configs/smoke_masked_horizon.json
uv run python src/05_model_training/train.py --config configs/smoke_masked_horizon.json
uv run python src/07_forecasting/forecast.py --config configs/smoke_masked_horizon.json
```

Run training and then forecasting. Every command
accepts `--config configs/config.json`. Edit that file for window lengths, column
roles, patch size, model dimensions, optimizer settings, baseline seasonality,
logging and plot settings.

Torch, Loguru, pandas, PyArrow, matplotlib and statsmodels are normal dependencies
in `pyproject.toml` and `uv.lock`. Windows uses official CPU Torch wheels for
preprocessing/embedding inspection; Linux uses CUDA 13.0 wheels, following
[uv's platform-specific index configuration](https://docs.astral.sh/uv/guides/integration/pytorch/).

The existing fused RWKV extensions require Linux/CUDA. On Windows the training/forecasting
script logs its handoff and launches the configured WSL interpreter using
`scripts/activate_rwkv7.sh`. No CPU model is substituted. In WSL, run directly:

```bash
source scripts/activate_rwkv7.sh
python src/05_model_training/train.py --config configs/smoke_masked_horizon.json
python src/07_forecasting/forecast.py --config configs/smoke_masked_horizon.json
```

The current WSL runtime has the pipeline dependencies installed. A fresh runtime
needs the [RWKV setup](../RWKV-7_setup/README_RWKV7_FAST_SETUP.md), plus
`python -m pip install loguru pandas pyarrow matplotlib statsmodels`.
Do not replace that separate runtime's established CUDA toolkit overrides with a
plain `uv sync`; consult the setup guide's compatibility notes.

## Stage contracts

Transform reads `data/01_generated/**/data.parquet` in sorted order, selecting up
to `data.max_tables` eligible tables. Explicit targets and known-future columns
are disjoint. `past_covariates: "auto"` selects the other columns except time,
excluding lag-derived columns by the configured patterns, as historical-only inputs. All selected columns
must be numeric. NaNs remain missing; infinities and invalid time ordering fail.

The final horizon of each table is held out for evaluation. Every training label
ends before that evaluation cutoff. This is temporal holdout smoke evaluation,
not cross-dataset zero-shot benchmarking. The per-window Parquet tables include history and
labels for inspection; tensor dictionaries distinguish them explicitly.

Embed normalizes from context-only mean/std, patches values, masks and relative
time, then adds target/past/known-future role embeddings. The saved artifact has:

- `encoder_state`: initial projection and role weights;
- `records[*].prepared`: features, roles, location/scale and patch dimensions;
- `records[*].initial_embeddings`: projected tensors `[batch, variables, padded patches, width]`;
- original input tensors and labels, kept separate from token features.

With `training.train_embeddings: true`, training recomputes projections from saved
features and learns them jointly with RWKV. The cached vectors are inspectable
initial snapshots, not trained vectors. With `false`, training freezes projection
weights and consumes the saved embedding vectors. Forecasting applies checkpoint
weights to raw context. CUDA computation uses bf16; normalization/loss/metrics use
float32 or float64.

Configuration signatures reject changed seed/data/model settings. Content hashes
reject newly processed windows with old embeddings and regenerated embeddings
with an old checkpoint. Rerun downstream stages after changing upstream artifacts.
Default artifact paths are overwritten on rerun; copy the configuration and change
its paths to retain multiple experiments.

## Logs and evaluation

Loguru writes the console and a unique `logs/<UTC>_<stage>_<id>/events.log` for each invocation. Each run also saves its configuration, code hashes, interpreter and success/failure status. Logs
include configuration, interpreter, Torch/CUDA status, column roles, raw shapes,
missing values, tensor shapes/dtypes/previews, elapsed times, parameter counts,
per-step training loss, gradient norm, warnings, RMSE and output paths. CUDA timers
synchronize before/after operations. These are diagnostic timings, not controlled
throughput benchmarks; first-call overhead and detailed logging affect timings.

Inspect complete saved tensors with:

```python
import torch

artifact = torch.load("data/03_embedded/embeddings.pt", weights_only=True)
print(artifact["records"][0]["prepared"]["features"])
print(artifact["records"][0]["initial_embeddings"])
```

Forecast reloads the checkpoint without training. It compares RWKV with Prophet
and last-observation carry-forward on identical origins and horizons. Carry-forward
repeats the last finite historical target value. Prophet uses only the historical
target and configured seasonality; missing historical values are interpolated,
and baseline failures are surfaced. RWKV also uses historical covariates.
Sampled holdout plots draw each forecast segment separately so unsampled intervals
are not shown as predictions. Parquet files contain actuals and all three point
forecasts; JSON reports pooled and per-table metrics and elapsed time.

The default three-step run demonstrates execution, not accuracy or calibration.

## Architecture and tests

[Chronos-2](https://www.amazon.science/blog/introducing-chronos-2-from-univariate-to-universal-forecasting)
inspired patch features feed RWKV time mixing plus cross-variable attention.
[TimesFM-3](https://research.google/blog/timesfm-3-a-zero-shot-foundation-model-for-multivariate-forecasting/)
inspired masked horizon patches allow single-pass decoding. Pinball loss trains
nine ordered quantiles. Variable counts can differ between windows. Lookahead
concatenation, categorical/static encoders, state caching and large-scale
pretraining remain unimplemented.

```powershell
uv run python -m unittest discover -s tests -p test_pipeline.py
```

```bash
source scripts/activate_rwkv7.sh
python -m unittest discover -s tests -p test_forecaster.py
python src/05_model_training/smoke_basin.py
```

CPU tests cover serialization, temporal holdout, label isolation, embedding
gradients and incompatible-artifact rejection. CUDA checks cover model behavior,
weight updates, checkpoint reload and forecasting on freshly generated SCM data.

## Audit details

See [the data and architecture audit](../audits/2026-10-24/PIPELINE_AUDIT.md) for
file naming, exact source/window mappings, ragged records, feature semantics,
measured parameter counts, two-axis tensor shapes, migration and test evidence.

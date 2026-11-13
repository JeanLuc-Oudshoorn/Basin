# Inspectable forecasting pipeline

Three Python files implement independently executable stages, without Kedro.
Training and forecasting share the third file but have separate commands. All
pipeline settings live in `configs/config.json`; relative paths resolve from the
repository root, including when launched from PyCharm.

| Stage | Script | Output |
|---|---|---|
| Transform | `src/03_transforming_input_data/transform.py` | `data/02_processed/windows.pt`, manifest and per-window CSVs |
| Embed | `src/04_creating_embeddings/embed.py` | `data/03_embedded/embeddings.pt` and manifest |
| Train | `src/05_model_training/train_forecast.py train` | `results/pipeline/training/model.pt`, loss CSV/JSON |
| Forecast | `src/05_model_training/train_forecast.py forecast` | `results/pipeline/forecasts/`: comparison PNGs, forecast CSVs, metrics JSON |

`src/06_forecasting/README.md` points to the forecast command. There is no longer
a `basin` Python package.

## Run stage by stage

```powershell
uv sync --extra data
uv run python src/03_transforming_input_data/transform.py
uv run python src/04_creating_embeddings/embed.py
uv run python src/05_model_training/train_forecast.py train
uv run python src/05_model_training/train_forecast.py forecast
```

Use `all` in the final script to train and forecast sequentially. Every command
accepts `--config configs/config.json`. Edit that file for window lengths, column
roles, patch size, model dimensions, optimizer settings, baseline seasonality,
logging and plot settings.

Torch, Loguru, pandas, PyArrow, matplotlib and statsmodels are normal dependencies
in `pyproject.toml` and `uv.lock`. Windows uses official CPU Torch wheels for
preprocessing/embedding inspection; Linux uses CUDA 13.0 wheels, following
[uv's platform-specific index configuration](https://docs.astral.sh/uv/guides/integration/pytorch/).

The existing fused RWKV extensions require Linux/CUDA. On Windows the third
script logs its handoff and launches the configured WSL interpreter using
`scripts/activate_rwkv7.sh`. No CPU model is substituted. In WSL, run directly:

```bash
source scripts/activate_rwkv7.sh
python src/05_model_training/train_forecast.py all
```

The current WSL runtime has the pipeline dependencies installed. A fresh runtime
needs the [RWKV setup](RWKV-7_setup/README_RWKV7_FAST_SETUP.md), plus
`python -m pip install loguru pandas pyarrow matplotlib statsmodels`.
Do not replace that separate runtime's established CUDA toolkit overrides with a
plain `uv sync`; consult the setup guide's compatibility notes.

## Stage contracts

Transform reads `data/01_generated/**/data.parquet` in sorted order, selecting up
to `data.max_tables` eligible tables. Explicit targets and known-future columns
are disjoint. `past_covariates: "auto"` selects the other columns except time,
including target-lag columns, as historical-only inputs. All selected columns
must be numeric. NaNs remain missing; infinities and invalid time ordering fail.

The final horizon of each table is held out for evaluation. Every training label
ends before that evaluation cutoff. This is temporal holdout smoke evaluation,
not cross-dataset zero-shot benchmarking. The per-window CSVs include history and
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

Loguru writes the console and `results/pipeline/logs/*.log` (append/rotate). Logs
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

Forecast reloads the checkpoint without training. It compares RWKV with
Holt-Winters triple exponential smoothing fitted to the same historical context,
using the same held-out horizon and finite-label RMSE. Baseline trend, seasonality,
damping and period are configured in JSON. At least two seasonal cycles are
required. Missing baseline context is interpolated using history only, with a
warning; baseline failures are surfaced. RWKV can use covariates; Holt-Winters is
univariate. Plots show context, origin, actuals, RWKV median/quantiles and baseline.
CSVs contain every forecast and quantile; metrics JSON records RMSE and durations.

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

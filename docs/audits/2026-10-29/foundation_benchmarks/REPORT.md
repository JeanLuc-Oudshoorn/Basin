# Configurable foundation-model benchmarks and holdout sampling

## Configuration and stage behavior

`src/06_holdout_test/holdout_test.py` still loads the exported trained RWKV model
and its saved dataset-disjoint holdout. RWKV always runs. Comparisons now come
from `holdout.benchmark_models`, defaulting to `['prophet', 'naive', 'chronos']`.
Supported additional choice: `timestral`. An empty list evaluates RWKV alone.
Unknown or repeated names fail rather than silently dropping a requested model.

Example smoke-test configuration (JSON uses double quotes):

```json
"holdout": {
  "benchmark_models": ["prophet", "naive", "chronos", "timestral"],
  "sample_percent": 1.0,
  "sample_seed": 7,
  "percentiles": [1, 25, 50, 75, 99],
  "inference_batch_size": 5,
  "progress_every_tables": 100,
  "model_cache_dir": ".cache/huggingface",
  "foundation_device": "cuda"
}
```

The actual config additionally pins each provider's model ID and revision under
`holdout.chronos` and `holdout.timestral`. `chronos.batch_size` counts input
variable streams according to its library API, default 32.

`sample_percent=100` is the default and evaluates the entire saved holdout.
Smaller percentages use a seeded SHA-256 ranking of source identities, select
`ceil(holdout_tables * percent / 100)` whole tables, and retain at least one.
Selection is independent of record order and the same seed yields nested samples
as the requested percentage grows. All providers receive the same selected
tables and up-to-five forecast origins. Sampling does not change the original
train/validation/test partition or retrain anything.

The percentage applies to **admitted test tables**, not all generated tables.
The current exported split has 10,783 test tables out of 107,828 eligible admitted
tables. Thus 1% evaluates 108 tables, rather than 150. The available and selected
counts are recorded and logged; no generated-data count is assumed.

Outputs remain in `results/447m_model/holdout_test`:

- `sample_selection.json`: percentage, seed, available/selected counts and sources.
- `model_used.json`: trained RWKV checkpoint identity and selected epoch.
- `test_metrics.json`: configured methods, provider metadata, sampling summary,
  pooled and per-table/per-origin scores.
- Prediction parquet columns and plotted methods follow the requested list.
  Metrics include RMSE, MAE and normalized RMSE for every selected method;
  normalized pinball remains an additional RWKV metric, not a provider comparison.
- Live progress and ETA describe the selected tables, with INFO every 100 tables
  and at completion. Detailed per-origin results remain in JSONL/DEBUG logs.

Only current aggregate metrics and the current selection manifest define a run.
Older per-table files beyond a smaller rerun's selected count can remain from an
earlier evaluation; they are not included when building the new aggregate.

## Model availability and size research

Amazon's official [Chronos-2 model card](https://huggingface.co/amazon/chronos-2)
describes a 120M model. Its anonymous Hub metadata reports **119,477,664**
parameters. A public 300M–500M Chronos-2 checkpoint was not available. The
implementation uses the actual Chronos-2 model rather than substituting another
Chronos family or mislabelling its size. Revision:
`29ec3766d36d6f73f0696f85560a422f50e8498c`.

Mistral's [Timestral-209M model card](https://huggingface.co/mistralai/Timestral-209M)
and [official repository](https://github.com/mistralai/timestral) describe its
209M model and historical-covariate interface. Anonymous Hub metadata reports
**209,312,256** parameters. Both repositories report `gated=false`, `private=false`;
their config files returned HTTP 200 without credentials. The adapters explicitly
use `token=False`; no sign-in or gated access acceptance is performed. Timestral
weights use the model card's Modified MIT license, and its inference code uses
Apache-2.0. Revision: `9874d252ab0aae16f7f7c999f98f98d6ad7d8365`.

Timestral remains optional in the model list, preserving the requested three
default comparison names. Actual loaded parameter counts, dtype, device, model
IDs and revisions are written into benchmark metadata, so the comparison does
not imply equal model capacity to the 447.5M RWKV model.

## Installation and runtime

`chronos-forecasting==2.3.2` is a regular dependency. Timestral 1.0.0 is provided
by the `benchmarks` extra, pinned to its public source commit
`a534bdbd244d64f3d9d743f46c7bf6fc330ed4ff`. Both packages are installed in the
Windows `.venv` and active WSL CUDA environment; `uv.lock` records dependencies.

```powershell
uv sync --inexact --extra data --extra validation --extra benchmarks
uv run python src/06_holdout_test/holdout_test.py
```

For the active external WSL virtual environment:

```bash
source scripts/activate_rwkv7.sh
python -m uv sync --active --inexact --extra data --extra validation --extra benchmarks
```

Each requested foundation model is loaded once per evaluation and used zero-shot.
Weights download to the configured cache on the project's E: drive; subsequent
runs reuse them. First-run downloads and existing artifact verification/model
loading happen before forecast progress starts and are not part of its ETA.
The existing 8 GiB allocator cap still applies; provider inference uses float32.
Setting `foundation_device="cpu"` runs external models on CPU if desired.

Chronos uses the official `Chronos2Pipeline.predict_quantiles` interface with
median forecasts, explicit 2,048-point context cap and `cross_learning=False`.
Each forecast-origin context is a separate dictionary with historical targets
and historical covariates. Related variables within one context can interact,
but independent origins/tables cannot share information. Timestral uses
`TimestralForecaster.forecast` with historical targets and `past_covariates`,
returning medians. Neither adapter receives future covariates or target labels.
All models see identical capped history and score identical real target rows,
including partial final patches. Prophet and naive remain univariate baselines.

## Verification

Unit checks cover configuration defaults, invalid model names, deterministic
sampling/rounding/seed changes, selected-provider metrics, naive-only execution,
covariate alignment and shape/finite-output checks. The original holdout
protocol tests explicitly use classical baselines to avoid downloading model
weights during unit tests.

`scripts/verify_foundation_benchmarks.py` runs an isolated 0.02% sample of the
actual saved holdout (three tables), explicitly enabling both foundation models
alongside Prophet/naive and the trained RWKV. Its outputs are under `smoke/`,
with a compact result in `gpu_verification.json` and execution log at
`logs/20261029_foundation_benchmarks_gpu.log`. Production weights and benchmark
reports are not modified by the smoke run.

CPU regression log: `logs/20261029_foundation_full_tests.log`.

Verification passed: 159 CPU-suite tests ran (152 passed, seven CUDA-only
tests skipped). The independent GPU smoke completed all five methods on three
sampled tables, 15 origins and 217 real forecast observations. Both weight
downloads succeeded anonymously and the loaded parameter counts matched the
Hub metadata. Scoring/export/plotting took 22.08 seconds, including first-shape
RWKV compilation, after provider loading; first-run artifact verification and
downloads took additional time. This tiny smoke confirms execution and is not
an accuracy ranking or full-holdout runtime estimate. Ruff, Bandit, Windows
`pip check` and `uv lock --check` passed.

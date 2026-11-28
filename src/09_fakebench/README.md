# FakeBench-TS stage

[FakeBench-TS](../../benchmarks/fakebench-ts/README.md) forecasts random
walks, where the best forecast is known exactly, and measures how much trend,
cycle or mean reversion a model invents. It lives in its own repository and is
included here as a git submodule:

```powershell
git submodule update --init benchmarks/fakebench-ts
```

Run from the repository root:

```powershell
uv run python src/09_fakebench/fakebench.py
```

This forecasts the 10,000 random walks of the `v3` dataset with Basin,
Chronos-2, Timestral and TimeH2O and scores them. Like the other GPU stages
it routes itself to WSL. On an idle RTX 5070 a run takes about half an hour:
most of it is loading the weights from disk and TimeH2O's forecasts (about 9
minutes, four passes per series); the forecasts of each of the other models
take one to two minutes.

## What is scored

Every series is a random walk whose steps have mean zero: 5,000 with
symmetric steps and 5,000 with skewed steps. The mean of every future value
is the last observed value, so under squared error the naive forecast is the
best forecast of every series. The headline is each model's RMSE divided by
the naive forecast's (1.000 is the best a forecast can expect), with mean
absolute error as a secondary measure. The
[FakeBench-TS README](../../benchmarks/fakebench-ts/README.md) has the
reasoning.

In 30% of the series with symmetric steps the size of the steps changes over
time (never their direction), and most series come with covariates that say
nothing about the target. Neither changes the best forecast. The headline
numbers use the forecasts made without covariates; `leaderboard.md` reports
what the covariates did to each model.

The `v3` data file is not kept in git. Create it once (a few minutes, 130 MB):

```powershell
cd benchmarks/fakebench-ts
uv run python -m fakebench_ts generate
```

`--dataset v1` runs the earlier dataset instead: 2,100 random walks, no
covariates, point forecasts only.

## The models

- **Basin** uses the checkpoint the holdout test would use
  (`holdout.checkpoint`, otherwise the newer of `model.pt` and the best
  weights in `latest_checkpoint.pt` under `paths.training`), with the same
  inference set-up. Its point forecast is the median. On `v3` it also gives
  its quantiles and is given the covariates. Basin has no input for future
  values, so of a covariate that is known into the future it reads the
  history only.
- **Chronos-2, Timestral and TimeH2O** use the model IDs, pinned revisions,
  device and cache folder under `holdout` in the configuration. TimeH2O's
  weights are gated: see the [holdout README](../06_holdout_test/README.md).

## Where the results are

Every run writes a new folder under `results/model/fakebench` (set
`paths.fakebench` to change it), named by its start time in UTC and Basin's
epoch and step, for example `20261127T095237Z_epoch3_step4526436`. A run never
writes into an existing folder, so earlier results stay as they were. Each
folder holds:

- `leaderboard.md`: the tables for all series and for the symmetric and the
  skewed half, with uncertainty bands, states of randomness, covariates, and
  history against horizon length
- `results.json`: all numbers, also by family, context length and forecast step,
  and each model's speed: after its forecasts every model forecasts 200
  series again one at a time, and `leaderboard.md` reports the median time
  of one forecast, by history length, next to the forecasts per second of
  the batched run
- `per_series.parquet`: every model's scores on every series
- plots (`error_vs_optimal_*`, `departure_by_horizon_*`, `examples_*`, ...)
- `forecasts/`: one file of forecasts per model
- `run.json`: the dataset, Basin's checkpoint, epoch and step, and where each
  model's forecasts came from

Basin is forecast again in every run. The published models do not change,
so their forecasts are copied from the most recent earlier complete run on
the same data instead of being computed again.

Runs made before 2026-11-27 used earlier datasets and absolute error against
the median as the headline; their folders are kept as they were.

### Copy in Azure Blob Storage

With `fakebench.result_upload` in the configuration (the same fields as
`holdout.result_upload`, and the same account from `.env`), every complete
run folder is also copied to the container, as
`results/model/fakebench/<run folder>/`. The container is checked before the
forecasts start; `results.json` is sent last, so a run folder in the
container that has it is complete; sizes and MD5s are compared afterwards.
Nothing is ever deleted from the container. To send a run again:

```powershell
uv run python src/09_fakebench/fakebench.py --upload <run folder>
```

## Options

| Option | Effect |
|---|---|
| `--models basin chronos2 timestral timeh2o` | which models to include; FakeBench-TS baselines such as `seasonal_naive` also work |
| `--dataset v1` | the earlier dataset of 2,100 walks (default `v3`) |
| `--limit 100` | forecast an even subset of the series for a quick trial; written under `trials/`, never uploaded |
| `--rerun` | forecast the published models again instead of copying their forecasts |
| `--batch-size 16` | series per forecast call |
| `--upload <run folder>` | only copy that finished run folder to the container again |

## History given to the models

Each model receives the whole context of a series: 32, 128, 512 or 2,048
points, a quarter of the series each, and forecasts 112 steps.
`leaderboard.md` has a matrix per model of history length against horizon
length: the first 32, the first 64 and all 112 steps of the same forecast.

This stage reads no training or holdout data. Do not run it while a training
run is using the GPU.

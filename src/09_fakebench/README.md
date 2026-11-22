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

This forecasts the 2,100 series of the `v1` dataset with Basin, Chronos-2,
Timestral, TimeH2O and TimesFM 3.0 and scores them. Like the other GPU stages it routes itself to WSL.
On an idle RTX 5070 the whole run took 75 seconds and peaked at 3.1 GB of GPU
memory.

- **Basin** uses the checkpoint the holdout test would use
  (`holdout.checkpoint`, otherwise the newer of `model.pt` and the best
  weights in `latest_checkpoint.pt` under `paths.training`), with the same
  inference set-up. Its point forecast is the median.
- **Chronos-2, Timestral, TimeH2O and TimesFM 3.0** use the model IDs, pinned revisions,
  device and cache folder under `holdout` in the configuration. TimeH2O's
  weights are gated: see the [holdout README](../06_holdout_test/README.md).
- The series come from the dataset published in the submodule,
  `benchmarks/fakebench-ts/data/fakebench_v1.parquet`.

## Where the results are

Every run writes a new folder under `results/model/fakebench` (set
`paths.fakebench` to change it), named by its start time in UTC and Basin's
epoch and step, for example `20261120T195246Z_epoch1_step1457590`. A run never
writes into an existing folder, so earlier results stay as they were. Each
folder holds:

- `leaderboard.md`: the tables, overall and by history and horizon length
- `results.json`: all numbers, also by family, context length and forecast step
- `per_series.parquet`: every model's scores on every series
- plots (`error_vs_optimal_*`, `departure_by_horizon_*`, `examples_*`)
- `forecasts/`: one file of forecasts per model
- `run.json`: Basin's checkpoint, epoch and step, and where each model's
  forecasts came from

Basin is forecast again in every run. The published models do not change,
so their forecasts are copied from the most recent earlier run on the same
data instead of being computed again.

## Options

| Option | Effect |
|---|---|
| `--models basin chronos2 timestral timeh2o timesfm` | which models to include; FakeBench-TS baselines such as `seasonal_naive` also work |
| `--limit 100` | forecast an even subset of the series for a quick trial |
| `--rerun` | forecast the published models again instead of copying their forecasts |
| `--batch-size 16` | series per forecast call |

## History given to the models

Each model receives the whole context of a series and nothing else: 128, 512
or 2,048 points, a third of the series each, and forecasts 112 steps.
`leaderboard.md` has a matrix per model of history length against horizon
length: the first 32, the first 64 and all 112 steps of the same forecast.

This stage reads no training or holdout data. Do not run it while a training
run is using the GPU.

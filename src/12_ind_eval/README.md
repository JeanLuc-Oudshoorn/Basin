# IND-Eval stage

[IND-Eval](https://huggingface.co/datasets/cognitedata/IND-Eval) (Haugland et
al., 2026, [arXiv 2610.16652](https://arxiv.org/abs/2610.16652)) is a
forecasting benchmark on real industrial time series. This stage runs Basin,
the naive forecast, Holt-Winters, Chronos-2, Timestral and TimeH2O on it.

The results of the first full run are in
[docs/benchmark_results/IND_EVAL.md](../../docs/benchmark_results/IND_EVAL.md).

## What the benchmark is

Eight sources are distributed: `3w` (oil wells), `electricity`, `mds` (Mesa
del Sol microgrid), `metropt3` (metro compressor), `neats` (offshore
drilling), `pronto` (multiphase flow), `solar` and `tablet_compression`
(pharmaceutical tablet presses). A ninth, ETT, ships only as scripts and is
not part of this stage. Together they hold about 200,000 series in 11 GB.

Every source is a folder of parquet files; one file is one **task** (a source
at one sampling rate or product code and one horizon). A row is one channel
of one evaluation window: `context` is the history, `target` the values that
follow it directly. The model is given the context and forecasts the target.
The benchmark's own horizons run from 8 to 1,024 points and its contexts to
4,096; this stage uses at most 2,048 points of history and forecasts at most
112 points, the same for every model (see below).

It is a forecasting benchmark only. The dataset card and the paper define no
anomaly-detection task, and the files hold no anomaly or fault labels. Two
sources, PRONTO and 3W, are built around fault events, but the events are
only used to choose where the forecasts start.

## Running it

```powershell
uv run python src/12_ind_eval/ind_eval_data.py     # download, about 11 GB, once
uv run python src/12_ind_eval/ind_eval.py          # all models
```

The data goes to `benchmarks/IND-Eval` (not kept in git), at the release
`ind_eval.revision` names. Like the other GPU stages, the run routes itself
to WSL. Do not start it while a training run is using the GPU.

Useful variations:

| Command | Effect |
|---|---|
| `--limit 20` | 20 series per task, written under `trials/`: a quick check that every model runs |
| `--models naive holtwinters` | only these models; a run of CPU models only stays on Windows |
| `--sources metropt3 pronto` | only these sources |
| `--resume <run folder>` | finish an interrupted run, or add models to a finished one |

A full run is long: Mesa del Sol alone has 115,000 series. Holt-Winters is fitted
anew to every series (10 processes by default) and is best run on its own:
`--resume <run folder> --models holtwinters`. `ind_eval.max_series_per_task`
caps the series per task for every model alike.

## How a series is forecast

- **One channel at a time.** Every model forecasts a channel from that
  channel's own history. The other channels of the window are not used, and
  PRONTO's three control inputs (rows with `role = covariate`) are neither
  forecast nor given to a model.
- **History: at most 2,048 points.** Every model gets the latest 2,048
  points of the context (`ind_eval.context_points`); a longer context is cut.
  This is Basin's limit, applied to all models so that they see the same
  history. The benchmark itself allows 4,096.
- **Forecast: at most 112 points.** Every model forecasts the first 112
  points of the target (`ind_eval.horizon`); a shorter target is forecast
  whole. This is what Basin forecasts in one call. A task with a horizon of
  512 or 1,024 is therefore scored on its first 112 points only, and the
  three tablet-compression horizons become 30, 60 and 112.
- **Missing values** in the history are filled by linear interpolation, the
  same for every model. Missing target values are not scored.
- **Naive** repeats the last value. **Seasonal naive** repeats the last
  cycle and is the reference every score is divided by; it is always run.
  **Holt-Winters** is the model of the holdout stage
  (`src/06_holdout_test/holt_winters.py`). These three give a point forecast
  only.

## Scores

As in the paper, as far as it states them (`ind_eval_scoring.py`):

- **MASE** of a series: mean absolute error of the 50% quantile, divided by
  the mean absolute change over one seasonal cycle in the series' context. A
  task's MASE is the mean over its series.
- **CRPS** of a task: the quantile loss at the levels 0.1 to 0.9, summed over
  the task and divided by the sum of the absolute target values.
- **nMASE and nCRPS**: the model's MASE and CRPS of a task divided by those
  of the seasonal-naive forecast on the same series. Below 1 is better than
  seasonal naive.
- A source's score is the geometric mean over its tasks; the overall score is
  the geometric mean over the sources, so every source counts the same.

The cycle length is the task's `seasonality` column (1 for most industrial
sources, so seasonal naive is the naive forecast there). Electricity and
solar do not state one; `ind_eval.seasonality_by_freq` sets it from the
sampling rate (96 for 15 minutes, 144 for 10 minutes, 24 for hourly, 7 for
daily, 1 for weekly). That table is our reading, not the paper's. A task
with a sampling rate that is not in the table is refused.

For the three point-forecast models the CRPS scores the point as if it were
every quantile, which favours the models that give quantiles. Compare them on
nMASE.

**Comparing with the published leaderboard.** The paper reports nMASE 0.755
and nCRPS 0.539 for Chronos-2 over nine sources. Our numbers will differ for
reasons that have nothing to do with the models: history and forecast length
are capped at 2,048 and 112 points here, ETT is missing, the paper's code was
not available to check its exact formulas, and the seasonal cycle of
electricity and solar is our choice. Compare models within one run
of this stage.

## Where the results are

Every run writes a new folder under `results/model/ind_eval` (set
`paths.ind_eval` to change it), named by its start time and Basin's epoch and
step:

- `leaderboard.md`: overall nMASE and nCRPS, both by source, and series per
  second for each model
- `results.json`: the same, and every task's MASE, CRPS, nMASE and nCRPS
- `scores/<model>/<source>__<task>.parquet`: the sums behind the scores for
  every series; the `.json` next to it records the series, the forecast time,
  and marks the task as finished
- `run.json`: the data release, Basin's checkpoint, and the history and
  forecast lengths the run used

Forecasts themselves are not kept, only the scores of every series.

**Repeated series are left out.** Within a source, the tasks for different
horizons often start at the same points (confirmed for tablet compression
and MetroPT-3). Cut to 112 points, a 512-point and a 1,024-point task of the
same windows are then the same history and the same target. Before the
first model runs, the stage reads every task once and marks each series
whose history and target are identical to one in an earlier task of its
source. Those are neither forecast nor scored, and a task that holds nothing
else is dropped. `run.json` and `leaderboard.md` record how many. The list
is kept next to the data (`duplicates_<settings>.json`), so it is made once
for a given history length, forecast length and subset.

The data is licensed per source (see `raw/<source>/LICENSE` in the dataset);
the collection itself is Apache-2.0.

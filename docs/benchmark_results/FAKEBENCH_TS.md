# Benchmark results: FakeBench-TS

[FakeBench-TS](https://github.com/JeanLuc-Oudshoorn/fakebench-ts) asks a
model to forecast series that cannot be forecast. Every series is a random
walk whose steps have mean zero, so the best forecast of every future value
is the last observed value. Anything else a model forecasts (a trend, a
cycle, a return to an earlier level) is invented, and costs accuracy.

| | |
|---|---|
| Basin weights | best weights after epoch 3 (step 4,526,436), 221.5 million parameters |
| Series | 10,000 random walks (dataset `v3`): 5,000 with symmetric steps, 5,000 with skewed steps |
| Forecast | 112 points from 32, 128, 512 or 2,048 points of history (a quarter of the series each) |
| Run | `results/model/fakebench/20261127T201702Z_epoch3_step4526436` |

## Accuracy

Both metrics are divided by the score of the naive forecast (the last
observed value) on the same series. For RMSE, 1.000 is the best a forecast
can expect, and 1.08 means 8% more error than necessary. Departure is how far
the forecast moves away from the last observed value on average, in typical
steps of the series: 0 means nothing was invented.

| Model | RMSE vs naive (95% interval) | MASE vs naive | RMSE vs naive, symmetric steps | RMSE vs naive, skewed steps | Departure |
|---|---|---|---|---|---|
| Naive (last value) | 1.000 | 1.000 | 1.000 | 1.000 | 0.00 |
| Timestral (209M) | **1.049** (1.032 to 1.072) | **1.032** | **1.014** | **1.098** | **1.73** |
| Chronos-2 (119M) | 1.067 (1.055 to 1.082) | 1.063 | 1.039 | 1.106 | 2.46 |
| **Basin** | 1.083 (1.067 to 1.104) | 1.064 | 1.032 | 1.153 | 2.72 |
| TimeH2O-v1 (109M) | 1.118 (1.093 to 1.144) | 1.072 | 1.020 | 1.246 | 2.44 |

- All four models stay within 5% to 12% of the best possible score. Basin is
  third.
- The intervals of neighbouring models overlap, so the order between
  neighbours is not certain. Timestral is clearly ahead of Basin and TimeH2O.
- Skewed steps are harder for every model than symmetric steps.
- Short histories cause most of the error. With 2,048 points of history the
  four models score between 1.015 and 1.046; with 32 points between 1.078
  and 1.239.

## Uncertainty bands

The band between a model's 10% and 90% quantiles should hold 80% of the
outcomes. The quantile loss is divided by that of the true quantiles, which
only someone who knows how the series was made can give: 1.000 is a floor.

| Model | Quantile loss vs true quantiles (95% interval) | 80% band holds | Band width vs true | Outcomes below the 10% / 50% / 90% quantile |
|---|---|---|---|---|
| Timestral | **1.060** (1.054 to 1.066) | **80.0%** | 1.06 | 10% / 50% / 90% |
| **Basin** | 1.085 (1.078 to 1.093) | 75.3% | 0.97 | 13% / 50% / 88% |
| Chronos-2 | 1.096 (1.089 to 1.103) | 71.0% | 0.94 | 15% / 50% / 86% |
| TimeH2O-v1 | 1.103 (1.094 to 1.112) | 74.8% | 0.99 | 13% / 50% / 88% |

Basin is second. Its bands, like those of Chronos-2 and TimeH2O, are somewhat
too narrow.

## Meaningless covariates

7,298 of the series come with covariates that say nothing about the target.
Each was forecast twice by the same model, with and without them. The right
response is no change at all. Every model is given the history of the
covariates only, never their values over the forecast horizon.

| Model | Forecast moved by (typical steps) | Departure without | Departure with | RMSE with vs without (95% interval) |
|---|---|---|---|---|
| **Basin** | **0.74** | 2.71 | 2.71 | 0.996 (0.993 to 0.999) |
| Timestral | 0.77 | 1.73 | 1.80 | 1.002 (0.998 to 1.006) |
| Chronos-2 | 1.55 | 2.44 | 2.03 | 0.982 (0.976 to 0.988) |

Basin and Timestral are disturbed the least by covariates without
information, and equally little. Chronos-2 moves its forecast about twice as
far. TimeH2O-v1 is not in this table: it does not use the history of a
covariate, only values known into the future, which no model is given here.

## Speed

All four models ran on the same GPU (RTX 5070). One forecast is one series
without covariates, one request at a time (median of 200 requests). The last
column is the rate when the whole benchmark is forecast in batches of 16.

| Model | One forecast (median) | 95% within | History 32 | History 128 | History 512 | History 2,048 | Forecasts per second, batched |
|---|---|---|---|---|---|---|---|
| **Basin** | to be measured again | | | | | | 407 |
| Chronos-2 | 17.2 ms | 18.5 ms | 17.6 ms | 17.2 ms | 17.2 ms | 16.8 ms | **561** |
| Timestral | 45.5 ms | 50.5 ms | 48.9 ms | 44.9 ms | 45.3 ms | 45.4 ms | 208 |
| TimeH2O-v1 | 159.4 ms | 171.4 ms | 156.9 ms | 159.7 ms | 158.7 ms | 161.0 ms | 46 |

Chronos-2 is the fastest for forecasts in batches. Basin's time for a single
forecast is left out: in this run it was measured after the batched run, when
Basin's memory budget for prepared computations was used up, so the single
forecasts ran on a slower path than they do in normal use. On the holdout
set, where single forecasts are timed first, Basin takes 6.1 ms.

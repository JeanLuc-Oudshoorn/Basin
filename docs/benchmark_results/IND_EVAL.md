# Benchmark results: IND-Eval

[IND-Eval](https://huggingface.co/datasets/cognitedata/IND-Eval) (Haugland et
al., 2026) is a public forecasting benchmark of real industrial time series
from eight sources. Basin was not trained on any of it. Unlike the
[holdout set](HOLDOUT_SET.md), this data does not come from Basin's own
generator.

| | |
|---|---|
| Basin weights | best weights after epoch 3 (step 4,526,436), 221.5 million parameters |
| Data | IND-Eval release v0.1.3: 162,244 series in 150 tasks from 8 sources |
| Forecast | The first 112 points after the history (fewer where the task is shorter), with the quantiles 10% to 90% of every point |
| Inputs | Every model is given the same: the latest 2,048 points at most of the series' own history. Other channels of the same system are not used, and no model is given future values. |
| Run | `results/model/ind_eval/20261128T104419Z_epoch3_step4526436` |

The sources:

| Source | What it is | Tasks | Series |
|---|---|---|---|
| 3W | Sensors of oil wells around fault events | 20 | 8,557 |
| Electricity | Electricity consumption of customers, from 15 minutes to weekly | 8 | 34,029 |
| Mesa del Sol (MDS) | A microgrid: solar, battery, generator and load | 3 | 89,586 |
| MetroPT-3 | The air compressor of a metro train | 3 | 2,850 |
| NEATS | Offshore drilling | 27 | 4,925 |
| PRONTO | A multiphase flow rig around fault events | 6 | 72 |
| Solar | Solar power production, from 10 minutes to weekly | 8 | 8,494 |
| Tablet compression | Pharmaceutical tablet presses, 25 products | 75 | 13,731 |

A task is one source at one sampling rate or product and one forecast length.
Series that repeat a series of an earlier task (37,432) are left out.

**How to read the scores.** Every score is the model's error divided by that
of the seasonal-naive forecast (repeating the last cycle) on the same series.
Below 1 is better than seasonal naive. Tasks are combined by geometric mean
within a source, and the sources by geometric mean, so every source counts the
same whatever its size. On the sources without a cycle (3W, MetroPT-3, NEATS,
PRONTO, tablet compression) seasonal naive is the naive forecast.

**Comparing with the published leaderboard.** The numbers here are not those
of the IND-Eval paper: history and forecast are cut to 2,048 and 112 points,
a ninth source (ETT) is not included, and the cycle length of electricity and
solar is set from the sampling rate. Compare models within this page.

## Accuracy of the median forecast

nMASE: the mean absolute error of the median forecast, scaled by the series'
own changes in its history, over that of seasonal naive. Lower is better.

| Source | Basin | Timestral | TimeH2O-v1 | Chronos-2 | Holt-Winters | Naive |
|---|---|---|---|---|---|---|
| **All sources** | 0.743 | 0.671 | **0.669** | 0.744 | 1.037 | 1.237 |
| 3W | 0.587 | 0.543 | **0.531** | 0.681 | 0.762 | 1.000 |
| Electricity | 0.798 | 0.683 | **0.644** | 0.664 | 1.066 | 2.228 |
| Mesa del Sol | 0.826 | **0.762** | 0.766 | 0.783 | 1.074 | 1.098 |
| MetroPT-3 | 0.368 | **0.297** | 0.298 | 0.399 | 0.943 | 1.000 |
| NEATS | 0.941 | **0.826** | 0.830 | 0.896 | 1.016 | 1.000 |
| PRONTO | 1.022 | 1.042 | 1.031 | 1.084 | 1.295 | **1.000** |
| Solar | 0.930 | **0.818** | 0.848 | 0.925 | 1.399 | 2.238 |
| Tablet compression | 0.728 | **0.696** | 0.706 | 0.739 | 0.882 | 1.000 |

### Share of series with a lower error than seasonal naive

| Source | Basin | Timestral | TimeH2O-v1 | Chronos-2 | Holt-Winters |
|---|---|---|---|---|---|
| **All series** | 56.9% | **63.3%** | 63.2% | 60.7% | 41.5% |
| 3W | 74.0% | 78.0% | **78.3%** | 71.4% | 65.9% |
| Electricity | 72.4% | 87.1% | **89.9%** | 87.3% | 44.9% |
| Mesa del Sol | 48.5% | **51.2%** | 50.6% | 49.3% | 37.7% |
| MetroPT-3 | 75.1% | 80.1% | **81.9%** | 74.4% | 35.1% |
| NEATS | 55.0% | **59.2%** | 59.1% | 54.8% | 32.8% |
| PRONTO | 50.0% | **58.3%** | 55.6% | 55.6% | 33.3% |
| Solar | 42.2% | **64.2%** | 57.8% | 49.1% | 22.8% |
| Tablet compression | 68.6% | **71.3%** | 70.8% | 69.1% | 58.2% |

"All series" counts every series once, so Mesa del Sol, with more than half
of the series, weighs most in that row.

What the tables show:

- **Basin is level with Chronos-2 and 11% behind Timestral and TimeH2O-v1**
  over all sources (0.743 against 0.744, 0.671 and 0.669).
- **Basin is clearly better than the classical methods.** Its error is 26%
  below that of seasonal naive and 28% below that of Holt-Winters.
  Holt-Winters is no better than seasonal naive overall.
- **The gap to the best model differs by source.** It is 5% on tablet
  compression, 8% on Mesa del Sol and 11% on 3W; 14% on NEATS and solar; and
  24% on electricity and MetroPT-3.
- **Basin is weakest where the series has a daily cycle.** On electricity it
  is last of the four pretrained models, and it beats seasonal naive on 72%
  of the series where the others do on 87% to 90%. On solar it beats
  seasonal naive on 42% of the series against 64% for Timestral.
- **Against Chronos-2 it depends on the source.** Basin is ahead on 3W,
  MetroPT-3, PRONTO and tablet compression, level on solar, and behind on
  electricity, Mesa del Sol and NEATS.
- **Nothing beats the naive forecast on PRONTO.** It has only 72 series, and
  the differences between the pretrained models there are small.
- **The gap is about the same for short and long forecasts**: 7% behind the
  best model on the tasks with a short forecast, 9% on those with a long one.

## Quality of the quantiles

nCRPS: the quantile loss over the levels 10% to 90%, summed over a task, over
that of seasonal naive. Lower is better. Holt-Winters and the naive forecast
give one value, which is scored as if it were every quantile; that favours
the models that give quantiles, so they are left out of this table.

| Source | Basin | Timestral | TimeH2O-v1 | Chronos-2 |
|---|---|---|---|---|
| **All sources** | 0.786 | 0.538 | **0.537** | 0.594 |
| 3W | 0.511 | 0.418 | **0.409** | 0.522 |
| Electricity | 0.626 | 0.518 | **0.487** | 0.507 |
| Mesa del Sol | 1.861 | **0.512** | 0.514 | 0.536 |
| MetroPT-3 | 0.329 | **0.265** | 0.267 | 0.353 |
| NEATS | 0.769 | **0.701** | 0.722 | 0.766 |
| PRONTO | 1.080 | 0.956 | **0.938** | 0.968 |
| Solar | 0.746 | **0.631** | 0.658 | 0.697 |
| Tablet compression | 1.198 | **0.563** | 0.566 | 0.597 |

**Basin's scores on Mesa del Sol and tablet compression are caused by a few
hundred series on which its outer quantiles are far too wide.** On those
series Basin's median forecast is about as accurate as that of the other
models, but its quantile loss is hundreds to millions of times its absolute
error, which can only happen when outer quantiles lie far outside the range
of the series. Because the score sums the loss over
a task, a few such series decide it:

- On Mesa del Sol a single series of 89,586 accounts for 96% of Basin's
  total quantile loss.
- On tablet compression ten series of 13,731 account for 79% of it.
- Basin's quantile loss is more than a hundred times its absolute error on
  318 series (0.2% of all): 198 of tablet compression, 109 of Mesa del Sol,
  9 of NEATS and 2 of electricity. Nearly all are forecasts of the full 112
  points; the tasks with a forecast of 30 points are hardly affected.

Leaving out the 3.5% of series on which any of the four models has a
quantile loss of more than three times its absolute error gives:

| | Basin | Timestral | TimeH2O-v1 | Chronos-2 |
|---|---|---|---|---|
| All sources | 0.619 | 0.538 | **0.537** | 0.591 |
| Mesa del Sol | 0.594 | **0.536** | 0.539 | 0.555 |
| Tablet compression | 0.592 | **0.560** | 0.563 | 0.594 |

Only Basin's score changes. On the other 96.5% of the series Basin's
quantiles are 5% behind those of Chronos-2 and 15% behind those of Timestral
and TimeH2O-v1, in line with the accuracy of its median. The scores of the
other three models hardly change, so they do not depend on a few series in
this way.

For a user this means: **Basin's median forecast can be relied on as the
tables above describe, but its outer quantiles at the far end of a 112-point
forecast are occasionally unusable**, and should be checked against the range
of the history before they are used for an alarm limit.

## Speed

Series forecast per second over the whole benchmark, in batches, on an RTX
5070 (Holt-Winters on the CPU with 10 processes).

| Basin | Chronos-2 | Timestral | TimeH2O-v1 | Holt-Winters |
|---|---|---|---|---|
| **662** | 635 | 281 | 21 | 64 |

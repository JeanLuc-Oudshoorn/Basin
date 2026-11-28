# Benchmark results

Results of Basin against published time-series foundation models. All
results below are of Basin's best weights after epoch 3 (221.5 million
parameters).

## Overall

| Benchmark | What it measures | Basin | Timestral | TimeH2O-v1 | Chronos-2 |
|---|---|---|---|---|---|
| [Holdout set](HOLDOUT_SET.md) | Median MASE over 49,475 tables (lower is better) | **4.31** | 4.67 | 4.76 | 5.06 |
| [FakeBench-TS](FAKEBENCH_TS.md) | RMSE against the best possible forecast of 10,000 random walks (1.000 is the best a forecast can expect) | 1.083 | **1.049** | 1.118 | 1.067 |
| [Operational benchmarks](OPERATIONAL_BENCHMARKS.md) | Median MASE on four public operational datasets (caster / CATS / cloud monitoring / UCR; lower is better) | 3.63 / 9.72 / 2.38 / 2.05 | **3.11** / **9.19** / 2.08 / 1.66 | 3.23 / 9.56 / 2.17 / **1.57** | 3.53 / 9.73 / **2.08** / 1.60 |
| [IND-Eval](IND_EVAL.md) | Error of the median forecast on eight sources of industrial data, over that of the seasonal-naive forecast (below 1 is better than seasonal naive) | 0.743 | 0.671 | **0.669** | 0.744 |
| [Holdout set](HOLDOUT_SET.md) | Median time of one forecast on an RTX 5070 | **6.1 ms** | 47.7 ms | 144.4 ms | 17.1 ms |

- **Holdout set.** Basin is the most accurate model, and its lead grows with
  the difficulty of the table. The holdout tables come from the same
  generator as Basin's training data, which favours Basin.
- **FakeBench-TS.** On series that cannot be forecast, all four models stay
  within 5% to 12% of the best possible score. Basin is third in accuracy,
  second in the quality of its uncertainty bands, and with Timestral the
  least disturbed by covariates that carry no information.
- **Operational benchmarks.** On real operational data from elsewhere Basin
  is behind the best published model at forecasting, by 6% to 30%
  depending on the dataset, and ahead of Holt-Winters and the naive forecast
  on three of the four datasets. At detecting anomalies the four pretrained
  models are close to each other.
- **IND-Eval.** On real industrial data Basin is level with Chronos-2 and 11%
  behind Timestral and TimeH2O-v1, and 26% better than the seasonal-naive
  forecast. It is weakest on series with a daily cycle (electricity, solar).
  On a few hundred of the 162,244 series its outer quantiles are far too
  wide at the end of the forecast.

## The benchmarks

| Document | Benchmark |
|---|---|
| [HOLDOUT_SET.md](HOLDOUT_SET.md) | Forecasts of the holdout tables: independent of the training data, from the same generator. Also compares Holt-Winters and the naive forecast. |
| [FAKEBENCH_TS.md](FAKEBENCH_TS.md) | [FakeBench-TS](https://github.com/JeanLuc-Oudshoorn/fakebench-ts): random walks, where the best forecast is known exactly. Accuracy, uncertainty bands, meaningless covariates and speed. |
| [OPERATIONAL_BENCHMARKS.md](OPERATIONAL_BENCHMARKS.md) | Four public datasets of operational data (an aluminium caster, a simulated industrial system, cloud service telemetry, and electricity, weather and spacecraft series): forecasting normal data and detecting anomalies, each dataset on its own. Also compares Holt-Winters and the naive forecast. |
| [IND_EVAL.md](IND_EVAL.md) | [IND-Eval](https://huggingface.co/datasets/cognitedata/IND-Eval): forecasting real industrial series from eight sources (oil wells, electricity, a microgrid, a metro compressor, offshore drilling, a flow rig, solar power, tablet presses). Accuracy, quantiles and speed. Also compares Holt-Winters and the naive forecasts. |

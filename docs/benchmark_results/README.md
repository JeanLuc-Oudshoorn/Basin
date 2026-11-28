# Benchmark results

Results of Basin against published time-series foundation models. All
results below are of Basin's best weights after epoch 3 (221.5 million
parameters).

## Overall

| Benchmark | What it measures | Basin | Timestral | TimeH2O-v1 | Chronos-2 |
|---|---|---|---|---|---|
| [Holdout set](HOLDOUT_SET.md) | Median MASE over 49,475 tables (lower is better) | **4.31** | 4.67 | 4.76 | 5.06 |
| [FakeBench-TS](FAKEBENCH_TS.md) | RMSE against the best possible forecast of 10,000 random walks (1.000 is the best a forecast can expect) | 1.083 | **1.049** | 1.118 | 1.067 |
| Both | Median time of one forecast on an RTX 5070 (holdout / FakeBench-TS) | **6.1 ms / 5.9 ms** | 47.7 ms / 42.9 ms | 144.4 ms / 157.7 ms | 17.1 ms / 15.5 ms |

- **Holdout set.** Basin is the most accurate model, and its lead grows with
  the difficulty of the table. The holdout tables come from the same
  generator as Basin's training data, which favours Basin.
- **FakeBench-TS.** On series that cannot be forecast, all four models stay
  within 5% to 12% of the best possible score. Basin is third in accuracy,
  second in the quality of its uncertainty bands, and the least disturbed by
  covariates that carry no information.

## The benchmarks

| Document | Benchmark |
|---|---|
| [HOLDOUT_SET.md](HOLDOUT_SET.md) | Forecasts of the holdout tables: independent of the training data, from the same generator. Also compares Holt-Winters and the naive forecast. |
| [FAKEBENCH_TS.md](FAKEBENCH_TS.md) | [FakeBench-TS](https://github.com/JeanLuc-Oudshoorn/fakebench-ts): random walks, where the best forecast is known exactly. Accuracy, uncertainty bands, meaningless covariates and speed. |

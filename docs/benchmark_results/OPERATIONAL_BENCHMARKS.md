# Benchmark results: operational benchmarks

Four public datasets of operational data, none of which Basin was trained on,
and two tasks: forecasting normal data, and detecting anomalies. Unlike the
[holdout set](HOLDOUT_SET.md), this data does not come from Basin's own
generator.

| | |
|---|---|
| Basin weights | best weights after epoch 3 (step 4,526,436), 221.5 million parameters |
| Cases | 10,543: a history of 128, 512 or 2,048 points (a third of the cases each) and the 112 points after it |
| Forecast | 112 points, with the quantiles 10% to 90% of every point |
| Inputs | Every model is offered the same: the history of the series and, on CATS and the caster, the history of five other channels of the same system. No model is given future values. Basin, Chronos-2 and Timestral use the other channels; TimeH2O, Holt-Winters and the naive forecast cannot. |
| Run | `results/model/operational_benchmarks/20261127T192353Z_epoch3_step4526436` |

The datasets:

| Dataset | What it is | Forecast cases | Anomaly cases |
|---|---|---|---|
| Micromill caster | Sensors of a continuous aluminium caster (Carnegie Mellon) | 3,168 | none: the data has no labels |
| CATS | A simulated industrial system with 200 injected anomalies | 1,200 | 2,244 |
| Microsoft cloud monitoring | Telemetry of software services, anomalies labelled by their engineers | 890 | 311 |
| UCR, operational series | The 35 series of the UCR anomaly archive that are operational data: electricity demand, air temperature, spacecraft telemetry | 2,520 | 210 |

Compare models within a dataset. The datasets differ too much to compare
with each other, so there is no combined score.

## Forecasting normal data

MASE of the median forecast: the mean absolute error divided by the mean
absolute one-step change in the history. Each cell is the median over the
cases of the dataset. Lower is better.

### Over all 112 steps

| Dataset | Basin | Timestral | TimeH2O-v1 | Chronos-2 | Holt-Winters | Naive |
|---|---|---|---|---|---|---|
| Micromill caster | 3.63 | **3.11** | 3.23 | 3.53 | 3.94 | 3.60 |
| CATS | 9.72 | **9.19** | 9.56 | 9.73 | 10.51 | 10.12 |
| Microsoft cloud monitoring | 2.38 | 2.08 | 2.17 | **2.08** | 2.67 | 2.94 |
| UCR, operational series | 2.05 | 1.66 | **1.57** | 1.60 | 3.32 | 5.74 |

### Over the first 32 steps

| Dataset | Basin | Timestral | TimeH2O-v1 | Chronos-2 | Holt-Winters | Naive |
|---|---|---|---|---|---|---|
| Micromill caster | 2.04 | **1.91** | 1.96 | 2.14 | 2.36 | 2.25 |
| CATS | 4.29 | **3.99** | 4.15 | 4.74 | 5.36 | 5.24 |
| Microsoft cloud monitoring | 1.30 | **1.22** | **1.22** | 1.24 | 1.46 | 1.74 |
| UCR, operational series | 1.30 | 1.14 | **1.07** | 1.09 | 2.29 | 4.20 |

### Share of cases with a lower MASE than the naive forecast (112 steps)

| Dataset | Basin | Timestral | TimeH2O-v1 | Chronos-2 | Holt-Winters |
|---|---|---|---|---|---|
| Micromill caster | 57.9% | **64.8%** | 62.0% | 59.1% | 41.5% |
| CATS | 65.4% | **68.6%** | 66.3% | 64.9% | 35.0% |
| Microsoft cloud monitoring | 60.1% | **66.7%** | 64.3% | 65.4% | 52.8% |
| UCR, operational series | 91.3% | **95.1%** | 94.7% | 93.1% | 69.2% |

What the tables show:

- **Basin is behind the best published model on every dataset.** Over 112
  steps it is 6% behind on CATS (level with Chronos-2), 14% to 17% on the cloud data and
  the caster, and 30% on the UCR series. On the holdout set, which comes from
  Basin's own generator, Basin leads; on data from elsewhere it does not.
- **Basin is clearly better than the classical methods** on three of the four
  datasets. On the caster it is level with the naive forecast over 112 steps
  and better over the first 32.
- **The gap is smaller over the first 32 steps** (6% to 8% on three datasets)
  and grows with the length of the forecast.
- **The gap is largest on strongly periodic series with a short history.** On
  the UCR electricity and spacecraft series Basin's MASE is 1.5 to 1.6 against
  1.1 to 1.2 for Timestral. With 128 points of history Basin scores 3.77 on
  the UCR series against 2.36; with 2,048 points 1.42 against 1.31.
- **Holt-Winters is worse than the naive forecast on the caster and CATS**,
  and better on the cloud data and the UCR series.

### Are the uncertainty bands right?

The share of normal points between a model's 10% and 90% quantiles. It should
be 80%.

| Dataset | Basin | Timestral | TimeH2O-v1 | Chronos-2 | Holt-Winters | Naive |
|---|---|---|---|---|---|---|
| Micromill caster | 78.0% | 81.0% | 79.1% | 76.0% | 88.5% | 79.8% |
| CATS | 77.2% | 79.9% | 75.5% | 73.1% | 90.7% | 63.5% |
| Microsoft cloud monitoring | 79.5% | 82.2% | 80.9% | 77.4% | 88.4% | 93.0% |
| UCR, operational series | 80.5% | 81.6% | 82.2% | 72.9% | 90.9% | 81.6% |

Basin's bands hold 77% to 81% of the points on every dataset. Only Timestral
is as consistent. The bands of Holt-Winters are too wide everywhere.

## Detecting anomalies

A model is given the history up to shortly before a labelled anomaly and
forecasts across it. A point is **outside the interval** when it lies below
the model's 10% quantile or above its 90% quantile, and a window is
**flagged** when at least one of its judged points is outside. The same rule
is applied to normal windows of the same series, which gives the false
alarms.

Holt-Winters and the naive forecast give no quantiles of their own. Theirs
are those of a normal distribution around the forecast. For the naive
forecast the spread is that of a random walk with the steps of the history,
widening with the square root of the step. For Holt-Winters it follows from
its one-step errors within the history and its fitted smoothing parameters.

- **Recall**: the share of anomalies that are flagged.
- **False alarms**: the share of normal windows that are flagged.
- **F1**: combines recall and precision, with one normal window for every
  anomaly. A model that flags every window scores 0.667.
- **Separation**: how often an anomaly lies further outside the interval than
  a normal window does. 0.5 is chance, 1 is always. Unlike the other three it
  does not depend on where the line for flagging is drawn, which makes it the
  fairest comparison of the models.

### CATS (2,244 cases)

| Model | F1 | Recall | False alarms | Separation |
|---|---|---|---|---|
| Basin | 0.692 | 86.5% | 63.4% | **0.812** |
| Timestral | **0.702** | 85.7% | 58.4% | 0.808 |
| TimeH2O-v1 | 0.697 | 88.5% | 65.5% | 0.802 |
| Chronos-2 | 0.696 | 89.5% | 67.8% | 0.801 |
| Holt-Winters | 0.633 | 55.3% | 19.4% | 0.757 |
| Naive | 0.662 | 82.5% | 66.9% | 0.777 |

### Microsoft cloud monitoring (311 cases)

| Model | F1 | Recall | False alarms | Separation |
|---|---|---|---|---|
| Basin | 0.818 | 99.2% | 43.2% | 0.976 |
| Timestral | 0.828 | 100.0% | 41.4% | 0.981 |
| TimeH2O-v1 | 0.823 | 100.0% | 42.9% | **0.984** |
| Chronos-2 | 0.802 | 100.0% | 49.5% | 0.977 |
| Holt-Winters | 0.863 | 97.5% | 28.4% | 0.964 |
| Naive | **0.878** | 85.8% | 9.6% | 0.973 |

### UCR, operational series (210 cases)

| Model | F1 | Recall | False alarms | Separation |
|---|---|---|---|---|
| Basin | **0.705** | 91.9% | 68.8% | 0.835 |
| Timestral | 0.704 | 91.4% | 68.4% | **0.890** |
| TimeH2O-v1 | 0.703 | 91.0% | 67.7% | 0.888 |
| Chronos-2 | 0.701 | 96.7% | 79.0% | 0.874 |
| Holt-Winters | 0.584 | 52.9% | 28.2% | 0.753 |
| Naive | 0.560 | 56.2% | 44.6% | 0.687 |

What the tables show:

- **The four pretrained models are close to each other on every dataset.**
  On CATS Basin separates anomalies from normal data best, by a small margin;
  on the UCR series it is behind the other three.
- **F1 says little here.** On CATS and the UCR series the pretrained models
  find most anomalies but also flag 58% to 79% of the normal windows, so
  their F1 is barely above the 0.667 of flagging everything. A long anomaly
  gives up to 96 judged points, and of 96 normal points one is almost always
  outside an 80% interval. The rule for flagging is too loose for long
  anomalies.
- **The cloud anomalies are easy.** Every model, the naive forecast included,
  separates them from normal data almost perfectly (0.96 to 0.98). The
  differences in F1 on this dataset come from how wide a model's interval
  is: the naive forecast's interval is too wide (it holds 93% of the normal
  points), which gives it few false alarms.
- **The pretrained models are clearly better than the classical methods on
  the UCR series**, and a little better on CATS.

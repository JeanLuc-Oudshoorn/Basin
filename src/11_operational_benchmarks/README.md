# Operational benchmarks stage

Four public datasets and two tasks, with the same models as the holdout test
(Basin, Naive, Chronos-2, Timestral, TimeH2O):

| | Forecasting normal data | Detecting anomalies |
|---|---|---|
| **CATS** | 1,200 cases | 2,244 cases (190 of the 200 anomalies) |
| **UCR, operational series** | 2,520 cases | 210 cases (35 anomalies) |
| **Microsoft cloud monitoring** | 890 cases | 311 cases (73 of the 281 anomalies) |
| **Micromill caster** | 3,168 cases | none: the data has no anomaly labels |

A case is one request to a model: a context of 128, 512 or 2,048 points and
the 112 points after it. Every model forecasts all 112 steps of every case.
Nothing here touches training data or the holdout.

## The data

Download the four archives into `benchmarks/` (they are not in git):

- [CATS](https://www.kaggle.com/datasets/patrickfleith/controlled-anomalies-time-series-dataset)
  into `benchmarks/CATS Dataset` (`data.csv`, `metadata.csv`)
- [UCR anomaly archive 2021](https://figshare.com/articles/dataset/UCR_Time_Series_Anomaly_Detection_datasets_2021_/26410744)
  into `benchmarks/UCR_TimeSeriesAnomalyDatasets2021`
- [Microsoft Cloud Monitoring Dataset](https://github.com/microsoft/cloud-monitoring-dataset):
  `git clone https://github.com/microsoft/cloud-monitoring-dataset benchmarks/cloud-monitoring-dataset`
  (used here at commit `6a07a53`)
- [Micromill Continuous Caster](https://doi.org/10.1184/R1/33432847): unpack
  the four CSV files into `benchmarks/micromill-caster`

**CATS** is one simulated system, recorded every second for 5 million seconds,
without noise. It is simulated data, not measurements. Here every point is the
mean of 10 seconds (`cats.seconds_per_point`), which gives 500,000 points per
channel. Of its 17 channels the 10 telemetry readings are forecast. It has 200
injected anomalies of 30 to 500 points (5 to 83 minutes). The metadata names
the channel each one starts in (root cause) and the channel it may spread to
(affected); both are cases, reported separately.

**UCR** is 250 separate series of 8,000 to 120,000 points with one labelled
anomaly each. Only 35 are operational data, and they come from three sources:

| Category | Series | What it measures |
|---|---|---|
| electricity | 11 | Italian electricity demand, 1995 to 1998 |
| weather | 13 | hourly air temperature at one Californian weather station |
| spacecraft telemetry | 11 | one telemetry channel of a NASA spacecraft |
| *left out:* physiology | 153 | electrocardiograms, respiration, blood pressure of patients and pigs |
| *left out:* human motion | 34 | force plates, accelerometers and markers on walking people |
| *left out:* biology | 28 | feeding signal of an insect, accelerometer on a whale |

The classification is `UCR_FAMILIES` in `operational_datasets.py`, taken from
the archive's own description of each series. Half of the 35 are copies of
the others with distortion or noise added by the archive's authors (the
`variant` column). `operational_benchmarks.ucr.categories` in the
configuration chooses the categories.

**Microsoft cloud monitoring** is 60 separate series from the production
telemetry of Microsoft services, with anomalies labelled by the engineers who
run them. It is not sensor data: it measures software systems.

| Category | Series | What it measures |
|---|---|---|
| cloud service telemetry | 35 | latency of backend calls (23), data ingress rate (5), database query rates (5), API request rate (1), failed requests (1) |
| software crash rate | 19 | crash rate of a Windows application by version and country |
| *left out:* commercial | 6 | count of in-app purchases |

The series are short: 23 have 720 hourly points, 19 have 176 to 1,118, and 12
have 6,000 to 20,000. Most hold several anomalies of a few points each, close
together, and 5 hold none. That is why only 73 of the 281 anomalies give a
case: the others lie in the first 144 points of their series or within 128
points of the anomaly before them. Only the longer series have cases with a
context of 512 or 2,048. The crash-rate series repeat some hours and miss others;
they are put on an even hourly grid and windows with a missing value are not
used. The classification is `CLOUD_DOMAINS` in `operational_datasets.py`.

**Micromill caster** is measured data from an industrial steel caster: 22
sensors (casting speed, mold level, tundish weight, superheat, strand surface
temperature, and the flow and pressure of the mold and seven cooling zones),
twice a second, in four recordings of one to three hours (14 heats). Every
sensor of every recording is a series, 88 in all. A 112-step forecast covers
56 seconds; `caster.samples_per_point` averages to a coarser step.

It has no anomaly labels, so it gives forecast cases only. Two periods are not
normal operation and no case comes from them or from the two minutes next to
them: the first 8 minutes of recording 2, before the start-up has reached the
pyrometer, and recording 1 from its breakout on (the mold empties 35 minutes
in and the caster stops). They are found by one rule, `CASTER_CASTING` in
`operational_datasets.py`: the strand moves, mold and tundish hold steel, the
pyrometer sees the strand and the superheat sensor gives a reading.

## Covariates

The models take at most five covariates. Every CATS case gets five of the
system's other 16 channels (commands, stimuli and the other telemetry), drawn
at random for that case, as past covariates: their values over the context,
nothing of the future. Every caster case likewise gets five of the other 21
sensors of its recording. The draw is seeded, so preparing again gives the
same cases. The UCR and cloud series are single series and get no covariates.

Basin, Chronos-2 and Timestral use the covariates. TimeH2O cannot use past
covariates and Naive uses none, so both forecast from the target's own history.
Set `covariates.count` to 0 to run every model without covariates.

## Prepare the cases

```powershell
uv run python src/11_operational_benchmarks/prepare_cases.py
```

This reads the archives (about two minutes) and writes `cases.parquet` and
`prepared.json` to `data/04_benchmarks/operational`
(`paths.operational_benchmark_data`). `prepared.json` lists every series with
its category and whether it was included, and counts the cases.

- **Forecast cases** lie in normal data: context and future stay at least 100
  points away from every labelled anomaly (16 points for the short cloud
  series). For CATS that also means away from anomalies in other channels.
  They are spread evenly over each series and at least 112 points apart: 40
  per CATS channel and context length, up to 24 per UCR or cloud series and
  12 per caster sensor and recording.
- **Anomaly cases** end their context 16 or 64 points before the labelled
  start of an anomaly. The model has therefore seen no part of the anomaly,
  and the anomaly begins inside the forecast. Only its points inside the
  forecast are judged: at most 96 or 48. With a lead of 16, three quarters of
  the CATS anomalies fit in the forecast completely.
- The last 128 points of the context and the lead must be free of other
  anomalies (`anomaly.normal_points_before`); 10 CATS anomalies follow another
  one too closely and have no case. An anomaly that ended earlier may lie
  further back in a long context. That is so for 21% of the CATS cases with a
  context of 512 and 65% of those with 2,048; the results are also given
  separately for cases with and without one.

## Run the models

```powershell
uv run python src/11_operational_benchmarks/operational_benchmarks.py
```

Like the other GPU stages it routes itself to WSL. Do not run it while
training or the holdout test is using the GPU. Basin uses the checkpoint the
holdout test would use; the published models use the IDs and revisions under
`holdout` in the configuration.

| Option | Effect |
|---|---|
| `--models basin naive` | which models to run; the baselines `naive`, `mean` and `median` need no GPU and run where the command is given |
| `--limit 200` | an even subset of the cases, for a quick trial |
| `--batch-size 16` | contexts per model call |
| `--rescore RUN` | score the saved forecasts of a run folder again, without running a model |

Each run writes a new folder under `results/model/operational_benchmarks`
(`paths.operational_benchmarks`), named by its start time and Basin's epoch
and step:

- `leaderboard.md`: the four tables, then the same by context length, by
  category, and for CATS by role of the channel, anomaly type and whether the
  context holds an earlier anomaly
- `results.json`: all numbers, also per series
- `per_case.parquet`: every model's scores on every case
- `examples.html`: four anomaly cases per dataset, with every model's median
  and interval against the actual values
- `forecasts/`: the nine quantiles of every model for every case

## How the tasks are scored

**Forecasting.** The point forecast is the median. MASE is the mean absolute
error divided by the mean absolute step of the context, over the first 32, the
first 64 and all 112 steps. The tables give the median over the cases. Compare
models within a dataset, not datasets with each other.

**Anomaly detection.** Every model gives the quantiles 10% to 90% of every
step. A point is **outside the interval** when it lies below the 10% quantile
or above the 90% quantile. Nothing is assumed about the shape of the forecast
distribution. Naive has no quantiles of its own; it uses those of the textbook
interval of a random walk (the spread of the context's steps, widening with
the square root of the step). A window is **flagged** when at least one of its
judged points lies outside the interval.

A calibrated model has 20% of normal points outside this interval, by
definition of the two quantiles. A long anomaly gives up to 96 judged points,
and among 96 normal points some are almost always outside. So on long
anomalies (CATS) the rule flags most normal windows too, and on short ones
(the cloud series, typically 5 points) it does so far less. Two settings change the
rule without running a model again (`--rescore`):

- `anomaly.min_share_outside`: flag a window only when more than this share of
  its judged points lie outside the interval.
- `anomaly.interval: "normal"`: use the median plus and minus
  `anomaly.interval_sigmas` standard deviations of the normal distribution
  that has the model's 10% and 90% quantiles.

- **Recall**: the share of anomaly cases that are flagged.
- **False alarms**: for each anomaly case, the share of forecast cases of the
  same series and context length that are flagged on the same steps.
- **Precision**: of the flagged windows, the share that hold an anomaly.
  Precision depends on how many normal windows there are for every anomaly.
  Here there is one: each anomaly case is set against one normal window of
  its series, and the outcome of that window is the average over all of them.
  In live use anomalies are far rarer than that, and precision would be lower.
- **F1**: the harmonic mean of precision and recall, the main number. With one
  normal window per anomaly, a model that flags every window scores 0.667
  (precision 50%, recall 100%); a useful detector has to beat that.
- **Separation**: how often the anomaly lies further outside the interval than
  a normal window does. 0.5 is chance, 1 is always. It does not depend on
  where the line for flagging is drawn.
- **Flagged before the anomaly**: a point outside the interval in the normal
  steps between the end of the context and the labelled start.
- **Points inside the interval** (forecast tables): 80% for a calibrated
  model.

## Settings

`operational_benchmarks` in `configs/config.json`: the models, the context
lengths, the cases per series, the margin around anomalies, the lead points,
the number of covariates, the seconds per CATS point, the categories of each
dataset and the rule for flagging. After changing a setting that decides which
cases exist, prepare the cases again; the stage refuses cases prepared with
other settings. The rule for flagging (`interval`, `quantiles`,
`interval_sigmas`, `min_share_outside`) only changes the scores.

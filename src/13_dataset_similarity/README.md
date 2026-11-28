# Stage 13: how similar are the synthetic series to the benchmark series?

Basin is trained on synthetic series only. This stage measures how much those
series resemble the series of the benchmarks, where they differ, and in which
direction the generator would have to move to close the difference. It runs
on the CPU and changes nothing outside its own output folder and
`docs/dataset_similarity/`.

```
python src/13_dataset_similarity/dataset_similarity.py
python src/13_dataset_similarity/dataset_similarity.py --steps analyse report
```

## What is compared

One series is one target as a model sees it: the latest points before a
forecast. A table with covariates contributes its target column only, because
that is what a model is asked to forecast. This is what makes tables with and
without covariates comparable.

| Group | Series |
|---|---|
| Synthetic | The target of every table that training reserves for validation (25,000) |
| FakeBench-TS | The random walks with a history of 512 or 2,048 points |
| Operational benchmarks | The forecast cases of stage 11 with a history of 512 or 2,048 points |
| IND-Eval | 3,000 series per source of stage 12, drawn at random over its tasks |

Every series is described on its latest 512 points, and again on its latest
2,048 points where it has them. A fixed length is needed because a short
series cannot show a slow cycle: with mixed lengths the groups would differ
by their length before anything else.

## Steps

| Step | What it does | Saves |
|---|---|---|
| `collect` | Reads the series of the four groups | `windows.npy`, `windows.parquet` |
| `describe` | Computes the descriptors of every series | `features_<points>.parquet` |
| `embed` | Chronos-2 encoder embeddings of a part of every group | `embeddings_512.npz` |
| `analyse` | Distances, coverage, separation, maps | `analysis_<points>.parquet`, `.json` |
| `report` | Figures, tables, interactive map | `docs/dataset_similarity/` |

The output folder is `results/model/dataset_similarity`. The report step
writes figures (`figures/`), every table of numbers (`tables_<points>.md`)
and an interactive 3-D map (`map_<points>.html`, which loads `plotly.min.js`
from beside it) to `docs/dataset_similarity/`. The written report there,
`REPORT.md`, is not generated: it is the reading of one run.

On a machine that cannot read the project's model cache (Windows cannot
follow the links WSL makes in it), pass a folder that holds the Chronos-2
`config.json` and `model.safetensors` with `--embedding-model`.

## The descriptors

A series is standardised (mean 0, spread 1) after its gaps are filled by
linear interpolation, and then described by 56 numbers in eight families:

| Family | Descriptors |
|---|---|
| Memory | Autocorrelation at 1, 4, 16 and 64 steps, autocorrelation of the changes and of their changes, long-memory exponent, decorrelation time, first dip of the autocorrelation, partial autocorrelations |
| Spectrum | Spectral entropy, spectral slope, share of variance in four bands of cycle length, mean frequency, share of the strongest cycle |
| Trend | Share explained by a straight line, trend strength, curvature |
| Seasonality | Seasonal strength, correlation one cycle back, how far the cycle is from a sine wave, how much its size varies, share of the cycle spent at its floor |
| Changes over time | Stability, lumpiness, largest level shift, largest change of spread, clustering of large changes, longest stretch above the mean |
| Distribution | Skewness, tail weight, two-level behaviour, the same for the changes, share of movement in the largest jumps, where the series sits most often, largest excursion, share of rises, resting at the floor or the ceiling |
| Sensor behaviour | Flat runs and the longest one, distinct values, share at the most common value, share at a limit, missing share, whether the values lie on a grid |
| Noise and complexity | Noise share, permutation entropy, sample entropy, median crossings, error of a three-point forecast, share of turning points |

Trend and seasonal strength, spectral entropy, the long-memory exponent,
stability and lumpiness are the six features that GIFT-Eval reports for its
datasets, defined as in Hyndman's `tsfeatures`. Twenty more follow `catch22`
and `tsfeatures` (decorrelation time, partial autocorrelations, crossing
points, sample entropy and others). The sensor-behaviour family is added
here because those sets lack it and operational data shows it.

In a distance every family counts equally, and each of the four groups
counts equally in the centre and spread used for standardising.

## The measures

- **Coverage.** A benchmark series is covered when its fifth-nearest
  synthetic series is no farther away than the fifth-nearest neighbour of 95%
  of the synthetic series is from them. A fifth of the synthetic series is
  set aside and treated like a benchmark: it must come out about 95% covered.
- **Separation.** Per descriptor, the chance that a benchmark series has a
  higher value than a synthetic one (0.5: no difference), and the share of
  benchmark series beyond the synthetic 1% to 99% range. A classifier with
  all descriptors gives one number per group (0.5: cannot tell them apart).
- **Kinds of uncovered series.** The uncovered series are grouped, and every
  kind is listed with what sets it apart and with examples.
- **Accuracy.** For the forecast cases of stage 11: whether Basin loses more
  against the other models on series far from the synthetic ones.
- **Second opinion.** The same coverage on Chronos-2 embeddings, which do not
  depend on the choice of descriptors.

## Scoring a change of the generator

`generator_realism.py` measures how easily generated series are told from real
ones. It draws tables with the settings of
`src/01_data_generation/generate_operational_data.py` as they are at that
moment, so a change is tested by editing the generator and running a trial:

```
python src/13_dataset_similarity/generator_realism.py reference          # once
python src/13_dataset_similarity/generator_realism.py sample --name trial
python src/13_dataset_similarity/series_embeddings.py --windows results/model/generator_realism/trials/trial/windows.npy --output results/model/generator_realism/trials/trial/embeddings.npy --device cuda
python src/13_dataset_similarity/generator_realism.py score --name trial
```

`reference` collects four groups of real series, at most 5,000 each:
FakeBench-TS, the operational benchmarks and IND-Eval as the `collect` step
above gathered them, and windows of the measured columns of the datasets in
`data/00_public`. `sample` draws 12,000 tables (in memory; nothing is written
to the data folders) and keeps the latest 512 points of every target. The
embedding command runs on the GPU in the WSL environment or on the CPU
(`--device cpu`, about three minutes); the reference folder needs it once
too. `score` reports:

- **Score.** For every group, the AUC of a LightGBM classifier that tells its
  series from the generated ones, on the descriptors and on the embeddings,
  on series it was not trained on (the mean over three divisions into five
  folds). The mean of the two, and the mean over the groups. 1.0: always told
  apart; 0.5: cannot be told apart. Lower is better.
- **Unseen sources.** The same with whole datasets held out, so that the
  classifier cannot recognise a source by its cycle length or resolution and
  has to rely on what real sources have in common. This is the check that a
  gain on the score is not a specialisation on the sources at hand.
- **Breadth.** The share of the series of the generator's first baseline
  sample that the trial still has series near (0.95: nothing lost). It needs
  the two baseline trials `baseline_s1` and `baseline_s2`.

Every trial keeps, per table, what the generator drew and how many real
series of each group score above it (`likeness.parquet`), which shows the
kinds of tables that look real and the kinds that do not. The trials and
their outcomes so far are in
[docs/dataset_similarity/GENERATOR_CHANGELOG.md](../../docs/dataset_similarity/GENERATOR_CHANGELOG.md).

## Checking a changed generator

Generate a few thousand tables with the changed generator into a folder and
point the stage at it:

```json
"dataset_similarity": {
  "output": "results/model/dataset_similarity_trial",
  "synthetic": {"source": "data/01_generated/trial"}
}
```

The coverage of every benchmark dataset and the descriptor tables then show
whether the change moved the synthetic series towards the benchmarks, before
any training. Use the same number of tables when comparing two generators:
more synthetic series lie closer together, which tightens the limit.

## Settings

All settings have defaults (`DEFAULTS` in `similarity_data.py`); a
`dataset_similarity` section in the configuration overrides them.
scikit-learn and LightGBM are project dependencies.

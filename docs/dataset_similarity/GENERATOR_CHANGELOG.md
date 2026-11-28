# Generator change log

Changes to the data generator that make its series harder to tell from real
series, and how much each one helped. Started 2026-11-28 on the branch
`feature/generator-realism`.

## How a change is judged

1. The generator, with its settings as they are after the change, draws
   12,000 tables. The latest 512 points of every target are kept (tables
   shorter than 512 rows are left out, about one in nine).
2. A classifier (LightGBM) is trained to tell these series from each of four
   groups of real series, one group at a time. It is scored on series it was
   not trained on (five folds; windows of one recording stay in one fold).
3. Each group is scored twice: once with the 56 descriptors of a series, once
   with its Chronos-2 embedding. The score is the area under the curve (AUC):
   1.0 means every generated series is told from the real ones, 0.5 means
   they cannot be told apart.
4. The score of a group is the mean of its two AUCs, and the score of the
   generator is the mean over the four groups. **Lower is better.**

A change is kept when the score falls by more than 0.001. Two runs of the
unchanged generator with different random tables differ by 0.0002, so 0.001
is well outside chance. A change that does not reach that is discarded.

The four groups of real series:

| Group | Series | What it is |
|---|---|---|
| FakeBench-TS | 5,000 | Random walks with 19 kinds of steps |
| Operational benchmarks | 2,536 | Caster, CATS, cloud monitoring, UCR operational series (stage 11) |
| IND-Eval | 4,960 | Eight industrial sources (stage 12) |
| Public datasets | 2,186 | Windows of the measured columns of the 21 datasets in `data/00_public` that are long enough |

What the score cannot say: whether Basin forecasts better after training on
the changed data. It only says the generated series look more like the real
ones to a classifier. See "Limits" at the end.

## Baseline

The generator as it was used for training (two runs, different tables):

| Run | Score | FakeBench-TS | Operational | IND-Eval | Public |
|---|---|---|---|---|---|
| Baseline, tables 1 | 0.9645 | 0.9902 | 0.9463 | 0.9689 | 0.9526 |
| Baseline, tables 2 | 0.9647 | 0.9903 | 0.9446 | 0.9687 | 0.9552 |
| **Baseline** | **0.9646** | 0.9903 | 0.9455 | 0.9688 | 0.9539 |

The descriptors tell the groups apart more surely than the embeddings do
(for example IND-Eval: 0.987 against 0.951). What the classifier used most at
the start: for the three real groups, how coarse the readings are (distinct
values, flat stretches, changes that are whole multiples of a smallest
change), how much large changes cluster, and how much slow drift a series
has; for FakeBench-TS, that successive changes of a random walk are
unrelated, where the generator's series are either smoother or noisier.

## Changes

| # | Change | Score | FakeBench-TS | Operational | IND-Eval | Public | Result |
|---|---|---|---|---|---|---|---|
| 0 | Baseline | 0.9646 | 0.9903 | 0.9455 | 0.9688 | 0.9539 | |
| 1 | Stored resolution on 60% of the columns | 0.9597 | 0.9895 | 0.9426 | 0.9578 | 0.9489 | Kept (-0.0049) |

### 1. Stored resolution

Real recordings are stored with a fixed number of digits, so a window of 512
points often holds far fewer than 512 distinct values: half of the IND-Eval
series have fewer than one distinct value per four points. In the generator
three quarters of the series had a distinct value at every point. Now 60% of
the continuous columns (target and covariates) are stored at a resolution of
6 to 20,000 levels over the range of the signal, drawn evenly on a
logarithmic scale. Setting: `RESOLUTION_PROBABILITY`.

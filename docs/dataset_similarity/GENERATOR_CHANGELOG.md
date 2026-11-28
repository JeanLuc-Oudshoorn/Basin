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
   This is done for three different divisions into folds and averaged,
   because the division alone moves an AUC by a few thousandths.
3. Each group is scored twice: once with the 56 descriptors of a series, once
   with its Chronos-2 embedding. The score is the area under the curve (AUC):
   1.0 means every generated series is told from the real ones, 0.5 means
   they cannot be told apart.
4. The score of a group is the mean of its two AUCs, and the score of the
   generator is the mean over the four groups. **Lower is better.**

A change is kept when the score falls by more than 0.0015. Two runs of the
unchanged generator with different random tables differ by 0.0008, so a
smaller difference can be chance. A change that does not reach that is
discarded, or tested again on other tables when it is close.

**Breadth.** Every trial also reports how much of what the generator made at
the start it still makes: the share of the series of the first baseline run
that have series of the trial as near to them as a second baseline run has.
It is 0.95 when nothing was lost. It is not part of the score; it is there to
see that a change adds kinds of series and shifts shares, without removing
what the generator could already do.

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
| Baseline, tables 1 | 0.9652 | 0.9898 | 0.9471 | 0.9686 | 0.9552 |
| Baseline, tables 2 | 0.9660 | 0.9904 | 0.9493 | 0.9686 | 0.9556 |
| **Baseline** | **0.9656** | 0.9901 | 0.9482 | 0.9686 | 0.9554 |

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
| 0 | Baseline | 0.9656 | 0.9901 | 0.9482 | 0.9686 | 0.9554 | |
| 1 | Stored resolution on 60% of the columns | 0.9606 | 0.9896 | 0.9460 | 0.9578 | 0.9488 | Kept (-0.0050) |
| 2 | Trend in 30% of the tables instead of 50% | 0.9599 | 0.9907 | 0.9449 | 0.9579 | 0.9463 | Kept (-0.0007, see below) |
| 3 | Dominant cycles (first cycle 3 to 12 times larger in half of the seasonal tables) and a size of its own for every repetition | 0.9591* | 0.9904 | 0.9407 | 0.9587 | 0.9467 | Discarded (+0.0007) |
| 4 | 3, plus cycle lengths of a day at common sampling rates (24, 48, 96, 144, 288) and 50% seasonal tables | 0.9586* | 0.9920 | 0.9397 | 0.9596 | 0.9431 | Discarded (+0.0002) |
| 5 | 4, plus noise sized to the cycle in 60% of the dominant cycles | 0.9595* | 0.9919 | 0.9451 | 0.9592 | 0.9420 | Discarded (+0.0011) |

\* Changes 3 to 5 were scored with one division into folds, against 0.9584
for change 2 scored the same way. After change 5 the score was made more
precise (three divisions, and one descriptor corrected: a change now counts
as a whole multiple of the smallest change within 5% instead of 1%, because
values stored as 32-bit numbers are not exact multiples). Rows 0 to 2 show
the scores recomputed that way; later rows use it from the start.

### 1. Stored resolution

Real recordings are stored with a fixed number of digits, so a window of 512
points often holds far fewer than 512 distinct values: half of the IND-Eval
series have fewer than one distinct value per four points. In the generator
three quarters of the series had a distinct value at every point. Now 60% of
the continuous columns (target and covariates) are stored at a resolution of
6 to 20,000 levels over the range of the signal, drawn evenly on a
logarithmic scale. Setting: `RESOLUTION_PROBABILITY`.

### 2. Fewer trending tables

In a window of 512 points most real series have no trend: a straight line explains 2% to 6% of the variance of the median real series, against 13% for the generated ones. Tables drawn without a trend already match the real series on this (1%), tables drawn with one do not (49%). The share of tables proposed without a trend goes from 50% to 70%. Setting: `TREND_FREE_FRACTION`. FakeBench-TS gets slightly worse (random walks do wander off), the three real groups better. With the more precise score the gain is 0.0007, which is no more than two runs of the same generator differ. It was kept because the three groups of real series all improved and the descriptor it targets moved to where the real series are; trends remain in 30% of the tables.

### 3. Dominant cycles

Only 14% of the generated series have a strong cycle, against 28% to 33% of the real ones. Making the cycle dominate in half of the seasonal tables raised that to 17%, but the score did not move: the classifier told electricity, solar and the UCR series from the generated ones as surely as before.

### 4. Calendar cycle lengths

Real cycles are 24, 48, 96 or about 144 points long; the generated ones were spread evenly from 6 to 170. Tested on top of change 3. No gain: the three real groups improved a little, FakeBench-TS lost as much.

### 5. Jagged cycles

Against electricity, the generated cycles were too smooth (the up and down pattern of real load is almost random from point to point). Adding noise in proportion to the cycle made the UCR series, which are smooth, easier to tell apart, and electricity no harder. Changes 3 to 5 were taken out again; the lesson is that one kind of series has to become several percent of all tables before its dataset gets harder to tell apart, so single kinds move the score little.

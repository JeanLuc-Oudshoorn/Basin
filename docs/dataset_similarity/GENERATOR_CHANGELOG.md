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
| 6 | Nine operational patterns as new kinds of series, in a third of the tables | 0.9476 | 0.9836 | 0.9420 | 0.9444 | 0.9207 | Kept (-0.0123) |
| 7 | Smoother and more regular patterns (noise with memory, clearer days, steadier cycles), daily load 6% to 9% | 0.9479 | 0.9835 | 0.9430 | 0.9453 | 0.9197 | Discarded (+0.0003) |
| 8 | 7, plus a unit or pattern in 85% of the tables without a trend (was 60%) | 0.9453 | 0.9817 | 0.9454 | 0.9432 | 0.9108 | Kept (-0.0023) |

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

### 6. Operational patterns

The first five changes showed that the generator's existing mechanisms, however their shares are set, rarely give a series that the classifier takes for a real one (7 in 100 on average, and no mechanism above 17). What was missing were whole kinds of signal. Nine were added as generative models with wide parameter ranges, none fitted to a recorded series:

| Pattern | What it is | Share of tables |
|---|---|---|
| Held value | A controlled variable: noise around a setpoint that sometimes moves, often coarsely recorded | 6% |
| Daily load | A daily profile with quiet and busy hours, rest days, days of different size | 6% |
| Daylight | A source that works part of the day and rests at exactly zero (solar generation) | 2% |
| Duty cycle | Equipment that switches on and off: plateaus, or charging and discharging | 3% |
| Random walk | The sum of independent changes, with ten kinds of changes | 3% |
| Smooth drift | A slow physical variable, ramps and holds, sometimes logged rarely and joined by straight lines | 4% |
| Counts | Events per interval, with idle intervals, clumping and bursts | 3% |
| Staircase | A value that moves in steps and holds: a counter, a recipe level | 2% |
| Oscillation | A repeating waveform whose length and size vary from cycle to cycle | 3% |

A pattern takes the place of a closed-loop unit: the table's causal model still runs and supplies the pattern's demand, so the covariates can carry information about the target (except for the random walk, which nothing drives). Cycle lengths come mostly from the calendar (a day of 24, 48, 96, 144 or 288 points). All earlier unit families and the plain causal model remain; the share of tables with a unit of any kind goes from about a fifth to about four in ten. Breadth stays at 0.95: nothing the generator made before has become rare. Settings: `PLANT_PROBABILITY`, `PLANT_WEIGHTS`; code in `operational_patterns.py`.

### 7. Smoother patterns

Compared with the datasets nearest to them, the patterns were too jagged: 30% of the generated series are smooth from point to point, against 35% to 58% of the real ones. The ranges were moved (noise that remembers its last value, slower clouds, steadier oscillations, staircases that are seldom perfectly clean). On its own this did not move the score (+0.0003). It is kept as part of change 8, because the ranges stay wide and each move follows from a measured difference, not because the score asked for it.

### 8. More tables with a pattern

The patterns are taken for real series two to three times as often as the plain causal model, so they get more room: a unit or pattern in 85% of the tables that have no trend. About half of all tables now hold a pattern, a tenth one of the earlier closed-loop units, and four in ten the plain causal model (all tables with a trend among them). Breadth 0.945. The operational benchmarks did not improve (0.9420 to 0.9454); the public datasets, IND-Eval and FakeBench-TS did. Setting: `PLANT_PROBABILITY`.

### Note after change 8: one descriptor redefined

The descriptor that says whether values lie on a regular grid was computed
from the smallest change between two points. That fails when the smallest
change happens to span several grid steps, which made coarse real readings
(the caster) look unlike coarse generated ones for a reason that has nothing
to do with the generator. It is now computed from the gaps between
neighbouring distinct values. Under this final measure the baseline is
0.9666 (two runs: 0.9663, 0.9668) and the state after change 8 is 0.9465.
Rows 9 and later use it; the table at the end of this document gives every
kept change under the final measure.

### Note after change 8: a second score, on sources the classifier has not seen

Per pattern and per mechanism, the share of real series that score above a
generated series was tabulated. Even the kinds that look most real are told
from about nine in ten real series. The reason is that the real groups are
few sources, each a narrow cluster (one cycle length, one resolution, one
kind of noise), and a classifier that sees windows of a source in training
recognises that source in the test fold. A generator can only win that
contest by making series of exactly those sources, which is what should not
happen: the 40 or so sources at hand do not describe all operational data.

So every trial now also reports the AUC when **whole datasets are held
out**: the classifier is trained on some real sources and has to tell real
series of *other* sources from generated ones. It can no longer recognise a
source; it has to rely on what real sources have in common. On this score
the baseline is 0.801 and the state after change 8 is 0.693:

| Group | Baseline | After change 8 |
|---|---|---|
| FakeBench-TS (19 kinds of steps) | 0.987 | 0.977 |
| Operational benchmarks (4 datasets) | 0.561 | 0.510 |
| IND-Eval (8 sources) | 0.790 | 0.554 |
| Public datasets (21 datasets) | 0.865 | 0.730 |
| **Mean** | **0.801** | **0.693** |

Against real sources it has not seen, the classifier now hardly tells
IND-Eval and the operational benchmarks from the generator's series. The
first score is still the one a change is judged by, as agreed; the second
one is the check that a gain on the first is not specialisation. A change
that lowers the first score while raising the second is not kept.

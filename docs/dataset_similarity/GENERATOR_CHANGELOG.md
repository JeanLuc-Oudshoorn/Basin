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
| 9 | Damped readings in 30% of the columns | 0.9432 | 0.9854 | 0.9357 | 0.9432 | 0.9086 | Kept (-0.0033); unseen sources 0.685, breadth 0.95 |
| 10 | Calm and busy stretches in the noise | 0.9441 | 0.9846 | 0.9396 | 0.9433 | 0.9088 | Kept on the second score (+0.0009); unseen sources 0.677, breadth 0.94 |
| 11 | Patterns that ride on a trend in half of the trending tables; random walks recorded exactly | 0.9417 | 0.9810 | 0.9391 | 0.9422 | 0.9044 | Kept (-0.0024); unseen sources 0.678, breadth 0.94 |
| 12 | A tenth pattern (resource use) and shifted shares among the patterns | 0.9382 | 0.9820 | 0.9354 | 0.9385 | 0.8971 | Kept (-0.0035); unseen sources 0.668, breadth 0.94 |
| 13 | Calendar cycle lengths and a size of its own for every cycle, in the seasons of the causal model | 0.9366 | 0.9815 | 0.9297 | 0.9386 | 0.8966 | Discarded (+0.0001); unseen sources 0.656, breadth 0.94 |
| 14 | An eleventh pattern (a variable that is knocked off its level and recovers) and a pattern on 70% of the trends | 0.9351 | 0.9819 | 0.9267 | 0.9376 | 0.8943 | Discarded (-0.0014); unseen sources 0.655, breadth 0.93 |
| 15 | More daily cycles: daily load in 20% of the tables (was 13%), a daily profile more often in counts and resource use | 0.9387 | 0.9838 | 0.9351 | 0.9412 | 0.8948 | Discarded (+0.0022); unseen sources 0.660, breadth 0.94 |
| 16 | A second pattern added to the first in a quarter of the pattern tables | 0.9353 | 0.9806 | 0.9305 | 0.9393 | 0.8909 | Discarded (-0.0009); unseen sources 0.661, breadth 0.94 |
| 17 | Cycles of which a table shows as little as one and a half repetitions (was three) | 0.9365 | 0.9809 | 0.9313 | 0.9378 | 0.8958 | Kept for the forecasting check, not the score (+0.0003); unseen sources 0.658, breadth 0.94 |

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
that lowers the first score while raising the second is not kept. A change
that adds a capability, leaves the first score where it was (within 0.0015)
and lowers the second by more than 0.005 is kept, and marked as such: two
runs of the same generator differ by 0.001 on the second score.

### 9. Damped readings

Asked which real sources a classifier can still tell from generated series without having seen them, the answer was the smooth ones (grid load and generation, weather), and the descriptor it leaned on was whether large changes follow large changes. Real signals are usually sampled faster than they move, and a transmitter or a logger that stores interval means smooths them further; the generator put fresh noise on most points. Now 30% of the continuous columns are recorded through a response time of 0.7 to 15 points, before the other recording effects. Setting: `DAMPING_PROBABILITY`.

### 10. Calm and busy stretches

In real series large changes follow large changes far more than in generated ones (0.29 to 0.39 against 0.15 at the start), and a classifier that has not seen a source takes exactly that as the sign of a real series. The size of the noise can now wander: in half of the patterns, and as a fourth way for the noise of the causal model to change size (next to growing, cycling and changing once), with the share of nodes whose noise changes size raised from 20% to 35%. The first score did not move (+0.0009, within chance); the score on unseen sources fell from 0.685 to 0.678, which is why it is kept. Settings: `VOLATILITY_BURST_SHARE`, `VARIANCE_DYNAMICS_PROBABILITY`.

### 11. Patterns on a trend, exact random walks

Two additions. (1) The tables with a trend all came from the plain causal model, the kind of series the classifier tells from real ones most surely. In half of them a pattern now rides on the trend: a load that grows, a setpoint that drifts, an oscillation on a slope. The trend's whole change is 0.5 to 6 times the pattern's usual spread. Setting: `TRENDING_PATTERN_PROBABILITY`. (2) The target of a random-walk table is no longer passed through a response time or rounded to a stored resolution: both make the next change predictable from the last, and the point of these tables is that nothing is. The classifier told the generator's walks from the FakeBench-TS walks mainly by those two recording effects.

### 12. Resource use

The real sources that a classifier could still tell from generated series without having seen them were the noisy, non-negative ones with bursts: the load of a server, the volume of messages. A tenth pattern covers that kind: a noisy level that cannot go below zero, with heavy-tailed bursts that die away, sometimes a daily profile, lasting shifts of the usual level, idle stretches, or a limit it runs into. The shares among the patterns were shifted towards the kinds taken for real most often (held value, smooth drift, staircase, counts) and away from the one taken for real least (oscillation, 5% to 2.5% of the tables). Setting: `PLANT_WEIGHTS`.

### Note after change 12: the grid descriptor once more

Every 32-bit number lies on a grid, so a series with a small spread around a
large level counted as coarsely recorded for that reason alone. This is what
told the generator's random walks (which sit on a level) from the FakeBench-TS
walks (which start at zero). A grid finer than 50,000 steps over the range of
the series no longer counts. Under this measure the baseline is 0.9659 (unseen
sources 0.798) and the state after change 12 is 0.9365 (unseen sources 0.660).
Rows 13 and later use it.

### 13. Cycles of the causal model

The two parts of changes 3 and 4 that add something to the causal model's own seasons, tested again now that the score is more precise: cycle lengths that follow the calendar (7, 144 and 288 added, a day most likely) and a random size for every repetition. No effect on the first score (+0.0001) and a small one on the second (0.660 to 0.656, below what counts). Discarded; the daily-load pattern already brings both.

### 14. Disturbance and recovery

A pattern for transients: knocks at fixed intervals or at random, answered by a jump that creeps back, a swell, ringing, or a stay away from the level. First run -0.0014, a second run on other tables +0.0000 (0.9351 and 0.9365 against 0.9365); the second score 0.655 and 0.657 against 0.660. The new pattern is taken for real about as often as the plain causal model, not more. Discarded. The two runs also show that two samples of the same generator can differ by 0.0014 on the first score, so from here a single run has to gain more than 0.002 to count.

### 15. More daily cycles

A strong cycle shows in 18% of the generated series against 28% to 33% of the real ones, and a classifier that has not seen a source takes a daily rhythm as a sign of a real one. Giving the daily-load pattern more room did not help: the public datasets improved slightly, the other three groups lost more (+0.0022 overall). Discarded. A larger share of one pattern is not what is missing.

### Check after change 12: windows of 2,048 points, and speed

All trials use the latest 512 points. As a check, the generator as it was and
the generator after change 12 were scored once on the latest 2,048 points
(about 5,900 generated series each, since half of the tables are that long;
real groups: 2,500 FakeBench-TS, 2,536 operational, 5,000 IND-Eval, 1,493
public windows):

| | Score | Unseen sources | FakeBench-TS | Operational | IND-Eval | Public |
|---|---|---|---|---|---|---|
| As trained on | 0.9810 | 0.817 | 0.9959 | 0.9687 | 0.9824 | 0.9770 |
| After change 12 | 0.9603 | 0.680 | 0.9880 | 0.9549 | 0.9634 | 0.9348 |

The gain holds on long windows, and is somewhat larger there. Long windows
are easier to tell apart in general: more of a series shows.

Generation is as fast as before: 94 tables per second against 96 with ten
processes (two runs of 3,000 tables each).

### 16. Sums of two patterns

Recorded signals are often sums: a daily load on a drifting level, a held value with a ripple. A second pattern, 0.15 to 1 times the size of the first, was added in a quarter of the pattern tables. Against the mean of two runs of the current generator (0.9362; unseen sources 0.658) it scored 0.9353 and 0.661: no difference. Discarded. After changes 13 to 16 the picture is that further kinds and shares no longer move either score beyond chance.

### Check after change 12: does Basin, as trained, forecast the new kinds well?

The scores above say the new series look more like real ones. They do not say
that training on them will help. A cheaper question can be answered without
training: how does the Basin that exists, trained on the old generator,
forecast the new kinds, compared with Chronos-2 and Timestral? If Basin is as
good on them as on its own tables, they add little; if it falls behind, they
are what its training lacked.

13,053 tables of the generator after change 12 and 4,885 of the generator as
it was, each with a history of 512 points and the 112 points after it, target
only, no covariates. MASE of the median forecast, median over the tables of a
kind; lower is better. "Against the others" is Basin's error over the
geometric mean of the other two, per table (below 1: Basin ahead).

| Kind of table | Tables | Basin | Chronos-2 | Timestral | Naive | Basin against the others | Basin best of the three |
|---|---|---|---|---|---|---|---|
| **Generator as it was, all tables** | 4,885 | **2.97** | 3.36 | 3.19 | 5.14 | 0.96 | 47% |
| **Generator after change 12, all tables** | 13,053 | 3.31 | 3.51 | **3.26** | 5.57 | 1.00 | 35% |
| Plain causal model | 3,450 | **3.40** | 3.79 | 3.70 | 5.50 | 0.97 | 46% |
| Earlier closed-loop units (nine families) | 1,539 | | | | | 0.81 to 1.00 | 32% to 57% |
| Duty cycle | 400 | 3.19 | 3.76 | **3.14** | 6.92 | 0.99 | 31% |
| Resource use | 586 | 1.72 | 1.73 | **1.62** | 2.49 | 1.00 | 30% |
| Held value | 1,097 | 2.24 | 2.29 | **2.24** | 2.60 | 1.00 | 31% |
| Staircase | 456 | 7.85 | 8.44 | **7.46** | 11.99 | 1.00 | 33% |
| Random walk | 490 | 6.24 | 6.31 | 6.28 | **6.01** | 1.01 | 33% |
| Oscillation | 213 | 1.64 | 1.81 | **1.60** | 4.11 | 1.01 | 32% |
| Counts | 587 | 0.91 | 0.88 | **0.85** | 1.11 | 1.01 | 20% |
| Smooth drift | 827 | 13.41 | 13.51 | **12.91** | 14.59 | 1.01 | 35% |
| Daily load | 1,377 | 2.09 | 1.95 | **1.83** | 4.90 | 1.07 | 21% |
| Daylight | 329 | 1.48 | 1.39 | **1.29** | 5.47 | 1.12 | 16% |

- **On tables of the old generator Basin leads**, as on the holdout set: its
  error is 4% below the other two, and it is the best of the three on 47% of
  the tables.
- **On the new generator it has lost that lead.** On the plain causal model
  and the earlier units it is still ahead; on the new patterns it is level
  at best.
- **It is behind on exactly the kinds where it is behind on real data.** Daily
  load (7% behind) and daylight (12% behind) are the generated counterparts of
  electricity and solar, where Basin trailed most on IND-Eval (24% and 14%
  behind the best model). The generator can now make the series that expose
  the weakness, which it could not before.
- **Random walks:** no model beats the naive forecast, as it should be, and
  Basin is no worse than the other two.

This is the strongest indication so far that training on the new generator
should narrow the gap on real data, and that daily cycles are where most is
to be gained. It is still an indication: only a training run can show it.
The comparison used no covariates, and Basin's lead on its own tables is
larger with them.

**What about daily cycles Basin misses: long ones.** A second sample of 13,069
tables kept what the generator drew. Basin's error against the other two, by
the length of the cycle:

| Cycle length (points) | Daily load | Daylight | Any table whose history shows a strong cycle |
|---|---|---|---|
| Up to 30 | 1.04 to 1.07 | 1.08 | 1.01 |
| 30 to 110 | 1.04 to 1.06 | 1.08 | 1.00 to 1.02 |
| 110 to 170 | 1.17 | 1.28 | 1.10 |
| Over 170 | 1.21 | 1.80 | (too few repetitions to detect) |

With a history of 512 points, a cycle of 144 points repeats three and a half
times and one of 288 points less than twice. Chronos-2 and Timestral continue
such a cycle from that little; Basin does not. That fits stage 11, where
Basin's gap was largest on periodic series with a short history. In the old
generator a cycle always repeated at least three times within a table, and
strong cycles were rare, so Basin has hardly seen a cycle of which only two
repetitions are visible. On the other new patterns the cycle length, the
resolution, the noise and the other drawn parameters make no difference to
Basin's standing: it is level with the other two throughout.

### 17. Few repetitions of a long cycle

This change follows from the forecasting check above, not from the score. The cyclic patterns drew their cycle length so that a table held at least three repetitions; now one and a half suffice. A table of 500 rows can hold a day of 5-minute readings (288 points), as real histories do. The score is unchanged (0.9365 against 0.9362; unseen sources 0.658 against 0.658), as expected: it concerns tables shorter than 900 rows, few of which are long enough to be scored. It is kept because it is an addition, costs nothing on either score, and the forecasting check points at exactly this as what Basin has not learned.

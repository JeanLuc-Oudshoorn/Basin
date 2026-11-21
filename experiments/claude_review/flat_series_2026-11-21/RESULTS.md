# Does REJECT_CONSTANT_OUTCOME explain the weakness on flat series?

2026-11-21. CPU only (training was running). Holdout run:
`results/model/holdout_test/20261120T172837Z_epoch1_step1457590`.

**Answer: no.** The filter removes about 0.8% of proposed tables, and only those
that are constant from start to end. Flat stretches inside tables remain and are
common. The gap to Timestral is on flat stretches that follow a varied history.

## Measured

`constant_rejection_rate.py` (6,000 proposals with the production settings,
filter off): 48 constant, 0.8%.

`flat_history_analysis.py` (all 5,689 holdout forecasts with a flat point):

- 5.2% of holdout points equal the last observed value; 15.5% of forecasts
  contain such points; 2.9% of forecasts are flat over the whole horizon.
- 1.5% of forecasts see a history that is constant throughout (the table moves
  elsewhere). These are not scored by the scaled error (MASE scale is zero).
  Basin's error there is about 0.0001 of the level; Timestral and Chronos-2
  return the last value exactly because they floor a zero scale at 1e-6 / 1e-5,
  Basin sets it to 1 (`index_patches.py`, `tokenize_next`).

Median absolute error on flat points, in history standard deviations, by the
length of the flat run that ends the history:

| Flat run | Forecasts | Still flat | Basin | Timestral | Chronos-2 |
|---|---|---|---|---|---|
| 1 | 2,384 | 6% | 0.242 | 0.219 | 0.240 |
| 2-4 | 820 | 16% | 0.126 | 0.067 | 0.122 |
| 5-16 | 657 | 35% | 0.058 | 0.016 | 0.061 |
| 17-64 | 506 | 55% | 0.031 | 0.0066 | 0.028 |
| 65-256 | 385 | 83% | 0.0065 | 0.0018 | 0.0116 |
| over 256 | 377 | 95% | 0.0032 | 0.0007 | 0.0055 |

`offset_and_p10.py`:

- Pooled p10 of the scaled error: Basin 0.2455, Timestral 0.1817 (as in the
  holdout metrics). With the last value after flat runs of 17 or more (3.6% of
  points): 0.205. With every flat point exact: 0.172.
- Level offset per flat forecast, evaluated weights against weights 88,574
  updates later: Spearman 0.60, same sign in 68%. Against Timestral's offset: 0.30.

## Code comparison

| | Basin | Timestral-209M | Chronos-2 |
|---|---|---|---|
| Scaling | running mean/std up to each patch, asinh | the same | mean/std of the whole context, asinh |
| Forecast frame | last patch's mean/std | the same | context mean/std |
| Zero spread | scale 1 | scale 1e-6 | scale 1e-5 |
| Patch | 16 | 32 | 16 |
| Time mixing | RWKV-7, attention every 4th layer | attention in all 24 layers | attention in every layer |
| Precision | BF16 | float32 | float32 |

Timestral scales exactly as Basin does and does not forecast relative to the
last value, so neither explains its advantage. Chronos-2 has attention in every
layer and float32 and is as weak as Basin, so neither explains it either. What
is left is training (data, loss, duration), which the inference packages do not
show.

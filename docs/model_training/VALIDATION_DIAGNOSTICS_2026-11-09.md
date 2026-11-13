# Validation diagnostics: November 9, 2026

This is a read-only investigation of validation checks 0–3 (steps 0, 77,131,
158,021 and 238,106). Training, checkpoints, configuration, objectives and data
were not changed. No GPU model was loaded or additional validation run started.

## Findings

The raw-scale RMSE increase is real, but does not describe the typical forecast.
One extremely large-amplitude table dominates the squared error. Separately,
tiny-history-to-large-future transitions dominate normalized loss and MASE.
Typical forecasts improved substantially from initialization, with much smaller
improvements after the first four hours. These facts can coexist.

### Typical validation forecasts

Calculated directly from the four `validation_check_*.parquet` files under
`results/model/training`. Each row is one of the 125,000 forecast requests,
not an entire table. These are medians across requests, not table medians.

| Active training hours | Median normalized pinball | Median RMSE | Median MAE | Median MASE | MASE p90 |
| --- | --- | --- | --- | --- | --- |
| 0 | 0.438174 | 10.6811 | 9.23255 | 3.39636 | 19.9955 |
| 4 | 0.201738 | 5.16316 | 4.00457 | 1.56102 | 3.96583 |
| 8 | 0.199517 | 5.06706 | 3.89122 | 1.53160 | 3.87601 |
| 12 | 0.198995 | 5.06064 | 3.88330 | 1.53012 | 3.85354 |

All four horizons retain configured weights 0.80, 0.16, 0.03 and 0.01.
Raw RMSE and MAE are computed in original units; normalized pinball divides by
context standard deviation; MASE divides absolute errors by the mean absolute
first difference of the context. They therefore emphasize different cases.

### The raw-error outlier

Source: `data/01_generated/2026-11-07/table_b9080ed9c679_589606/data.parquet`.
Its five origins contribute **99.614% of squared validation error** and
**89.729% of absolute validation error** at check 3. At origin 32 its RMSE
increased from 2,552,429 at check 2 to 27,050,674 at check 3; its MASE rose
from 6.153 to 64.681. This is an actual bad forecast, not merely display noise.

The first 32 observed outcomes range from about 472 to 7,045,399, with standard
deviation 1,334,854. The following 64 outcomes reach 8,518,128. Its raw units
are orders of magnitude larger than typical tables.

Excluding this table for diagnosis only gives aggregate MAE of 75.322, 28.630,
31.237 and 29.607 across checks 0–3. Excluding its five requests gives RMSE
4,857.631, 4,769.164, 4,985.473 and 4,879.399. These exclusions have **not**
been applied to reported validation, model selection or training.

### Why normalized means look almost stationary

The five highest-loss requests contribute **53.069% of total normalized pinball
loss** at check 3. The largest is
`table_b9080ed9c679_535757`, origin 256: loss 70,099.508, MASE 653,373.
Its historical outcome standard deviation is approximately 0.0000377 and its
history ranges from 0.00000618 to 0.000193. The following 64 observations
reach 35.367. A modest error in original units becomes huge when divided by
the tiny historical scale. This transition exists in the input data itself.

Another example, `table_b9080ed9c679_182348`, origin 64, has historical
standard deviation 0.0001202 followed by future outcomes reaching 26.258.
Its normalized loss is 13,840.521 and MASE 458,701 at check 3.

Keeping all positive standard deviations, as requested, is scale-consistent;
restoring a fixed absolute threshold would not resolve the underlying regime
transitions in a principled way. It has not been restored. An unexpected
future scale change can be extremely difficult to predict from past data;
these observations alone do not prove it is impossible or a generator bug.

As a sensitivity check only, dropping the five highest-loss requests gives
mean normalized pinball 1.10411, 0.813805, 0.811871, 0.810755. Dropping the
five highest-MASE requests gives mean MASE 11.8067, 5.00274, 4.95465, 4.94299.
There remains a long tail beyond these five cases.

### Training loss is an online cumulative statistic

`timed_training.py` stores `epoch_loss_sum / epoch_mass`. Each optimizer update
adds its pre-update loss weighted by effective supervised label mass. This is
not a fresh evaluation of a fixed training subset with the latest weights.
Different checkpoints include different tables and earlier model states.
It is also not directly comparable with validation: training supervises every
eligible causal anchor, whereas validation uses five fixed origins per table.

The timestamped resumed journal
`logs/rwkv7_train_20261108_190532_203118Z_pid23856_0fdb0d94.jsonl` was streamed
through step 238,106. Unique update events gave:

| Step interval | Available updates | Median update loss | p90 update loss | Maximum update loss | Largest 10 updates' share of unweighted loss sum |
| --- | --- | --- | --- | --- | --- |
| 1–77,131 | 59,861 | 0.259312 | 0.716896 | 3,239,934 | 50.42% |
| 77,132–158,021 | 80,890 | 0.257191 | 0.706103 | 3,572,055 | 56.32% |
| 158,022–238,106 | 80,085 | 0.256272 | 0.707315 | 11,228,380 | 60.61% |

The first interval lacks the first 17,270 updates, which preceded this resumed
log. These update medians and unweighted tail shares are diagnostic statistics;
they are not substitutes for the point-mass-weighted history training loss.
The largest loss, at step 226,475, has logged pre-clipping gradient norm 0.1611.
A very large pinball loss therefore does not itself establish exploding
gradients: the loss value grows with residual magnitude while its derivative
with respect to a normalized prediction is bounded by the quantile slope.

## Interpretation and next steps

There is no evidence here of a broad collapse across validation requests.
There is evidence of one worsening large-scale forecast and a heavy tail that
dominates mean statistics. Progress after the first check is genuinely slow
even in the medians; this should not be dismissed as just metric noise.

For future changes, add median/p90 and tail-contribution diagnostics alongside
the existing honest full-set metrics. Investigate the generator mechanisms
behind extreme scale transitions and the large-amplitude outlier separately.
Compare checkpoints on those exact origins before changing the objective or
normalization. Do not silently delete difficult validation cases or change the
training objective mid-run based on this investigation.

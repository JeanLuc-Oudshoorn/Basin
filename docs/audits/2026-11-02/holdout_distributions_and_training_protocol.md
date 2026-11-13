# Holdout distributions and training protocol clarification

## Completed training versus exported weights

Read from each run's `training/early_stopping.json`:

| Model | Completed epochs | Completed optimizer updates | Exported best epoch | Exported best update |
| --- | ---: | ---: | ---: | ---: |
| Archived 447M | 8 | 97,288 | 7 | 85,127 |
| Current 307M | 5 | 73,050 | 4 | 58,440 |

The exported `model.pt` contains the best validation weights, not the last
epoch's weights. Epoch zero is a validation baseline, not a trained epoch.
The datasets differ, so epoch counts alone are not equal exposure. The
archived exported weights received about 46% more optimizer updates. These
runs alone cannot isolate capacity, exposure, admission policy or objective.

## Exact prediction horizons

The current configuration sets `model.output_patches=16` and
`model.multi_horizon_weight=0.5`. During training, `train.forward_next` combines
ordinary next-patch pinball loss with an auxiliary pinball loss that samples
one additional future lead per input token. The additional sampled lead is
1..15 relative to the ordinary next patch, so it supervises patches 2..16
(up to 256 timesteps ahead). It does not evaluate all 16 patches at every
position. Available labels are masked; training labels exclude validation.
When no extra labels exist, the next-patch objective has full weight.

The auxiliary branch requires both `model.training` and
`prepared.is_training`. Full validation calls `model.eval()`, marks inputs as
non-training and scores only the final held-out 16-point patch. Holdout also
uses eval mode and calls `forecast_contexts`, whose head emits one patch,
at each of the up to five independent sampled origins. Later leads are not
scored. The archived model has one output patch and no auxiliary objective.

Thus validation and holdout agree on the one-patch horizon. There is a
training-versus-evaluation emphasis difference, not a hidden long-horizon
holdout metric. Validation uses the final patch from training tables, whereas
holdout uses separate tables and multiple context lengths. Validation's
normalized quantile pinball loss and holdout's primary MASE also differ.

The earlier controlled continuation experiment found next-patch-only training
better than the mixed objective on its diagnostic sample. This supports an
objective tradeoff, but does not establish the cause of the full gap or replace
a from-scratch controlled experiment. See
[the regression report](forecast_regression/REPORT.md).
No objective, scheduler, checkpoint or trained weights were changed here.

## Resuming a completed run

`training.resume="auto"` loads `latest_checkpoint.pt`, including optimizer,
scheduler and early-stopping state. Since the 307M run already has one bad
epoch and patience one, an unchanged rerun performs no further optimization.
It can repeat final export using the recorded best weights.

To continue intentionally, increase
`training.early_stopping.patience_epochs`, for example from 1 to 3, leaving
`training.resume="auto"` and all identity-bound settings unchanged. Patience
is deliberately outside checkpoint identity. The restored bad-epoch count
remains one; this does not reset training or its history. With improvement it
can continue longer, but improvement is not guaranteed. Resumption starts
from the last epoch checkpoint (epoch 5), not the exported best epoch 4.

The restored ReduceLROnPlateau scheduler uses patience one: PyTorch reduces
learning rate after two consecutive unimproved validations. Early stopping
with patience one prevents reaching that reduction. Patience three allows
lower-learning-rate epochs to be attempted. No patience change was made as
part of the statistics request.

## Additional summaries and density plots

The existing `pooled` MASE is a mean over scored points with valid historical
scales. It is not an equal-table mean. Pooled RMSE is the square root of the
pooled squared errors. Existing pooled values and the leaderboard are retained.

`table_summary.models.<model>.<metric>` adds mean, median, p10 and p90 of the
saved per-table scores, weighting each table equally. Metrics include MASE,
RMSE, MAE, normalized RMSE and RWKV normalized pinball. Finite counts and
undefined counts are explicit. No epsilon floor, winsorization or trimming is
introduced. Percentiles use NumPy's linear interpolation. A table's own score
still pools its valid scored observations across origins and targets.

New completed holdout runs automatically write `error_distributions.html`,
with overlaid transparent model densities and one panel per error metric.
Each curve uses equal-table weights and integrates to approximately one on
the finite plot grid. Gaussian KDE uses Silverman's robust bandwidth and
reflection at the nonnegative boundary. Constant samples are markers, not KDEs.
Only RWKV has pinball scores; other metric panels include all evaluated models.

Default visualization estimates density in `log(1 + error)` space. This keeps
tails visible; it does not transform the stored metrics. Hover shows original
error. Y is density per transformed unit. Set
`plot.error_distribution.axis="linear"` for raw-space densities. `enabled`
and `grid_points` (512 by default) are also configurable. Extreme points remain
included; finite grid resolution can undersample a particularly narrow peak.

Generated summaries and plots for both existing 260-table completed runs:

| Method | Equal-table MASE mean | Median | p10 | p90 |
| --- | ---: | ---: | ---: | ---: |
| RWKV 307M | 2.3247 | 1.7240 | 0.9338 | 3.8785 |
| RWKV 447M | 1.7464 | 1.3169 | 0.7510 | 3.0100 |
| Naive | 2.6717 | 2.1750 | 1.3370 | 4.1883 |
| Chronos-2 | 1.7548 | 1.3189 | 0.6871 | 2.8320 |
| Timestral | 1.6551 | 1.2449 | 0.6436 | 2.7495 |

The 94% concentration diagnosis belonged to the larger 20% evaluation, not
these 2% samples. Medians and percentiles describe typical tables but do not
remove failures or invalidate the original pooled metric.

Outputs: `results/model/holdout_test/error_distributions.html` and
`results/447m_model/holdout_test/current_holdout/error_distributions.html`.
Both `test_metrics.json` files now contain the new statistics. No forecasts
were rerun. To regenerate without loading any models:

```powershell
uv run python src/06_holdout_test/error_distribution.py
uv run python src/06_holdout_test/error_distribution.py --config configs/holdout_447m_current.json
```

## Verification

Four focused distribution tests cover exact known percentiles, equal-table
weighting, undefined metrics, nonnegative unit-mass reflected density, constant
samples, extreme-tail retention and unchanged pooled/individual metrics.
The 29 existing `test_holdout*.py` tests also passed, including the one-patch
request contract and saved-split checks. Existing forecasts were reused to
generate both sets of plots and summaries on CPU.

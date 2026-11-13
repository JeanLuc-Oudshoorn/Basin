# Fixed four-token weighted holdout evaluation

Effective 2026-11-03. This supersedes sampled horizon lengths in holdout only;
training, validation, checkpoint weights and saved dataset splits are unchanged.

## Request and scoring policy

- Preserve the existing five percentile origins and historical context cap.
- Every model receives a 64-step request (four 16-point patches), every time.
- Read weights from `model.horizon_loss_weights`: `[0.8, 0.16, 0.03, 0.01]`.
  Missing weights in archived configurations use this default. Invalid weights
  are rejected. Legacy holdout horizon draws no longer affect evaluation.
- Each token's finite actuals share its coefficient equally. Renormalize over
  tokens with any observations; every origin-target contributes unit mass.
- MASE, MAE, normalized pinball, MSE and normalized MSE use those point weights.
  RMSEs take the square root of the aggregated squared error. MASE scales are
  still estimated exclusively from historical context. Zero scales remain
  undefined, with explicit coverage counts and no epsilon replacement.
- Always retain and plot 64 forecast points. Missing future actuals are null
  in Parquet and have zero metric weight. Never invent labels or expose future
  actuals/covariates. An archived one-patch model rolls out for the full horizon.
- Graph admission counts the actual fixed-four-token request shapes, including
  intermediate model passes for archived one-patch autoregressive rollouts.

## Artifacts

`test_metrics.json` excludes per-table and per-origin lists. `detail_files`
links `table_metrics.json`, `origin_metrics.json`, `test_predictions.parquet`
and `request_latencies.parquet`. Existing per-table prediction Parquets remain.
The protocol is labeled `four_token_weighted_v1`.

`table_summary.models.<model>.<metric>` gives equal-table mean, median, p10,
p90 and valid/undefined counts. Leaderboard rows expose the four table-MASE
statistics directly. Pooled metrics additionally expose weighted point-level
`median_absolute_scaled_error`, `absolute_scaled_error_p10` and
`absolute_scaled_error_p90`. A median of table means is a different statistic
from a median across individual scaled absolute errors. Table percentiles use
linear interpolation; point-level quantiles use an inverse weighted empirical CDF.

`error_distributions.html` retains equal-table transparent KDE overlays.
`latency_distributions.html` adds independent normalized overlays for every
model, using all scored request repeats; exclude warmups, retain capture costs
and outliers. The default axis is log(1 + milliseconds); hover shows milliseconds.
Both HTML artifacts embed Plotly for offline use. Latency plot configuration
is `plot.latency_distribution.{enabled,axis,grid_points}`.

Plot regeneration accepts both legacy embedded table lists and compact metrics.
The distribution regeneration CLI preserves old detailed lists in sidecar JSON
before removing them from the main metrics. Regeneration does not synthesize
unrequested horizons or change old pooled scores: rerun holdout for this protocol.

## Verification

CPU suite: 261 tests completed successfully, 12 CUDA-only tests skipped. Checks cover:

- Equal 64-point requests to all models, including final origins and partial tails.
- Full forecast storage past the last actual, zero influence from missing labels.
- Known constant token errors `[1,2,3,4]`: weighted MASE/MAE 1.25, MSE 1.87,
  normalized pinball 0.625, weighted scaled-error median 1 and p90 2.
- Compact output with sidecar details, both distribution plots and existing
  forecast plot regeneration.
- Graph profile planning for direct four-patch heads and legacy autoregression.

No full production holdout was run or overwritten. No kernels or model weights
were changed; the end-to-end adapter smoke uses lightweight deterministic
forecasts rather than downloading/loading foundation models. Historical latency
and error headlines are not directly comparable with the new fixed-horizon,
weighted protocol.

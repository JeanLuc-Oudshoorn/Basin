# Short-horizon forecasting, rolling use and holdout evaluation

Effective 2026-11-02 for the new four-token Basin model.

As of 2026-11-05, preprocessing preserves every positive finite standard
deviation; the former `<=1e-5 => 1.0` substitution is removed from both
tokenizers. Only zero or invalid scales use the safe unit fallback. This is
shared by training and inference and takes effect when existing checkpoints
are loaded in a new process. Holdout and forecasting standard-deviation-based
metrics also retain tiny positive scales. MASE denominators are unchanged.

## Direct forecasts and longer requests

A token contains 16 consecutive observations. Basin returns 1, 2, 3 or 4 future
tokens (16, 32, 48 or 64 timesteps) from one observed context and one backbone
pass. Later direct tokens do not require observed future targets or covariates.
Partial horizons request enough patches and trim excess predictions.

For a longer request, Basin forecasts in blocks of at most **64 timesteps**,
appends each block's median forecast to its target history and makes the next
block forecast. Thus 128 timesteps take two sequential block forecasts. The
last 2048 history points are retained; unknown future covariates are marked
missing rather than supplied from actual future observations. This is
autoregression between blocks, not a guarantee of equally accurate forecasts
at every distance. Repeated forecasts can accumulate errors and lose
uncertainty; nine marginal quantiles are not a coherent sampled joint path.

Inference CUDA graph replay remains available. Its cache distinguishes output
patch counts, so 16- and 64-point requests do not reuse an incompatible head
graph. Architecture and lead weights come from the loaded checkpoint. Old
447M exports remain one-patch models; old 307M exports remain 16-patch models.
The new default config does not retroactively change those weights or limits.

The inference cache now allows up to 256 profiles, with a 2.5-GiB tracked cache
budget (1.5 GiB until 2026-11-15, see below), configured by `training.inference_graph_cache_size` and
`training.inference_graph_memory_gib`. The earlier eight-profile limit left
most requests in a varied holdout uncached. Profiles distinguish variable count,
padded context length and requested output patches. No continual eviction or
recapture is performed. Admission uses observed allocator/device-memory growth
and input storage; the cache budget is separate from total process VRAM and
does not include model weights or unrelated compiler/driver memory. First-use
captures can still cause slower individual requests.

### Padding to common lengths and the cache budget (2026-11-15)

Every context is right-padded to the next multiple of 16 patches (256 points)
for the RWKV kernel, in training batches, validation and holdout alike. The
padded length is part of the graph key, so all real lengths in one bucket share
a graph. The buckets end at 256, 512, 768, 1024, 1280, 1536, 1792 and 2048
points. A context that is at most 10% shorter than 2048, 1536, 1024 or 512
therefore always replays the graph of that length (the windows are 12%, 17%,
25% and 50% wide); `tests/test_graph_profiles.py` pins this. Variable count
stays part of the key: series with different covariate counts are never padded
into one attention block.

On 7,500 tables sampled from the 2026-11-13 generator the holdout planner needs
70 shapes for 36,510 requests, 59 of them recurring at least eight times and
covering 99.9% of requests. Padding costs 3.7% extra variable-patches in
training. Shape variety is therefore not what limits replay; the cache budget
was. Capturing all 70 shapes for the 221.5M model took 2.03 GiB
(`experiments/claude_review/inference_graph_memory.py`, RTX 5070), while the
former 1.5-GiB budget is used up after about 33 shapes covering about 95% of
requests. The archived 500M holdout shows the same: 33 of 51 admitted shapes
captured, 95.3% replay, 4,091 `memory_limit` fallbacks. The budget is now 2.5 GiB. Replay took 6.8 ms per request against
17.9 ms forward-only, weighted by request shape, on an untrained model.

For holdout specifically, `holdout.graph_min_profile_requests` defaults to 8.
The evaluator counts expected profile reuse from table lengths, roles, existing
origins and the fixed four-token horizon, without reading future values or
making extra forecasts. Only profiles meeting that count are admitted for
capture. This avoids paying capture cost for a shape requested once or twice.
`graph_profile_plan` records expected coverage; `low_reuse` fallbacks are
intentional. Ordinary forecast API calls have no such advance request plan
and retain capture-on-first-use behavior. Set the holdout threshold to 1 to
capture every encountered profile that fits the count and memory limits.

`test_metrics.json`, the live progress file and initial `model_used.json` now
record `rwkv_inference` execution statistics (initial provenance uses
`inference_execution`): replay requests/coverage, capture count/time, fallback
counts/reasons, profile count and tracked cache bytes. Cache saturation emits
one warning per reason; per-profile capture details are DEBUG. These are runtime
settings and require no retraining or regeneration of prepared artifacts.
Execution counters include warmup calls, whereas scored latency summaries
exclude them. For a one-patch archived model, each autoregressive block is a
separate engine request. Coverage is therefore coverage of model passes rather
than necessarily a fraction of top-level forecast API requests.

## Why prioritize nearby outcomes

The 80%, 16%, 3%, 1% lead weights reflect a deliberate operational priority.
Users can refresh forecasts when new observations arrive: waiting until
November provides additional evidence for a December sales forecast compared
with making that forecast in September. Near-term electricity prices and
operational monitoring often require hours or days of predictions. This
does not remove the value of planning farther ahead; it allocates finite
training capacity toward the expected use cases. Accuracy and latency across
multiple horizons still need empirical evaluation.

## Anomaly-detection use

When evaluating an already recorded sequence, use a chronological rolling
procedure. Start with an observed warmup prefix, predict the next token from
that prefix, then compare the forecast with the actual next 16 values. Only
after scoring may those actual values enter context for the next prediction.
Actual future covariates must remain hidden at prediction time unless the
application genuinely knows them in advance.

This differs from a long synthetic rollout: each new anomaly-scoring forecast
uses observed history rather than the previous median forecast. Large forecast
deviations or quantile-interval violations can be candidate anomaly signals.
Threshold calibration, false-positive control, seasonality and noise require
separate treatment; this change supplies forecasts, not a complete anomaly
detector. Anomalies may also contaminate later observed context. No new
automatic anomaly labeling is introduced here.

## Holdout origins and requested horizons

Whole datasets remain disjoint from training. Use up to five distinct patch
starts per table, using the first
eligible origin after two context tokens, starts nearest the 25th, 50th and
75th row percentiles, and the final eligible start. As of 2026-11-04 every
eligible start must leave **64 real future observations**. Short tables use however
many origins exist; tables shorter than 96 points (two context plus four future
tokens) are skipped with explicit counts and a warning. Each origin uses actual history strictly before it,
capped at 2048 points, regardless of earlier holdout forecast errors.

Effective 2026-11-03, **every origin requests four tokens (64 timesteps) from
every model**. There are no horizon draws. The four coefficients in
`model.horizon_loss_weights` (80%, 16%, 3%, 1%) weight the errors of consecutive
tokens, just as they weight training and validation loss. Legacy
`holdout.horizon_probabilities` and `horizon_seed` no longer select requests.
An archived configuration without model loss weights uses the same default
four coefficients; a one-patch archived model rolls out autoregressively.

The final origin is `floor((N - 64) / 16) * 16`; it is patch aligned and
always leaves four full tokens to score. For N=2048 it is 1984, replacing the
previous 2032 start. For N=589 it is 512, leaving 13 unused trailing points
because the next aligned origin would leave fewer than 64. The 25/50/75%
starts are selected from the same eligible range, without duplicates. All
64 forecast points are retained and plotted against actuals. Each token's
metric is averaged over its finite actuals; only genuine missing values need
masking and possible renormalization, never an unavailable end-of-table tail. No synthetic
actuals, future covariates, or previous origins' predictions enter the context.

## Results and interpretation

MASE remains the leading metric. MASE denominators use only the observed
historical target context, independently of forecast horizon. Within each
origin and target, a finite point in token k gets weight
`w[k] / observed_points_in_token[k] / sum(observed_token_weights)`.
Pooled MASE, MAE, normalized pinball, and mean squared errors use these weights;
RMSE and normalized RMSE take the square root after aggregation. Each
origin-target contributes equal total weight. Overlapping forecasts from nearby origins are separate requests;
an observation may be scored at more than one origin.

Results include `by_forecast_token` summaries for leads 1 through 4,
equal-table distribution summaries and
interactive density/comparison plots. Raw prediction parquet files include
`forecast_tokens`, `forecast_horizon`, `forecast_token`, and `metric_weight`.
`test_metrics.json` contains aggregate results only, with protocol version
`four_token_complete_weighted_v2` and skipped-short-table counts. Per-table and per-origin
scores are saved separately in `table_metrics.json` and `origin_metrics.json`,
linked from `detail_files`. Request latency records contain the
same horizon and scored length for every model. Timings still include adapter
preparation, transfers and forecast generation, excluding shared input loading,
scoring and plotting. The rotating model order and progress/ETA are retained.

The leaderboard includes pooled MASE and weighted point-level
`median_absolute_scaled_error`. It also exposes `table_mase_mean`,
`table_mase_median`, `table_mase_p10`, and `table_mase_p90`: each table gets one
vote in these four statistics. A median of per-table mean scaled errors differs
from the median of individual scaled absolute errors. Pooled point-level
`absolute_scaled_error_p10` and `absolute_scaled_error_p90` use an inverse
weighted empirical CDF; per-table percentiles use linear interpolation.

Every completed run writes `latency_distributions.html`, an offline interactive
Plotly KDE overlay with transparent model colors. It includes every scored
request repeat, including shape capture costs, and excludes warmups.
`plot.latency_distribution` accepts `enabled`, `axis` (`log1p` or `linear`),
and `grid_points` (default 512). No tails are clipped. Each model density has
unit area in the displayed coordinate system; hover shows milliseconds.

These weighted scores and fixed-64-point latencies are a new protocol. They
must not be compared directly with historical sampled-horizon or one-token
headlines. Saved old predictions cannot recreate unrequested later tokens;
rerun holdout evaluation for the new protocol. Retraining is not required.

## Replacing previous holdout outputs

Each new `holdout_test.py` run removes the previous generated metrics,
prediction Parquets, request latency files, plot selections and forecast/error/
latency HTML/PNG/SVG plots from its configured output directory. This happens
after loading the selected checkpoint, before writing the new model identity
and forecasts. A log message records the number of removed files. Thus running
with fewer plot examples, fewer tables, different targets or a new model cannot
leave previous generated plots beside the new result. User notes and independent
subdirectories are preserved. There is no implicit archive; logs remain under
`logs/`. Existing old outputs remain until the next holdout run is started.

See [the training specification](../model_training/SHORT_HORIZON_TRAINING.md)
for label masking, weighted validation and migration commands.

# Short-horizon weighting: Basin, Chronos-2 and Timestral

Research checked on 2026-11-02. This is an architecture/objective comparison
and recommendation; no production training settings or checkpoints were changed.

## Horizon capability versus evaluation horizon

Current Basin uses 16 timesteps per patch and 16 direct output patches (256
timesteps), with a 50/50 next-patch versus sampled-later-patch training objective.
Validation and the current holdout continue to score only the first 16 points.
The additional training lead is sampled uniformly from relative leads 1..15
before missing/unavailable labels are masked. The split is an objective weight,
not a guarantee that half of the gradient norm comes from each component.

Neither comparison checkpoint has a 64-timestep direct forecast limit:

| Model | Output patch length | Maximum direct patches | Direct horizon |
| --- | ---: | ---: | ---: |
| Basin current 307M | 16 | 16 | 256 |
| amazon/chronos-2 | 16 | 64 | 1,024 |
| mistralai/Timestral-209M | 32 | 32 | 1,024 |

Sources: [Chronos-2 checkpoint configuration](https://huggingface.co/amazon/chronos-2/blob/main/config.json)
and [Timestral official model card](https://huggingface.co/mistralai/Timestral-209M).
The number 64 in Chronos-2's `max_output_patches` counts patches, not timesteps.
An architecture's output capability does not establish its accuracy at that
horizon, nor determine which horizons should matter for Basin's use case.

## Chronos-2

The [Chronos-2 paper, sections 3.2 and 3.3](https://arxiv.org/html/2510.15821v1)
describes direct multi-patch quantile predictions. Training samples the number
of output patches per batch. Pinball loss is averaged over forecast steps and
batch items; missing targets and known covariates are excluded. The report does
not specify an 80/16/4 lead-weighting policy. It starts with a smaller maximum
output length and extends that during a later training stage alongside longer
contexts. The [public loss implementation](https://github.com/amazon-science/chronos-forecasting/blob/main/src/chronos/chronos2/model.py)
also averages across the forecast horizon rather than applying a geometric
per-patch weighting vector. Public fine-tuning data code uses its configured
prediction length; this should not be confused with the paper's pretraining
horizon-sampling procedure.

## Timestral

The [Timestral paper, sections 3.1 and 4.1](https://arxiv.org/html/2610.37585v1)
describes Decoupled Horizon Forecasting: causal context representations are
shared across separate forecast blocks. Block lengths are sampled, and a
length-k block trains distances 1 through k. Prediction tokens cannot read
the observed target values inside their own forecast block. Keeping observed
context and prediction tokens separate allows many origins to share context
computation without leaking future observations into predictions.

Timestral uses pinball loss with equal averaging over scored target positions
and quantile levels, without an auxiliary objective. The distribution of block
lengths controls how often each distance receives supervision. The report does
not provide a fixed 80/16/4 lead weighting or a complete numerical block-length
sampling distribution. Therefore an exact per-lead effective training weight
cannot be inferred from the publication alone.

Both models explicitly train multiple horizons. Their results do not imply
that long-horizon supervision is inherently harmful. Basin's simpler shared
head with lead offsets and its available data/training budget differ, so the
best weighting must be measured locally.

## Recommended Basin experiment

The proposed objective is well aligned with prioritizing short forecasts:

`L = 0.80 * L1 + 0.16 * L2 + 0.04 * L3`

Here Lh is the mean normalized pinball loss on the h-th future 16-point patch,
predicted using the same historical context. Do not feed actual future patches
back as context while scoring later leads; doing so would change the task.
Missing labels, padding and the final validation patch must remain excluded.
Available leads need an explicit renormalization policy for short tables so
that they retain useful training weight.

This objective directly supervises 48 timesteps. If the intended direct horizon
is 64, a possible four-patch allocation is 80/16/3/1. Retaining a fourth output
with zero direct supervision would not establish reliable 64-step capability.
Alternatively the three-patch model can roll forward for a longer forecast;
this incurs another model call and possible error accumulation.

Weighted sampling can estimate a multi-lead objective with less head work,
whereas evaluating all short leads gives a more deterministic weighted loss.
They need not have identical speed or gradient variance. No training speedup
has been measured for either proposed implementation.

Compare the proposal with a next-patch-only control, using the same starting
weights, batches, seed, optimizer settings and update budget. Track 16/32/48/64
step metrics on a separate development split while retaining the existing
one-patch validation criterion. Do not treat repeated tuning against the
already-inspected holdout as a new unbiased final evaluation.

The prior regression audit's controlled continuation favored next-patch-only
training over the current mixed objective. That supports testing the objective
change, but does not prove it explains the entire 447M/307M gap. Neither the
proposed 80/16/4 weights nor the four-patch alternative has been trained here.
Changing model configuration/objective also changes checkpoint identity: the
existing exact-resume mechanism deliberately rejects that change. A separate
weights-only warm start with a new optimizer/run, or a fresh training run,
would need to be used rather than presenting it as an unchanged resume.

## Recursive one-patch forecasts

Training only the next patch and repeatedly appending its forecast is a valid
recursive strategy. It may be preferable for Basin, but stronger next-patch
accuracy does not guarantee stronger long-horizon accuracy. Training sees real
histories while rollout sees generated histories, so drift and prediction
artifacts can accumulate. Even perfect conditional means or medians do not
generally yield the correct farther-horizon mean or median when substituted
recursively into a nonlinear model: a point estimate discards uncertainty over
possible intermediate futures. Sampling coherent future trajectories and
aggregating them can propagate uncertainty, but Basin's marginal per-point
quantiles do not themselves specify a joint distribution over patch paths.

Direct forecasting learns each lead conditional on the original observed
history. Shared parameters allow this without a separate model per timestep;
the target at a later lead is the appropriate conditional forecast, not a
guaranteed reconstruction of inherently unpredictable events. Longer-horizon
objectives can nevertheless compete with near-term performance at finite
capacity and training budgets.

The statistical tradeoff is discussed in
[Ben Taieb and Hyndman, ICML 2014](https://proceedings.mlr.press/v32/taieb14.html).
No strategy universally dominates. Recursion also needs a policy for future
unknown covariates; actual future covariate observations cannot be supplied in
a past-only evaluation. At horizon 1,024 with patch size 16, one-patch rollout
requires 64 sequential generation steps, whereas a direct head can emit the
horizon together. Recurrent state caching can reduce work per step without
removing that dependency. A fair Basin experiment should measure next-only
recursive forecasts and weighted short-block forecasts at identical origins
and horizons (16, 32, 48, 64 and beyond), scoring both accuracy and latency.

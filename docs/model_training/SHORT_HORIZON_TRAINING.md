# Four-token training and validation policy

Effective 2026-11-02 for new runs using `configs/config.json`.

Superseded as the default on 2026-11-15: `configs/config.json` now trains a 220M
hybrid model that forecasts seven tokens (112 timesteps) with equal weight per
timestep through a direct head. This document describes the four-token,
80/16/3/1 protocol of the archived 500M model, which `configs/legacy_500m.json`
still selects. See [the 220M architecture and recipe](HYBRID_220M_2026-11-15.md).

Default protocol update, 2026-11-06: the model architecture and four loss weights
remain the same, but full training tables now replace reserved-tail training.
The fixed 25,000-table in-sample panel is evaluated hourly at five origins.
See [hourly training](HOURLY_TRAINING.md). Reserved-tail and epoch-validation
sections below document the compatible legacy `data.validation_policy="reserved_tail"`
protocol, not the new default.

Normalization update, 2026-11-05: both tokenizers now preserve every positive,
finite historical standard deviation, including values below `1e-5`. Unit
scale is used only for zero or invalid/unavailable scales. This applies to
targets and covariates during training and inference. No positive-amplitude
cutoff or configurable floor remains. Existing checkpoints use the new runtime
preprocessing when loaded; resuming does not reset their weights or optimizer.
See the [normalization audit](../audits/2026-11-05/normalization_threshold.md).

## Product objective

Basin deliberately prioritizes short-term forecast accuracy. Operational users
often need the next hours or days and can refresh a forecast as new observations
arrive. A September forecast for December may inform an early decision, but a
November forecast for December has more recent information. Electricity-price
forecasting commonly emphasizes tomorrow or the next several days. Forecasting
also supports anomaly detection by comparing short forecasts with observations
as they become available. These examples motivate the weighting; they are not
claims that long-term forecasting has no value or that this weighting is
universally optimal.

## Architecture and configuration

The default has 394,914,080 parameters: 22 RWKV-7 temporal layers
with cross-variable attention, width 1024, jointly trained patch projections,
nine quantiles and up to 2048 historical observations. Each non-overlapping
token covers 16 timesteps. The shared forecast head now has three learned lead
offsets, for a total of **four output patches / 64 timesteps per context pass**.
The first patch uses the original head without an offset. Later patches are
predicted from that same causal hidden state plus their learned lead offset;
they do not consume actual or generated intermediate future target values.

The [capacity verification](CAPACITY_400M.md) records the complete parameter
count and why 22 layers is the largest depth below 400M with these dimensions.

```json
"data": {"horizon": 64},
"model": {
  "patch_size": 16,
  "output_patches": 4,
  "horizon_loss_weights": [0.8, 0.16, 0.03, 0.01]
}
```

These are excerpts, not a complete runnable configuration. In this pipeline,
`data.horizon` is the reserved validation tail length. The legacy
`multi_horizon_weight` option is ignored when explicit `horizon_loss_weights`
are provided. Existing checkpoints without explicit weights retain their
original sampled-auxiliary objective and original direct-output capacity; merely
updating the default JSON does not rewrite a saved model.

## Causal training examples

For every training input position, predict up to four consecutive future
patches using only the observed prefix at that position. In schematic form:

```text
Observed context:  P1 ... Pt
Lead 1 target:     P(t+1)      weight 0.80
Lead 2 target:     P(t+2)      weight 0.16
Lead 3 target:     P(t+3)      weight 0.03
Lead 4 target:     P(t+4)      weight 0.01
```

The RWKV backbone computes all causal prefix states in parallel during
training; it is not rerun four times per prefix. Four deterministic head
evaluations supervise the leads that have labels. This replaces the former
50/50 next-patch versus randomly selected distant-patch objective. All four
leads use normalization statistics from the same observed prefix. Actual
outcomes and time-matched covariates are inputs; lag feature columns remain
excluded.

The precise loss is a weighted mean of normalized pinball errors over valid
label points and quantile levels:

`L = sum(w[h] * valid * pinball) / (Q * sum(w[h] * valid))`.

The sum spans batch items, target variables, causal anchors, within-patch
points and forecast leads (the quantile dimension appears only in the
numerator). With equally observed labels at all leads, this is
`0.80 L1 + 0.16 L2 + 0.03 L3 + 0.01 L4`. Missing and unavailable labels are
masked and their weight is removed from the denominator. Padding is never
a target; labels cannot wrap around to earlier tokens. At the end of a
training chunk, unavailable future leads are not supervised. In particular,
neither actual targets nor covariates from the reserved validation tail are
included in training windows.

Physical microbatches use their effective weighted label mass when accumulating
gradients. A split update therefore has the same loss weighting and gradients
as its unsplit equivalent, rather than accidentally assigning equal weight
to differently sized microbatches. Raw next-patch target-point counts remain
separate for progress reporting; `target_loss_weight` reports effective mass.
Epoch training loss is aggregated using this same mass. These weights are
objective coefficients, not promises about each lead's share of gradient norm.

## Four-patch validation

For a table with N observations, transformation sets `cutoff = N - 64`.
All training labels and covariates precede this cutoff. Validation uses the
last up-to-2048 points preceding the cutoff as context, then predicts the
final **64 actual points in one pass**. It does not reveal one validation
patch to predict the next. Right-aligned training patches retain chronological
order; partial prefixes use masked left padding.

To leave at least one causal next-patch training pair before four held-out
patches, new training tables must contain at least **96 observations** (six
complete tokens). Shorter tables are excluded with a logged/audited reason.
This supersedes the former 48-point training minimum. Dataset-level holdout
selection still reserves 10% of eligible admitted sources; changing eligibility
can change the ranked split. Holdout evaluation now also requires at least
96 observations so every selected origin can leave four complete future
tokens (updated 2026-11-04).

Full validation runs at epoch zero and after each epoch, with the same
80/16/3/1 weighted pinball formula. Batched validation aggregates effective
weighted mass, so missing labels and batch boundaries do not alter the metric.
Validation RMSE remains an unweighted pooled RMSE across all observed points
of the four patches; only the loss uses lead weights. Best-checkpoint selection,
the learning-rate scheduler and early stopping monitor the weighted validation
loss. Existing epoch-frequency and patience settings are otherwise unchanged.

The admission screen also reserves the final 64 points, using `data.horizon`.
Its own screening forecast length and admission thresholds are unchanged.
Changing the reserved tail changes the screening signature, so old reports
cannot silently admit tables after inspecting the new validation labels.

## Migration and preservation of existing runs

The old 16-lead checkpoint, its optimizer state and its one-patch validation
history do not match this architecture or split. Earlier 17-layer four-token
checkpoints also cannot resume the new 22-layer architecture. Exact resume rejects the
change; `train.py` now checks the input signature before replacing existing
run reports. This is a new training experiment, not a continuation of the
old optimizer trajectory. Do not set `resume=false` in an old output directory
unless overwriting that run is intended.

Before running the new stages, move the old default result folder to an unused
archive location (the following destination must not already exist):

```powershell
Move-Item -LiteralPath .\results\model -Destination .\results\model_archive_before_395m
uv run python .\src\03_transforming_input_data\transform.py
uv run python .\src\04_patch_indexing\index_patches.py
uv run python .\src\05_model_training\train.py
```

Transformation automatically refreshes stale screening when configured to do
so. Old processed and patch-index artifacts have incompatible signatures and
must be regenerated. The new run uses the standard `results/model/training`
directory and ordinary epoch checkpoints. No special patch-training folder
or alternate training entrypoint is introduced. The archived 447M weights and
configuration retain their legacy architecture. Its current-holdout evaluation
configuration now requests the same fixed four-token horizon as the default benchmark.

## Holdout reporting update (2026-11-03)

Holdout now requests all four tokens at every existing origin and uses
`model.horizon_loss_weights` as metric coefficients, rather than probabilities
of choosing a shorter request. The model, training loss, validation partition,
checkpoint identity, and optimizer state are unchanged by this reporting update.
Existing weights can be evaluated without retraining or rebuilding data.

For each holdout origin-target, compute each token's mean error over available
finite actuals, combine using 80/16/3/1, then average across origin-target pairs.
MASE, MAE, normalized pinball and squared errors use these weights; RMSE applies
the square root after aggregation. As of 2026-11-04, origins are restricted to
starts with four complete future tokens remaining, rather than scoring a
shortened final tail. Only genuine missing actuals need masking. Forecasts always
contain all 64 points for plotting. This holdout aggregation is explicit and
does not change the existing validation RMSE convention described above.

`test_metrics.json` now contains aggregates and links to detailed artifacts.
It includes equal-table mean/median/p10/p90 for each error metric, alongside
pooled weighted point-level median absolute scaled error and p10/p90. These
statistics distinguish widespread errors from an extreme tail. Latency
distributions are plotted separately from error distributions, using scored
requests only. See the forecasting specification for schema and boundaries.

See [the forecasting policy](../forecasting/SHORT_HORIZON_FORECASTING.md) for
autoregressive continuation, holdout sampling and anomaly-detection semantics.
See [the implementation audit](../audits/2026-11-02/four_token_policy.md) for
verification and smoke-run artifacts.

# Flat-context forecasts: scale fallback, learned patch shape and holdout fixes

## Findings from frozen-weight probes

The checkpoint is the same export used by the latest saved holdout:
SHA-256 `17b5f38a3667514e4c18fca83bab20201e764e516e2dd360d1d0528f4a6b59fc`.
It trained on 114,132 sources for 15 completed epochs, exporting its best
epoch-12 weights. No weights or production normalization policy were changed.

The tests used the real GPU model in BF16, 64-point forecasts, and the normal
forward-only inference engine with graph capture disabled. Attention compilation
was also disabled to isolate preprocessing from execution optimizations. Thus
probe values can differ slightly from the production compiled/replayed output;
the amplitude problem survives without graph replay. The first native-kernel
probe and repeated forward-only-kernel probe show the same findings. Both runs
have timestamped run contexts under `logs/`.

## Why tiny input movements can produce a forecast around 0.18

Basin's numerical pipeline is:

1. Calculate the causal historical mean and standard deviation for each variable
   at each token boundary.
2. Normalize historical values as `(value - mean) / scale`.
3. The learned backbone and forecast head predict normalized residual quantiles.
4. Return forecasts as `mean + scale * normalized_prediction`.

In `src/04_creating_embeddings/embed.py`, `tokenize_next` replaces **every
standard deviation <= 0.00001 with 1.0**, not merely an exactly zero standard
deviation. The older tokenizer uses the same rule. For table `019943`, origin
32, the actual historical standard deviation is **0.00000231004**, but the
encoder/output scale becomes **1.0**. The mean is approximately -0.000007228.
A learned normalized deviation around 0.18 therefore becomes a raw deviation
around 0.18. Multiplication by the actual historical scale would make that
same normalized deviation approximately 0.000000416 instead.

This rule avoids division by zero, but it is an inappropriate amplitude proxy
for a positive, tiny-variance series. It is discontinuous and depends on the
measurement units. A series expressed in different units can cross the fixed
threshold and receive very different behavior.

Controlled results at the first origin of `019943`:

| Probe | Historical std | Encoder scale | Forecast range across all 64 points |
| --- | ---: | ---: | --- |
| Normal pipeline | 0.00000231004 | 1 | -0.04395 to +0.17968 |
| Diagnostic positive-std normalization | 0.00000231004 | 0.00000231004 | -0.000008690 to -0.000006019 |
| Target amplitude multiplied by 0.1 | 0.000000231004 | 1 | -0.04395 to +0.17968 |
| Target amplitude multiplied by 10 | 0.0000231004 | 0.0000231004 | -0.000021846 to +0.000004864 |

Multiplying the tiny input movements by ten makes the predicted movements
dramatically **smaller**, because it crosses the fallback threshold. This
provides direct evidence of a scale-handling problem, beyond insufficient
data coverage alone. Both the normalized input features and output scale were
changed in the diagnostic normalization variant, using the same frozen weights.
Weighted MAE at this particular origin falls from 0.06720 to 0.00001142. This
is a controlled example, not a demonstrated whole-holdout improvement.

The diagnostic preserves any positive prefix standard deviation, retaining
unit scale only for exactly zero variance. It is not a production-ready fix:
near-zero numerical precision, truly constant/all-missing histories, sudden
future regime changes, covariates and training-loss normalization must all be
handled consistently. Inference-only rescaling should not silently redefine
the preprocessing policy on which a checkpoint was trained.

## Why the forecast still wiggles on an exactly flat target

The numerical issue explains amplitude, but not the entire shape. The model
does not enforce `constant history => constant median forecast`. Even a zero
target is combined with learned role embeddings, time features, model weights
and optional covariates. Its hidden representation is not constrained to zero.
The forecast head emits 16 separately learned future positions. It can learn a
nonflat default shape when context does not constrain the answer well.

The same head is reused for subsequent patches with learned lead embeddings.
On `019943`, the four predicted patch shapes have pairwise correlations above
0.9997. Thus the apparent periodicity is nearly a repeated learned **patch
shape**, not evidence that the model discovered a historical seasonal cycle.
Removing covariates does not eliminate it: an exactly zero, target-only context
still produces roughly -0.0469 to +0.1563 in this checkpoint. This rules out
covariates as the sole source of that behavior.

This is not a fundamental inability of RWKV to forecast constants: the existing
head can represent a flat forecast. It is a learned behavior of these weights
and this unconstrained shared patch head, amplified by unit output scale.
Fixing positive-variance normalization alone does not resolve the exactly
constant case, because its true standard deviation is still zero.

## Table `072778` is a different case

At its first origin (32), target std is **0.0000391236**, above the fallback
threshold. Its forecast stays tiny, approximately -0.000403 to -0.000375.
The enormous MASE there mainly comes from the actual future jumping toward
8.89 after a quiet history, rather than a large invented forecast amplitude.
Changing the positive-std fallback leaves this forecast unchanged.

By origin 48 the observed context already includes a sharp rise to about 5.97,
with std approximately 1.60. Basin predicts approximately 6.90 to 11.25 with a
wiggling patch shape, despite no demonstrated historical cycle. Four future
patch shapes again correlate above 0.9998. The head is extrapolating a learned
shape from a short rising segment. Better data/objective/head calibration may
help here; this specific forecast cannot be blamed on the tiny-variance rule.
An observed rising segment does not establish the timing of a later turn.

## Would more data solve this?

A deterministic sample of 512 **training** tables found:

- 19 tables had at least one near-flat training prefix (std <= 0.00001).
- 15 had near-flat initial two-token contexts.
- 201 of 32,845 examined prefix tokens were near flat: approximately **0.61%**.

This is a sample of raw target prefix amplitudes, not a census or an exact
measurement of effective gradient weights. It excludes the reserved final
64 observations. It demonstrates that such contexts are present, but rare.
There was no search through holdout labels to select this training sample.

More representative data and training can improve learned priors and flat-case
calibration. More data from the same mixture, however, leaves the discontinuous
unit-dependent preprocessing rule intact and does not guarantee removal of
patch-shaped hallucinations. The prioritized next work is:

1. Design and validate consistent scale-aware normalization, including a separate
   policy for truly constant histories; compare unit-rescaled copies of inputs.
2. Add targeted flat/near-flat and transition cases to training and regression
   evaluation, separating stable futures from genuine future regime changes.
3. If repeated patch shapes remain, evaluate a head/base-forecast residual or
   continuity-aware training change with controlled ablations. Do not force all
   flat histories to remain flat when historical covariates may predict a change.

The current task leaves production normalization, model architecture and
checkpoint weights untouched. The structural/numerical issue is now supported
by controlled evidence rather than attributed solely to dataset size.

## Holdout changes implemented

### Complete future blocks

`sampled_origins` now accepts `prediction_patches`; holdout and its graph planner
both pass four. Eligible starts are `32,48,...` up to
`floor((N-64)/16)*16`. The first eligible, closest distinct 25/50/75% starts,
and final eligible are chosen from this restricted set. There is no repetition
when fewer than five eligible origins exist.

For N=2048, the final origin moves from 2032 to **1984**, giving 64 actuals.
Nonaligned lengths retain patch alignment, possibly leaving a short unused
tail. At least 96 points are required for two context tokens plus four future
tokens. Shorter saved holdout tables are skipped with a warning and explicit
counts; no short forecast silently replaces the full evaluation block.
Internal missing values remain masked. Loss coefficients are unchanged.
The result records protocol `four_token_complete_weighted_v2`.

### Stale output cleanup

`holdout_test.run` invokes `clear_holdout_outputs` after loading a valid model
and before publishing its identity or predictions. Known generated root files
are deleted: prior metrics, prediction Parquets, latency files/selections and
forecast/error/latency plots, including plots for no-longer-selected tables or
targets. Cleanup runs every time and logs its count. No archive is created.
User notes and independent subdirectories are preserved; no recursive folder
deletion occurs. Production outputs were not deleted during this investigation;
the cleanup is verified in temporary output directories and applies on the
next normal holdout invocation.

## Evidence and reproduction

- [Frozen-weight probe results](evidence.json)
- [Interactive actuals and normalization probes](normalization_probes.html)
- `probes.parquet`: all 64 point forecasts for each diagnostic case.
- `training_sample.parquet`: sampled training sources and prefix counts.
- `scripts/investigate_flat_contexts.py`: reruns the GPU probes and training sample.
- `tests/test_holdout_output_cleanup.py`: generated-file cleanup and preservation.
- `tests/test_holdout_protocol.py`: complete-block origin selection and history cap.
- `tests/test_four_token_policy.py`: identical full horizons, short-table skips,
  weighted scoring, saved predictions and distribution artifacts.

Logs are saved under `logs/*_flat_context_investigation_*`. Compile/cache/temp
locations use the existing WSL activation settings on drive E.

Verification completed: 263 CPU-suite tests ran successfully (12 CUDA-only
tests skipped); 17 frozen-checkpoint GPU probe cases completed successfully.
Pre-commit syntax, JSON, whitespace, Ruff and Bandit checks passed. No full
production holdout was rerun, no production results were deleted, and no model
parameters or normalization defaults were changed.

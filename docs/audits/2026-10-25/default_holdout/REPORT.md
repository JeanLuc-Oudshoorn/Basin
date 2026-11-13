# Default training protocol and approximately 450M model

Implemented in the ordinary `src/05_model_training/train.py`. There is no new
pipeline branch or patch directory. Shared code lives in `training_protocol.py`
in that same training directory and `holdout.py` in `src/06_forecasting`.
The separate forecasting command replays the checkpoint's saved test partition.

## Dataset isolation

1. Select raw date folders through `configs/config.json`, then apply the existing
   Prophet-vs-naive admission gate. The transform stage still requires a current
   validation report; this change does not introduce automatic Prophet admission.
2. Exclude tables shorter than `max(48, 3 * patch_size)` real timesteps. Padding
   does not make a short dataset eligible. Log these exclusions in the processed
   manifest. Training also enforces this minimum on previously cached records.
3. At the start of every normal training invocation, sort eligible admitted source
   keys by SHA-256 of `seed:source_key`. Keys include the date-relative table path.
   Reserve `ceil(N * training.holdout_fraction)` tables, bounded to leave at least
   one table for training. The default fraction is **0.1**. Thus 10 of 100 tables
   are test tables; 1 of 6 is a test table. At least two eligible tables are required.
   Selection is independent of window order and outcomes; changing the source pool
   can change the partition. New transform artifacts include root-relative keys;
   older artifacts fall back to their stored source identifiers.
4. Remove **every** training window belonging to those test sources. Their final
   tokens are also excluded from all training-time validation. Initial test context
   is never used in a gradient update. The cached embedding stage contains only
   untrained projection snapshots and per-window causal statistics: it does not
   learn on the test sources. The neural projection learns jointly on training
   records only, with the backbone and output head.

The processed artifact's `split="evaluation"` means a table's final-token record.
It becomes either a training-time validation record or an unused test-table
representative when the trainer partitions whole datasets. The saved
`dataset_split.json`, also included in `model.pt`, is authoritative. It records
train/test source identifiers, content hashes, seed, rounding, counts and keys.
No test target or test score drives optimization, scheduling, or model selection.

This is an out-of-sample test **conditional on admission**: Prophet previously
screened the admitted sources using an earlier portion of each table. It is not
an unbiased benchmark over every generated dataset. No claim of generalization
quality follows from the synthetic smoke test below.

## Training and validation

The patch size is 16 observations. For every training table, the final 16 real
observations form its validation label, and never appear in a training label or
training input. Earlier next-token pairs are trained in parallel with causal
RWKV processing and prefix-specific normalization. Long tables are chunked into
at most 128 input tokens with one additional token supplying the shifted label;
adjacent chunks share one context token. Padding is masked, not a synthetic label.
The pre-existing right-aligned patch convention for non-multiple lengths remains.
Only outcomes and time-matched observed covariates are used; lag columns remain
excluded.

The regular trainer measures **full validation over all non-test tables** before
the first update and after **every** optimizer step, in evaluation mode without
gradients. Loss is normalized pinball loss, weighted by finite label observations,
across all configured quantiles. RMSE is computed by pooling squared errors over
the same observations. A per-table breakdown, step and duration are saved in
`validation_history.json`. The update-batch training loss is measured before its
update; validation is measured after the update. They have different context/label
distributions and should not be interpreted as the same diagnostic.

AdamW and the existing warmup/cosine scheduler remain in use. Only training-window
count determines an epoch and the planned number of optimizer updates. Training
still recomputes embeddings for each update when `train_embeddings=true`.

## Final rolling test

The checkpoint is saved and the `training_complete` journal event emitted first.
Then the trained model evaluates the reserved sources, checking their Parquet
hashes against the transformed source versions before reading them.

For a 64-observation table, the test origins are:

| Context rows | Predicted and scored rows |
|---|---|
| `[0, 32)` (tokens 1–2) | `[32, 48)` (token 3) |
| `[0, 48)` (tokens 1–3) | `[48, 64)` (token 4) |

The next origin reveals the previous token's **actual values**, not model
predictions. Each origin predicts one whole token (16 observations), rather than
one scalar timestep. An incomplete final token is forecast and scored only on its
real finite observations. At each origin, all methods receive only the final
2,048 observed history points; older observations and covariates are discarded.
No future target or future covariate is supplied to the neural model.

Both baselines are refitted at every origin on the same target history. The
comparison uses univariate Prophet and triple exponential smoothing (Holt-Winters)
with configured trend and seasonality. RWKV additionally receives observed
historical covariates. Holt-Winters requires two seasonal cycles in its first
32-point context, so the trainer checks `2 <= seasonal_periods <= patch_size`
before initializing the model. There is no silent switch to nonseasonal Holt.
Missing historical target values are interpolated for baseline fitting only;
missing test labels are excluded consistently. Fit failures or nonfinite
predictions fail the evaluation instead of silently omitting a competing method.
The final comparison always runs both baselines, including Prophet; the legacy
standalone forecast toggle does not disable the requested post-training test.

Artifacts under the configured forecast path:

- `test_predictions.parquet`: source, target, origin, timestep, context length,
  actual, predictions from all three methods, context scale, normalized pinball.
- `test_table_NNN.parquet` and `test_table_NNN_target_N.png`: detailed predictions
  and comparison plots for up to the configured maximum number of tables.
- `test_metrics.json`: per-origin, per-table and pooled RMSE, MAE, normalized RMSE;
  normalized pinball loss is also recorded for RWKV's quantiles. Pool squared
  errors first, then take their square root; do not average per-origin RMSE.
- `test_metrics_progress.json`: progress after each completed test table.

A copy of the final metrics is saved to the training output directory. Loguru
events and automatic timestamped JSONL remain under `logs`. JSONL contains split,
initial/full validation, update metrics, `training_complete`, test progress,
`out_of_sample_evaluation`, then `run_complete`. Its final record includes pooled
test metrics separately from validation loss. An evaluation failure after the
checkpoint is saved produces `run_error` and can be retried with `forecast.py`.
Evaluating all rolling origins and refitting both baselines can take much longer
than a short neural training run.

## Model dimensions and tied weights

The default measured model has **447,482,144 trainable parameters**, approximately
450M, with 25 RWKV-7 layers and width 1,024. Each RWKV FFN has width 4,096. Temporal
RWKV uses 16 heads of size 64. Each layer also has same-time variate attention
with 16 heads of size 64; it mixes covariates and targets within the same table,
never across tables. Patch embeddings have output dimension 1,024. Context remains
2,048 observations = 128 tokens, not 2,048 tokens.

**Weights are not tied.** The active wrapper uses the vendored RWKV blocks, not
its text embedding or text vocabulary head. Its learned input residual projection
maps `3 * 16 = 48` continuous features per patch to width 1,024, with a separate
three-role embedding. Its independent output residual projection maps width 1,024
to `16 * 9 = 144` forecast values (16 points, nine quantiles). These are different
representations and dimensions, so language-model vocabulary embedding tying is
not directly applicable. `architecture.json` explicitly records untied embeddings
and every parameter tensor shape.

## Running the normal pipeline

Model configuration changes invalidate cached artifacts. Keep the configured
existing output directories (their historical `300m` suffix does not determine
capacity). Run the stages again using the main configuration:

```powershell
uv sync --extra validation
# If the selected raw data has not yet been screened:
uv run python src/02_data_validation/validate_generated_data.py --config configs/config.json
uv run python src/03_transforming_input_data/transform.py --config configs/config.json
uv run python src/04_creating_embeddings/embed.py --config configs/config.json
uv run python src/05_model_training/train.py --config configs/config.json
# Optional replay of the same saved holdout, with no retraining:
uv run python src/06_forecasting/forecast.py --config configs/config.json
```

On Windows, the train and forecast entry points route to the existing WSL CUDA
environment. Preprocessing can run in the Windows uv environment. Historical
checkpoints are not resized: changing their architecture requires a new run.

## Verification

`tests/test_holdout_protocol.py` checks deterministic table isolation, exact 10%
for a 20-table pool, small-pool rounding, minimum length, missing validation,
rolling origins and partial tails, pooled metrics, full gradient-free validation,
and actual-history/covariate alignment including truncation beyond 2,048 points.
The existing next-patch split test now uses an eligible 48-point short table.

`scripts/verify_training_protocol.py` exercises the **same normal trainer**, using
the default 447.5M architecture on the RTX 5070, ten synthetic 64-point tables,
two optimizer updates, full validation on nine tables, and final rolling test on
one table (two origins, 32 scored points). Prophet and Holt-Winters are real fits.
It also loads the saved checkpoint through the normal forecasting entry point and
checks that replay metrics match exactly. Admission is explicitly disabled for
this synthetic wiring test, which does not measure admission or forecast quality.
Temporary source and checkpoint files are removed afterwards; the script regenerates
them. Measured architecture, split, losses, validation histories, final metrics,
embedding optimization and a comparison plot are retained alongside this report.

To rerun the GPU smoke test:

```bash
source scripts/activate_rwkv7.sh
python scripts/verify_training_protocol.py
```

Recorded verification on 2026-10-25:

- Complete CPU regression suite: **92 tests, OK, six CUDA-dependent skips**;
  `logs/holdout_full_tests.log` contains the complete output.
- The separate full-size GPU smoke test passed on the RTX 5070. Full validation
  pinball losses were **0.553768 → 0.305327 → 0.297186**, at initialization and
  after the two updates respectively. Each sweep scored 144 validation values
  from nine tables. Both embedding updates had nonzero gradients and measured
  projection weight changes.
- One reserved table produced two test origins and 32 scored observations.
  Pooled RMSE: RWKV **0.586842**, Prophet **4.11663e-8**, Holt-Winters
  **7.38490e-6**. This synthetic fixture is deliberately an additive trend and
  period-16 sinusoid, exactly matching both baseline families. Two neural
  optimizer updates are a protocol check, not a competitive accuracy experiment.
- Checkpoint replay produced identical pooled metrics. See
  [smoke_verification.json](smoke_verification.json),
  [validation_history.json](validation_history.json), and
  [test_metrics.json](test_metrics.json).
- Journal event order, final test summary, and validation/test separation were
  checked in `logs/rwkv7_train_20261025_141459_656253Z_pid365_a6ec9ede.jsonl`;
  [journal_verification.json](journal_verification.json) retains the verification.
- Ruff lint/format checks and Bandit passed on the changed Python files.

![Full-size smoke test: rolling forecasts and actuals](smoke_forecasts/test_table_000_target_0.png)

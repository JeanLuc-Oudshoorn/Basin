# Prophet admission gate, embedding training and scaling plan

## Admission rule and stage order

The default `configs/config.json` enables dataset screening before transformation.
For each source, admission requires a **strictly lower RMSE than last-observation
carried forward** in either Prophet's univariate or multivariate scenario (or both).
An equality is a rejection, including a perfect zero-error naive forecast. A failed
variant never qualifies; a successful other variant can still admit a table.

```powershell
uv sync --extra data --extra validation
uv run python src/02_data_validation/validate_generated_data.py
uv run python src/03_transforming_input_data/transform.py
uv run python src/04_creating_embeddings/embed.py
uv run python src/05_model_training/train.py
uv run python src/06_forecasting/forecast.py
```

The validation entry point now reads the shared configuration instead of hardcoded
paths and parameters. It reuses `ProphetValidator.evaluate_table`; it does not
introduce another fitting implementation. The reusable validator retains its
standalone `run()` API and default `reserved_tail_rows=0` for compatibility.
All stage invocations save Loguru events, configuration, source hashes and status
under unique `logs/` run directories. Screening records RMSE, improvements, errors,
train/test lengths, covariate names, source hashes and elapsed time for each table.

## Two chronological boundaries, with no final-test selection leakage

For a source with N rows and a 16-point forecast horizon:

1. Reserve rows `[N-16, N)` for the final model evaluation. These outcome values
   are excluded from the screening scores and from model training.
2. On the earlier N-16 rows, fit Prophet to the first floor(0.8*(N-16)) rows.
   Evaluate against the remaining earlier rows and compare with the last observed
   training outcome carried forward for that same screening horizon.
3. If either scenario wins, allow transformation of the source. Model training
   then uses all pre-final-test rows with the existing causal next-patch objective.

The inner screening holdout is a **data-selection validation slice**, not a second
untouched test set. Its observations may subsequently be used for model training.
The final 16 observations remain untouched until the final forecasting stage.
A unit test changes those reserved outcomes to NaN and changes the reserved
covariates, confirming unchanged screening scores and forecast inputs.

Default multivariate screening uses `covariate_mode="forecast"`: each nonconstant
covariate is forecast with Prophet from screening training history, then used in
the target forecast. It never supplies actual future covariates. This aligns with
the current RWKV historical-covariate policy. `known_future` remains configurable
for explicitly conditional screening, but is a different information policy.
Lag-derived columns are excluded, including lagged covariates. Prophet regressors
remain contemporaneous; lag relationships are not manually engineered.

The current validator is specific to `outcome` and `time_index`; the transform gate
rejects a configuration claiming different target/time roles rather than screening
the wrong target. It selects numeric `covariate_*` columns; generator tables use
that naming. Extending to arbitrary target/column schemas requires adapting the
validator too.

## Persistence and failure handling

The configured Parquet report is
`results/02_data_validation/learnability_2026-10-23/metrics.parquet`.
Screening checkpoints progress to a separate Parquet file; only a completed run
atomically replaces the authoritative report. Stable relative source keys work
across Windows and WSL. SHA-256 binds each row to its exact source file. A separate
screening signature binds seed, validation settings and reserved horizon.

`transform.py` recomputes wins from finite, nonnegative scores; it does not trust
previous integer win flags. It rejects stale source hashes, missing rows, failed
fits, changed screening settings, and reports that did not reserve the final
forecast horizon. A missing report or duplicate source keys is an error. No
admitted tables is an error. Each admission and rejection is logged; both
`validation_selection.json` and the processed manifest preserve decisions and the
report hash. The gate runs before any `max_tables` sampling limit. Legacy explicit
configurations may disable the gate, and that is logged.

Current output directories use `447m_model`, preserving the completed
unscreened 100-table experiment and checkpoint. The pipeline artifact signature
includes the validation policy, so old unscreened embeddings cannot be accidentally
used with the current configuration. Normal processed-to-embedded and
embedded-to-checkpoint hash checks remain in place.

## Executed screening

All 100 source tables in `data/01_generated/2026-10-23` were evaluated successfully:

| Screening result | Tables |
|---|---:|
| Both scenarios beat naive | 79 |
| Univariate only | 2 |
| Multivariate only | 1 |
| Neither | 18 |
| Admitted total | **82** |

All settings come from `config.json`. Automatic calendar seasonalities are disabled
because these synthetic row positions are not calendar dates. The default custom
period list is empty, so screening does not assume the previous benchmark's period
16. Periods can be supplied explicitly in `validation.seasonal_periods`; two
training cycles are required by the validator. Neither SCM metadata nor the final
test outcome is used to choose a period.

This is an operational **Prophet-based selection heuristic**, not proof that
rejected data is inherently unlearnable. It can exclude useful nonlinear or lagged
relationships Prophet cannot capture, and a one-split strict win can occur by
chance. It also favors the patterns represented by Prophet. Apply it to a training
pool; keep a separate, unfiltered unseen-dataset evaluation suite when measuring
foundation-model generalization. More robust screening would use multiple inner
forecast origins and a configurable minimum improvement.

## Embedding dimensions and joint optimization

The current patch encoder already produces **960-dimensional** embeddings, above
the requested approximate 512 dimensions. Each variable's 16 observations produce
48 causal input features: normalized values, observed masks and time positions.
A residual projection `48 -> 3840 -> 960`, plus `48 -> 960` skip, creates each token;
a learned role embedding is added. Parameters are shared between tables and
variables, allowing different lengths, covariate counts and meanings.

`embeddings.pt` contains deterministic causal features, labels, initial projection
weights and **initial inspection snapshots**. With `training.train_embeddings=true`
(the default), every training update recomputes the projections from these saved
features. Forecast loss backpropagates through the output head, all RWKV/variate
layers, and into the projection and role embeddings. The optimizer includes these
parameters, and the final model checkpoint saves their learned state. Forecasting
uses that trained checkpoint. There is no need for a separately pretrained or
frozen embedding model. Consuming frozen snapshots is an explicit optional mode,
not the default.

Training now logs embedding tensor shape/device and skip-projection gradient norm
on every update. `embedding_optimization.json` records output dimension, projection
parameter count, number of updates with nonzero skip gradients and the L2 change
in skip weights from initialization. This is a practical diagnostic in addition
to existing gradient-flow tests; it is not a claim that one epoch learns a useful
representation.

## GPU observations and next steps

The original run's log records `Device=NVIDIA GeForce RTX 5070`, bf16 CUDA model
weights, 301,774,368 trainable parameters and 2.473 GiB peak PyTorch allocation.
Training explicitly requires CUDA and bf16 and uses the third-party CUDA kernels;
it cannot silently fall back to CPU. Validation, transformation and initial
embedding preparation run on CPU. Windows GPU stages launch the configured WSL
CUDA runtime.

Each optimizer update currently processes **one table/chunk**, with all its
variables and next-token positions. Much of the time series is only tens to
hundreds of tokens. The debug logger copies statistics to CPU, and the timing
helper synchronizes CUDA before and after each subsection. These are plausible
contributors to low utilization, not a measured utilization diagnosis. Memory
allocation alone does not measure compute occupancy. Inspect a CUDA/compute graph
and collect a CPU/CUDA profiler trace before changing kernels. PyTorch documents
[CPU/CUDA activity profiling](https://docs.pytorch.org/docs/stable/profiler.html).

Recommended order:

1. Overfit a small set of clearly structured, admitted series and track training
   and validation loss separately. Check precision and optimizer behavior: current
   weights **and Adam moments are bf16 without FP32 master weights**, which is a
   smoke configuration rather than a finalized pretraining optimizer.
2. Profile and batch tables with similar token/variable counts. This requires
   explicit masking of padded variables in cross-variable attention, not merely
   concatenating unrelated tables. Use quieter logging and fewer synchronizations
   for throughput runs; retain detailed audit mode for diagnosis.
3. Establish a genuinely disjoint dataset/generator-seed evaluation pool with
   naive, seasonal-naive, smoothing and Prophet baselines; use training-only period
   selection instead of assuming one period for all sources.
4. Then scale varied training data and optimizer updates. The original 111k rows
   and 100 updates are extremely small for 301.8M parameters. More data should
   include diverse periods, lag relationships, noise levels and covariate patterns,
   not merely more nearly identical tables. Retain an unfiltered evaluation suite
   to expose screening bias.

The prior unscreened run had **Prophet** lowest mean RMSE (36.43 versus 42.43
for smoothing and 43.27 for RWKV). **Smoothing had the lowest median RMSE**
(8.86 versus 9.56 for Prophet and 11.90 for RWKV). Thus the best method depends
on the metric, and the six selected plots need not represent aggregate performance.
Fixed-period smoothing is limited when tables have different periods. Prophet
likewise needs appropriate custom seasonality choices for arbitrary synthetic
periods; see the [official seasonality and regressor documentation](https://facebook.github.io/prophet/docs/seasonality%2C_holiday_effects%2C_and_regressors.html).

## Verification

Tests exercise strict OR admission, ties/zero baseline, missing/failed/nonfinite
scores, independent partial failures, stale content, duplicate keys, settings/seed
changes, transformed-record membership, lag exclusion, and unchanged final-test
splits. Validator tests include real Prophet fits and final-test isolation.
Existing patch/pipeline tests continue to cover causal features, padding, chunk
coverage and gradients. The machine-readable `evidence.json` records actual run
counts, artifact hashes, embedding-update evidence and test logs.

## Completed filtered pipeline smoke run

The filtered pipeline completed transformation, embedding preparation, one CUDA
training epoch and final-patch forecasting for **all 82 admitted tables**. It used
**5,888 supervised transitions in 82 optimizer updates**, and peak allocated GPU
memory was **2.470 GiB**. The 960-dimensional projection has 3,922,560 parameters; its skip
weights changed by L2 **0.091755**, with nonzero gradients on **82/82** updates.
This verifies actual embedding updates, not just inclusion in the optimizer.

All 82 held-out forecasts and both baselines succeeded; six plots were saved under
`results/447m_model/forecasts/`. This cohort differs from the earlier
100-table benchmark, so comparing its aggregate metrics with that earlier run
would not isolate an effect of screening. Raw scores are in
[forecast_summary.json](forecast_summary.json). The trained checkpoint is
`results/447m_model/training/model.pt`.

Validation: **5 validator tests, 2 admission/integration tests, 2 legacy pipeline
tests, and 4 CPU patch tests passed**; the CUDA patch test was skipped in the native
CPU test run. The real 301.8M filtered training and forecasting jobs passed on CUDA.
Ruff checks/formatting, Bandit on changed production files, and uv lock validation
passed. Full evidence and log paths are in [evidence.json](evidence.json).

# Covariate-informed Prophet in holdout testing

The holdout Prophet comparison previously used only the historical outcome.
RWKV, Chronos-2 and Timestral already received historical targets and time-matched
past covariates. This was a baseline simplification, not a Prophet limitation.

Prophet supports additional regressors, but requires their values over the
forecast horizon. The [official documentation](https://facebook.github.io/prophet/docs/seasonality%2C_holiday_effects%2C_and_regressors.html)
explains that these must be known or separately forecast. Its regressor mechanism
is a single-outcome regression, rather than a joint multivariate sequence model.

## Current default

`prophet.use_covariates=true` and `prophet.covariate_forecast="prophet"`:

1. At each selected origin, read only the same capped historical rows available
   to RWKV and the other foundation models.
2. Interpolate missing historical covariate values using historical rows only.
   Skip constant or entirely missing covariates; reject infinities.
3. Fit a univariate Prophet model to each usable historical covariate and predict
   the next-token regressor values. No future table rows are read for this.
4. Add the historical covariates as regressors to the outcome Prophet model, fit
   on historical outcomes and predict using those forecast regressor values.

Regressor forecasts are computed once per origin and reused if a table has
multiple targets. Internal regressor names avoid collisions with Prophet's
reserved columns. No explicit lag features are added. Nonconstant binary
covariates are forecast with the same method rather than using their actual
future values. This matches the validator's history-only regressor-forecast
approach, although seasonal settings, origins and metrics still differ.

The fast option `prophet.covariate_forecast="naive"` carries the last historical
covariate value forward. `prophet.use_covariates=false` restores univariate
Prophet. With no usable covariates the model naturally remains univariate.
The naive *outcome* benchmark is unchanged in all modes.

Each origin records the source covariate names used/skipped. Benchmark metadata
records the covariate mode so new metrics can be distinguished from the old
univariate comparison. Scores can improve or worsen: uncertain future regressor
forecasts add their own errors. Runtime increases from one Prophet fit to up to
one fit per usable covariate plus the outcome fit, at every selected origin.

Stan output files are written in per-fit temporary directories under
`.cache/prophet_holdout` on the project drive and removed after prediction.
The optional `holdout.prophet_temp_dir` can override that location. Detailed fit
timing stays at DEBUG and the every-100-table INFO progress policy is retained.

## Verification

Tests cover regressor forecasts being supplied to the outcome model, matching
historical lengths, no supplied future outcomes, imputation, constant/missing
column skipping, internal names, univariate/naive overrides and temporary-output
cleanup. Logs: `logs/20261029_prophet_regressor_tests.log` and
`logs/20261029_prophet_covariate_protocol_tests.log`.

A real Prophet smoke uses 64 historical rows, two varying covariates and one
constant covariate, and predicts a 16-row horizon. Result: `smoke.json`; detailed
log: `logs/20261029_prophet_regressor_smoke.log`. It exercises actual covariate
and outcome fits without loading the GPU models or modifying production results.
Full regression log: `logs/20261029_prophet_covariate_full_tests.log`.
All 164 regression tests completed: 157 passed and seven CUDA-only tests were
skipped in the CPU environment. The real Prophet smoke passed all three fits
(two covariates plus the outcome). Ruff and Bandit checks passed. The benchmark
default test now checks the fallback configuration independently of the user's
active model list; the user's four-model list and 2% sample were preserved.

Existing result files describe the previous run; a new evaluation is required
to compare the covariate-informed model. A running worker retains its loaded
implementation and does not change behavior mid-run.

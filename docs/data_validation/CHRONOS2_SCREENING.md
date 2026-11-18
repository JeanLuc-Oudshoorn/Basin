# Chronos-2 Small admission screening

When screening is enabled, Stage 02 uses `validation.engine="chronos2"` and the 28M-parameter
`autogluon/chronos-2-small` model, pinned at
`ddec01313e50b6bc58ebaa92ede81bc24a3d9f9a`. The installed
`chronos-forecasting==2.3.2` package supplies its pipeline.

A successfully evaluated table passes exactly when:

```text
chronos2_mae <= 1.15 * baseline_mae
```

The tolerance is `validation.max_relative_degradation=0.15`. Equality passes.
Naive repeats the last observed target for the entire screening horizon.
Both errors use the same actual outcomes, and Chronos uses its median forecast.
MASE and its historical scale remain diagnostic columns. Where the scale is
positive, the relative MAE and MASE tests are equivalent. A constant context
does not automatically pass: it must meet the MAE rule. If naive MAE is zero,
Chronos MAE must also be zero. Invalid data/evaluation failures are not admitted.

## Skipping screening

Set `validation.enabled=false` to bypass the predictive screening gate. This is
now selected in `configs/config.json` for processing large generated cohorts.
`transform.py` admits all selected source tables without reading screening
reports, hashing sources for admission, or loading a screening model. It records
`screening_skipped=true`, the reason and candidate/admitted counts in
`validation_selection.json` and the transform manifest. Previously rejected and
never-screened tables are both eligible. Normal transformation checks for valid
data, input roles and sufficient sequence length still apply.

You can proceed directly to transform and patch indexing. If you run the validation
script anyway, it logs the skip and exits successfully without scanning tables
or starting WSL. It does not fabricate forecast scores or change the latest
screening report. Existing reports are preserved.

Set `validation.enabled=true` to restore screening with the configured model and
15% rule. `auto_run` is a separate control: when screening is enabled,
`auto_run=false` requires an existing current report; it does not let all tables
pass. Skipping screening also does not disable the training/holdout evaluation.

## Inputs and temporal boundaries

The screener uses `data.targets`, `time_column`, `past_covariates`,
`known_covariates` and `excluded_column_patterns` from the training configuration.
Exactly one target is currently required. With `past_covariates="auto"`, all
columns other than target, time, known covariates and excluded columns are
historical covariates. The configured lag-column exclusions also apply here.
Explicit roles must be disjoint and must not include excluded columns.
The current configuration has no known future covariates.

Reserve the final `data.horizon=16` rows for downstream final-token evaluation.
Split the remaining prefix at `floor(validation.train_fraction * prefix_length)`.
Use up to `validation.chronos2.context_length=2048` historical observations ending
at that cutoff, then forecast the next `validation.forecast_horizon=16` points
(or fewer if the available prefix ends earlier). The target and historical-only
covariates are sliced at the cutoff. Only columns explicitly declared as known
covariates may supply values over the forecast horizon. Reserved target values
never enter screening. Input histories must be finite and times strictly increasing.

## Execution and outputs

```powershell
uv run python src/02_data_validation/validate_generated_data.py
uv run python src/03_transforming_input_data/transform.py
uv run python src/04_patch_indexing/index_patches.py
```

One model is loaded once, with CUDA selected automatically through the existing
Windows-to-WSL launcher. Weights stay in `.cache/huggingface` on the project drive.
Native source discovery and its five-second heartbeat remain enabled.
Tables are prepared in bounded groups and batched by forecast horizon and
covariate role counts. Covariate lookup names are normalized inside each table,
so different generated names do not fragment otherwise compatible batches;
reports retain original column names. Pipeline `batch_size=64` counts target and covariate series;
it is not a promise of 64 multivariate tables per GPU batch. Cross-learning
between different tables is disabled. Model group attention can use covariates
within a table. No per-table model fitting is performed.

Reports retain timestamped cohort folders and the atomic `latest.json` pointer.
New score columns are `chronos2_mae` and `chronos2_mase`, with baseline errors,
source hashes, covariate column lists/counts, temporal boundaries and decisions.
Screening signatures bind model revision, input roles, exclusions, tolerance,
split, context, horizon and scaling settings. Bolt/Prophet scores cannot be reused
as Chronos-2 scores. `--retry-errors` only reuses unchanged successful results
under the same policy. `transform.py` runs the new screener if its report is
missing or stale, unless `validation.auto_run=false` requires an explicit run.

An already running Bolt process keeps its loaded configuration and model.
Restart validation to use Chronos-2; changing the files does not switch a live
process. Existing reports remain in their original run folders. Rerun transform
and index_patches to apply the new selection before training.

## Verification

`tests/test_chronos_screening.py` checks the 15% boundary, zero-baseline and
constant-history behavior, covariate schema batching, historical/future availability,
reserved-tail isolation, invalid covariates, retry/gate integration, and signature
binding. `scripts/verify_chronos_screening.py --manifest <source-index.json>` runs
the pinned model on four actual tables without updating production admission
reports. Its default output is `results/chronos2-screening-smoke.json`.

Model and API references: [model card](https://huggingface.co/autogluon/chronos-2-small),
[Chronos-2 pipeline](https://github.com/amazon-science/chronos-forecasting/blob/main/src/chronos/chronos2/pipeline.py).

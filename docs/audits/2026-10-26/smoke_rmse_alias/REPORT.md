# Authorized smoke-test compatibility report

The user explicitly requested using the completed RMSE selection for a large
training smoke test, exposing its values under MAE column names rather than
repeating Prophet validation. No MAE measurements or new forecasts were computed.

`scripts/prepare_rmse_smoke_report.py` verified the completed source report's
checksum, source selection, screening settings, row count and signatures. It
copied the RMSE numbers into `baseline_mae` and `multivariate_mae` and retained
the original RMSE columns. The original report and summary under `selected_dates`
remain unchanged. The source report checksum is
`b0afe00bffff490fda2a4d44ebbc138cc2db15179a2424510d64a14815e7649b`.

The compatible report is registered as the selected cohort's latest report:

`results/02_data_validation/2026-10-25_231326cf77e5/20261026T183136_068973Z_smoke_rmse_alias_804ddcf0/metrics.parquet`

All 150,000 records are retained, including exactly the same 107,828 admissions
and 42,172 rejections. The report stores `smoke_test_only=true`,
`actual_scoring_metric=rmse`, `metric_alias=rmse_as_mae`, the original signature,
and the source report path/checksum. The current signature is used only to make
this explicitly authorized compatibility report usable by transformation.
Settings, summary, status and the latest pointer all record its provenance.

Transformation retains the smoke-test provenance in its selection audit and
marks next-patch records with `screening_smoke_test_only`. Embedding retains those
record fields. Future validator runs still compute actual MAE; retry mode never
reuses marked smoke rows as MAE measurements.

The 37 focused validation, date-selection, transformation, token-inventory and
holdout tests pass. An added integration test verifies smoke provenance survives
transformation and that proper validation refits all marked rows. The full
dataset readiness check and original-checksum verification are recorded in
`results/02_data_validation/smoke-readiness.log`. The transformation/training
stages themselves were not started by this conversion.

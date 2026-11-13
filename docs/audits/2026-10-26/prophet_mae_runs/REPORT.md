# MAE screening and independent dated reports

The next validation run admits a table when multivariate Prophet MAE is finite,
nonnegative, and no greater than last-observation-carried-forward MAE. Ties pass;
failed fits do not. MAE is the mean absolute pointwise forecast error over the
same chronological screening horizon. RMSE remains a secondary diagnostic.
Improvement percentages and admission flags now refer to MAE. The downstream
transformation gate uses the same rule. Screening schema version 3 prevents
reuse of older RMSE-based scores as MAE scores.

Default configuration enables `validation.run_folders` and uses
`results/02_data_validation` as the results root. Each cohort gets a folder
containing the canonical selected dates and a source-selection identifier.
Each execution, including retries, creates its own UTC timestamp/UUID subfolder.
This preserves same-day reruns as well as runs on different dataset dates.

Each run stores final Parquet metrics, an append-only progress CSV, settings
including the full config and exact dataset dates, a summary, and a status file.
Execution dates are recorded separately from synthetic-data dates. Status updates
every 100 completions and at termination. An atomic `latest.json` pointer is
published only after metrics, summary and final status have been written.
Transformation resolves the latest finished report for the configured cohort;
an interrupted run does not change this pointer. Legacy explicit-path mode is
available through `run_folders=false` and continues archiving replaced files.

The existing background storage-repair process was not restarted or altered:
its already-loaded code and config continue the original RMSE run, writing the
existing `selected_dates` reports. New MAE runs use the new folder layout.

Verification covers a forecast winning on MAE while losing on RMSE, downstream
admission despite contrary RMSE columns, rejection of RMSE-only reports,
same-date reruns preserving earlier bytes, separate date selections, automatic
downstream report discovery, independent retry outputs, and failed runs leaving
the prior pointer intact. Existing parallel, leakage, disk cleanup, and retry
checks also pass.

A real three-table CLI smoke run completed with MAE, two admitted tables and no
failed fits. Its report is under `results/prophet-mae-smoke/2026-10-25_cff39287b220/`.
The transformation CLI then successfully consumed the cohort's latest report.
Ruff lint/format and Bandit checks pass for the changed production files.

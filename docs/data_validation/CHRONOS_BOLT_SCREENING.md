# Chronos-Bolt Small admission screening

This describes the retained legacy `chronos_bolt` engine. The default is now
[Chronos-2 Small with covariates and a 15% tolerance](CHRONOS2_SCREENING.md).

Stage 02 now defaults to `validation.engine="chronos_bolt"`. A successful table
is rejected for low predictive information only when:

```text
Chronos-Bolt MASE > (1 + max_relative_degradation) * naive MASE
```

The default tolerance is `0.1`: exactly 10% worse still passes. Better forecasts
and ties pass. Naive means the last historical observation carried forward
throughout the forecast horizon. If naive MASE is zero, only a zero model MASE
passes. Missing/invalid data or evaluation errors are reported separately; they
are not admitted as successful forecasts.

MASE uses the mean absolute lag-`m` difference of the historical target context
as its denominator. Both methods use the same scale and actuals, with
`validation.mase_seasonality=1` by default. For positive scales the comparison
is also algebraically equivalent to comparing their MAEs at the same tolerance.
A constant historical context makes MASE undefined: a valid table is retained,
marked `decision="undefined_mase_keep"`, and its MASE fields remain null. This
does not silently substitute another denominator or claim that Bolt won.

## Temporal protocol

Read only the configured target and time columns. Require finite outcomes and
strictly increasing finite times; do not use SCM metadata, lag columns or future
covariates. Reserve the final `data.horizon` observations before screening so
the downstream final-token validation outcomes never influence admission.
Split the remaining prefix at `floor(train_fraction * prefix_length)`, with
`train_fraction=0.8`. Use up to `chronos_bolt.context_length=2048` historical
observations ending at that split and forecast the next
`validation.forecast_horizon=16` observations, or however many remain before
the reserved tail. The remainder of the prefix beyond that screening forecast
is not scored. Thus admission tests one bounded forecast request per table,
rather than forecasting the entire last 20% as the old Prophet screener did.
The report records source length, split, context length, forecast start/stop,
and actual forecast length for auditing.

[Chronos-Bolt Small](https://huggingface.co/amazon/chronos-bolt-small) has
47,718,016 parameters. It is a **univariate** model; this filter does not test
whether a table is predictable only through covariates. Model training and
holdout foundation-model comparisons continue to use their covariates. The
official [adapter implementation](https://github.com/amazon-science/chronos-forecasting/blob/main/src/chronos/chronos_bolt.py)
supports batched, variable-length target histories and median forecasts.

## Configuration and execution

All settings live in `configs/config.json` under `validation`:

```json
{
  "engine": "chronos_bolt",
  "auto_run": true,
  "max_relative_degradation": 0.1,
  "mase_seasonality": 1,
  "forecast_horizon": 16,
  "train_fraction": 0.8,
  "chronos_bolt": {
    "model_id": "amazon/chronos-bolt-small",
    "revision": "772f3d25d38aec6d914c8949dab4462e2d46f5d8",
    "batch_size": 64,
    "context_length": 2048,
    "device": "auto",
    "cpu_threads": 4,
    "cache_dir": ".cache/huggingface"
  }
}
```

One resident model processes bounded batches of independent target histories.
Short tables with different forecast lengths are grouped by horizon. There are
no spawned model copies. `validation.workers` applies only to legacy Prophet
screening. `device="auto"` chooses CUDA when available and CPU otherwise;
the standard validation/transform CLI routes Windows execution through the
existing WSL GPU launcher. Set `device="cpu"` for a CPU-only run.
The installed `chronos-forecasting` dependency already provides Chronos-Bolt;
no new package is needed. Weights download anonymously into the configured
project cache on first use.

```powershell
uv run python src/02_data_validation/validate_generated_data.py
uv run python src/03_transforming_input_data/transform.py
uv run python src/04_creating_embeddings/embed.py
```

On Windows, the validation and transformation entry points discover source files
natively before launching WSL. They save a run-local relative file list under
`.cache/source_indexes` and pass it to the WSL process. WSL loads that list rather
than recursively scanning every table directory across `/mnt/e`. The index is
bound to the configured source root and selected dates, and each new launch
creates a fresh snapshot. Actual table files are still opened and hashed during
screening; missing or changed content is not hidden by the index. Nested table
folders retain the same recursive selection semantics.

Discovery logs its phase immediately and emits a heartbeat every five seconds.
The first batch and model loading also log before starting their work. For the
150,000-table October 31 cohort, native discovery/index creation took 17.25 s and
WSL loaded the index in 0.95 s; the prior WSL recursive walk took about 11.5 min.
These are observations during the live run, not controlled cold-cache benchmarks.
The current run can continue normally once discovery finishes; the new index
handoff takes effect on subsequent launches, with no new configuration required.

`transform.py` automatically validates when the selected cohort's report is
missing, has a different screening signature, lacks required Bolt columns, or
contains changed/missing sources. Set `auto_run=false` to require a current
report explicitly. The old Prophet reports cannot be reused as Bolt scores.
`--retry-errors` reuses unchanged successful rows with the same signature and
reevaluates errors, changed sources and missing tables. CLI forwarding preserves
this flag when using WSL. Legacy Prophet support remains explicit compatibility
mode, not the default.

## Audit trail

Each validation run keeps its existing timestamped cohort folder and atomic
`latest.json` pointer. Reports include `baseline_mase`, `chronos_bolt_mase`,
`mase_scale`, raw MAEs, `mase_defined`, `admitted`, `decision`, source SHA-256,
model ID, screening signature and temporal boundaries. The signature binds the
model/revision, metric, threshold, split, horizon and scaling settings, so changing
policy invalidates prior decisions. `settings.json` preserves the full config;
`summary.json` includes actual model parameter count, device and load duration.
Per-table progress is durable in CSV; INFO reports every 100 tables and at
completion with elapsed time and ETA. Detailed records are DEBUG in run logs.

Changing the filter affects newly transformed/embedded data; it does not modify
an existing model. Rerun transform and embed before training on the new cohort.
Existing models remain testable using their own saved split and schema policy,
even though their old screening signature differs from the current config.
The complete verification trail is in the
[startup and screening audit](../audits/2026-10-30/bolt_screening_startup/REPORT.md).

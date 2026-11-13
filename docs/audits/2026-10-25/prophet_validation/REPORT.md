# Faster Prophet screening

Stage 02 now fits only the multivariate target model. Stage 03 admits a table
when that fit succeeds and its finite, nonnegative RMSE is no greater than the
last-observation baseline. Exact ties, including zero-error ties, pass. The
reserved final test horizon and exclusion of lag regressors remain in effect.

`validation.workers` controls table-level process concurrency: `null` chooses
up to four workers, `1` runs serially, and a positive integer selects an explicit
count (capped by the number of tables). Spawned workers inherit native thread
limits of one. Jobs are bounded to twice the worker count, results are collected
as they finish, and final reports retain input order. Worker count does not
affect the screening signature. The screening schema version increased, so
reports generated under the previous policy must be regenerated.

Progress now appends CSV rows instead of rebuilding the entire accumulated
Parquet report after every table. The configured cohort contained 38,897 table
folders when inspected, making the former quadratic reporting cost significant.
Stage 02 writes `metrics_progress.csv` beside the final `metrics.parquet` and
`summary.json`. The standalone batch API appends to `metrics.csv` during the
run, then writes final sorted CSV and Parquet reports.

The current config already disables uncertainty simulation and daily, weekly,
and yearly calendar seasonalities. No holiday data or country holidays are
supplied. Prophet only includes holidays when explicitly configured; see its
[holiday documentation](https://facebook.github.io/prophet/docs/seasonality%2C_holiday_effects%2C_and_regressors.html).
No optimizer tolerances, trend capacity, or regressor forecasting assumptions
were changed to obtain the speedup.

## Measured sample

The same first 24 sorted tables from `2026-10-25` were evaluated with the
configured `forecast` covariate mode, seed 7, 80% training split, and 16 reserved
tail rows. This Windows machine has 12 logical CPUs (6 physical cores), Python
3.12.13, Prophet 1.5.0, and CmdStanPy 1.3.0.

| Evaluator | Seconds |
| --- | ---: |
| Original serial, two target fits | 9.86 |
| Multivariate target only, serial | 7.63 |
| Multivariate target only, four workers | 3.32 |

The observed overall speedup was **2.97x**. All 24 tables completed successfully,
and multivariate RMSE matched the original evaluator exactly (maximum absolute
difference 0). Source hashes and per-table scores are in
[benchmark.json](benchmark.json).

These are single sample timings of evaluation, including Parquet reads and
source hashing, rather than a full-cohort throughput estimate. The original
timer includes its first Prophet import; the revised serial timer starts after
imports, and the concurrent timer includes process startup and worker imports.
The concurrent sample overlapped short test runs. Reporting costs are excluded.

With K usable covariates, the configured `forecast` mode still needs K
covariate models plus one target model per table. The previous evaluator needed
K+2 fits. `known_future` avoids covariate fits but assumes future covariates are
available, changing the benchmark's meaning.

## Verification

The 14 focused tests cover chronological splitting, reserved-tail isolation,
lag/constant regressor exclusion, covariate forecasting without future inputs,
one target fit, zero-baseline ties, failed fits, multivariate-only downstream
selection, source/config freshness, worker environment restoration, serial vs
spawned-worker numerical agreement, and parallel reporting across date cohorts.
Ruff lint and formatting checks pass for the changed Python files.
The actual Windows CLI also completed a separate three-table smoke run with
two spawned workers, producing three successful records and consistent progress,
final metrics, and summary files.

```powershell
python -m unittest discover -s tests -p test_prophet_validator.py
python -m unittest discover -s tests -p test_learnability_gate.py
python -m unittest discover -s tests -p test_generated_dates.py
```

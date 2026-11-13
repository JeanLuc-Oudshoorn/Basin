# Covariate-aware Chronos-2 Small screening

The default validation engine is now `chronos2`, using
`autogluon/chronos-2-small` pinned to commit
`ddec01313e50b6bc58ebaa92ede81bc24a3d9f9a`. The pin was read from the model
repository API and verified by loading the actual weights in the existing WSL
environment with `chronos-forecasting==2.3.2`. No dependency change was required.

The admission rule is `chronos2_mae <= 1.15 * baseline_mae`, including equality.
Both forecasts share the same target outcomes and naive means persistence of
the final historical target. MASE remains diagnostic; constant historical
contexts must meet the MAE criterion rather than automatically passing.

The validator shares the existing bounded screening loop and temporal protocol,
but supplies target, historical covariates and explicitly known future covariates
to Chronos-2. Current config has no known future covariates. Input roles and
lag-column exclusions match the transformer. Original column names are retained
in reports; numeric model inputs normalize lookup names to batch independent
tables with different original names. Batches use matching forecast horizons and
past/known covariate counts. Cross-table learning is disabled.

The transform gate, summaries, retries, WSL forwarding and native source indexes
all support the new engine. Screening signatures now also bind input roles and
exclusions. Old Bolt/Prophet reports cannot be reused as Chronos-2 scores. Existing
dated run folders are preserved; an old live process keeps its loaded policy.
The legacy Bolt engine and its verification script remain available.

## Verification

- Windows: 29 tests pass across Chronos-2, legacy Bolt, source discovery,
  generated-date selection and legacy Prophet gate integration.
- WSL: all six Chronos-2 screening tests pass.
- Ruff lint/format and Bandit pass on changed Python files.
- Actual GPU smoke: all four selected generated tables forecast successfully;
  three pass and one fails the new threshold. The model reports 27,934,624
  parameters on CUDA. Covariate counts are 0, 1, 1, 1. The three covariate tables
  have different original names and were forecast in one compatible group.
- Latest real smoke took 40.49 seconds including 38.11 seconds of model loading.
  Logged prediction calls took 1.965 seconds for the first univariate table
  (including inference warmup) and 0.145 seconds for the three covariate tables.
  This small smoke is an integration check, not a cohort throughput benchmark.

Evidence: `results/chronos2-integration-tests.log`,
`results/chronos2-wsl-tests.log`, `results/chronos2-screening-smoke.json` and
`results/chronos2-screening-smoke.log`. The smoke script only reads raw sources
and writes its own diagnostic output; it does not update the production
admission report or latest pointer. The complete cohort was not rerun here.

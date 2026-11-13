# Investigation: 307M forecasting accuracy regression

2026-11-02. Read-only checkpoint investigation plus a disposable controlled fine-tuning experiment. Production weights, configuration and holdout outputs were not modified.

## Findings

There are two separate problems: a genuine learned-model accuracy regression on the matched typical-case sample, and extreme MASE sensitivity to nearly constant histories. CUDA graph replay is not the cause in the matched tests. The archived 447M model remains competitive on the same new 2% sample where the 307M model is substantially worse. However, the old model fails even more severely on some new near-constant cases outside that sample; it is not universally better on the new distribution.

The change was broader than reducing 25 layers to 17. The new model also trains a 16-patch auxiliary forecasting task with 50% loss weight, uses different training data/screening, changes batch ordering, and exports its best epoch after four passes instead of seven. The existing checkpoints alone cannot assign a unique causal percentage to model size versus those training changes.

The attached `previous_holdout.txt` is the **new 307M model's 20% holdout run**, not the competitive historical 447M run. Its values match the recovered 307M journal. The actual archived 447M result is in `results/447m_model/holdout_test/test_metrics.json`.

## 1. Same new tables, same origins, same labels

260 tables from 2026-10-31; 1,295 forecast origins; 18,851 scored observations, of which 18,707 have defined MASE. Competitor forecasts are reused from the saved run, so every row is compared against the exact same observations.

| Model / execution | MASE | RMSE | MAE |
|---|---:|---:|---:|
| 307M native, original attention | 2.283011 | 61.8391 | 18.3940 |
| 307M compiled, no replay | 2.283471 | 61.8522 | 18.3966 |
| 307M fast kernel, no replay | 2.283471 | 61.8522 | 18.3966 |
| 307M fast kernel + replay | 2.283471 | 61.8522 | 18.3966 |
| 447M compiled | 1.715314 | 40.0995 | 12.4752 |
| Chronos-2 | 1.723114 | 38.5686 | 12.3315 |
| Timestral | 1.628067 | 37.3796 | 11.7456 |
| Naive | 2.649944 | 94.1034 | 24.5044 |

On these identical cases the new model's MASE is 33.1% higher and RMSE 54.2% higher than the archived model's. This is a real regression. It persists with replay disabled and with the original RWKV forward path. Raw RMSE across the two different historical datasets is not a matched comparison.

## 2. Inference and checkpoint checks

Four execution variants were checked for each model and cohort: original RWKV + original variable attention, original RWKV + compiled variable attention, the forward-only recurrence without graph replay, and that recurrence with replay. Both models used an eight-profile inference cache, including cache saturation/fallback, different table lengths and covariate counts.

Across **17,336 forecast executions**, all scored median predictions from replay, the forward-only path without replay, and the standard compiled path match exactly. Original attention introduces small BF16 differences, but its aggregate scores show essentially the same regression. This does not prove bitwise equivalence for every possible input or every unrecorded quantile; it does directly rule out replay as the explanation for the observed MASE/RMSE gap.

Both exported checkpoints match every tensor in their corresponding `best_weights.pt`: selected epochs 4 (307M) and 7 (447M). Embeddings were jointly trained; the new run records nonzero embedding gradients at all 73,050 updates. The wrong checkpoint, frozen embeddings and dropped variable attention do not explain the result.

The old raw 2026-10-25 directory is absent. Historical inputs were recovered from `data/02_processed/447m_model/windows.pt`. It contains full target history and covariates only up to the reserved final 16 observations. **204 final partial-patch origins requiring unavailable covariates were excluded**, without imputing them. The comparison below therefore uses 872 recoverable origins / 13,952 observations across 216 old tables, not the complete old run.

| Model on the recoverable old cases | MASE | RMSE |
|---|---:|---:|
| 307M | 2.367522 | 84.6567 |
| 447M current execution | 1.891076 | 71.7877 |
| 447M saved historical predictions | 1.891076 | 71.7877 |

The archived checkpoint reproduces its historical predictions exactly on these recovered inputs. The 307M model is worse on both old and new tables, so the new evaluation cohort alone is insufficient to explain the regression.

## 3. Why the 20% headline is alarming

The complete 20% result was recovered from `logs/rwkv7_holdout_test_20261102_054242_743315Z_pid3696_a6cf0231.jsonl`; it had been overwritten in the shared output directory by the subsequent 2% run. It contains 2,594 tables and 186,640 scored observations.

The largest origin contributes **64.3%** of total RWKV scaled absolute error; the largest five contribute **94.0%**. These are genuine scored errors, amplified by nearly zero historical naive errors, not a JSON aggregation bug.

| Diagnostic exclusion only | RWKV MASE | Naive | Chronos-2 | Timestral |
|---|---:|---:|---:|---:|
| Drop top 0 RWKV contributors | 38.2141 | 24.9356 | 23.9585 | 23.8585 |
| Drop top 1 RWKV contributors | 13.6309 | 3.3210 | 2.3438 | 2.2442 |
| Drop top 5 RWKV contributors | 2.3100 | 2.6929 | 1.7541 | 1.6561 |
| Drop top 10 RWKV contributors | 2.2021 | 2.5686 | 1.6640 | 1.5848 |

These exclusions are sensitivity diagnostics, **not replacement scores**. The original unfiltered scores remain the valid record of the configured metric. The 2% and 20% samples are different populations; the latter includes rare pathological cases missing from the former.

For `table_285516885900_019943`, origin 144 has historical naive MAE about `5.82e-7`; a future change produces absolute errors around 0.15 for all models, yielding MASE near 250,000. At origin 32 on the same table, historical naive MAE is `4.88e-7`. RWKV's absolute MAE is only 0.0603 in original units, but becomes MASE 123,720; naive MAE is approximately `1.16e-5`. This second case is a real near-constant-series forecasting weakness, not solely common unpredictable future movement.

`mase_scale()` correctly uses historical lag-one naive MAE and excludes exact-zero denominators. Nonzero denominators close to zero remain extremely influential. In addition, `tokenize_next()` replaces historical standard deviations at or below `1e-5` with **1.0**. On those prefixes a modest learned normalized residual can become much larger than the tiny data variation. This normalization rule predates the new model, but the new data makes the corner case visible. Changing it only at inference would change the trained input/output representation and must be evaluated carefully.

The MASE definition and historical-naive scaling follow [Forecasting: Principles and Practice, scaled errors](https://otexts.com/fpp3/toolbox.html#scaled-errors). Our near-zero sensitivity findings follow from the formula and the measured prefixes. MASE 1 refers to historical one-step naive error; it does not force a 16-step out-of-sample carry-forward baseline to score 1.

### The old model also fails the new pathological cases

Both original frozen checkpoints were run on the 20 worst 307M origins. At origin 32 of `table_285516885900_019943`, the 447M model has absolute MAE **1.1692** and MASE **2,398,415**, versus 307M MAE **0.0603** and MASE **123,720**. At origin 144 both are poor: 447M MASE **276,352**, 307M **284,785**. The 307M outlier predictions reproduce the saved 20% journal metrics, confirming that the same cases were reconstructed.

The 20 measured origins' nonnegative scaled errors, divided by **all 185,344 MASE-eligible observations** in the 20% run, already contribute **231.763** to the 447M model's hypothetical full-20% MASE. Therefore its complete-20% MASE would be **at least 231.763**, even if it predicted every remaining point perfectly. This is a lower bound, not a measured complete 447M 20% evaluation. For comparison, those same 20 origins contribute 36.079 of the new model's measured 38.214.

Consequently, reverting to the 447M checkpoint would not fix the catastrophic 20% mean MASE. The changed data distribution plus the normalization/scaling corner case is sufficient to produce an enormous headline for the old model too. The ordinary 2% comparison and the near-constant-tail problem must be reported separately; do not extrapolate the favorable old-model ranking in that small sample to the whole new holdout.

## 4. Training changes and evidence

| Item | Archived 447M run | New 307M run |
|---|---|---|
| Capacity | 447,482,144; 25 layers | 307,307,808; 17 layers |
| Width / patch / history cap | 1024 / 16 / 2048 | Same |
| Training data | 2026-10-25 | 2026-10-31, broader generator |
| Admission | Prophet RMSE <= naive RMSE, through explicit smoke-test MAE aliases | Covariate-aware Chronos-2 Small MAE <= 1.15 × naive MAE |
| Admitted / candidates | 107,828 / 150,000 (71.9%) | 129,671 / 150,000 (86.4%) |
| Best epoch / update | 7 / 85,127 | 4 / 58,440 |
| Supervision | Next patch only | 50% next-patch + 50% sampled longer-lead pinball |
| Output training | One patch | Leads up to 16 patches (256 observations), shared head with lead offsets |
| Physical microbatch / optimizer batch | Up to 4 / 8 | Up to 8 / 8, same 1536 variable-token cap |
| Shuffling | All batches shuffled | Shuffled blocks up to 128 similar batches |
| Training CUDA replay | Not used | Disabled for the successful run |
| Learning rate reached | 1e-4 | 1e-4, never reduced |

The old screening report explicitly records `actual_scoring_metric=rmse`, `metric_alias=rmse_as_mae`, `smoke_test_only=true`. It must not be described as a true Prophet MAE experiment. The new checkpoint records **Chronos-2 Small**, not Chronos-Bolt; its recorded tolerance is **15%**, not 10%. [The official model card](https://huggingface.co/autogluon/chronos-2-small) identifies this small model as approximately 28M parameters. The benchmark's Chronos-2 is the larger approximately 120M model. These are confirmed provenance differences, not configuration changes made by this investigation.

The extra forecasting objective was added with the direct multi-patch head. It changes what the backbone and shared decoder learn, despite leaving the first-patch output shape unchanged. For stochastic future covariates or sharp regime changes, predicting distant patches is a different task from predicting the next patch. Its effect must be ablated rather than attributed to inference compilation.

New training loss is not comparable to the old next-patch-only loss. In epoch one, the new median update loss is 0.421, but the mean is 12.45; the 99th percentile is 381.6 and maximum 26,111.9. The largest 1% of updates account for 69.9% of logged point-weighted loss, versus 11.5% in the old run. Pinball gradients with respect to predicted normalized values are bounded, so this loss concentration does **not** imply those cases contribute 69.9% of parameter gradients. It does show that the loss average is dominated by a very different, heavy-tailed target distribution.

Validation monitors only the final next patch, whereas training includes early, shorter prefixes and the added long-lead objective. The new best validation pinball loss is 0.1909, versus 0.1288 on the old, different training cohort; those numbers cannot isolate capacity loss.

The next-patch gap grows with context on the matched new sample: above 1024 history points, MASE is 2.017 for 307M versus 1.204 for 447M. At the first 32-point origin it is 2.273 versus 2.108. The regression is not restricted to newly introduced near-constant starts.

Early stopping and scheduling currently conflict: `ReduceLROnPlateau(patience=1)` needs two bad validations to reduce LR, but early stopping terminates after the first bad epoch. Both recorded runs therefore finish without any LR reduction. The new model also has about 31% fewer optimizer updates at its selected best checkpoint, alongside its smaller capacity and changed supervision.

## 5. Controlled short intervention

Both arms start from the exact same best 307M state, use fresh AdamW at `1e-5`, the same seed, the same 1,024 shuffled batches from 8,192 **training-only** tables, the same clipping and physical microbatch limits, and no training/inference graph replay. The final validation patch is excluded from these training inputs. Only the auxiliary loss weight differs between arms. Production model files are never saved or replaced.

| Checkpoint / auxiliary weight | New-cohort MASE | New RMSE | Old-subset MASE | Old RMSE |
|---|---:|---:|---:|---:|
| Original 307M | 2.283471 | 61.8522 | 2.367522 | 84.6567 |
| Fine-tune weight 0.5 | 2.236317 | 61.4188 | 2.356871 | 81.3643 |
| Fine-tune weight 0.0 | 2.094393 | 55.0695 | 2.181365 | 79.0355 |

During this continuation, removing auxiliary weighting improves new-cohort MASE by **6.35%** and RMSE by **10.34%** relative to the control arm. That is positive causal evidence that the 50% auxiliary objective contributes to the next-patch tradeoff. It does not quantify its from-scratch effect or close the full gap to 447M.

This is a short continuation experiment, not a controlled from-scratch retraining or a model-size ablation. Both arms also change learning rate, reset optimizer state and use globally shuffled batches relative to the production run. Their difference isolates auxiliary weighting **during this continuation**; improvement over the original checkpoint cannot be assigned uniquely to LR, renewed optimization or shuffling. It does not establish which objective would win after full training. Test results are diagnostic and must not become an unreported hyperparameter-selection loop.

## 6. Causal conclusion and next actions

**Confirmed:** the large 20% headline is driven by near-zero MASE denominators and real near-constant-series forecast errors; those new pathological cases also defeat the old model, sometimes more severely. Separately, the 307M learned weights genuinely underperform on the matched 2% sample and recoverable old cases. Replay and the inference-only recurrence are exonerated for the observed scores. Simply reverting to 447M or generating more of the same data is not a demonstrated remedy for both problems.

**Not identified separately:** the share caused by parameter reduction, new training distribution/screening, auxiliary supervision, changed ordering, or shorter optimization. These were changed together. No honest inference-only experiment can determine their independent training effects. The short intervention is evidence about continuation, not proof that size is harmless or that the auxiliary task alone caused the regression.

Recommended next baseline: keep the inference speedups, freeze the raw tables/split/metric origins, and train a 307M **next-patch-only** model with the old supervision and ordinary batch shuffling. Give LR reduction at least one epoch to work before stopping. Compare its learning curves and the frozen 447M checkpoint on the same scorecard. Then add the long-horizon objective, new screening and blocked ordering one at a time; use multiple seeds before attributing small differences to model size. Restore/test an old-data 307M baseline from archived processed tensors if isolating the size change is the immediate priority.

Keep standard MASE unmodified in the audit, but also show per-table medians, percentiles, near-constant-prefix counts, absolute errors and the largest scaled-error contributors. Do not silently discard difficult tables or epsilon-clip denominators and continue calling that the same MASE. A regularized variant would need a separate name and consistent treatment across all models. Review constant-prefix normalization on a dedicated test suite before retraining; do not switch its representation only for inference.

Preserve each holdout run in a timestamped directory or snapshot its final metrics. The current shared output path overwrites the previous sample's aggregate file; the JSONL journal was essential to recover the 20% experiment.

## Plots and evidence

- [Matched forecast examples](matched_forecast_examples.html): four deliberately selected regressions, not a representative random sample.
- [Near-constant outliers](near_constant_outliers.html): both original trained checkpoints on the largest contributors, in absolute units.
- `*_on_*.json` / `.parquet`: matched metrics and every scored median prediction for all four execution variants.
- `*_equivalence.parquet`: per-origin quantile differences; old-cohort experiments additionally record full quantile comparisons against the compiled path.
- `recovered_20_percent_metrics.json`, `twenty_percent_diagnostics.json`, `twenty_percent_worst_origins.parquet`: recovered full 20% result and sensitivity analysis.
- `objective_ablation_settings.json`, `objective_ablation_results.json`, `objective_weight_*`: exact training-only sample and disposable intervention results.
- `export_verification.json`, `*_checkpoint_metadata.json`: export equality and saved run configurations.
- `verification.json`: source hashes, executed forecast counts and numerical checks.

Reproduction scripts: `scripts/recover_regression_inputs.py`, `scripts/investigate_forecast_regression.py --model model` (and `--model 447m_model`), `scripts/extract_holdout_regression_evidence.py`, `scripts/ablate_regression_objective.py`, `scripts/inspect_regression_outliers.py`, then `scripts/report_forecast_regression.py`. GPU scripts require the activated WSL environment; run GPU jobs sequentially.

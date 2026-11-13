# Removing the positive standard-deviation cutoff

User-requested change, effective 2026-11-05. The actual prior cutoff was
`1e-5` (0.00001), not 0.0001. Both positive cutoffs are now absent.

## Implementation

- `src/04_creating_embeddings/embed.py`: masked-horizon and causal next-patch
  tokenizers use the computed scale whenever it is finite and strictly positive.
  They retain unit scale only when the computed scale is zero or invalid.
  No epsilon, lower-amplitude bound or replacement threshold is introduced.
- This shared tokenizer code handles targets and covariates during training,
  validation, forecast inference and CUDA graph input preparation. Kernels,
  graph replay and model parameter shapes are unchanged.
- `src/06_holdout_test/holdout.py`: standard-deviation-normalized secondary
  metrics no longer substitute unit scale for tiny positive context deviations.
- `src/07_forecasting/forecast.py`: scaled-RMSE reporting no longer clamps
  positive context deviations to 1e-5. Zero/invalid values use unit scale.
- MASE continues to use its historical mean-absolute-difference denominator,
  with the existing undefined-scale handling. Its formula was not changed.

## Scope and existing artifacts

This is a runtime preprocessing change applied to existing checkpoints as
well as new training. It does not reset training automatically. The current
lazy raw-window pipeline recomputes tokenizer features/scales; its processed
and embedding preparation artifacts need not be regenerated solely for this
code change. Restart an already-running Python process to pick up the code.
Exported weights and old holdout outputs are not modified by this edit.

Old weights were trained under the previous normalization policy, so validate
their new results on a fresh holdout run. The preceding frozen-weight probes
support the amplitude correction but do not establish an aggregate accuracy
improvement. Exactly flat series still use unit scale and can exhibit the
learned patch shape described in the 2026-11-04 investigation; removing the
positive cutoff does not claim to solve that separate issue.

Computed variances remain subject to normal floating-point limitations. This
change removes the explicit positive threshold; it does not replace the
existing causal variance estimator or introduce a new constant-series policy.

## Verification

`tests/test_normalization_scales.py` verifies both tokenizer objectives:

- Positive target/covariate scales at amplitudes 1e-3, 1e-6, 1e-8 and 1e-12
  match the actual standard deviation rather than 1.0.
- Multiplying tiny inputs by 100 preserves normalized features and multiplies
  the scale by 100, even below the former cutoff.
- Constant and all-missing histories retain finite features and safe unit scales.

Focused tests passed. The full suite completed successfully: 266 tests,
including 12 CUDA-only skips. All pre-commit checks passed (syntax, whitespace,
Ruff and Bandit). No production training or full holdout run was started.

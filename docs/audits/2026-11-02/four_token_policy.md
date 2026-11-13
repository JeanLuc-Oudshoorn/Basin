# Four-token policy implementation and verification

Date: 2026-11-02. Scope: default model configuration, transformation, weighted
training/validation, sampled-horizon holdout, backward compatibility and docs.

## Requested policy and implementation

- Four direct output patches, each 16 points, with 80/16/3/1 loss coefficients.
  Longer forecasts roll forward in blocks of up to 64 using median feedback.
- Training computes all four valid short-lead losses at each causal anchor.
  The former stochastic distant-lead auxiliary branch remains only for legacy
  checkpoint configurations that do not have explicit horizon weights.
- Transformation reserves the final 64 points from targets and covariates.
  New training datasets require at least 96 observations; shorter datasets
  are audited as excluded. Right-aligned patching and temporal ordering remain.
- Full epoch-zero and epoch-end validation predicts the reserved block from
  one original observed context. It uses the same weighted normalized pinball
  objective. RMSE still pools unweighted observed errors over all 64 points.
- Weighted valid-label mass normalizes loss and gradient accumulation. Missing
  and unavailable labels have no contribution. CPU mass calculation preserves
  accumulation equivalence without adding per-microbatch GPU synchronization.
- Holdout origins remain unchanged. Deterministic SHA-256 horizon draws use
  the source key, origin and configured seed, with probabilities 80/16/3/1.
  Unavailable complete horizons are removed and probabilities renormalized;
  the final partial patch still requests 16 and scores observed points only.
- Every model receives the same historical context and horizon. A legacy
  one-patch model rolls out to satisfy multi-patch requests. Metrics contain
  realized horizon counts, `by_forecast_tokens` accuracy and latency by horizon.
  No additional lead weighting is applied to holdout accuracy metrics.
- `configs/holdout_447m_current.json` adopts the same holdout horizon policy;
  the archived weights and their architecture remain unchanged.
- Saved configurations without explicit weights retain their original model
  behavior. Checkpoint identity accepts the missing legacy optional field as
  `None` but rejects migration into the new architecture/objective. Training
  checks incompatible resume signatures before replacing run reports.

The default measured model is **307,295,520 parameters**, width 1024, 17 layers.
Three lead offsets replace the former fifteen. Inference graph replay remains
enabled by default and keys include output-patch count. Training graph replay
remains disabled by default. Other execution defaults are unchanged.

Specifications and migration steps:
[training](../../model_training/SHORT_HORIZON_TRAINING.md),
[forecasting](../../forecasting/SHORT_HORIZON_FORECASTING.md).

## Native test suite

`python -m unittest discover -s tests -p 'test_*.py'` completed 244 tests:
233 passed, 11 CUDA-only tests skipped in the native CPU environment.

New tests cover known weighted loss values, missing-tail normalization,
effective mass and causal lead alignment, weighted microbatch-gradient
equivalence, four-patch validation batching, exclusion of the final 64 target
and covariate points, minimum-length filtering, seeded horizon frequencies,
boundary handling, identical requests across all four methods and legacy
one-patch recursive inference. Existing one-patch fixtures explicitly retain
their legacy horizon rather than inheriting the new default 64-point tail.

## CUDA end-to-end smoke runs

Both ran on the local RTX 5070 using the configured WSL environment, bf16 and
the real third-party RWKV forward/backward kernels. They invoked transformation,
embedding preparation, the actual decorated `train()` entrypoint, full
validation, checkpoint/export, graph-replay forecasting and the separate
holdout entrypoint. Synthetic raw data and model/optimizer artifacts were
isolated under `.cache/tmp/wsl` on the project drive, then removed. Production
models, prepared data and existing holdout results were not overwritten.

| Run | Model | Training | Weighted validation loss |
| --- | --- | --- | --- |
| Small smoke | width 64, 2 layers | 2 epochs, 10 updates | 0.613596 → 0.607415 → 0.601029 |
| Full architecture | width 1024, 17 layers, 307,295,520 parameters | 1 epoch, 6 updates | 0.607017 → 0.542749 |

Each run generated 12 synthetic tables, reserved two whole holdout tables and
validated all ten training tables' 64-point tails. The full architecture run
included 2048-point source tables. It used physical microbatch one, optimizer
batch two and validation batch four; compiled variate attention was disabled
to isolate the policy without compilation warmup. Training used the configured
8-GiB allocator budget. This is not a full-data throughput or maximum-microbatch
memory benchmark.

Both compared ordinary and graph-replay forecasts at horizons 16, 32, 48, 64,
80 and 128. **Maximum absolute prediction difference was zero in every case**.
The longer requests exercised block autoregression beyond the new 64-step
direct limit. The separate holdout tests used RWKV and naive to avoid external
model downloads; unit adapter tests covered identical requests for Chronos-2
and Timestral as well. The smoke scores are functionality checks, not evidence
of useful forecast accuracy or a full retraining of the production model.

Artifacts:

- [Small smoke summary](four_token_policy_smoke/summary.json).
- [Full-size smoke summary](four_token_policy_full_size_smoke/summary.json).
- [Small holdout metrics](four_token_policy_smoke/holdout/test_metrics.json).
- [Full-size holdout metrics](four_token_policy_full_size_smoke/holdout/test_metrics.json).
- Each holdout directory includes prediction/timing parquet files, provenance,
  a comparison HTML plot and an error-distribution HTML plot. Raw synthetic
  source paths refer to removed temporary inputs and are diagnostic provenance.

Detailed logs are under
`logs/20261102T154421_817025Z_four_token_policy_smoke_e72a67cd` and
`logs/20261102T155019_276829Z_four_token_policy_smoke_b91e835c`.
Training JSONL journals were automatically created as
`logs/rwkv7_train_20261102_154422_898579Z_pid523_1875113b.jsonl` and
`logs/rwkv7_train_20261102_155020_394354Z_pid503_1b3e3d21.jsonl`.
Normal model export is retained only for the duration of each isolated smoke;
the default `results/model` still contains the pre-change trained model until
the user archives it and trains the new policy.

Reproduce inside WSL after sourcing `scripts/activate_rwkv7.sh`:

```bash
python scripts/smoke_four_token_policy.py
python scripts/smoke_four_token_policy.py --full-size
```

The script writes its own audit directories and logs, never the normal model
directories. Reproduction replaces its smoke artifacts. Epoch checkpoints,
optimizer moments, RNG state, validation history and best-weight export are
exercised through the normal trainer. Accuracy improvements on the real corpus
and speed at the default microbatch size remain unmeasured by these smokes.

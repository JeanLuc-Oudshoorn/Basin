# 301.8M next-patch training and final-token evaluation

## Executed experiment

The model was trained from scratch for **one complete epoch** on every table in
`data/01_generated/2026-10-23`: **100 tables**, **111,088 original rows**, lengths
**33–2,039**, with no source table skipped. The final **16 observations of each
table** were held out. All **6,795 eligible earlier next-patch transitions** were
used in **100 optimizer updates**. The held-out 1,600 observations were used only
for evaluation after training. All 100 tables have successful RWKV, exponential
smoothing and Prophet results.

The checkpoint is `results/next_patch_300m/training/model.pt` (603,753,815 bytes).
Measured CUDA-stage elapsed time was approximately 63.80 seconds for training,
including initialization/checkpoint I/O, and 48.30 seconds for evaluation, including
model loading, fitting both baselines and writing outputs. These are diagnostic
run timings, not controlled throughput benchmarks. Peak PyTorch-allocated GPU
memory was **2.473 GiB** on the RTX 5070; this excludes driver/display usage and
reserved-but-unused allocator memory.

## Exact model

| Setting | Value |
|---|---:|
| Allocated trainable parameters | **301,774,368** |
| RWKV temporal blocks | **19** |
| Cross-variable attention modules | **19** |
| Hidden width | **960** |
| RWKV feed-forward width | **3,840** |
| RWKV heads | **15 × 64** |
| Variate attention heads | **12 × 80** |
| Observations per patch | **16** |
| Maximum historical observations | **2,048** |
| Maximum input tokens | **128** |
| Quantiles per output point | **9**, from 0.1 through 0.9 |

Input residual projection is `48 -> 3840 -> 960` plus a skip connection. The head
maps each token's 960-dimensional state to `16 × 9 = 144` next-patch outputs through
a residual MLP. Parameters do not depend on table length or variable count. The
full inventory is in [architecture.json](architecture.json). The model size is
approximately 300M rather than adding unused parameters to hit exactly 300,000,000.

## How this matches language-model training

Let a table's patch tokens be `P0, P1, ..., P(K-1)`, where `P(K-1)` is the test patch.
Training takes inputs `P0 ... P(K-3)` and shifted targets `P1 ... P(K-2)`.
At position `j`, causal RWKV state sees only `P0 ... Pj`, and the head predicts
`P(j+1)`. Computing every position together is the parallel equivalent of separately
running `[P0] -> P1`, `[P0,P1] -> P2`, etc. There is no need to duplicate every prefix
on disk. Tests compare the parallel predictions with independent prefix calls.

Tokens are continuous patch embeddings. Supervision is pinball regression against
the next patch's **outcome values**, not a vocabulary softmax or regression against
a moving embedding target. Covariates are time-aligned observed input sequences;
their future values are not supplied or supervised unless they are explicitly
listed as additional targets. Lag-derived columns are excluded before tokenization.
This objective currently rejects known-future covariates; the historical
`masked_horizon` mode remains available for the earlier feature set.

For evaluation, all pre-test patches `P0 ... P(K-2)` are supplied, and the final
token's head predicts `P(K-1)`. Evaluation history includes earlier training labels,
which are genuinely observed by the forecast origin; the final patch itself never
enters a training example or normalization statistic.

## Token boundaries, padding and context cropping

Patches are aligned to the **end** of the table, ensuring that the final test patch
always contains the final 16 real observations. If the length is not divisible by
16, only the first patch is partial and left-padded with missing-value masks.
For a 33-row table this gives a one-observation first token, a full second token
used as its training label, and a full third token reserved for testing.

Short histories are retained rather than discarded. Projected sequences are
right-padded to the next multiple of 16 tokens for the fused CUDA kernel, up to
128 tokens. They are not needlessly expanded to 128 tokens every time. Padding
has no supervised target, and outputs are sliced to real token positions. Causal
time processing prevents trailing padding from affecting earlier predictions.

At forecast time, histories longer than 2,048 observations are cropped to their
**last 2,048 points**, with covariates cropped identically, before computing
normalization. Values in the discarded prefix are ignored, including invalid
values there. Tests verify exact equality to explicitly cropped inputs, including
a 2,500-observation case. The current dataset's maximum length is below 2,048, so
this boundary is validated with tests rather than inferred from the evaluation.

Longer training files are divided into chunks of at most 128 input patches plus
one label patch; adjacent chunks share one context patch so no supervised transition
is duplicated or dropped. State resets at chunk boundaries. This is blockwise LM
training, not exact sliding-window recurrence for every token of an arbitrarily
long stream. No chunking was needed for these 100 source tables.

## Causal normalization: avoiding subtle leakage

Whole-history normalization, used by the previous masked-horizon model, would leak
future values when training all next-token positions at once. It is therefore not
used in this mode. For each variable and token position `j`, mean and standard
deviation are computed from observed points only through the **end of token j**.
The token's values are normalized using those statistics. Its next-patch label is
also normalized using **j's statistics**, never those of the label patch.

Cumulative sums/counts use float64 for numerical stability, then statistics and
loss use float32. Missing values do not contribute. Near-zero variance uses unit
scale. Masks and relative positions accompany normalized values. Relative time is
measured against the fixed 2,048-point limit rather than eventual sequence length,
so adding later patches does not change the earlier embeddings.

The two-dimensional structure is unchanged: per-variable causal RWKV along time,
then full attention across variables at the same patch position within a table.
It cannot attend to another table or a later patch. The first partial patch's mask
lets the network distinguish missing padding from actual zero-valued observations.

## Stages and training details

The existing four commands remain independently runnable:

```powershell
uv run python src/03_transforming_input_data/transform.py
uv run python src/04_creating_embeddings/embed.py
uv run python src/05_model_training/train.py
uv run python src/06_forecasting/forecast.py
```

`configs/config.json` now selects this run. The previous small configuration is
preserved as `configs/smoke_masked_horizon.json`. The `stride` and
`train_windows_per_table` fields apply only to that legacy objective and are ignored
in next-patch mode; `max_tables: null` means every source table.

Processed and embedded files live in their `next_patch_300m` subdirectories; older
experiments were not overwritten. `windows.pt` stores independent full pre-test
series/chunks and evaluation records. `embeddings.pt` holds independent tokenized
records, shifted labels, prefix statistics and initial embedding snapshots.
Labels are not features. Embeddings are recomputed from saved causal features
during training so projection weights receive gradients; snapshots are not stale
inputs to a supposedly trainable projection.

The optimizer is AdamW, learning rate `1e-4`, weight decay `0.01`, gradient clip
`1.0`, with one seeded shuffled pass over the 100 records. Each update averages
the pinball loss over that record's observed label points and quantiles. Thus
tables receive one update each, not weighting proportional to table length across
the epoch. Weights, gradients and AdamW moments use bf16, with no FP32 master
weights; this is a functional test configuration rather than a finalized large-scale
pretraining optimizer. Activation checkpointing uses PyTorch's recommended
[non-reentrant checkpoint interface](https://docs.pytorch.org/docs/stable/checkpoint.html).
Model-only recovery checkpoints are written every 25 updates; optimizer-state
resume is not implemented.

Forecasting longer than one patch uses median-feedback autoregressive rollout,
reapplying the 2,048-point crop. Unavailable future covariates become missing.
The evaluated 16-point horizon requires only one pass; longer-horizon quantile
bands are not calibrated joint uncertainty estimates.

## Baselines and evaluation protocol

All three methods receive the same pre-test target history (at most 2,048 points).
RWKV additionally uses observed covariates. Exponential smoothing and Prophet are
univariate baselines fitted separately for each table. No held-out values are used
to choose seasonal periods or hyperparameters.

- Exponential smoothing: additive trend and additive seasonality, fixed period
  16. Where fewer than two seasonal cycles are observed, an explicitly logged
  nonseasonal Holt trend model is used. The exact model kind is recorded per row.
- Prophet: synthetic daily dates starting at `2000-01-01` represent row positions,
  not actual calendar semantics. Automatic daily/weekly/yearly seasonality is off;
  a period-16 component with Fourier order 3 is configured. Changepoint prior is
  0.05, seasonality prior 10, and uncertainty simulation is disabled. The
  [official Prophet fit/predict API](https://facebook.github.io/prophet/docs/quick_start/)
  is used. These baseline settings are untuned; a different seasonal prior can
  materially change the comparison.

RMSE is computed for each table's final 16 outcomes. The mean below averages
those table-level RMSEs; median uses the same 100 scores. Scaled RMSE divides each
table's RMSE by the standard deviation of its observed context (floor `1e-5`)
before averaging, so it is less dominated by large-unit tables.

| Method | Mean RMSE | Median RMSE | Mean scaled RMSE |
|---|---:|---:|---:|
| RWKV, one epoch | 43.2684 | 11.8981 | 0.8455 |
| Exponential smoothing | 42.4299 | 8.8627 | 0.9071 |
| Prophet | 36.4268 | 9.5588 | 0.7559 |

RWKV has lower RMSE on **29/100** tables versus exponential smoothing and **25/100**
versus Prophet. There is no claim of state-of-the-art accuracy: 111k observations
and 100 updates are tiny relative to a 301.8M-parameter model. The final-token
holdout measures within-dataset forecasting, not zero-shot unseen-dataset ability.

Six plots were chosen by evenly spaced source-table indices, not by model success:

- [Table 000](../../../../results/next_patch_300m/forecasts/table_000_evaluation_310_target_0.png)
- [Table 019](../../../../results/next_patch_300m/forecasts/table_019_evaluation_325_target_0.png)
- [Table 039](../../../../results/next_patch_300m/forecasts/table_039_evaluation_1521_target_0.png)
- [Table 059](../../../../results/next_patch_300m/forecasts/table_059_evaluation_1088_target_0.png)
- [Table 079](../../../../results/next_patch_300m/forecasts/table_079_evaluation_761_target_0.png)
- [Table 099](../../../../results/next_patch_300m/forecasts/table_099_evaluation_174_target_0.png)

## Audit evidence and validation

This folder preserves [configuration](config.json), [source/split manifest](data_manifest.json),
[parameter inventory](architecture.json), [training history](loss_history.json),
[all 100 evaluation rows](metrics.json), [aggregate results](summary.json), and
[coverage/hashes/test evidence](validation.json). Full runtime logs and run-specific
source hashes remain under `logs/`; Windows launchers and WSL workers each record
their own invocation. Every final source-table test row has both baseline scores.

Tests cover parallel-vs-prefix predictions, future-label isolation, causal scaling,
partial first patches, short-history padding, exact tail cropping, complete and
nonduplicated chunk target coverage, final-token exclusion, projection/backbone
gradients and checkpointed backward execution. Existing legacy pipeline and model
tests remain supported. A post-run validation adjustment ignores invalid values
in discarded prefixes; it does not alter retained values or this run's numerical
path. See `validation.json` for final test results.

Final verification: **5 next-patch tests passed on CUDA, 6 legacy model tests passed on CUDA, and 2 legacy pipeline tests passed**. Ruff, formatting, Bandit and the uv lock check passed. A zero-patch-size configuration-validation regression was caught and fixed; its separate successful rerun is recorded in `logs/300m_legacy_model_tests.log`. Neither validation-only fix changes this experiment's valid-data numerical path.

# Sampled holdout evaluation, inference execution and epoch-zero baseline

## Requested protocol and implementation

The final comparison now contains **RWKV, Prophet and naive last-observation
carry-forward**. Holt-Winters was removed from both the dataset holdout and the
legacy saved-horizon forecast path, and its obsolete training prerequisite and
default configuration were removed. Prophet is fitted independently at each
selected origin using only the historical target. The naive forecast repeats
the last finite historical target across the forecast patch. RWKV receives the
historical outcome and time-matched covariates. All methods score the same finite
future observations. Missing historical baseline values are interpolated only
within the available history; no future values enter fitting or normalization.

`training_protocol.sampled_origins` chooses at most five distinct patch starts:

1. The first eligible origin, after two context tokens.
2. Eligible starts nearest 25%, 50% and 75% of the table's row count.
3. The final eligible origin, forecasting the last token.

Ties choose the earlier start. Choices are ordered and reserve distinct starts
for remaining cases. If at most five origins exist, all are used once. This
means the middle selections can move away from their nominal percentiles in
short tables to avoid duplicate cases. Origins remain aligned to the original
16-row patch grid. Partial final patches retain the previous protocol: forecast
one token, score only real rows. Context is capped to the final 2,048 observations
for all methods, as before. Examples (zero-based forecast start rows):

| Rows | Selected starts |
|---:|---|
| 48 | 32 |
| 64 | 32, 48 |
| 160 | 32, 48, 80, 112, 144 |
| 2048 | 32, 512, 1024, 1536, 2032 |
| 2056 | 32, 512, 1024, 1536, 2048 (8 real target rows) |

Every table remains in the holdout; only forecast origins are sampled. The
existing 10% whole-table split is unchanged. Sampled pooled scores describe these
selected horizons and should not be compared as identical coverage to the old
all-origin pooled scores. Token inventory schema 2 uses the selected origins and
invalidates reports made with the old protocol or different percentiles.

Settings in `configs/config.json`:

```json
"holdout": {
  "percentiles": [1, 25, 50, 75, 99],
  "inference_batch_size": 5,
  "output_subdir": "sampled_holdout"
}
```

Outputs go to `paths.forecasts/sampled_holdout` to avoid mixing them with the
interrupted all-origin evaluation. Each table has a prediction parquet; final
`test_predictions.parquet` and `test_metrics.json` combine the selected cases.
Plots show separate forecast segments, leaving unsampled intervals disconnected.
Console and JSONL events report completed/total tables, origin count, elapsed
time and ETA; `test_metrics_progress.json` is a compact live summary. ETA uses
average elapsed time per completed table and can vary with subsequent table
lengths and Prophet fitting times. This implementation starts a fresh evaluation;
it does not reuse the old all-origin partial results or implement evaluation resume.

To evaluate the saved model without training:

```powershell
uv run python src/06_forecasting/forecast.py
```

## Inference research and execution changes

PyTorch documents that `inference_mode` also disables view tracking and version
counter updates, beyond disabling gradient recording. It is now used for both
validation paths. See [PyTorch's inference-mode design](https://github.com/pytorch/rfcs/blob/master/RFC-0011-InferenceMode.md).

PyTorch's [MultiheadAttention documentation](https://docs.pytorch.org/docs/stable/generated/torch.nn.MultiheadAttention.html)
describes its inference fast path and recommends `need_weights=False` for
optimized attention. The existing code already satisfies that setting. We
benchmarked it against the existing compiled Q/K/V projection and math-SDPA
implementation used in training. [torch.compile documentation](https://docs.pytorch.org/docs/stable/generated/torch.compile)
explains compilation caching, shape guards and the memory tradeoff of CUDA-graph
modes. We retained the existing default compilation mode and bounded CUDA memory
budget rather than introducing CUDA graphs or another architecture.

- `training.compile_variate_inference=true` now enables the measured compiled
  attention during validation/forecasting, independently of
  `training.compile_variate`. Both use the same learned weights and state-dict
  names. Wider axes above eight variables and nonzero dropout use the eager path.
- `inference.forecast_contexts` prepares each independent historical context on
  CPU, batches up to five, and gathers each sequence's own last real hidden
  position. Each member retains its own causal normalization, roles and context
  boundary. Only time is padded; variables never attend across tables or origins.
- Single-context next-patch inference now projects/sorts only the final hidden
  position through the forecast head instead of every historical position.
- Batching is bounded, and forecasting applies the same configured CUDA allocator
  budget as training. First calls for new shapes can incur compilation overhead;
  caches stay in the repository's `.cache/torchinductor` directory by default.
- No model dimensions, recurrent equations, attention connectivity, patch size,
  model weights or checkpoint parameter names changed. BF16 kernel execution and
  summation order can change rounding and potentially very close metric rankings.
  Set `training.compile_variate_inference=false` for the original attention path.

Persistent recurrent-state reuse was not implemented. Exact semantics require
handling the rolling context cap, prefix normalization, time features and
cross-variable mixing; casually caching state would not establish equivalence.

## Measurements and correctness checks

`scripts/benchmark_inference.py` uses the full 447,482,144-parameter model with
random initialization and 128 real validation tables from the existing bounded
sample. One sweep warms up each case, then two sweeps are timed with CUDA
synchronization. No training or production weights are modified.

| Validation execution | Tables/second | Sample loss |
|---|---:|---:|
| Previous `no_grad`, eager attention | 77.20 | 0.6196433461 |
| `inference_mode`, eager attention | 81.74 | 0.6196433461 |
| `inference_mode`, compiled attention | 90.56 | 0.6196160258 |

The compiled case is about **17.3% higher throughput** (14.8% less elapsed time)
than the original sampled path, excluding initial compilation. The bounded
five-context experiment measured 102.4 ms separately versus 98.9 ms batched with
eager attention. Batching alone showed only a modest benefit in this case; the
largest overall evaluation savings come from fewer origins and removing repeated
Holt-Winters fits. These are short experiments, not a measured full-epoch speedup.
Raw results: `inference_benchmark.json`.

`scripts/verify_sampled_evaluation.py` then loaded the actual saved trained model:

- 128-table validation loss: eager **0.1262469799**, compiled **0.1262695840**;
  relative difference **0.0179%**.
- Eight trained-model context checks comparing separate and batched execution:
  maximum relative prediction-vector difference **0.82%**, below the 2% BF16
  tolerance. CPU tests additionally compare float32 predictions at tight
  tolerances and verify batch-member independence.
- Three real held-out tables, 15 origins, 223 observed forecast points: **2.50 s**
  after model initialization and preceding compile warmup, including Prophet
  fitting, metrics, parquet exports and two plots. This small smoke is not an
  accuracy study or a reliable full-corpus time estimate.
- Evidence: `gpu_verification.json`, `smoke/sampled_holdout/`,
  `logs/20261029_sampled_gpu_verification.log` and its timestamped smoke JSONL.

## Epoch-zero validation

`training.validate_before_training=true` is now the default. Fresh training
runs evaluate the entire existing validation set before the first optimizer
update, write `validation_epoch_00000.parquet`, and add epoch 0 to
`validation_history.json`. The row has `phase="before_training"`, step 0 and
`training_loss=null`. Later epoch rows include the baseline validation loss and
the absolute loss reduction from it. The JSONL `validation_baseline` event is
flushed and synced to disk.

Epoch zero is diagnostic: it does not advance the scheduler, early-stopping
counter, completed-epoch count, or best trained-checkpoint selection. RNG state
is restored after the baseline. Ordinary checkpoints carry the baseline in
their validation history, and resumed runs do not repeat it. An older checkpoint
without epoch zero logs that it is unavailable; a trained model's score is never
mislabelled as an untrained baseline. The existing completed run cannot gain a
historical epoch-zero measurement retroactively.

CPU tests cover origin selection, short/partial tables, naive missing-value
behavior, input alignment/capping, batching equivalence, full validation,
baseline timing, history persistence and resume. The CUDA epoch-recovery smoke
also passed with epoch zero, compiled inference and physical microbatching:
`resume/gpu_verification.json`; log `logs/20261029_epoch0_gpu_resume.log`.
The CPU regression suite ran 150 tests: 143 passed and seven CUDA-only tests
were skipped in that environment. The separate GPU checks above exercised the
actual backend. Log: `logs/20261029_sampled_full_tests.log`. Ruff formatting/lint
and Bandit checks on the changed runtime and experiment files passed.

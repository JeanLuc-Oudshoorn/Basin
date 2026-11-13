# Holdout startup and Chronos-Bolt admission

## Startup findings

The existing WSL log at
`logs/20261030T185141_415225Z_holdout_test_34f57f1b/events.log` shows:

| Phase | Observed wall time |
| --- | ---: |
| Open embedding index, hash its referenced processed corpus | 226.667 seconds |
| Open/reconstruct `windows.pt` (4,146,222,689 bytes) and reach model metadata | 36.723 seconds |
| Initialize/load RWKV model | 42.728 seconds |
| Load Chronos-2 | 9.103 seconds |
| Load Timestral | 26.892 seconds |

The holdout stage started at 19:51:41 and reached the first table loop at
19:57:37, about 356 seconds later, excluding the earlier Windows/WSL import
and launcher delay. Opening the embedding index called `open_patch_index`,
which hashed and opened the entire processed tensor corpus and reconstructed
all records just to select a few held-out tables. The checkpoint was also
read once as metadata and again by the model loader.

## Startup changes

`holdout_test.py` now memory-maps the model export once. The saved split already
contains original test source paths, stable source keys and SHA-256 digests.
It validates unique sources, complete checksums and train/test disjointness,
then applies the existing seeded sampling. Only selected source Parquet headers
are read for dimensions and roles, using the checkpoint's original target,
covariate and excluded-column settings. Each selected raw source is still
content-hashed before forecasting, and the exported checkpoint is hashed for
model provenance.

No embedding or processed tensor file is required for holdout. This also means
changing the current screening policy does not invalidate an older trained
model's own sealed test split. Training/embedding artifact checks remain in
their original upstream stages. `select_test_records` remains available for
legacy callers, but the holdout runner does not call it on the full corpus.

`RWKVForecaster.load` accepts the already-open checkpoint, keeping its original
model initialization, dtype conversion and state-copy order. Missing state keys
still fail, and training initialization is unchanged.

An additional experiment using meta-device construction and `assign=True`,
following [PyTorch's loading recommendations](https://docs.pytorch.org/tutorials/recipes/recipes/module_load_state_dict_tips.html),
passed a CPU stand-in parity check but changed actual GPU RWKV predictions by
up to 0.082611083984375 on this sample. It was **reverted**, not shipped. This
audit records the failed parity check rather than claiming the optimization
preserves the vendored backend's behavior. The accepted change removes corpus
processing and duplicate checkpoint reads without changing weight-loading math.

Measured metadata/schema-only probe of the real checkpoint, 10,783 saved test
tables, 216 selected tables:

| Runtime | Checkpoint metadata | Sampling + selected schemas | Total |
| --- | ---: | ---: | ---: |
| Windows CPU | 0.910 seconds | 0.069 seconds | 0.979 seconds |
| WSL CPU | 1.049 seconds | 0.575 seconds | 1.625 seconds |

These timings intentionally exclude interpreter imports, checkpoint hashing,
GPU initialization, model construction, external model loading and compilation.
They replace the corpus-dependent part of startup, not the entire launch time.
The raw probes are `startup_probe.json` (Windows) and `startup_probe_linux.json`.

A complete real one-table GPU smoke
succeeded with the training corpus paths set to nonexistent locations. It took
68.598 seconds including runner startup and predictions; RWKV load took
42.453 seconds. `gpu_holdout_first_verification.json` preserves that initial
baseline. Final verification below records the restored original loader and
compares its pooled RWKV metrics against that result.

## Admission changes

The default screener is now `amazon/chronos-bolt-small`, pinned to revision
`772f3d25d38aec6d914c8949dab4462e2d46f5d8`, anonymously downloaded to the
project's `.cache/huggingface` directory on E:. Installed Chronos already
supports this model, so no extra package is required. A real CPU load measured
47,718,016 parameters. This is a univariate target-only filter, not a model of
covariate-dependent predictability. The main multivariate RWKV training and
foundation-model holdout adapters keep their covariates.

Reject a valid scored table only for
`chronos_bolt_mase > 1.1 * baseline_mase`. The tolerance, lag and model settings
are configurable under `validation`. Exactly 10% worse, ties and improvements
pass. If a defined naive MASE is zero, a positive Bolt MASE fails. Undefined
MASE from a constant historical context retains the valid table, explicitly
flagged; it is not reported as a win. Invalid data/evaluation failures remain
separate errors and are not admitted as successful scores.

The final `data.horizon` outcomes are excluded from screening. Split the
remaining prefix at 80%, cap historical context at 2048, and predict up to
the next configured 16 rows. This bounds the screening forecast to one patch;
the prior Prophet screener forecast its entire remaining prefix. Temporal
boundaries and actual scored horizon are persisted per row. No covariates,
lagged outcomes, SCM metadata or reserved final-token labels enter the model or
MASE denominator. Historical lag-1 naive MAE is the default scale; both
forecasters use the same scale and actuals.

One model batches up to 64 independent tables, grouping short-table horizons.
It does not launch one model per worker. CPU threads and device are configured;
Windows CLI validation/transform use the existing WSL launcher. Retry flags
are preserved when forwarded. INFO progress/ETA appears every 100 tables and
at completion, with full per-table DEBUG logs and durable progress CSV.

Transformation automatically screens reports that are absent, old Prophet
reports, mismatched policies, incomplete cohorts or changed source content.
The screening signature binds the model/revision, MASE rule, tolerance and
temporal settings. Retry reuses only successful unchanged real scores under
that exact signature. Completed report pointers are published atomically;
interrupted runs cannot replace a prior finished report. Set `auto_run=false`
to fail instead of automatically refreshing.

The legacy Prophet engine remains explicit compatibility mode for historical
reports/tests. Default runs use Bolt. Existing trained models can still be
tested immediately. Using the new admission filter for future training requires
new transform/embed artifacts. Full behavior/configuration is documented in
[CHRONOS_BOLT_SCREENING.md](../../../data_validation/CHRONOS_BOLT_SCREENING.md).

## Verification artifacts

- `scripts/verify_bolt_and_startup.py`: bounded, reproducible metadata profiling
  and real CPU Bolt screening on copies of three generated tables, with outputs
  in `screening/`. All three evaluations succeeded; no full cohort was screened
  and no production reports or transformed datasets were replaced.
- `scripts/verify_holdout_fast_start.py`: one sealed holdout table with actual
  RWKV, Chronos-2, Timestral and naive predictions. The corpus paths are absent
  deliberately. Outputs go to `holdout_smoke/`, separate from user results.
  `gpu_holdout_verification.json` records end-to-end time, loading time, metrics
  and maximum change against the initial successful RWKV metrics.
- Unit tests cover threshold boundaries, zero/undefined denominators, source
  changes, reserved-tail leakage, bounded batching, stale-policy refresh,
  retry reuse, schema reconstruction and inference without training artifacts.
  Loader tests verify bit-identical CPU predictions, materialized weights and
  rejection of incomplete checkpoint state.

Final verification:

- Full suite: **183 tests, 176 passed, 7 CUDA-dependent skips** on the Windows
  CPU runtime, 23.668 seconds. Separate real CUDA execution is recorded above.
- Applicable pre-commit checks passed on changed code/docs/config: AST, JSON,
  merge/private-key checks, whitespace, EOF, LF endings, Ruff lint/format and
  Bandit. TOML/YAML checks had no matching files.
- Final original-loader GPU smoke: **68.427 seconds** for startup plus all
  predictions/scoring, about **59.686 seconds** from runner entry to the table
  loop. RWKV loading remained **42.693 seconds**; its initialization/backend
  work was not bypassed. No training tensor artifacts were opened.
- All five pooled RWKV accuracy metrics matched the first successful run
  exactly: maximum absolute reference-metric difference **0.0**. The failed
  meta-loader experiment is excluded from the implementation.
- The earlier same-machine log took approximately **356 seconds** to reach
  the table loop. This is a practical before/after observation, not a controlled
  cold-cache benchmark: external model load times and filesystem caching vary.
  Interpreter imports and WSL launch add time outside these runner measurements.
- Production holdout outputs, trained model, raw generated tables and prior
  screening reports were left intact; real smokes used separate audit outputs
  and copies of three source tables.

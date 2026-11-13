# Removing the exact-token audit from training's critical path

## Diagnosis

The reported `embed:tokenize_next` messages occurred during the automatic
**exact token-inventory audit**, before model initialization or gradient updates.
The normal trainer previously called `write_token_inventory` unconditionally.
The audit visits every input and label patch, hashes contents and source
positions, and inserts digest sets into SQLite to compute exact uniqueness and
overlap. This is much more work than preparing a patch for the model.

The worker run `logs/20261027T195420_214209Z_train_bca1e05b` ended its training
stage timer after approximately **2,469 seconds**, still showing audit tokenizer
messages. Its shared JSONL journal,
`logs/rwkv7_train_20261027_195333_006166Z_pid4180_a3b06185.jsonl`, contains only
launcher/start events and no model-initialized or optimizer-update event. These
files are left unchanged as evidence.

The audit had three avoidable costs:

- It accessed lazy `record['prepared']` merely to compare a token count. This
  computed normalization, masks and positions for a whole window even though the
  audit hashes raw patches rather than learned embeddings.
- SQLite used durable transactions on the mounted project drive for a disposable
  scratch index, adding synchronization and random I/O to a large corpus scan.
- Dataset split logging printed the complete source and digest dictionaries to
  the console, despite their already being saved in audit artifacts.

The CPU thread limit was also applied only after the audit. Normal training does
not require exact uniqueness counts to compute any prediction, loss or gradient.

## New default

The main configuration now contains:

```json
"token_inventory": {
  "mode": "cached"
}
```

This field is under `training`. Modes are:

| Mode | Startup behavior |
|---|---|
| `cached` (default) | Reuse a matching exact report. If absent, stale or invalid, record that exact counts are unavailable and proceed without scanning. |
| `exact` | Explicitly recompute the complete exact audit before constructing the model. This can still take a long time. |
| `off` | Skip inventory generation and cache lookup. |

Cache identity checks the report schema, processed-data checksum, pipeline
signature, patch size, embedding width and complete dataset-split audit. A
different source cohort, seed or holdout partition cannot silently reuse old
counts. Cached results describe the frozen processed cohort; they do not bypass
the final forecast evaluator's raw holdout-file checks.

`token_inventory_status.json` records mode, cache availability, reason, report
path, training/validation/test counts and whether a corpus scan occurred. Missing
unique counts are not replaced by guesses. Existing reports are retained rather
than deleted when stale; status identifies whether the report applies to the
current run. The model's actual parameter count is still measured regardless
of inventory availability.

Generate exact counts explicitly when wanted:

```powershell
uv run python src/05_model_training/token_inventory.py --config configs/config.json
```

The next normal training run can reuse that report if identities match. The
standalone command intentionally computes exact counts even when normal startup
is configured as `cached`.

## Other performance changes

Exact auditing now counts raw patches without invoking the lazy tokenizer.
Previously materialized feature records can still have their token counts
checked; lazy records are counted directly from raw shapes and patch alignment.
Existing exact-count tests verify the same results for duplicates, missing data,
partial patches, roles, split overlaps and rolling test coverage.

The scratch SQLite database disables journaling and synchronization, with a
bounded 64 MiB page cache. It exists only inside a temporary directory and is
discarded after the audit. These settings do not affect persistent model/data
artifacts or an external database. The database remains on the configured training
drive; it is not relocated to C: or an unbounded RAM-backed index. Large exact
audits remain costly and are not promised to become fast merely by this change.

The trainer sets `cpu_threads` before loading and optional auditing. Dataset-split
console logging shows counts instead of all source paths; complete membership
and hashes remain available in the split artifact and JSONL. The lazy-index
loader already checks the full processed-file checksum, so the trainer no longer
reads the same multi-GB file a second time solely to repeat that verification.
Legacy artifacts retain the explicit checksum guard.

Training still tokenizes the **current microbatch on demand**. It does not skip
patch preparation, change targets or masks, introduce an embedding cache, change
the optimizer, alter full-epoch validation, or change the dataset holdout policy.

## Real-data measurement

`scripts/benchmark_training_startup.py` ran in the configured **WSL runtime** on
the actual indexed corpus. No model was allocated, no optimizer ran, and no new
training run was started. Benchmark output writes were isolated in an audit
directory, leaving production training outputs unchanged.

| Measurement | Result |
|---|---:|
| Total indexed windows | 215,656 |
| Training windows | 97,045 |
| Validation tables | 97,045 |
| Reserved test tables | 10,783 |
| Indexed artifact loading and checksum | 52.52 seconds |
| Metadata-only dataset split | 0.445 seconds |
| Default cold-cache inventory decision | 0.00855 seconds |
| Corpus tokenizer calls during that decision | **0** |
| Sampled real training windows prepared | 256 |
| Sample feature-preparation duration | 0.0464 seconds |
| Sample feature-preparation throughput | **5,517 windows/second** |
| CPU threads | 4 |

See [benchmark.json](benchmark.json) and
`logs/20261027_training_startup_benchmark.log`. The benchmark forbids tokenizer
calls during startup with an explicit assertion, then separately prepares a
seeded sample of real training windows. Feature throughput is measured after
artifact loading/checksumming and excludes model forward/backward, optimizer work,
exact hashing and SQLite counting. It is **not an end-to-end model training rate**.
Startup still needs to load metadata, verify the large artifact and initialize
the 447.5M model; it is not instantaneous.

To repeat this bounded CPU-only measurement in WSL:

```bash
source scripts/activate_rwkv7.sh
python scripts/benchmark_training_startup.py --samples 256
```

## Verification and use

New regression tests prove that default startup reaches model construction
without an inventory scan or redundant checksum, cache reuse rejects changed
artifacts and partitions, explicit exact mode still generates reports before
model construction, and exact auditing never tokenizes lazy records. Existing
stale-input and exact-count tests remain in place. Ruff, formatting and Bandit
checks cover the changed files.

Restart `train.py` to use the updated startup path. Code edits do not modify an
already running Python worker. Transformation and `embed.py` do not need to be
rerun because only training/audit behavior changed.

Final verification: **124 regression tests, OK, six CUDA-dependent skips**, in
`logs/20261027_training_startup_tests.log`. The WSL real-data benchmark exited
successfully. Ruff lint/format checks and Bandit passed. The existing prepared
index signature still matches the current configuration. No production model
training was launched or running worker automatically terminated during this fix.

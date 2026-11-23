# Full-table training, periodic validation and hourly checkpoints

Default protocol effective 2026-11-06, implemented through the normal `train.py`.
No alternate pipeline folder or training entrypoint is needed.

Update, 2026-11-15: the default model and recipe changed (220M hybrid model, 7
forecast tokens, optimizer parameter groups, warm-up/cosine schedule, weight
averaging, robust validation monitor). The timed trainer, checkpoints, clocks and
recovery described here are unchanged. Differences that matter when reading this
guide: validation forecasts span 7 tokens instead of 4; each check also reports
the 10th, 50th and 90th percentile of table MASE and their geometric mean
`table_mase_profile`, which selects the best weights and counts the strikes
towards early stopping; with `training.weight_average` set, the
averaged weights are what is validated and saved as best; a warm-up/cosine run
needs a fixed `epochs` or `steps` budget and validates its final weights when
that budget ends. Settings quoted below are those of the archived 500M run
(`configs/legacy_500m.json`). See
[the 220M architecture and recipe](HYBRID_220M_2026-11-15.md).

## Configuration

Relevant settings in `configs/config.json`:

```json
{
  "data": {"validation_policy": "sampled_training_origins"},
  "training": {
    "resume": "auto",
    "epochs": null,
    "steps": null,
    "checkpoint_interval_seconds": 3600,
    "validation_interval_seconds": 18000,
    "max_run_hours": null,
    "validation_sampling": {
      "tables": 25000,
      "seed": 7,
      "percentiles": [1, 25, 50, 75, 99]
    },
    "early_stopping": {"patience_checks": 3, "min_delta": 0.0, "monitor": "loss"},
    "scheduler": {
      "name": "reduce_on_plateau",
      "factor": 0.5,
      "patience_checks": 1,
      "min_learning_rate": 0.00001
    }
  }
}
```

Set `max_run_hours` to e.g. `12` to stop this invocation after approximately
12 elapsed hours inside the training loop. `null` removes that duration cap;
early stopping and explicit epoch/update limits still apply. The duration cap
can be changed when resuming. The default batch/microbatch and gradient
accumulation options are retained. No partially accumulated optimizer update
is saved as though it had completed.

## Optimizer precision and continuation

BF16 model parameters use `FP32MasterAdamW`: forward/backward remain BF16,
but AdamW updates, first/second moments and master weights remain FP32.
Small updates accumulate in the master weights before being copied to BF16.
The master weights are saved inside every recovery checkpoint, preserving
updates that are too small to appear in the BF16 model copy yet.

`training.resume="auto"` also accepts the previous direct-BF16 AdamW
checkpoints. On that one-way upgrade, model weights initialize the masters,
saved moments are promoted to FP32, and optimizer steps, learning rate,
scheduler, stopping history and batch cursor are retained. Other model/data
and optimization compatibility checks remain enforced. Previously rounded-away
updates cannot be reconstructed. This fixes update precision; its effect on
forecast accuracy must be measured through subsequent validation.

From the project root in PowerShell, continue with:

```powershell
.\.venv\Scripts\python.exe .\src\05_model_training\train.py --config .\configs\config.json
```

Windows routes the GPU work to the configured WSL environment. FP32 optimizer
storage increases VRAM usage and recovery checkpoint size. FP32 staging gradients
are processed in bounded 64-MiB buckets and released after each bucket. A single
parameter larger than that budget is processed alone. This avoids staging the
entire model's FP32 gradients simultaneously; master weights and moments remain
FP32. Bucket sizing is an execution detail and does not change checkpoint format,
logical gradient accumulation, clipping, optimizer step counters or scheduling.
See the [optimizer staging OOM fix](OPTIMIZER_STAGING_OOM_2026-11-11.md).

## Clocks and save order

Both intervals count **active training time**, including CPU update
preparation and optimizer execution, excluding validation and checkpoint writes.
They are checked at completed optimizer updates, so the maximum normal cadence
overshoot is one accumulated update. Active time and last-check times persist
across restarts. Validation does not wait for an epoch to finish, and an epoch
ending early does not trigger an extra validation sweep.

The duration cap instead counts elapsed time per invocation inside `fit_timed`,
including baseline validation, resume loading, scheduled validation and saving.
Dataset/model initialization before the loop is outside this clock. It is a
**soft limit**: finish an in-progress update/check/save safely, then stop.
A long validation or disk write can therefore extend elapsed time past the cap.
If the limit lands when a validation check is due, complete that check and save
its updated scheduler/selection state before exiting. This can extend elapsed
time by one validation sweep plus saving. A pending check from an interrupted
run likewise completes before the next training update. No training update
is silently executed before that pending check.

When scheduled validation is due:

1. Finish the optimizer update and close the bounded CPU prefetch generator.
2. Atomically save current model/optimizer state and mark validation pending.
3. Evaluate the complete fixed panel, with no competitors loaded.
4. Update best-weight selection, early stopping and plateau scheduler.
5. Save again with the completed validation and updated scheduler state.

If validation fails or the process is killed during it, the earlier checkpoint
remains usable. On resume, the check is repeated before the next training
update, preserving the intended learning-rate and stopping trajectory. An
unexpected kill between scheduled checkpoints loses work since the latest
successful checkpoint; a planned duration stop always writes a final checkpoint.

Existing epoch-zero baseline validation remains configurable through
`validate_before_training`. Its RNG effects are isolated, as are all scheduled
checks. Set it false for a quicker diagnostic start.

### Background checkpoint writes, cloud copy and summarised logging (2026-11-17)

Measurements and tests: [training speed, 2026-11-17](TRAINING_SPEED_2026-11-17.md).

| Setting (`training.*`) | Default in code | `configs/config.json` | Effect |
| --- | --- | --- | --- |
| `background_checkpoints` | `true` | `true` | The state is copied to host memory (about 2 s for the 221.5M model) and written by a background thread while training continues. "Atomically save" in the steps above then means "copy the state, start writing"; the file replaces the previous checkpoint only when complete. A run that fails or is interrupted still finishes the write in flight. If host memory is short, the write blocks as before. |
| `checkpoint_upload` | off | `{"container": "checkpoints"}` | While a checkpoint is written, the same bytes are uploaded to Azure Blob Storage as `<paths.training>/latest_checkpoint.pt`. The local file stays the one training resumes from. See below. |
| `validation_cache` | `false` | `"~/.cache/basin/validation_panels"` | The fixed validation panel is saved in `validation_panel.bin` during the first check; later checks read it instead of hashing, opening and slicing every source table again. Same windows, same scores. `true` keeps it in the training folder; a path keeps it elsewhere (one subfolder per training folder). WSL's own disk is read 20 times faster than its mount of the Windows drive. |
| `log_every_tables` | `null` (every update) | `1000` | One journal record and one console block per that many tables, plus one before every checkpoint and validation. Sums cover every update. Per-update `window_ids`/`sources` are not written in this mode; `loss_history.json` and the checkpoint still hold the last 1,000 updates. Use `null` for per-update diagnostics (`scripts/diagnose_epoch_throughput.py`). |
| `fused_master_optimizer` | `true` | `true` | One-pass CUDA kernel for the FP32-master AdamW update; bit-identical to the three-pass update, which remains the fallback. |

Cloud copy. Credentials come from `AZURE_ENDPOINT` and `AZURE_STORAGE_KEY`, read
from the environment or from `.env` in the project root (git-ignored). The WSL
environment needs `azure-storage-blob` (`scripts/setup_rwkv7.sh` installs it).

- Before training starts, a small blob is written, read back and deleted. Wrong
  credentials, a missing container or no network stop the run there.
- The first checkpoint of every run is written, uploaded and verified before
  training continues; a failure stops the run with the local checkpoint intact.
- Later upload failures are logged as errors (`checkpoint_upload_failed` in the
  journal) and do not stop training; the next checkpoint is uploaded again.
- Every block carries an MD5 that the service checks, the blob replaces the
  previous one only when all blocks have arrived, and the stored size and
  whole-file MD5 are compared with the local file afterwards.
- The upload never reads the file back from disk, and it cannot hold the local
  write up for long: a network that stalls for two minutes ends the upload, not
  the write.
- An upload still running when the next checkpoint is due is abandoned; the
  previous cloud copy stays in place.
- A run that is killed leaves the last complete local checkpoint and the last
  complete cloud copy in place, plus an unfinished `.latest_checkpoint.pt.*.tmp`
  file that the next start removes.

```bash
python src/05_model_training/checkpoint_upload.py check       # credentials and container
python src/05_model_training/checkpoint_upload.py list        # blobs, sizes, step metadata
python src/05_model_training/checkpoint_upload.py download --to results/model/training/latest_checkpoint.pt
```

`download` verifies the SHA-256 recorded at upload before replacing the target.

## Data and fixed validation panel

Transformation now uses each training table through its final observation as
gradient labels, subject to causal context/lead masks. There is no final
four-token holdback. Training still chunks long series at the configured
context limit, never shuffles observations within a table, and preserves
partial-prefix masking. Full-table training admits sources with at least
48 points (or the configured patch-dependent minimum).

The existing configured dataset-level holdout fraction is unchanged. Whole
holdout sources remain excluded from gradients and from the monitoring panel.
Changing corpus membership still changes the reproducible ranked split; the
export saves the actual split for subsequent holdout testing.

After reserving the independent test fraction, select 25,000 independent tables
using seeded SHA-256 ranking and remove every selected table from training.
Selection uses a bounded top-k heap, and origins are constructed only for the
chosen sources. Each chosen table must support five distinct origins with two
context tokens and four complete future tokens: at least 160 points for
16-point patches. Shorter tables remain available for training, but not this
panel. A smaller corpus selects fewer tables and logs the reduced count, while
always leaving at least one training table; no eligible validation tables is an error.

Use the first eligible origin, starts nearest the 25/50/75% row percentiles,
and the final eligible patch-aligned origin. All five leave 64 future points.
Each uses only actual historical targets and time-matched past covariates,
capped at 2048 observations; no future covariates enter the encoder. Every
request predicts four patches in one pass, with 80/16/3/1 weighted normalized
pinball loss. Reports also include weighted RMSE/MAE/MASE. Missing labels are
masked and zero MASE scales remain undefined.

This is **dataset-disjoint validation**: no window, context or label from any
reserved validation table is used in gradient training. Validation informs
learning-rate scheduling, early stopping and best-model selection. The separate
test set is reserved for final evaluation. Metrics use `in_sample=false` and
`dataset_disjoint=true`. The older `sampled_training_origins` config identifier
is retained for artifact compatibility; it now always uses disjoint tables.
The fixed selection makes successive validation checks comparable; it is not
redrawn at every checkpoint.

`training/dataset_split.json` records separate `train_sources`,
`validation_sources` and `test_sources`, actual counts/fractions, plus the
`dataset_disjoint_v1` policy. `training.validation_sampling.tables` sets the
validation count (25,000). `training.holdout_tables` sets a fixed number of
independent test tables (50,000); training stops with an error rather than
shrink that set when too few independent tables exist. Without it,
`training.holdout_fraction` sets the reserved test fraction of independent
tables. Test selection happens first and is unaffected by changing the
validation count. Augmented families remain training-only: neither a coarse,
aggregate or sign-inverted copy nor the original it was made from can enter
validation or test.

Start a fresh run with `training.resume=false` (or a new training output
directory) when migrating from overlapping in-sample validation: those weights
may have trained on the newly reserved validation tables. Auto-resume rejects
the old policy. This split-only change does not require rebuilding compatible
transform/index_patches artifacts. Existing saved split/results are not retroactively
rewritten or made unbiased.

In JSON, write `"resume": false` with no quotes around `false` for a fresh
start. Quoted `"true"`/`"false"` values are rejected with an explanatory error;
they are not checkpoint filenames. Use `"resume": "auto"` for later restarts.
Leaving `false` configured starts from scratch on every invocation.

Validation descriptors contain metadata only. At evaluation, read one source
at a time, construct its five windows, and group them into bounded compatible
CPU buckets for batched GPU execution. Tokenized contexts live only until
their batch is consumed. No full 125,000-context embedding cache is created.
Transformation also stores only source metadata for evaluation entries under
this policy, rather than duplicating raw tail/context tensors for every table.
The patch indexing stage indexes these entries without tokenizing them.

Sharded corpora use tensor-free `.record_metadata` sidecars for startup scans and
batch planning. New transformation spills publish these automatically. Older
artifacts build missing sidecars once, without an upstream stage rerun. Startup
reports corpus sizes, periodic metadata progress and split phases. The first
migration can still take time; subsequent passes avoid tensor deserialization.
These indexes live alongside processed data on the project drive. See the
[startup investigation](../audits/2026-11-08/training_startup.md) for evidence and
measurement limits.

The passes that run before the first update (two for the split, one for the
check that validation tables are not trained on, two for batch planning, two
for the checkpoint digests) read a few small fields of every record. Those
fields are kept in a start-up index, `.startup_index` beside the processed
shards: flat arrays that the passes read instead of the JSON sidecars. The
first start after a corpus is transformed builds it with one pass over the
sidecars; later starts check its checksums and reuse it. The whole validation
and test records, which a start also needs, are saved there after the first
read. The index is bound to the ordered shard checksums and is rebuilt when it
does not match them; deleting the folder only costs the next start one rebuild.
On the 23-million-record corpus (2026-11-23) the index is 3.7 GiB, took 8
minutes to build, and a resume reached its first update after 16 minutes
instead of 58, with the same split, batch order and checkpoint identity
(`src/03_transforming_input_data/startup_index.py`).

Augmentation-family screening resolves shared date folders once and uses the
generated table names, avoiding millions of per-source path-resolution calls
over WSL. It reports indexing, directory-scan and classification progress about
every 30 seconds. Directory scans still cover raw siblings that were not
admitted. See the [family-screening investigation](../audits/2026-11-08/family_screening_startup.md).

## Recovery and artifacts

For large sharded corpora, the default `training.storage_locality=true` shuffles
shard sections and tables within each compatible shape bucket, retaining each
shard in the bounded tensor cache while it is consumed. Windows are covered once
per epoch and their internal chronology is preserved. This changes batch order
and partial-batch counts compared with global shuffling; checkpoint identity
binds the setting and physical shard layout. Compact record indices, cached shape
plans and streamed hashes/audits reduce repeated scans and peak startup memory.
See the [million-table pipeline review](../audits/2026-11-08/pipeline_scaling.md)
for measurements, compatibility and remaining linear costs. The metadata disk
cache survives restarts, but the split and in-memory batch plan are reconstructed.

Run and resume with the same command:

```powershell
uv run python .\src\05_model_training\train.py
```

`results/model/training/latest_checkpoint.pt` contains:

- Current weights and separate best validated weights.
- Full AdamW state, including moments, step counters and parameter-group options.
- Learning-rate scheduler state/plan and early-stopping counters.
- Torch CPU/CUDA, Python and NumPy RNG states.
- Current epoch and **next batch-group offset** in its deterministic shuffle.
- `loop_state.epoch_windows`: training windows consumed in this epoch, and
  `loop_state.epoch_target_points_processed`: supervised target points consumed
  in this epoch. The global `points_processed`/`windows_processed` counters span
  all epochs. Target-point counts include valid supervised labels across leads
  and overlapping windows; they are not unique raw rows. The batch cursor,
  rather than a raw-row offset, controls exact restart order.
- Active-time counters, unfinished epoch loss totals, processed-window/point
  counters and pending-validation status.
- Recent update history and aggregate validation history.

Resuming reconstructs the same shuffled batches and starts at the saved cursor,
without preparing/skipping every already-consumed batch. No updates are replayed.
Model/corpus/validation identity, logical batch settings, dtype/device type and
PyTorch version are checked. Physical microbatch size/token limits may be
lowered on resume after a memory failure; this preserves the logical batch
cursor and optimizer state, but tensor-size changes can affect rounding and
stochastic RNG assignment. Exact numerical continuation assumes the same
execution environment and deterministic behavior of its kernels; it is not a
promise of bitwise equivalence across different hardware or library versions.

### Graph replay for validation (measured 2026-11-15, off by default)

`training.validation_cuda_graphs=true` replays one CUDA graph per full
validation batch shape; leftover batches use the plain forward. Weights are read
at replay time, so graphs survive optimizer updates and the weight average
without recapture (checked by changing every weight in place: no recapture,
forecasts within the usual rounding distance of the plain path).
`validation_graph_memory_gib` (default 2.0) bounds the cache and
`validation_graph_retain=false` frees it after each check.

It is off because it does not pay. At the validation batch size of 16 the GPU
work dominates the launch overhead that replay removes: on the 221.5M model,
replay was faster only for the smallest shapes (one to three variables, at most
512 points) and equal or 3-8% slower for the rest, about 9% less forward time
overall. That forward time is small to begin with: about 29 ms per batch, or
roughly four minutes for all 125,000 forecasts. Graphs for batches are also
large; capturing every shape overflowed the 12 GB card. A slow validation check
is therefore spent outside the model's forward pass. Script and first-run log:
`experiments/claude_review/validation_graph_replay.py`.

The current config checkpoints after one hour of active training and validates
after five hours. These cadence settings may also change on resume without
discarding model weights, optimizer state or the saved batch cursor. When the
validation interval changes, a saved pending check is recalculated against the
new interval and the last completed validation time. For example, an interrupted
one-hour validation is no longer immediately due after switching to four hours;
training continues until cumulative active time reaches the new deadline.
With unchanged cadence, an interrupted validation is rerun as before. Scheduler
and early-stopping patience remain counts of validation checks, so those checks
now span more training time.

## Dedicated GPU memory and physical microbatches

The current default targets up to 11 GiB of PyTorch allocator memory on the
12 GiB GPU. At startup the limit is reduced if necessary to leave 0.5 GiB
beyond currently occupied desktop/external memory. This replaces the old
8 GiB limit and its additional 80% capacity clamp. Both live tensors and
cached reservations count toward the limit; this is not a limit on dataset size.
Other applications can consume more VRAM after startup, so this cannot guarantee
that Windows will never page GPU allocations under every external workload.

```json
{
  "batch_size": 8,
  "gradient_accumulation_steps": 1,
  "microbatch_size": 8,
  "microbatch_token_limit": 3072,
  "auto_microbatch": true,
  "cuda_memory_limit_gib": 11.0,
  "cuda_memory_headroom_gib": 0.5,
  "checkpoint_token_threshold": 1792,
  "resume": "auto"
}
```

These are fields inside `training`. The logical optimizer update still covers
eight tables. Its physical execution uses at most eight at a time, further
limited by `tables × variables × padded patch tokens <= 3072` where possible.
One table is the minimum, even when that single table exceeds the token limit.
For 2,048 observations and five covariates, 128 patches × six variables permits
four tables per physical microbatch, with two backward passes per eight-table
update. Shorter/narrower tables can use all eight tables at a time. A trial with
four as the maximum was safe but unnecessarily split these small inputs;
the final eight-table maximum also passed the full-model GPU stress test.

Activation checkpointing reduces retained layer intermediates by recomputing
them during backward. The threshold is an approximate activation budget, not
a hard VRAM guarantee: retained layers are
`min(layers, threshold × layers // (B × V × T))`. The selected 1792 budget
retains more intermediates than the earlier 1024 policy; the larger token cap
reduces splitting. These defaults passed a 480-update, 48-shape stress test with
zero OOM retries, and measured approximately 8% more throughput on recorded
batches. See the [benchmark and safety report](../time_benchmark/TRAINING_THROUGHPUT_2026-11-08.md).
They do not change the model architecture, data, loss weights or forecasting
graph replay.

If forward/backward still raises a CUDA OOM, the trainer clears partial
gradients, releases failed activations, restores the update's RNG state, and
retries the *entire same logical update* with a smaller physical microbatch.
No examples are dropped, and loss gradients retain the original label-mass
weighting. Successful smaller limits are remembered per variable/patch shape
and saved inside the hourly checkpoint. If even one table fails, the trainer
tries checkpointing every layer once, where enabled. Exhausting that option
raises an error; optimizer-step OOMs also raise, because partially applied
optimizer steps cannot safely be retried. Training graph capture must be off
for this recovery path; the default remains off.

Every 100 updates, INFO logs show allocated/reserved/peak VRAM, physical embedding
shapes and recovery count. Full training data remain on disk in shards; CPU
preparation holds the current update plus one prefetched update, and only the
current physical microbatch moves to CUDA. Millions of tables are not loaded
onto the GPU. See the [memory investigation](../audits/2026-11-08/training_memory.md).

The latest 1,000 update records are retained in checkpoint/JSON history to bound
long-run memory. The full timestamped training progress remains in `logs/*.jsonl`.
Global counters remain cumulative across restarts. Detailed validation origin
metrics are stored in `validation_check_XXXXX_step_XXXXXXXXX_TIMESTAMP_ID.parquet`.
Every check also saves its own aggregate JSON with the same stem. These files
have unique names, including on retries or fresh runs in the same directory.
`validation_history.json` retains the ordered aggregate history for the resumed
training trajectory; it is replaced atomically after each check, preserving
earlier entries. Plot `loss` against `active_training_seconds / 3600` (or `step`);
`completed_at_utc` records the completion timestamp. Baseline check zero is
included when enabled. Individual snapshots remain available even if a check
finishes but the process dies before the following checkpoint save; a repeated
check after resume creates a separate snapshot without overwriting that attempt.

Keep `training.resume="auto"` to continue the saved epoch cursor and history.
Older hourly checkpoints lacking the explicit epoch target-point counter still
resume at their exact batch cursor; that diagnostic count is `null` until the
next epoch rather than reporting a guessed historical count.
`training_plan.json` records clocks, sampling counts and schedule settings.
Progress logs show update counts and countdowns to validation, saving and exit.
They also show epoch completion percentage and an epoch training ETA, based on
active training time per processed window. The estimate excludes future
validation/checkpoint pauses and can vary as different table shapes are reached.
For an already running process that predates this console message, view the
same recorded ETA without restarting training:

```powershell
uv run python .\scripts\watch_training_progress.py --follow
```

The structured dataset-split log links the complete split JSON instead of
embedding millions of source names in a single training-journal event.

`model.pt` is still the forecasting export: it selects the best positively
trained validated checkpoint. If a short run stops before any trained validation,
it exports current weights explicitly as unvalidated, rather than the untrained
baseline. Exported step/epoch provenance is recorded separately from the latest
optimizer cursor. Always resume from `latest_checkpoint.pt`, not `model.pt`.

## Migrating from reserved-tail artifacts

The default data policy changes the processed/patch-index signature and gradient
labels. Recreate those two stages for the new corpus/policy:

```powershell
uv run python .\src\03_transforming_input_data\transform.py
uv run python .\src\04_patch_indexing\index_patches.py
```

Start this as a new training run: old reserved-tail checkpoints cannot resume
the changed corpus/objective protocol exactly. Remove the old latest checkpoint
or select a fresh training output directory; no existing files are deleted by
this implementation. Automatic admission-screen reuse still depends on its
separate screening signature; this new sampling flag alone does not invalidate
otherwise matching screening scores.

Legacy configs using `data.validation_policy="reserved_tail"` retain epoch
validation and epoch checkpoints. Older `patience_epochs` keys remain accepted;
the new default uses `patience_checks` so cadence is explicit.

All originals with coarse or sign-inverted copies, and those copies themselves,
are training-only. Neither validation nor holdout can select them, including
when a copy did not pass screening. The holdout fraction applies to independent
tables. This split rule is checked against raw sibling directories automatically,
including for existing processed artifacts. See the
[augmentation isolation audit](../audits/2026-11-06/augmentation_isolation.md)
for eligibility, audit fields and checkpoint compatibility.

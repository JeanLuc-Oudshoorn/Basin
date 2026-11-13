# Transformation: from generated tables to training records

The transformation stage converts generated time-series tables into the tensors
and source metadata used by the embedding and training stages. It determines
which variables the model sees, how a series becomes next-patch training
examples, and which observations each example contains.

The entry point is [`transform.py`](../../src/03_transforming_input_data/transform.py).
This guide describes the current `model.objective="next_patch"` workflow,
including its resumable checkpoints. The script also retains an older
fixed-window forecasting branch; the checkpoint mechanism described here applies
to the next-patch branch.

## Where it fits in the workflow

```text
Generated data.parquet tables
    -> optional admission screening
    -> transform.py: select columns, check data, construct training records
    -> windows.pt index + tensor shards on disk
    -> embed.py: prepare the patch index
    -> train.py: prepare patch features, learn embeddings and train the model
```

Transformation is important because a forecasting model needs examples with
explicit input roles and causal prediction boundaries. Passing complete raw
tables directly to training would leave those decisions unresolved. The saved
source identities and configuration signature also help downstream stages detect
stale or mismatched artifacts.

Transformation does not fit the forecasting model or learn embeddings.
Normalization, missing-value masks and learned patch projections are prepared
later as records are used. The offline embed phase builds an index rather than
saving embedding vectors for the entire corpus.

## Running the stage

From the repository root:

```powershell
uv run python src/03_transforming_input_data/transform.py --config configs/config.json
```

After successful completion, continue with:

```powershell
uv run python src/04_creating_embeddings/embed.py --config configs/config.json
uv run python src/05_model_training/train.py --config configs/config.json
```

On Windows, enabling Chronos admission screening routes transformation through the
configured WSL environment. With screening disabled, transformation can run
directly in the local Python environment.

## What happens to each dataset

### 1. Discover and optionally screen source tables

`paths.raw` identifies the generated-data root. `data.generated_dates` selects
date folders beneath it; the script recursively finds `data.parquet` files in
those folders. If the date setting is absent, it scans the configured raw root.
Generation plots and SCM specifications are not training inputs.

When `validation.enabled=false`, all discovered candidates proceed to structural
checks. This is the current configuration. When screening is enabled, the script
uses the selected validation engine's admission report. With Chronos-2, a table
passes when its forecast MAE is no more than the configured relative tolerance
worse than last-observation carry-forward (15% with the current settings).
Missing or stale Chronos reports can trigger screening automatically when
`validation.auto_run=true`.

Screening is a dataset-quality policy, separate from constructing tensors. It
can add substantial runtime, including source-file hashing even when a report
already exists. Transformation checkpoints do not checkpoint that upstream work.
See [data admission screening](../../README.md#data-admission-screening).

`data.max_tables` limits the admitted tables processed by transformation. It is
applied after admission screening, so it does not limit the initial discovery or
screening pass.

### 2. Check the table and select input columns

For the next-patch branch, the script:

- Requires the configured time column to contain unique, increasing, nonmissing
  values. It checks the existing order; it does not sort the table.
- Selects `data.targets`, normally `outcome`.
- Selects historical covariates from `data.past_covariates`. With `"auto"`, these
  are the remaining columns after excluding targets, time and excluded patterns.
- Excludes lag-derived columns matching `data.excluded_column_patterns`. Such
  columns can expose future target information in forecasting workflows.
- Rejects infinite values in selected inputs. Missing observations represented
  by `NaN` remain available for downstream missing-value handling.
- Skips tables that are too short for the selected training/validation policy.

The next-patch branch currently requires `data.known_covariates=[]`. It uses
time-aligned observations of historical covariates.

### 3. Construct causal next-patch records

With `model.patch_size=16`, one patch represents 16 consecutive timesteps. The
model learns to predict subsequent target patches using earlier observations.
For example, a sequence of patches `A, B, C, D` supplies transitions `A -> B`,
`A,B -> C`, and `A,B,C -> D`, subject to the context limit.

With a 2,048-point context, a training chunk contains at most 2,048 input points
plus one 16-point target patch. Longer series are divided into chunks. Adjacent
chunks share one patch at the boundary so that supervised transitions are not
duplicated. Short or incomplete initial patches are aligned to the right;
metadata records how much left padding the later tokenizer needs.

The script creates float32 tensors with shape
`[batch=1, number_of_variables, number_of_timesteps]`, keeping target and
historical-covariate tensors separate. Each record also stores its source hash,
row boundaries, variable roles and table identity.

`data.train_windows_per_table` and `data.stride` belong to the older fixed-window
branch. They do **not** limit the number of next-patch training chunks.

### 4. Record the evaluation policy

`data.validation_policy` controls how transformation treats the end of a series:

| Policy | Training data | Evaluation records |
| --- | --- | --- |
| `sampled_training_origins` (current) | Retains the full series for causal training transitions. | Saves descriptors; training later selects fixed forecast origins from training tables. These metrics are in-sample monitoring. |
| `reserved_tail` | Excludes the final `data.horizon` observations from gradient labels. | Saves a context/target batch for that reserved tail. |

With the current 16-point patches, 64-point horizon and four output patches,
evaluation requests can cover four future patches. The horizon must contain
whole patches and fit within the configured model outputs. The data and model
context-length settings must agree.

Transformation prepares records for all admitted sources. **The training stage
performs the actual training/test split by source dataset**, using the seed and
`training.holdout_fraction`. Records from test sources are excluded from gradient
updates. Augmentation families receive additional training-only restrictions.
A record marked `split="train"` in `windows.pt` therefore does not by itself
mean its source will be used to update the final model.

## Outputs

Paths below are relative to `paths.processed`, currently
`data/02_processed/model`.

| Output | Purpose |
| --- | --- |
| `windows.pt` | Small index of tensor shards, their checksums/counts and the configuration signature. Consumed through `read_artifact()` by embed and training. |
| `manifest.json` | Human-readable table metadata, selection audit, exclusions and training-transition counts. Interpret boundaries using each table's `validation_policy` and `training_stop`. |
| `validation_selection.json` | Admission-screening decisions, or an explicit record that screening was disabled. |
| `.transform_checkpoints/<identity>/` | Dataset shard files (`batch_*.pt`) and the recovery index `progress.json`. Referenced shards remain essential after completion. |
| `table_*.parquet` | Optional selected-column exports when `data.export_processed_tables=true`; currently disabled. |

Each invocation also creates a run folder under `paths.logs`, containing the
configuration, event log, source-code provenance and run status.

## Progress and automatic resume

Progress bars show completed/total **tables**, percentage, tables per second,
elapsed time and ETA. They update every 100 tables, after five seconds when a
table finishes, and at completion. Inventory checking, checkpoint restoration
and transformation have separate bars. Five million rows do not necessarily
mean five million progress units: a table can contain many rows.

Checkpoint cadence is independent of progress logging. Configure it with this
top-level section in `configs/config.json`:

```json
{
  "transform": {
    "batch_size_tables": 1000,
    "batch_memory_limit_mb": 64,
    "checkpoint_every_tables": 100000,
    "checkpoint_interval_seconds": null
  }
}
```

These defaults save every **100,000 completed tables**, plus the final partial
checkpoint batch. Set a positive time interval in seconds to enable a time trigger. If both
triggers are enabled, the first one reached causes a save. `null` disables an
individual trigger; checks happen between tables. With both disabled, only the
final checkpoint is saved.

The memory buffer has separate limits: it spills to disk after 1,000 tables or
approximately 64 MiB of tensor data, whichever is reached first. Tensor bytes
are checked after each record, so one record can exceed the target; a 10,000-record
cap also limits metadata-only buffers. These small writes release tensor memory
without advancing the recovery checkpoint. With the settings above, a forced exit
can still repeat up to 100,000 tables even though many small shards were written.
Uncommitted shards are discarded on restart and regenerated from the checkpoint.

To resume an interrupted run, rerun the same command. The script checks the
source inventory and verifies the saved shard checksums without loading their
tensors, then continues after the last committed table. Work after the latest
checkpoint, including an unfinished table, is repeated. The console prints the
checkpoint path and resume position.

Checkpoint identity includes the dataset signature, transformation code, raw
root, ordered source names, sizes and modification times. Changes to those
inputs use a separate checkpoint directory. Keep source files unchanged during
a run. Changing only the checkpoint intervals preserves compatibility.

The sharded implementation has a new checkpoint identity. Earlier in-progress
checkpoints from the single-file implementation are not migrated automatically;
completed single-file artifacts remain readable by embed and training.

Each checkpoint commits references to the new shards. The final `windows.pt`
index is atomically published after all tables are checkpointed, and the table
manifest is streamed to JSON. Finalization never merges the tensor corpus into
RAM. If it is interrupted, restarting can retry it without transforming saved
tables again. Wait for the successful completion message before starting embed.

Embed and training use `read_artifact()` to access the index. Records retain
lightweight references to their batches; a bounded cache opens one shard at a
time using memory mapping. Older single-file `windows.pt` artifacts remain
readable. Directly calling `torch.load()` on a new index returns its metadata,
not a materialized `records` collection.

## Runtime and storage considerations

Runtime includes file discovery, optional screening, inventory checks, Parquet
reads, tensor construction, hashing and writes. Many small source files,
especially access through WSL to a Windows drive, can make filesystem overhead
significant. A progress bar at 100% for transformation can be followed by a
separate final-output write.

The shard files are the processed dataset itself, so there is no second full
tensor copy for the final output. **Do not delete a shard directory referenced by
`windows.pt`**, even after successful completion. Keep the index and referenced
files together when moving or backing up the dataset. Old unreferenced run
directories are separate from the current dataset.

Transformation memory now depends primarily on its output buffer and the largest
individual source table, rather than all training tensors. Each source Parquet
table is still read in full. The file inventory and shard descriptors also grow
with dataset size; the configured MiB target is not a cap on total process RAM.
Training still builds source partitions, batching information and record metadata
in memory, while the bulk tensor data remains on disk. A single very large table
or an extremely large metadata inventory can therefore still require substantial
RAM. The older fixed-window branch retains its previous memory behavior.

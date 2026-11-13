# Embedding preparation: batches, progress and recovery

[`embed.py`](../../src/04_creating_embeddings/embed.py) prepares the patch index
between transformation and model training. It inspects window shapes, checks
patch counts, records per-window metadata and saves a reproducibly initialized
encoder. The learned embedding projections run later during training; this stage
does not fit an embedding model or cache embedding vectors for the corpus.

Run it after transformation completes:

```powershell
uv run python src/04_creating_embeddings/embed.py --config configs/config.json
```

## Batches and memory

The stage processes records in configurable batches and streams metadata to a
JSONL file one record at a time. It does not construct an additional list of
all records, features or embedding vectors.

With the sharded transformation format, the reader keeps at most one source
shard mapped at a time. A batch can end inside a shard; that shard stays cached
until it has been consumed, avoiding repeated deserialization of the same file.
The cache is released when moving to another shard and after the final batch.
Raw training tensors remain in their original transformation shards.

Memory still includes the encoder, the input shard's metadata, the shard index
and small output buffers. Batch size is not a hard limit on total RAM. Older
single-file `windows.pt` inputs remain supported, but their entire serialized
metadata collection is loaded by PyTorch; use sharded transformation output for
large datasets.

## Configuration

The top-level `embedding` section in `configs/config.json` defaults to:

```json
{
  "embedding": {
    "cpu_threads": 2,
    "batch_size_records": 1000,
    "checkpoint_every_records": 100000,
    "checkpoint_interval_seconds": null,
    "progress_every": 1000
  }
}
```

| Setting | Meaning |
| --- | --- |
| `cpu_threads` | CPU threads used during preparation. |
| `batch_size_records` | Number of windows/descriptors inspected per batch. No whole-batch tensor collection is allocated. |
| `checkpoint_every_records` | Commit progress after this many additional records; `null` disables the record trigger. |
| `checkpoint_interval_seconds` | Optional time trigger; `null` disables it. |
| `progress_every` | Progress-bar interval in records. |

Counts must be positive integers; the optional time interval can also be a
positive fractional number of seconds. If both checkpoint triggers are enabled,
the first one reached causes a save. Triggers are checked after each record,
and a final checkpoint is always saved. With both triggers disabled, progress
is committed only at completion.

Progress counts **records**, not tables or rows. One source table can produce
multiple training windows plus an evaluation descriptor. The bar shows
completed/total records, percentage, records per second, elapsed time and ETA.
It updates at the configured record interval, after five seconds when a record
finishes, and at completion. Progress logging and checkpoint frequency are
independent.

## Resuming after an interruption

Rerun the same command. Recovery files live under `paths.embedded`, currently:

```text
data/03_embedded/model/
  .embedding_checkpoints/<identity>/
    progress.json
    windows.jsonl
```

`progress.json` contains the committed record count, metadata byte offset,
cumulative statistics and checksums of committed output ranges. The working
`windows.jsonl` is appended incrementally. At each checkpoint, the script flushes
and syncs that file before atomically publishing the new recovery position.

On restart, the script verifies the saved metadata prefix, removes any incomplete
or uncommitted tail and seeks directly to the next record. It does not re-index
the committed records or retain them in RAM. Verification reads the saved JSONL
prefix in bounded blocks and can take time for a large file. Work since the most
recent checkpoint is repeated.

The checkpoint identity binds the processed artifact's content hash, dataset
signature, record count and embedding implementation. Changing the input data
or implementation starts a separate recovery directory. Changing batch size,
CPU threads, logging frequency or checkpoint intervals preserves compatibility.
Keep the processed input and its referenced transformation shards unchanged
during preparation.

If final output writing fails after indexing completes, a rerun restores the
completed position and retries finalization without inspecting all records again.
Only one embed run can write to an output directory at a time.

## Completed outputs

| File under `paths.embedded` | Purpose |
| --- | --- |
| `embeddings.pt` | Small `lazy_patch_index_v1` artifact with the processed-input hash, record count, signature and initial encoder weights. Training uses its existing loader. |
| `windows.jsonl` | One metadata row per window or evaluation descriptor, including token counts and eventual embedding shape where applicable. |
| `manifest.json` | Configuration, total/resumed record counts, memory information and output statistics. |

Final outputs are published through temporary files and atomic replacement.
The metadata copy is streamed in bounded blocks. Wait for the completion message
before starting training:

```powershell
uv run python src/05_model_training/train.py --config configs/config.json
```

The embedding recovery folder retains a second copy of the JSONL metadata so
finalization can be retried. It can be removed after successful completion if
recovery is no longer needed. **The referenced transformation shard files must
remain available**: they contain the training data and are not disposable
embedding checkpoints.

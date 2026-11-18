# Patch indexing: checks, progress and recovery

[`index_patches.py`](../../src/04_patch_indexing/index_patches.py) runs between
transformation and model training. It verifies the transformed shards, inspects
the shape of every window, checks its patch count against the model's limits,
records one metadata row per window and saves the patch index: a small file
that binds the processed data to the model configuration and holds a
reproducibly initialized patch encoder.

The stage was called `embed.py` until 2026-11-17. It never produced embeddings:
the learned patch projection runs inside the model and is optimized during
training. No embedding model is fitted here and no embedding vectors are
cached for the corpus.

Run it after transformation completes:

```powershell
uv run python src/04_patch_indexing/index_patches.py --config configs/config.json
```

## How the corpus is inspected

For a sharded transformation output the work is split in two:

- One thread in the main process computes the SHA-256 checksum of each shard,
  in file order, and compares it with the shard index. A hard disk therefore
  reads the shards as one sequential stream. A mismatch stops the run with
  `Corrupt processed shard`.
- Worker processes open each verified shard through the normal memory-mapped
  loader and inspect its records. The main process writes their results in
  record order.

The checks and the output are the same as when everything is inspected in one
process; only the elapsed time changes. Every shard is still read once in full
for its checksum, so on a hard disk the stage takes about as long as reading the
processed dataset once.

Worker processes are used when at least eight shards remain to be inspected.
Smaller jobs, and older single-file `windows.pt` inputs, are inspected in the
main process one record at a time. Single-file inputs remain supported, but
their entire serialized metadata collection is loaded by PyTorch; use sharded
transformation output for large datasets.

Each worker keeps one shard mapped at a time. The main process holds only
finished metadata rows, for at most twice as many shards as there are workers.
Raw training tensors remain in their original transformation shards.

## Configuration

The top-level `patch_index` section in `configs/config.json` defaults to:

```json
{
  "patch_index": {
    "cpu_threads": 2,
    "workers": null,
    "batch_size_records": 1000,
    "checkpoint_every_records": 100000,
    "checkpoint_interval_seconds": null,
    "progress_every": 1000
  }
}
```

| Setting | Meaning |
| --- | --- |
| `cpu_threads` | PyTorch CPU threads in the main process. |
| `workers` | Worker processes that open shards. `null` uses half the logical CPUs (at most 8) when eight or more shards remain and none otherwise; `0` always inspects in the main process. |
| `batch_size_records` | Records inspected per batch when no worker processes are used. No whole-batch tensor collection is allocated. |
| `checkpoint_every_records` | Commit progress after this many additional records; `null` disables the record trigger. |
| `checkpoint_interval_seconds` | Optional time trigger; `null` disables it. |
| `progress_every` | Progress-bar interval in records. |

Counts must be positive integers (`workers` may be zero); the optional time
interval can also be a positive fractional number of seconds. If both checkpoint
triggers are enabled, the first one reached causes a save. Triggers are checked
after each record, and a final checkpoint is always saved. With both triggers
disabled, progress is committed only at completion.

These settings are not part of the dataset signature. A configuration that still
uses the former names (`paths.embedded`, the `embedding` section) is rejected
with a message naming the new ones.

Progress counts **records**, not tables or rows. One source table can produce
multiple training windows plus an evaluation descriptor. The bar shows
completed/total records, percentage, records per second, elapsed time and ETA.
It updates at the configured record interval, after five seconds when a record
finishes, and at completion. Progress logging and checkpoint frequency are
independent.

## Resuming after an interruption

Rerun the same command. Recovery files live under `paths.patch_index`, currently:

```text
data/03_patch_index/model/
  .patch_index_checkpoints/<identity>/
    progress.json
    windows.jsonl
```

`progress.json` contains the committed record count, metadata byte offset,
cumulative statistics and checksums of committed output ranges. The working
`windows.jsonl` is appended incrementally. At each checkpoint, the script flushes
and syncs that file before atomically publishing the new recovery position.

On restart, the script verifies the saved metadata prefix, removes any incomplete
or uncommitted tail and continues with the next record, which can be in the
middle of a shard. It does not re-index the committed records or retain them in
RAM. Verification reads the saved JSONL prefix in bounded blocks and can take
time for a large file. Work since the most recent checkpoint is repeated.

The checkpoint identity binds the processed artifact's content hash, dataset
signature, record count and the stage's implementation. Changing the input data
or implementation starts a separate recovery directory. Changing batch size,
worker count, CPU threads, logging frequency or checkpoint intervals preserves
compatibility. Keep the processed input and its referenced transformation shards
unchanged during the run.

If final output writing fails after indexing completes, a rerun restores the
completed position and retries finalization without inspecting all records again.
Only one run can write to an output directory at a time.

## Completed outputs

| File under `paths.patch_index` | Purpose |
| --- | --- |
| `patch_index.pt` | Small `lazy_patch_index_v1` artifact with the processed-input hash, record count, signature and initial encoder weights. Training opens the corpus through it. |
| `windows.jsonl` | One metadata row per window or evaluation descriptor, including token counts and the shape its embedding will have in training. |
| `manifest.json` | Configuration, total/resumed record counts, memory information and output statistics. |

Final outputs are published through temporary files and atomic replacement.
The metadata copy is streamed in bounded blocks. Wait for the completion message
before starting training:

```powershell
uv run python src/05_model_training/train.py --config configs/config.json
```

The recovery folder retains a second copy of the JSONL metadata so finalization
can be retried. It can be removed after successful completion if recovery is no
longer needed. **The referenced transformation shard files must remain
available**: they contain the training data and are not disposable checkpoints.

Model files exported before the rename record the index checksum as
`embedding_sha256`; newer ones use `patch_index_sha256`. Forecasting accepts
both.

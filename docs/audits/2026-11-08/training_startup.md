# Silent sharded-corpus startup

The live WSL trainer (PID 329) was not deadlocked. `ps` showed a running process
averaging approximately 73% of one CPU core, with about 2 GiB resident RAM and
no training updates yet. `/proc/329/io` showed approximately 68 GB read; its
memory mappings later identified shard 787 of 1,151. The prepared index contains
10,966,424 records. Small `windows.pt`/`embeddings.pt` files are indexes, not the
entire corpus. The tensor shards hold the large underlying dataset.

Startup's split code iterates record metadata several times. The old sharded
record accessor deserialized entire tensor shards to return each record's
metadata, despite needing only source/split/shape information. A single-shard
cache bounds RAM but repeats expensive deserialization across corpus passes.
Batch planning also accessed real tensors just to inspect their dimensions.
Neither operation previously reported incremental progress. One busy core
appears as low overall CPU utilization; GPU utilization is expected to remain
low until the actual model/validation stages.

Changes:

- Transformation spills now publish tensor-free per-shard metadata sidecars.
- Sharded metadata access reads those sidecars, independent of the bounded
  tensor cache. Existing processed data builds the cache from one verified
  tensor read per missing shard, so no transform/embed rerun is required.
- Sidecars are atomically published under the processed directory's
  `.record_metadata/<shard-sha256>.json`, on drive E for this project. They are
  derived indexes, add disk usage, and can be regenerated. Shard identity and
  record counts are checked; actual tensor access retains full shard checksum
  verification. They do not contain learned embeddings or change training labels.
- Batch role/length grouping uses existing role and context-row metadata,
  avoiding tensor loads when constructing batches. Legacy examples without
  those fields retain the existing tensor-based fallback.
- Startup reports the record/shard count, metadata progress every 30 seconds
  at shard boundaries, and explicit split-phase messages.

The running Python process has the old modules loaded and was left running.
To use the change it must be stopped and restarted. Its already completed scan
is not a durable training checkpoint and cannot be adopted by the new loader.
The first pass over uncached legacy shards still incurs migration cost; future
metadata-only passes avoid that tensor scan. This does not establish the total
startup time or training throughput of the entire eleven-million-record corpus.

Measured the real first 10,000-record shard using Windows CPU while leaving WSL
training running: cold metadata construction 3.074 s, reopened metadata scan
0.145 s (about 21x faster), and planning its 651 batches 0.009 s. The metadata
sidecar is 10.9 MB. See `metadata_startup_sample.json`. This one-shard result
does not include whole-corpus hashing, family screening, Python record-list
memory, model loading or baseline validation. No production model/checkpoint
was replaced; only the first derived metadata sidecar was created by the sample.

Regression tests prove new-artifact metadata iteration and batch planning make
zero tensor-shard loads; legacy artifacts migrate once and reopen without tensor
deserialization; retained batch references still return the original tensors;
and bounded shard lifetimes and transform/embedding recovery remain intact.

Verification: 311 tests completed with no failures and 12 CUDA-only skips;
pre-commit checks passed. An existing resume regression fixture now explicitly
selects `resume="auto"`, rather than inheriting the user's fresh-start config.

# Holdout startup optimization, November 12, 2026

The user's running evaluation was not stopped, restarted or used for another
GPU experiment. No production outputs or training weights were overwritten.
CPU-only tests and a small WSL metadata benchmark were run separately.

## Observed delay in the existing process

`logs/20261112T065701_493941Z_holdout_test_561cc74c/events.log` reports:

- Split/checkpoint metadata: 236.9 seconds, including loading a roughly 2 GB
  JSON split with millions of irrelevant training paths and dictionary entries.
- Augmentation lineage scans: about 369 seconds across 5.48 million directory
  entries, although only 118,395 sources belong to the saved holdout.
- Serial selected Parquet header reads: 589.9 seconds for 17,760 tables.
- Then a silent full SHA256 pass over the 8 GB training recovery checkpoint,
  including optimizer and unused current weights. The user reported another
  twenty minutes at this stage. This last duration is reported, not a completed
  measured hash time. No forecasts had begun at that log boundary.

## Changes

1. `startup_cache.load_split` uses ijson to stream metadata. It retains only
   holdout sources/maps and compact sorted 128-bit train/validation membership
   indexes. Both leakage checks remain active. It calculates the split SHA256
   during its full parse and persists a cache invalidated by split size/mtime/
   ctime. The initial pass locates test sources before filtering the large maps;
   subsequent launches do not scan the JSON again. Membership index files have
   checksum validation and are memory-mapped on Linux. Windows reads them into
   memory to avoid locking temporary files against cleanup.
2. Lineage results are cached by source membership and every scanned directory's
   fingerprint, including referenced alias target directories. A changed raw
   directory invalidates the result. A directory changing during a scan raises
   an error rather than publishing that result as a valid cache.
3. Eight threads read pending Parquet headers. Schema caches are keyed by saved
   source checksum and original column policy. Actual raw data hashes continue
   to be verified during evaluation, before forecast generation. A changed raw
   file is not accepted merely because a cached schema exists.
4. Recovery evaluations fingerprint tensor names, dtypes, shapes and contents
   from `best_state_dict`. They do not read optimizer tensors for provenance.
   `checkpoint_sha256=null`, `evaluated_weights_sha256=<digest>` and an explicit
   scope avoid mislabeling a tensor fingerprint as the complete file hash.
   Exported model files retain complete-file SHA256 provenance. Size and both
   timestamps invalidate saved checksums. The split's full SHA256 is retained.
5. `RWKVForecaster.load` imports backend setup outside a meta-device context,
   constructs the architecture without initializing real random CPU weights,
   allocates destination tensors and strictly restores all checkpoint values.
   This also avoids the transient allocation of an unnecessary CPU model.
6. Cold streaming/header/checksum work logs progress every thirty seconds.

All caches stay in the project on E. No generated raw dataset or result archive
is deleted. ijson 3.6.0 was installed in native and WSL environments and added
to pyproject/uv.lock. Sampling percentage and seed, model architecture, trained
weights, scoring and request-latency boundaries remain unchanged.

## Bounded timing evidence

WSL CPU-only benchmark, no visible CUDA devices, 128 real sources from the
previous archived holdout selection. Each uncached-schema trial used a new
schema-cache directory; OS file caches were not flushed. Trial order was
one worker, eight workers, one worker, eight workers:

| Workers | First trial seconds | Second trial seconds |
| --- | --- | --- |
| 1 | 7.602 | 4.817 |
| 8 | 0.132 | 0.135 |

Reusing schema metadata took 0.0030–0.0035 seconds for those 128 records.
A synthetic split with 50,000 training sources and 500 holdout sources took
0.153 seconds to stream/build and 0.018 seconds to load warm. These measurements
do not establish full-corpus startup duration and must not be extrapolated as
an end-to-end speedup. First use still builds caches and scans lineage.

Raw measurements: [cpu_benchmark.json](cpu_benchmark.json).
The ephemeral reproduction script is `.cache/tmp/benchmark_holdout_startup.py`,
launched through `.cache/tmp/benchmark_holdout_startup.sh` in the existing WSL
runtime. It does not call model forecasting or read training record payloads.

All three actual vendored RWKV constructor classes were separately extracted
from their source AST and used for a CPU meta-device construction test. The
500,056,352-parameter architecture constructed in 4.69 seconds, with every
parameter on meta and no CUDA backend invocation. This checks construction,
not GPU inference performance.

## Verification and limits

- Full native CPU suite: 343 tests passed, 13 skipped, before the final switch
  from full recovery-file hashing to selected-weight fingerprinting.
- Final focused holdout suite: 41 tests passed, including the added fingerprint
  cache/invalidation test.
- Tests cover overlap rejection, changed split invalidation, lineage changes,
  checksum invalidation, existing output cleanup behavior, and strict checkpoint
  restoration with bit-exact predictions using the CPU test backbone.
- No second full GPU holdout or GPU model-loading trial was started while the
  user's evaluation was running. Full real-corpus cold/warm startup timings
  therefore remain to be measured on the next launch.
- The currently stuck invocation already imported the older implementation;
  edits cannot remove a hash read already in progress. It was left untouched.

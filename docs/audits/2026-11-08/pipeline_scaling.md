# Training pipeline review: 5,483,212 sources

## Restart and cache status

The current processed index has 10,966,424 records in 1,151 shard entries.
All 1,097 non-empty shards have `.record_metadata/<sha256>.json` sidecars.
The other 54 shards have zero records and are never visited by record iteration.
This was checked against the actual index and filesystem, without re-reading
the tensor corpus. Thus the completed tensor-to-metadata conversion is reusable
on restart. Metadata is not rebuilt, and transform/embed need not be rerun.

The trainer still reads cached JSON metadata to construct the split. It does
not persist a completed, restartable `split_datasets` execution: restart begins
startup again, using the cache. Calling `record_metadata` again in logs does not
mean rebuilding the cache. Family directory scans are refreshed each run so
newly generated copies cannot silently leak into validation/test.

## Findings and changes

| Area | Scaling issue | Change |
|---|---|---|
| Source discovery | Recursive raw-file discovery would be expensive | Normal training uses the prepared index; transform-only discovery is not repeated |
| Metadata loading | Full tensor deserialization for metadata | Existing tensor-free disk cache is reused; actual tensors stay lazy |
| Family screening | Millions of `Path.resolve` filesystem calls | Shared date-prefix resolution, one directory scan, phase progress; preceding fix retained |
| Validation candidate filtering | `source in train_sources - family_sources` recomputed a huge set difference for every record | Set memberships are computed once outside record loops |
| Splitting | Repeated full metadata scans for IDs, hashes, windows and audits | Two passes: source summaries, then selected windows; IDs selected before retaining validation records |
| Test selection | Sorting every independent source by its SHA rank | Bounded `heapq.nsmallest` for the requested test count; identical deterministic membership |
| Source-set memory | Redundant copies and repeated source sorting | Reuse sets in place; one final sorted source list per reported partition |
| Training-record memory | Retaining millions of rich dictionaries | `RecordSelection`: eight-byte indices plus lightweight `RecordRef` handles; shard caches own metadata/tensors |
| Batch planning | Shape/length sorting repeated each epoch and on resume | `BatchPlanner` caches sorted shape groups; each epoch reshuffles handles |
| Shuffled data I/O | Global random access defeats the single-shard tensor cache | `training.storage_locality=true` visits each shard in one shuffled section per epoch |
| Checkpoint fingerprint | Sorting rich sharded records and joining all IDs into a huge string | Streamed ordered identity for compact records; streamed ID hash, with physical shard layout bound when locality is enabled |
| JSON audits | Building a multi-gigabyte JSON string before writing | Buffered streaming JSON, atomic replacement, byte-count heartbeat for large source audits |
| Hourly validation | 25,000 raw tables / 125,000 forecasts | Remains bounded, streamed and Basin-only; this is substantive evaluation work, not a startup metadata scan |
| Token inventory | Exact counting would scan every raw token | Existing default `mode=cached` remains; exact counting is opt-in |
| Checkpoint/model export | Large weight/optimizer files and full split audit | Remain real linear I/O costs; no repeated corpus scan added to hourly saves |

The per-record set-difference expression was a bug in the earlier split code.
It made filtering proportional to records times source-set size, rather than
ordinary linear membership checks. The two-pass rewrite removes it completely.

## Storage-local batches

New default config: `training.storage_locality=true`. Table order is shuffled
within compatible role/length buckets, shape runs are shuffled within a shard,
and shard order is shuffled each epoch. All windows remain covered once per
epoch; observations within each window stay chronological. Batches never mix
different role counts. Gradient accumulation and physical microbatch limits
continue to apply. Normalization, architecture, labels, lead weights and split
membership are unchanged.

Shard-local mixing is less global than arbitrary table shuffling: nearby
generator outputs and augmentations can share a shard. This is the explicit
I/O-locality tradeoff, not an assertion of identical training accuracy. Set the
flag false for globally shuffled batches when starting a compatible new run.

This deliberately changes training batch membership/order and can change the
number of partial batches/updates per epoch. It is not a claim of identical
optimization trajectories to global shuffling. The setting, streamed dataset
identity and storage layout are checkpoint-bound. Newly created checkpoints
resume the same shuffled order and intra-epoch cursor exactly under the same
execution environment. Incompatible older sharded checkpoint identities are
rejected; they are not silently continued under a different order. Eager legacy
datasets retain their existing canonical identity path. No production weights
or checkpoints were reset or written during this investigation.

The compact source selection is rebuilt from prepared artifacts on restart,
then the saved epoch/cursor picks the next optimizer group. Cached shape plans
live in memory for the duration of a training invocation, rather than being
re-sorted after every epoch. Model checkpoints retain optimizer, scheduler,
RNG, completed-window/point counts and pending-validation state as before.

## Evidence

`scripts/benchmark_pipeline_scaling.py` reads a bounded sample of existing
shards on CPU. On Windows, eight real training windows from two shards,
batch_size=1, gave:

| Metric | Global shuffle | Storage-local shuffle |
|---|---:|---:|
| Tensor-shard loads | 5 | 2 |
| Tensor read/deserialization seconds | 9.454 | 3.733 |
| Planning seconds | 0.183 | 0.219 |

Record coverage matched. This is approximately 2.53x faster data loading for
that sample, not a whole-training speedup. The benchmark ran with an existing
startup process in the background and warm OS filesystem caches. It does not
time GPU forward/backward, compilation, hourly validation, or the whole corpus.
See `pipeline_scaling_sample.json`. With locality, focused tests enforce one
tensor-shard load per visited shard and zero tensor loads during batch planning.

Additional regression tests enforce exactly two metadata passes through split,
identical split audits for reversed input order, eight-byte selection indices,
seeded epoch shuffle/coverage, and streamed ID hashes matching the prior join
hash when storage binding is disabled. Changing physical storage layout changes
the locality-bound hash. An end-to-end CPU test trains actual compact sharded
records with dropout and gradient accumulation, stops mid-epoch, resumes, and
compares weights, optimizer moments, scheduler, RNG and loss sequence exactly
against uninterrupted training. JSON failures preserve the previous audit file.

## Remaining expected costs

Startup is still proportional to corpus size: cached JSON reads, directory
enumeration, source-key/checkpoint hashing and large source audits take time.
The audit still records all source IDs and hashes, so its Python source maps and
exported metadata grow with table count. The one-shard tensor cache bounds raw
tensor residency; one tensor-shard deserialization per visited shard per epoch
remains. None of these steps can be inferred to be instantaneous from the sample.
The full 5.48-million-table GPU training throughput and peak RAM were not
benchmarked here. A completed whole-run split cache is not implemented.

Verification: full suite completed 319 tests with no failures and 13 skips
(CUDA-only cases and a Windows directory-symlink restriction). Pre-commit checks
passed after normalizing the benchmark JSON's final newline. The final JSON
writer also emits that newline directly. Focused CPU tests exercise real
compact-record training, gradient accumulation and mid-epoch resume, rather
than only mocking the optimization loop. The live production startup process
was left running; it must be restarted to load these module changes.

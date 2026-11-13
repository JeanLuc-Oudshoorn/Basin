# Augmentation-family screening at five million sources

After the metadata cache completed, the live WSL trainer (PID 1329) remained
running in augmentation screening. The old helper first evaluated
`(ROOT / source).resolve()` for all 5,483,212 sources, retaining millions of
Path objects. It emitted no progress until the entire function completed.
On `/mnt/e`, resolving every path component incurs expensive filesystem work;
this can dominate startup even though series contents are not read. A process
can look mostly idle overall while issuing serialized filesystem queries.

The fix indexes the generated `date/table/data.parquet` layout using string
operations and resolves each distinct date prefix once. It scans each canonical
date directory once, then classifies sources by generated directory names.
Memory now holds date mappings, family names and excluded source references,
instead of a Path object for every input source. Relative/absolute date aliases
share the scan. Actual table-directory symlinks found during the scan are
resolved individually so aliases retain family identity. This optimization
targets generator-produced regular `data.parquet` files; it does not inspect
arbitrary per-file symlinks or numerical similarity between independent series.

All prior exclusion rules remain: originals with coarse/sign-inverted copies
and the copies themselves are training-only, including when sibling copies
were not admitted. Repeated augmentation suffixes remain excluded. No persistent
family cache is introduced: newly generated siblings are discovered each run.
There is no model, label, split fraction or optimizer change.

INFO messages now report phase starts, progress about every 30 seconds, elapsed
time and ETA when the phase's total is known. Directory listing has an unknown
total and reports entries scanned rather than a misleading percentage. Completion
is also reported. No per-table filesystem-resolution loop remains.

A read-only sample of 1,000 real paths on the WSL `/mnt/e` project drive measured
5.750 s for the old per-source resolve versus 0.00576 s for shared-prefix indexing,
about 999x faster for that operation, with identical resulting paths. See
`family_path_resolution_sample.json`. This is not a whole-screening benchmark:
enumerating millions of raw directories and classifying all sources still costs
time, and concurrent training/startup filesystem contention affects timings.

The live process was left running and cannot adopt Python module edits already
loaded in memory. Restarting the trainer applies this fix and reuses completed
`.record_metadata` sidecars; there is no transform/embed rerun or metadata-cache
invalidation. Existing training checkpoints and result files were not touched.

Regression tests cover 20,001 source paths across two date directories and
assert exactly two path resolutions, one scan for mixed relative/absolute date
aliases, directory-link family identity where symlinks are supported, plus the
existing family-isolation, unadmitted sibling and disjoint validation tests.

Verification: 314 tests completed without failures; 13 were skipped (12 CUDA
tests and one directory-symlink test unavailable under this Windows account).
Pre-commit checks passed. The live training process was not stopped or modified.

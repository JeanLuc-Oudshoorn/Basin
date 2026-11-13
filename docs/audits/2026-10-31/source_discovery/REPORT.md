# Silent WSL source discovery

The live Chronos-Bolt validation process (WSL PID 363) was active, spending time
in Windows-drive filesystem calls, while its log remained at the stage-level
`status.json` write. Chronos-Bolt loading is lazy; no model load or forecasting
had started at that point. The selected October 31 cohort had 150,000 table
directories. Recursive `Path.rglob` discovery over `/mnt/e` took about 687.7 s,
from 20:49:58 until the first selected-folder message at 21:01:25 local time.

The process was neither killed nor restarted. It subsequently loaded
Chronos-Bolt Small on CUDA and began producing successful evaluations, with
regular progress updates every 100 tables. Model-load metadata recorded
47,718,016 parameters and 13.19 s load time. Its live log is
`logs/20261031T194957_798875Z_validate_ee1f199f/events.log`.

## Fix

Windows validation and transformation entry points now discover files natively
and pass a run-local file index through `--source-manifest` when launching WSL.
The index uses relative paths, binds the source root and canonical date selection,
and preserves recursive ordering. The receiving process passes it to nested
validation/transformation through a process-local environment variable. Each
new launcher invocation creates a fresh snapshot; it is not a persistent cache
that silently excludes newly generated tables on later runs.

Index loading rejects traversal, absolute/drive-qualified paths, duplicates,
unselected dates and mismatched cohorts. It checks the root and selected date
directories but does not stat or resolve every table path. Normal screening
still reads and hashes each actual table, so the index does not bypass source
freshness or evaluation checks.

Native discovery logs its start and a five-second heartbeat, including matched
file count. First-batch preparation and model loading also announce themselves
before work begins, including before importing Chronos.

## Verification

Native discovery/index creation for the actual 150,000 tables took 17.25 s;
loading the resulting 8.7 MB index from WSL took 0.95 s. The measurements are
single observations during an active run, not a controlled cold-cache comparison.
See [benchmark.json](benchmark.json).

The Windows source-discovery/Bolt/date-selection suite passed all 19 tests.
The WSL source-discovery suite passed five tests and skipped the Windows-only
launcher test. Tests cover recursive selection/order, metadata-free index loads,
run-local membership, cohort binding, unsafe/duplicate paths and forwarding of
portable paths by both launchers. Ruff and Bandit checks pass.

The current validation uses its already-loaded code and has progressed beyond
the scan. Leave it running; future launches use the fix automatically.

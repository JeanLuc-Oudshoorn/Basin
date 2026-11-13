# Prophet disk exhaustion and recovery

The run starting October 25 at 21:07 local time reached the end at October 26
01:22, but its process-level `complete` status concealed 51,420 failed table
evaluations. Of 150,000 tables, 98,580 evaluated successfully: 70,806 passed
screening and 27,774 lost to the baseline. The remaining 51,420 were failures,
not evidence of unusable predictive information. 51,418 errors explicitly mention
disk exhaustion; the other two are CmdStanPy `fd_out` initialization errors in
the same run. The Parquet row count and SHA-256 matched the original summary.
See [findings.json](findings.json).

Windows TEMP/TMP pointed to C:/Users/.../AppData/Local/Temp. CmdStanPy creates a
process-level temporary directory on import and normally retains model CSV and
stdout files until the worker exits. Long-lived screening workers accumulated
enough output to fill C:, even though source tables and metrics lived on E:.
PyCharm's logs independently record failures to write its own cache/log files
to C: during the same period. At diagnosis C: had recovered approximately 11.4
GiB free; large validator temp directories were no longer present.

## Changes

- Temp storage is selected on the project drive before Prophet/CmdStanPy import.
  The default root is `.cache/prophet`. A per-fit temporary output directory is
  removed after predictions are copied, on success and on failure.
- Storage exhaustion stops validation instead of becoming another rejected row.
  Summary fields distinguish successful and failed evaluations.
- `--retry-errors` verifies source hashes and screening signatures, retains
  unchanged successful scores, and retries failed, changed and missing tables.
- Previous reports, summaries and progress CSVs are archived on E: before
  progress is replaced. Final metrics are still atomically replaced.
- The local PyCharm module excludes generated data, results, logs and `.cache`.

A recovery run was launched in the background with `--retry-errors`. Its log is
`results/02_data_validation/storage_repair/stderr.log`; its process metadata is
in `process.json` in that directory. The original final report remains in place
until recovery finishes; progress is checkpointed separately. The report will
not describe recovery as fully successful while error rows remain.

## IDE storage

The active IDE is PyCharm 2025.3.3 installed on E:, but its system directory is
C:/Users/Stardustnuke/AppData/Local/JetBrains/PyCharm2025.3. This is independent of
the installation/project location, as described in
[JetBrains' directory documentation](https://www.jetbrains.com/help/pycharm/directories-used-by-the-ide-to-store-settings-caches-plugins-and-logs.html).

`scripts/relocate_pycharm_storage.ps1 -WaitForExit` was launched hidden and waits
for PyCharm to close. It then copies the entire existing system directory to
E:/JetBrains/PyCharm2025.3/system, preserving Local History and other IDE state.
Only after a successful copy does it update custom properties and VM options
for system storage, logs and Java temporary files on E:. Configuration backups
and the original C: system directory are retained. It does not delete C: data or
change global Windows environment variables. If PyCharm reopens while copying,
the script waits for another exit and repeats the copy before changing settings.

Close PyCharm and wait until
`E:/JetBrains/PyCharm2025.3/migration-status.json` says `complete` before reopening.
The script never forcibly terminates the IDE. The IDE settings update cannot
take effect in an already-running JVM.

## Verification

All 17 focused Prophet/date-selection/admission tests passed. Additional coverage
checks per-fit cleanup on success/failure, environment restoration, disk-full
fail-fast behavior, selective retries, hash freshness and report archival.
A fresh real-Prophet process evaluated three tables successfully, confirmed its
CmdStanPy temp root was on E:, and left no per-fit output directories. Ruff and
Bandit passed, and the relocation script passed PowerShell syntax parsing.

# Holdout console logging

`holdout.progress_every_tables` defaults to 100 in `configs/config.json`.
The evaluator reports one INFO progress line at tables 100, 200, etc., and at the
final table if it falls between intervals. Lines include completed/total tables,
percentage, forecast-origin count, elapsed time and ETA. Start/end summaries and
unexpected warnings/errors remain visible.

Per-origin metrics, per-table RWKV timings, Prophet fit timings and progress-file
write messages use DEBUG. `timed` and `write_json` accept an optional log level;
their default remains INFO for other stages. The Prophet and CmdStanPy standard
loggers temporarily filter below WARNING during fitting, then restore their
previous levels. This suppresses chain start/done messages forwarded from WSL.

Detailed metrics still go to the JSONL journal. `test_metrics_progress.json`
still updates after every table. Model predictions and evaluation coverage are
unchanged. A currently running Python worker needs restarting to load the new
logging policy.

The regression check processes 201 small mocked forecast tables and verifies
that progress INFO messages occur only at 100, 200 and 201 and contain an ETA.
Verification log: `logs/20261029_holdout_logging_tests.log`.

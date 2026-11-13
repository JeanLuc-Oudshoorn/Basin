# Separate holdout-test stage

## Stage boundaries

The pipeline now has `src/05_model_training`, `src/06_holdout_test` and
`src/07_forecasting`. The previous forecasting directory was renamed from
`06_forecasting` to keep the numeric ordering. Its shared holdout evaluator moved
to `06_holdout_test/holdout.py`; the new executable is
`06_holdout_test/holdout_test.py`.

`train.py` finishes after selecting the best validation weights, exporting
`model.pt`, writing training metrics and closing its training journal. It no
longer imports the holdout evaluator, fits Prophet, writes final test metrics,
or blocks training completion on out-of-sample evaluation. The 10% dataset split,
epoch-zero validation, full epoch validation, scheduler, early stopping and
training checkpoint/resume behavior are unchanged.

## Running the test

```powershell
uv run python src/06_holdout_test/holdout_test.py
```

The default model is `paths.training/model.pt`, currently
`results/447m_model/training/model.pt`. Training replaces this
export after each completed run, so it automatically represents the latest
exported trained model, using that run's best validation epoch. This selection
does not scan unrelated experimental files by modification time. A custom
`--config` can select a different training directory.

`latest_checkpoint.pt` serves optimizer recovery; the holdout stage loads the
exported `model.pt`. If training has only produced recovery checkpoints and no
export, the test stage raises a clear error rather than initializing a model or
silently choosing an arbitrary file. If a newer run has not exported yet, any
existing export remains the selected model; `model_used.json` identifies it.

Testing replays the split saved in the model, verifies the embedding checksum,
requires exactly one evaluation record per saved test source and checks those
records' hashes against the saved split. The evaluator verifies each source
parquet hash before scoring. It never draws a new random holdout partition.

## Outputs and logs

`paths.holdout_test` defaults to
`results/447m_model/holdout_test`. Older configurations without this
key use the training directory's sibling `holdout_test`. All test outputs go
directly into this directory:

- `model_used.json`: model path/checksum, best epoch/update and test-table count.
- `test_metrics.json`: pooled, per-table and per-origin RWKV/Prophet/naive scores.
- `test_predictions.parquet` and `test_table_*.parquet`: forecasts and actuals.
- `test_table_*_target_*.png`: comparison plots.
- `test_metrics_progress.json`: completed/total tables, elapsed time and ETA.

The up-to-five-origin protocol and its settings are unchanged. Obsolete
`holdout.output_subdir` was removed from the default config because the output
directory now has its own path key. Existing partial results in former forecast
directories are retained and are not copied or treated as results of a new run.
Rerunning testing replaces the fixed result filenames; timestamped ordinary logs
and JSONL journals remain separate per invocation under `logs/`.

`run_context` now recognizes `holdout_test` as a journal-owning stage. Windows
and WSL share the same journal identity, including success/failure events. Source
provenance includes both new stage directories. The existing forecasting command
is now `src/07_forecasting/forecast.py`; modern model exports still delegate to
the holdout evaluator for compatibility. The explicit holdout command is the
recommended way to test the trained model.

## Verification

`tests/test_holdout_stage.py` exercises the real training export path with a
small CPU stand-in and asserts that no holdout evaluation occurs and the journal
completes. It separately exercises the new entry point, automatic model path,
saved-split filtering, missing/duplicate/changed records, model provenance and
the separate successful holdout journal. Existing holdout protocol tests cover
prediction origins, covariate alignment, context caps, partial tails and metrics.

Logs: `logs/20261029_holdout_stage_entry_tests.log`,
`logs/20261029_separate_holdout_tests.log`, and
`logs/20261029_separate_holdout_full_tests.log`.

The bounded GPU workflow in `scripts/verify_training_protocol.py` now invokes
training and the new holdout entry point separately, using ten temporary 64-row
tables and isolated audit outputs. It uses the full configured model, exports
the trained weights, then reloads them through the new stage and runs the real
Prophet/naive comparison. Evidence is under `gpu_smoke/` and
`logs/20261029_separate_holdout_gpu.log`. Production model and data files are not
modified by this check.

Final results: the CPU regression suite passed 147 tests with seven CUDA-only
tests skipped (154 total). The full-size GPU smoke passed: training returned
after two optimizer updates and export; the separate stage reloaded that export
and scored two eligible origins on the isolated test table using real Prophet
and naive forecasts. Ruff and Bandit checks passed for the new stage.

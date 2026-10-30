# Forecasting stage

Run `uv run python src/07_forecasting/forecast.py` for the existing saved-horizon
forecasting workflow. This folder was previously named `06_forecasting`.

For the normal trained model's dataset holdout test, use the dedicated stage:

```powershell
uv run python src/06_holdout_test/holdout_test.py
```

Training saves the model and exits; it does not run that test automatically.
The holdout stage selects `paths.training/model.pt` and the split stored in it,
and writes to `paths.holdout_test` (default
`results/next_patch_300m_validated/holdout_test`). It compares RWKV, Prophet and
last-observation carry-forward at up to five origins per test table.

For compatibility, calling `forecast.py` with a modern checkpoint containing a
saved dataset split still delegates to the holdout evaluator and writes to the
same holdout output directory. Both commands avoid training.

See `src/06_holdout_test/README.md` and `docs/forecasting/FORECASTING.md`.

# Holdout test stage

Run from the repository root:

```powershell
uv run python src/06_holdout_test/holdout_test.py
```

This automatically loads `paths.training/model.pt`, the latest model exported
by training (using that run's best validation weights). It replays the test split
stored in that model and verifies the embedding and source checksums. It does
not train or draw a new split. `train.py` now stops after saving its artifacts.

The default output is `results/next_patch_300m_validated/holdout_test`, configured
by `paths.holdout_test`. It contains metrics JSON, prediction parquet files,
interactive Plotly HTML plots, live progress/ETA and `model_used.json` recording the model's
checksum and selected epoch. Ordinary logs and a timestamped JSONL journal are
written under `logs/`, including when Windows routes execution to WSL.

Each test table contributes up to five one-token forecasts at the configured
origins, comparing RWKV, Prophet and last-observation carry-forward. All protocol
settings remain under `holdout` in `configs/config.json`.
`holdout.benchmark_models` selects comparisons (RWKV is always included):
`["prophet", "naive", "chronos"]` by default; add `"timestral"` to include it.
`holdout.sample_percent` defaults to 100; set it to e.g. 1 for a seeded 1% sample
of the saved holdout tables. `sample_selection.json` records the selected sources.
All models use those same sources, origins and historical context.
Prophet now uses historical covariates by default (`prophet.use_covariates=true`).
It forecasts each nonconstant covariate with Prophet, then uses those forecasts
as regressors for its outcome prediction (`prophet.covariate_forecast="prophet"`).
No observed future covariates are supplied. Set `covariate_forecast="naive"`
to carry covariates forward for a faster comparison, or `use_covariates=false`
to reproduce the univariate baseline. Extra fits increase evaluation time.

Holdout comparison plots use a separate reproducible random sample of the
evaluated tables: `plot.max_plots=10`, `plot.sample_seed=7`. Each plot overlays
actuals and every model selected in `holdout.benchmark_models`, plus RWKV.
If fewer tables are evaluated, all are plotted; set `max_plots=0` to disable.
`plot_selection.json` lists the chosen sources and their HTML filenames.
Open the HTML files in a browser to zoom, pan, hover over predictions and toggle
models using the legend. Plotly is embedded, so the files work offline.

Convert an existing completed run's saved random examples without rerunning any
models:

```powershell
uv run python src/06_holdout_test/plotting.py
```

This reuses the saved selection and parquet predictions, updates plot filenames
in the manifests, and leaves forecast values and scores unchanged. Original PNG
files are retained. Future holdout runs generate HTML automatically.

Chronos-2's public checkpoint is 119.5M parameters; no public 300M–500M Chronos-2
checkpoint was available when implemented. Timestral-209M is available without
authentication and is an optional dependency:

```powershell
uv sync --inexact --extra validation --extra benchmarks
```

Both weights are downloaded anonymously into `.cache/huggingface` on first use.
Chronos is a regular project dependency. Model IDs, pinned revisions and runtime
parameter counts are recorded in the metrics report. See the
[foundation benchmark audit](../../docs/audits/2026-10-29/foundation_benchmarks/REPORT.md).
Console progress is reported every 100 tables and at the final table, including
percentage, elapsed time and ETA (`holdout.progress_every_tables`). Per-origin
metrics and timing use DEBUG; Prophet/Stan fit chatter is suppressed while
warnings and errors remain visible. The live progress file still updates after
every table.

An epoch recovery checkpoint (`latest_checkpoint.pt`) serves training resume;
testing uses the exported `model.pt`. If no export exists, this stage explains
that training must finish first. Results from earlier evaluations stay in their
previous directories; rerunning this stage replaces its fixed output filenames.

See [the stage separation audit](../../docs/audits/2026-10-29/separate_holdout_stage/REPORT.md).

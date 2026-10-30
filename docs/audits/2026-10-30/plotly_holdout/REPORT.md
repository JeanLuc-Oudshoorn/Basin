# Interactive holdout comparison plots

Holdout evaluation now renders Plotly HTML files instead of Matplotlib PNGs.
Plotly 6.9.0 is installed in the Windows and WSL environments, declared as a
regular dependency and recorded in `uv.lock`. Matplotlib remains installed for
other existing visualization workflows.

Each selected example shows actual values and every benchmark model. Independent
forecast origins are separated by gaps in each model trace, so the graph does
not imply predictions between sampled origins. Each model has one legend item
that toggles all its forecast segments. Hover text shows forecast origin,
timestep and predicted value. Range slider, scroll-wheel zoom, drag zoom/pan and
reset controls are available. Forecast values and metrics are unchanged.

`plot.max_plots=10` and `plot.sample_seed` still control selection. Filenames are
`test_table_<index>_target_<index>.html`, recorded in `plot_selection.json` and
`test_metrics.json`. The JavaScript is embedded for offline viewing, following
[Plotly's HTML-export documentation](https://plotly.com/python/interactive-html-export/).
Each file is several MB; no browser server, CDN or GPU is needed to view it.

To convert the saved example selection of a completed evaluation:

```powershell
uv run python src/06_holdout_test/plotting.py
```

The converter reads per-table parquet predictions, original table observations
and the saved model/plot selections. It verifies that table identities match
before rendering. It updates only plot metadata in the metrics/selection files,
not scores or predictions. Existing PNGs remain. The current completed run's
ten plots were regenerated successfully; models were not loaded or run again.

Verification checks all selected traces, forecast-gap preservation, range slider,
embedded offline JavaScript and regeneration with unchanged aggregate scores.
Logs: `logs/20261030_plotly_tests.log`, `logs/20261030_regenerate_plotly.log`.
Pre-commit hooks are run before committing. The earlier bulk change is already
committed as `8a37724`; this change is a separate commit and does not add unrelated
untracked experiment/configuration files or generated results.

Final verification: 165 regression tests completed (158 passed, seven CUDA-only
tests skipped). All pre-commit hooks passed after their formatting fixes were
restaged, including Ruff, Bandit and line-ending checks. Logs:
`logs/20261030_plotly_full_tests.log`, `logs/20261030_plotly_precommit.log`.

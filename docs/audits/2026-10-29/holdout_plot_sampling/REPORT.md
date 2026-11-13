# Random example plots

Holdout plotting now samples evaluated tables with an independent seeded hash
ranking rather than taking the first tables. Settings in `configs/config.json`:

```json
"plot": {
  "max_plots": 10,
  "sample_seed": 7
}
```

Existing width, height and DPI settings still apply. At most `max_plots` tables
are chosen; if fewer tables are evaluated, all are chosen. A value of zero
disables plotting. Each target in a selected table receives one PNG containing
actuals and all evaluated model forecasts, with consistent model colors and
separate lines for each forecast origin. Unsampled time intervals are not joined
by forecast lines.

`plot_selection.json` in `results/447m_model/holdout_test` records
the seed, chosen table sources and PNG filenames. The selection is also recorded
in `test_metrics.json`. It is made from the evaluated holdout sample, and does
not affect metric coverage or require extra inference. A separate `plots:` hash
prefix prevents using the same ranking as the holdout dataset sample itself.

The change applies when the evaluation script is next launched. Existing files
from earlier evaluations remain; the new selection manifest identifies the
current run's examples.

# Generated-data date selection

## Configuration

The default configuration now separates the generated-data root from the selected
date folders. Edit these fields in configs/config.json:

```json
{
  "paths": { "raw": "data/01_generated" },
  "data": { "generated_dates": ["2026-10-25"] }
}
```

This is a fragment of the existing config; retain the other fields. One date or
multiple dates use the same list format. The committed default selects only
2026-10-25, the date currently present in the generated-data root. For multiple
dates, use a list such as ["2026-10-25", "2026-10-26"] once both folders exist.

Dates must be unique, real calendar dates in YYYY-MM-DD form, with a nonempty list.
An invalid, missing or empty selected date folder is an error. Dates are processed
in ascending order, with deterministic table ordering inside each date. Changing
the list order alone does not change the dataset or invalidate artifacts. Wildcards
and automatic inclusion of future folders are intentionally unsupported: an
explicit list provides a reproducible cohort.

## Shared pipeline selection

The shared discover_source_tables helper in transform.py is used by the Prophet
validation entry point and both transform modes. Only data.parquet files inside
the selected date folders are discovered. Other sibling dates are ignored.

The existing predictability gate then admits only selected tables where a valid
Prophet variant beats the naive forecast. The optional max_tables cap remains a
global cap after admission, not a per-date cap; its default null uses all admitted
tables. Date selection does not automatically invoke Prophet validation.

The validation report now uses date-qualified relative source keys, for example:

```text
2026-10-23/table_001/data.parquet
2026-10-25/table_001/data.parquet
```

Those are different sources even when table directory names match. Keys are
relative to paths.raw, work across Windows/WSL, and remain protected by existing
source-content hashes and screening-settings signatures. A combined report may
contain extra dates, but transformation considers only currently selected paths.

The default validation output is now
results/02_data_validation/selected_dates/metrics.parquet. Previous date-specific
reports are preserved. Run the stages in order after selecting a new cohort:

```powershell
uv run python src/02_data_validation/validate_generated_data.py
uv run python src/03_transforming_input_data/transform.py
uv run python src/04_creating_embeddings/embed.py
uv run python src/05_model_training/train.py
```

Embedding and training inherit the selected cohort from processed/embedded
artifacts; neither independently scans raw data. Every invocation's config is also
saved in the existing run logs and training JSONL journal.

## Artifact safety and audit trail

For configs using generated_dates, the pipeline signature includes the normalized
raw root and sorted selected dates, as well as the existing seed/data/model and
validation settings. A different date list or root rejects old artifacts before
embedding or model initialization. Existing processed-to-embedded and
embedded-to-checkpoint content-hash checks continue to apply.

The source selection is saved explicitly in processed and embedded payloads,
manifests, the validation summary and admission audit. Per-record source paths
also retain their date folder. No raw datasets, prior checkpoints, cached artifacts
or historical reports were rewritten by this implementation.

The new default configuration needs freshly generated validation, transformation
and embedding outputs before training, even for the same selected day: the source
key layout and signature now express explicit date selection. Cached old data is
not silently retagged or accepted under the new configuration.

## Compatibility

Historical configs without data.generated_dates retain their recursive paths.raw
selection and exact previous signature formula. Thus archived experiments pointing
directly at one date folder remain usable. An explicitly present null or empty
list is an error rather than an implicit request to train on every date.

Each --config file is independent. The regular pipeline uses configs/config.json;
capacity experiments with their own config files retain their recorded source
settings. Add the same root/list fields to such a config to opt it into explicit
date selection, and regenerate its upstream artifacts accordingly.

## Verification

Three dedicated tests cover:

- Two selected dates with the same table name and an excluded third date, through
  screening, transformation, embedding and training-artifact loading. The screening
  scores are stubbed to isolate selection; real Prophet fitting is unchanged.
- Changing the selection causes the actual train(config) entry point to reject
  stale embeddings before CUDA initialization. Root changes also alter signatures,
  while list reordering does not.
- Invalid lists/calendar dates, duplicates, missing/empty folders and legacy
  single-folder behavior, including exact old-signature compatibility.

All three tests passed. Regression suites passed two admission tests, two pipeline
tests, five JSONL journal tests, and four CPU next-patch tests; one CUDA-only
next-patch test was skipped. This totals 16 passing tests and one skip.
No accuracy experiment or new model training was needed for source selection.

Source hashes, selected-date discovery evidence and log paths are in evidence.json.

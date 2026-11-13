# Model folder naming

2026-10-31. The default model's results now live in `results/model`, with
`training`, `holdout_test` and `forecasts` subdirectories. Archived 447M results
live in `results/447m_model` with the same three stage subdirectories.

`configs/config.json` uses `data/02_processed/model`,
`data/03_embedded/model` and `results/model`. `configs/legacy_447m.json`
uses matching `447m_model` folders. The existing archived processed and embedding
directories were renamed along with their results directory. Default processed
and embedded directories are created by their pipeline stages when needed.

Changes cover active configurations, scripts, user documentation, historical
audit text and saved text run metadata/logs. Current usage examples point to
the default model; historical benchmark inputs point to the archived 447M model.
The root README contains the complete stage and artifact tree.

Only directory names and text references were changed. Binary checkpoint,
embedding-index, Parquet and window files were not rewritten, so their contents
and binary checksums are preserved. This is not a model conversion or retraining.
Artifact loading resolves processed, embedding and checkpoint locations through
the configuration. Training checkpoint identity does not include these renamed
output-directory paths; source dataset paths remain unchanged.

Historical text files now spell paths using their current locations. Their
original timestamps, metrics and recorded hashes were retained. A recorded hash
of an edited historical text configuration describes its original bytes, not
the path-renamed text; binary-artifact hashes still describe unchanged bytes.
`migration.json` lists every text file changed by the mechanical replacement and
records former folder names as underscore-separated name components. Subsequent
README and current-usage edits clarify the default/archive distinction.

Verification checks configuration JSON, expected folder locations, disappearance
of former directory names, absence of former path spellings in maintained source
and runtime text, and repository formatting/static checks. Generated caches,
virtual environments, Git history, vendored sources and binary artifacts are
outside the text migration.

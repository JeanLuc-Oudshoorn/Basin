# Training JSONL module location

Moved `training_log.py` from `src/03_transforming_input_data` to
`src/05_model_training`. It is a reusable training utility, not a standalone
script. No extra command is needed to create a progress journal.

`train.py` imports the utility and decorates `train(config)` with
`@logged_training("train")`. Therefore both the regular CLI and a direct Python
call create a timestamped JSONL automatically, even if data loading subsequently
fails. The CLI's outer `run_context` reuses the same journal. Windows launchers
and WSL workers share its identity and file, avoiding duplicate run logs.
Files remain under the configured `paths.logs` directory (normally `logs/`).

The shared stage context now imports the training journal only for training or
capacity diagnostics. Transform and embedding stages retain their normal Loguru
logs without depending on a logging module in the preprocessing directory.
Updated the worker subprocess test to import from the new training location.

Validation: all six training-journal tests passed, including automatic logging
from the actual training functions, unique filenames, exception records, nested
contexts, and shared launcher/worker journals. Ruff lint and formatting checks
passed for the moved module and updated callers.
The complete regression suite also passed: 110 tests, six CUDA-dependent skips.
Output is saved in `logs/20261026_training_log_relocation_tests.log`. The transform,
embedding and training CLI entry points were checked independently with `--help`.

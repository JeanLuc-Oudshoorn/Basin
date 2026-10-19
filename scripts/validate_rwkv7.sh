#!/usr/bin/env bash
set -euo pipefail
source "$(dirname -- "${BASH_SOURCE[0]}")/activate_rwkv7.sh"
RWKV_PREFER_CACHED_EXTENSIONS=0 python -c \
    'from rwkv7_backend import load_rwkv; load_rwkv(); print("RWKV backbone built")'
RWKV_PREFER_CACHED_EXTENSIONS=1 PYTHONPATH="$BASIN_PROJECT_ROOT${PYTHONPATH:+:$PYTHONPATH}" \
    python scripts/verify_cached_rwkv7.py
RWKV_PREFER_CACHED_EXTENSIONS=1 python smoke_rwkv7_timeseries.py
python -m pip freeze > requirements-rwkv7-observed.txt
sha256sum third_party/rwkv-lm/RWKV-v7/train_temp/src/model.py \
    third_party/rwkv-lm/RWKV-v7/train_temp/cuda/* > rwkv7-source-sha256.txt
# The guide deliberately overrides cuda-toolkit's runtime version. Record it.
if python -m pip check > docs/rwkv7-pip-check.txt 2>&1; then
    cat docs/rwkv7-pip-check.txt
else
    cat docs/rwkv7-pip-check.txt
    python - <<'PY'
from pathlib import Path
lines = Path('docs/rwkv7-pip-check.txt').read_text().splitlines()
assert lines and all(
    line.startswith('cuda-toolkit 13.0.2 has requirement nvidia-cuda-runtime==13.0.96')
    and '13.3.29' in line for line in lines
), 'Unexpected dependency conflict; inspect docs/rwkv7-pip-check.txt'
print('Only the documented CUDA runtime override conflict was found.')
PY
fi

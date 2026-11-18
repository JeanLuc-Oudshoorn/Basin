#!/usr/bin/env bash
# End-to-end smoke test of the current architecture on a small corpus.
# Run from Bash inside Ubuntu-24.04:  bash scripts/smoke_v2.sh [stage ...]
# Stages (default: all): transform index train holdout
set -euo pipefail
source "$(dirname -- "${BASH_SOURCE[0]}")/activate_rwkv7.sh"
CONFIG="${BASIN_SMOKE_CONFIG:-configs/smoke_v2.json}"
stages=("$@")
[ ${#stages[@]} -eq 0 ] && stages=(transform index train holdout)
for stage in "${stages[@]}"; do
  case "$stage" in
    transform) python -u src/03_transforming_input_data/transform.py --config "$CONFIG" ;;
    index) python -u src/04_patch_indexing/index_patches.py --config "$CONFIG" ;;
    train) python -u src/05_model_training/train.py --config "$CONFIG" ;;
    holdout) python -u src/06_holdout_test/holdout_test.py --config "$CONFIG" ;;
    *) echo "Unknown stage: $stage" >&2; exit 2 ;;
  esac
done

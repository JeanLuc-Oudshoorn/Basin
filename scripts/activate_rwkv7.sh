#!/usr/bin/env bash
# Source this file from Bash inside Ubuntu-24.04.
export BASIN_PROJECT_ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
export BASIN_VENV="${BASIN_VENV:-$HOME/projects/basin-runtime/.venv-wsl}"
source "$BASIN_VENV/bin/activate"
unset CUDA_HOME CUDA_PATH
export TORCH_EXTENSIONS_DIR="$HOME/.cache/torch_extensions/basin-py312-torch212-cu130-sm120-v1"
export MAX_JOBS=4
export TORCH_CUDA_ARCH_LIST='12.0'
export RWKV_SKIP_LM_LOSS=1
export RWKV_EXTENSION_VERBOSE=1
export RWKV_PREFER_CACHED_EXTENSIONS=1
mkdir -p "$TORCH_EXTENSIONS_DIR"
cd "$BASIN_PROJECT_ROOT"

#!/usr/bin/env bash
set -euo pipefail
PROJECT_ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
export BASIN_VENV="${BASIN_VENV:-$HOME/projects/basin-runtime/.venv-wsl}"
sudo apt-get update
sudo apt-get install -y python3.12-venv python3.12-dev build-essential
if [[ ! -f "$BASIN_VENV/bin/activate" ]]; then
    python3.12 -m venv "$BASIN_VENV"
fi
source "$BASIN_VENV/bin/activate"
python -m pip install 'pip' 'wheel' 'setuptools==70.2.0'
python -m pip install 'torch==2.12.0+cu130' --index-url https://download.pytorch.org/whl/cu130
python -m pip install 'pytorch-lightning==1.9.5' 'torchmetrics==1.9.0' 'ninja==1.13.0' 'packaging==26.2'
python -m pip install --no-deps \
    'nvidia-cuda-nvcc==13.3.73' 'nvidia-cuda-crt==13.3.73' \
    'nvidia-nvvm==13.3.73' 'nvidia-cuda-runtime==13.3.29' \
    'nvidia-cuda-cccl==13.3.3.4.1'
nvidia-smi
python -c 'import torch; print(torch.__version__, torch.version.cuda); print(torch.cuda.get_device_name()); print("bf16:", torch.cuda.is_bf16_supported())'
cd "$PROJECT_ROOT"
if [[ ! -f third_party/rwkv-lm/RWKV-v7/train_temp/src/model.py ]]; then
    python scripts/bootstrap_rwkv7.py
fi

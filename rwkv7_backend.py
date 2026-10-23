import importlib
import os
import sys
from pathlib import Path

import torch

from rwkv7_cuda_paths import configure_cuda_home


def load_rwkv(*, ctx_len=1024, inference_only=False):
    if ctx_len % 16:
        raise ValueError("RWKV context length must be divisible by 16")
    if not torch.cuda.is_available() or not torch.cuda.is_bf16_supported():
        raise RuntimeError("This backend requires CUDA with bf16 support")

    configure_cuda_home()
    os.environ.update(
        {
            "RWKV_MY_TESTING": "x070",
            "RWKV_KERNEL": "@rwkv3",
            "RWKV_CTXLEN": str(ctx_len),
            "RWKV_HEAD_SIZE": "64",
            "RWKV_HEAD_L2WRAP_CE_CHUNK": "0",
            "RWKV_FLOAT_MODE": "bf16",
            "RWKV_JIT_ON": "0",
            "RWKV_INFERENCE_ONLY": "1" if inference_only else "0",
        }
    )
    os.environ.setdefault("RWKV_PREFER_CACHED_EXTENSIONS", "1")
    torch.backends.cudnn.enabled = True
    torch.backends.cudnn.benchmark = True
    torch.backends.cuda.matmul.allow_tf32 = True

    train_dir = (
        Path(__file__).resolve().parent
        / "third_party"
        / "rwkv-lm"
        / "RWKV-v7"
        / "train_temp"
    )
    sys.path.insert(0, str(train_dir))
    previous_dir = Path.cwd()
    os.chdir(train_dir)  # The vendored CUDA source paths are relative.
    try:
        return importlib.import_module("src.model")
    finally:
        os.chdir(previous_dir)

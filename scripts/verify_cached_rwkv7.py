"""Fail if importing the backbone calls the extension builder, even for a check."""

import time

import torch.utils.cpp_extension as cpp_extension


def unexpected_build(*args, **kwargs):
    raise RuntimeError(f"Unexpected extension build/check: {kwargs.get('name')}")


cpp_extension.load = unexpected_build
from rwkv7_backend import load_rwkv  # noqa: E402 -- install the build guard before importing

started = time.perf_counter()
load_rwkv()
print(
    f"RWKV module import: {time.perf_counter() - started:.2f}s; builder was never called"
)

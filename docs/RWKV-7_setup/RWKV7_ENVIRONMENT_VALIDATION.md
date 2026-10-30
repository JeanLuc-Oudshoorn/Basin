# Basin environment validation

Validated on 2026-10-19 in this Basin checkout using the ordered steps in
`README_RWKV7_FAST_SETUP.md` and `scripts/setup_rwkv7.sh`.

| Component | Observed |
| --- | --- |
| OS | Ubuntu 24.04.4 LTS under WSL2 |
| Python | 3.12.3 |
| Virtual environment | `/home/stardustnuke/projects/basin-runtime/.venv-wsl` |
| GPU | NVIDIA GeForce RTX 5070, 12 GB, compute capability 12.0 |
| Windows driver | 610.88 |
| GCC/G++ | 13.3.0 |
| PyTorch | 2.12.0+cu130; CUDA build 13.0 |
| CUDA nvcc / CRT / NVVM | 13.3.73 / 13.3.73 / 13.3.73 |
| CUDA runtime / CCCL | 13.3.29 / 13.3.3.4.1 |
| Lightning / TorchMetrics | 1.9.5 / 1.9.0 |
| Ninja / packaging / setuptools | 1.13.0 / 26.2 / 70.2.0 |
| DeepSpeed | Not installed; optional for the custom training loop |

The environment and extension cache reside on the Linux filesystem. The source
remains in the user's Windows Basin checkout. Source-import timing may differ
after moving or cloning the project.

## Results

`bash scripts/validate_rwkv7.sh` completed with exit status 0:

- All seven backbone CUDA operator libraries compiled in a fresh Basin cache.
- A fresh Python process replaced `torch.utils.cpp_extension.load` with a
  function that raises on any call. All seven libraries loaded directly and
  the builder was never called.
- The timed `load_rwkv()` call took **0.63 seconds**, after importing
  `torch.utils.cpp_extension` and the standalone backend. This is the guide's
  module-import measurement, not total Python process startup or training speed.
- The exact guide smoke test passed with output shape **(2, 32, 4)** and loss
  **1.8579**. Forward, backward, finite-gradient assertions, a nonzero attention
  output-weight gradient, and an AdamW update all completed.
- `python -m pip check`: **No broken requirements found.** The guide's deliberate
  CUDA runtime override is installed; pip did not report a conflict here.
- Bash syntax checks, Python compilation of Basin's setup/loader/smoke files,
  and Git whitespace checks of Basin-authored files passed. Original upstream
  whitespace is preserved in vendored files.

The persistent extension-cache root is:

```text
/home/stardustnuke/.cache/torch_extensions/basin-py312-torch212-cu130-sm120-v1
```

The generated CUDA overlay is:

```text
/home/stardustnuke/.cache/mamba_cpp_build_overlays/mamba-cuda-build-a4d315884e1b
```

See `requirements-rwkv7-observed.txt`, `rwkv7-source-sha256.txt`, and
`docs/rwkv7-pip-check.txt` for the installed versions, post-adaptation source
hashes, and dependency-check output. The freeze is an observation, not a
replacement for the ordered Torch/toolkit installation recipe.

This validates the backend and a small synthetic training loop. A 400-million-
parameter forecasting model and its training memory/throughput remain to be
implemented and measured.

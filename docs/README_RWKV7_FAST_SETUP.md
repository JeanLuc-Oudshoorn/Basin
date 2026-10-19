# Recreating this project's fast RWKV-7 environment

This guide records the setup found in this checkout on 2026-10-19 and gives a
copyable recipe for another project, including a continuous-valued time-series
training smoke test. The main startup optimization is **loading already-built
CUDA operator libraries with `torch.ops.load_library()`**, bypassing repeated
`torch.utils.cpp_extension.load()` / Ninja checks. Initial compilation is still
required once for each compatible build configuration.

## What we changed here

| Change | Where it lives | What it fixes |
| --- | --- | --- |
| Direct loading of cached `.so` operator libraries | `_cached_extension_library_path()` and `_load_extension()` in [the patched model](third_party/rwkv-lm/RWKV-v7/train_temp/src/model.py) | Repeated Ninja checks on later process launches |
| Inference import without Lightning, DeepSpeed, or the LM loss extension | `RWKV_INFERENCE_ONLY` branches in the same file | Unneeded training dependencies and loss-extension import during prediction |
| Persistent, space-free CUDA and Torch library paths | `configure_cuda_home()` and its helpers in [the experiment](examples/rwkv7_experiment.py) | Ninja/linker problems with the PyCharm path containing spaces |
| A `libcudart.so` alias and matching CUDA compiler/header wheels | The CUDA overlay and `.venv-wsl` | Missing runtime linker name and missing/incompatible CCCL headers |
| Skip selected default initializers before immediately restoring checkpoint weights | `skip_default_parameter_initialization()` in [prediction](examples/rwkv7_predict.py) | Work spent initializing weights that will be overwritten |

The current defaults matter:

- `examples/rwkv7_predict.py` sets `RWKV_INFERENCE_ONLY=True` and
  `RWKV_PREFER_CACHED_EXTENSIONS=True` before importing the vendored model.
- `examples/rwkv7_experiment.py` defaults both options to `False`.
  Its `configure_rwkv_environment()` **overwrites/removes the corresponding
  environment variables** according to those Python settings. Setting only a
  shell variable does not enable its cache shortcut: change the Python constant
  or set `experiment.RWKV_PREFER_CACHED_EXTENSIONS = True` before calling
  `experiment.import_rwkv_model_module()`.
- The current `examples/rwkv7_4096_experiment.py` also defaults the cache shortcut
  to `False`; `examples/rwkv7_long_context_experiment.py` enables it.

These switches are local patches, not universal RWKV package options. Copy the
patched source in this working tree. A fresh upstream clone or `pip install rwkv`
does not reproduce this implementation. The vendored upstream starting commit is
`952102498e9ed367ea0a59ee64106916d474d30f`, with additional local changes. In
particular, some current changes to `src/model.py` are uncommitted.
The inspected `src/model.py` SHA-256, before the time-series adaptation below,
is `63ca38c31bd9a09f1cd5107a0b7d7213c2e697e1b8b5f6cfa0e0519d53b61a1b`.

### Why some Ninja output can remain

The fast loader only bypasses builds for extensions loaded with
`is_python_module=False`. With `RWKV_KERNEL=@rwkv3`, these are the seven libraries:

```text
rwkv7_clampw_v3
rwkv7_cmix_bf16_v5
rwkv7_tmix_mix6_bf16_v5
rwkv7_tmix_kk_pre_bf16_v5
rwkv7_tmix_lnx_rkvres_xg_bf16_v1
rwkv7_tmix_a_gate_bf16
rwkv7_tmix_vres_gate_bf16_v3
```

Normal LM training also imports `rwkv7_l2wrap_ce_bf16_v2`, a **Python extension
module** whose returned object provides `.forward()` and `.backward()`. The
existing shortcut intentionally does not bypass its Ninja check. Do not replace
that load with `torch.ops.load_library()`: it would return no Python module.
Inference skips it. The time-series recipe below adds a separate option to skip
this unused LM loss while preserving training initialization and autograd.

Ninja messages about checking or having no work do not necessarily mean CUDA is
being recompiled. PyTorch's loader emits build instructions and invokes Ninja
even when the existing outputs are up to date. That behavior is described in
the [PyTorch extension documentation](https://docs.pytorch.org/docs/2.12/cpp_extension.html).

## 1. Match the runtime

The environment inspected here has:

| Component | Observed value |
| --- | --- |
| OS | Ubuntu 24.04 under WSL2 |
| Python | 3.12.3, Linux virtual environment `.venv-wsl` |
| GPU | NVIDIA GeForce RTX 5070, 12 GB, compute capability 12.0 |
| Windows NVIDIA driver | 610.88 |
| PyTorch | `2.12.0+cu130`; `torch.version.cuda == '13.0'` |
| C++ compiler | Ubuntu GCC/G++ 13.3.0 |
| `nvidia-cuda-nvcc` | `13.3.73` |
| `nvidia-cuda-crt` / `nvidia-nvvm` | `13.3.73` / `13.3.73` |
| `nvidia-cuda-runtime` | `13.3.29` |
| `nvidia-cuda-cccl` | `13.3.3.4.1` |
| `ninja` | `1.13.0` |
| `pytorch-lightning` / `torchmetrics` | `1.9.5` / `1.9.0` |
| `deepspeed` | `0.19.3` (installed here; optional for the custom loop below) |

The CUDA 13.3 compiler/toolkit wheels and Torch's CUDA 13.0 build are different
version numbers. This table describes the working local combination; it is not
a requirement that every RWKV-7 installation use these versions. Torch's
`cuda-toolkit==13.0.2` dependency metadata requests an older CUDA runtime wheel;
our runtime-wheel replacement can therefore produce a `pip check` conflict.
The commands below make that replacement explicitly, after installing Torch.
Do not silently substitute newer packages and reuse the old binaries.

Use a WSL interpreter in the other project's IDE/run configuration. A native
Windows `.venv/Scripts/python.exe` cannot use these Linux `.so` files. The strict
Ubuntu/Python/venv checks in this project's `experiment_runtime.py` are local
experiment policy; the standalone loader below does not import that module.

From **PowerShell**, enter the existing distribution:

```powershell
wsl.exe -l -v
wsl.exe -d Ubuntu-24.04
```

If setting up another machine, install WSL2/Ubuntu 24.04 first and use an NVIDIA
Windows driver that supports its GPU and CUDA runtime. CUDA in WSL uses the
Windows driver; follow [NVIDIA's WSL setup instructions](https://docs.nvidia.com/cuda/wsl-user-guide/index.html).

All following shell blocks are **Bash inside Ubuntu-24.04**, in the same shell
unless stated otherwise:

```bash
set -euo pipefail
export MAMBA_SOURCE='/mnt/e/PyCharm/PyCharm Community Edition 2022.2.3/Projects/Mamba'
export RWKV_PROJECT="$HOME/projects/rwkv7-timeseries"
mkdir -p "$RWKV_PROJECT"
cd "$RWKV_PROJECT"

sudo apt-get update
sudo apt-get install -y python3.12-venv python3.12-dev build-essential
python3.12 -m venv .venv-wsl
source .venv-wsl/bin/activate

python -m pip install 'pip' 'wheel' 'setuptools==70.2.0'
python -m pip install 'torch==2.12.0+cu130' \
  --index-url https://download.pytorch.org/whl/cu130
python -m pip install 'pytorch-lightning==1.9.5' 'torchmetrics==1.9.0' \
  'ninja==1.13.0' 'packaging==26.2'
python -m pip install --no-deps \
  'nvidia-cuda-nvcc==13.3.73' 'nvidia-cuda-crt==13.3.73' \
  'nvidia-nvvm==13.3.73' 'nvidia-cuda-runtime==13.3.29' \
  'nvidia-cuda-cccl==13.3.3.4.1'

nvidia-smi
python -c 'import torch; print(torch.__version__, torch.version.cuda); print(torch.cuda.get_device_name()); print("bf16:", torch.cuda.is_bf16_supported())'
```

`--no-deps` applies only to the explicitly listed toolkit wheels, whose compiler,
CRT, NVVM, runtime, and header components are all supplied above. The Torch CUDA
wheel is available from the [official cu130 wheel index](https://download.pytorch.org/whl/cu130/torch/).
DeepSpeed is needed if you retain the vendored DeepSpeed checkpointing/optimizer
paths; the smoke test uses `grad_cp=0` and an ordinary PyTorch optimizer. The
vendored model imports DeepSpeed automatically when it is installed, so leaving
it out of a custom-loop environment also avoids that import overhead.

Keep the new project's environment, source, and build cache on the WSL Linux
filesystem when possible. This is an additional improvement over the current
`/mnt/e` environment, whose imports can still be slow even after skipping Ninja.
Microsoft documents the [filesystem performance advantage](https://learn.microsoft.com/en-us/windows/wsl/filesystems).
If reusing the current interpreter temporarily, select
`$MAMBA_SOURCE/.venv-wsl/bin/python`; create a new virtual environment rather than
copying the old environment directory to another path.

## 2. Copy the patched RWKV source

Run this once in the new project, before a destination `train_temp` exists:

```bash
mkdir -p third_party/rwkv-lm/RWKV-v7
test ! -e third_party/rwkv-lm/RWKV-v7/train_temp
cp -a "$MAMBA_SOURCE/third_party/rwkv-lm/RWKV-v7/train_temp" \
  third_party/rwkv-lm/RWKV-v7/
cp "$MAMBA_SOURCE/third_party/rwkv-lm/LICENSE" third_party/rwkv-lm/LICENSE
```

Preserve the upstream notices/license when vendoring. Keep the whole `cuda/`
directory along with `src/model.py`; copying just Python code is insufficient
for the first build or subsequent rebuilds. Record these copied files in the
other project's version control, including the local cache-loader patch.

For **continuous time-series training**, apply this small change to the copied
model. It lets your regression/forecasting loss replace LM cross entropy without
pretending the backbone is inference-only:

```bash
python - <<'PY'
from pathlib import Path

p = Path('third_party/rwkv-lm/RWKV-v7/train_temp/src/model.py')
s = p.read_text()
old = '''if os.environ.get("RWKV_INFERENCE_ONLY") == "1":
    L2WRAP_CE_CUDA_V2 = None'''
new = '''if (
    os.environ.get("RWKV_INFERENCE_ONLY") == "1"
    or os.environ.get("RWKV_SKIP_LM_LOSS") == "1"
):
    L2WRAP_CE_CUDA_V2 = None'''
assert s.count(old) == 1, 'Expected the local patched RWKV source; inspect before editing'
p.write_text(s.replace(old, new))
PY
```

`RWKV_SKIP_LM_LOSS` is an **additional adaptation introduced by this recipe**.
It is not already present in the Mamba source. With it enabled, do not call
`l2wrap_cross_entropy()` or the vendored LM `training_step()`; use your own loss.
For LM training, omit this adaptation or leave `RWKV_SKIP_LM_LOSS=0`.

## 3. Extract the CUDA path helpers without importing the LM experiment

This creates `rwkv7_cuda_paths.py` in the new project by copying the exact helper
functions from the current experiment. It does not import the experiment, its
data loaders, tokenizer, logging utilities, or runtime guard.

```bash
python - <<'PY'
import ast
import os
from pathlib import Path

source = (Path(os.environ['MAMBA_SOURCE']) / 'examples/rwkv7_experiment.py').read_text()
names = {
    'prepend_env_path', 'ninja_safe_overlay_root', 'ninja_safe_build_path',
    'ninja_safe_cuda_home', 'ensure_symlink', 'ensure_versioned_library_alias',
    'ninja_safe_torch_lib_path', 'require_cuda_cccl_headers',
    'configure_torch_cpp_extension_paths', 'configure_cuda_home',
}
functions = [n for n in ast.parse(source).body
             if isinstance(n, ast.FunctionDef) and n.name in names]
assert {n.name for n in functions} == names
header = '''from __future__ import annotations
import hashlib
import os
from pathlib import Path
import sys
import torch

PROJECT_ROOT = Path(__file__).resolve().parent
'''
Path('rwkv7_cuda_paths.py').write_text(
    header + '\n\n' + '\n\n'.join(ast.get_source_segment(source, n) for n in functions) + '\n'
)
PY
```

These helpers find `nvcc` in the active virtual environment's
`lib/python3.12/site-packages/nvidia/cu13/bin/`. They create persistent overlays
under `~/.cache/mamba_cpp_build_overlays` when needed, link `bin`, `include`,
`nvvm`, and runtime libraries, and supply `lib/libcudart.so` if only a versioned
runtime name exists. They also update **both** `os.environ['CUDA_HOME']` and
`torch.utils.cpp_extension.CUDA_HOME`, plus `TORCH_LIB_PATH`, so an earlier Torch
import cannot leave the extension builder using stale paths.

If `CUDA_HOME` or `CUDA_PATH` already points at a different valid toolkit, the
helper prefers it. Unset those variables in the shell to use the venv wheels:

```bash
unset CUDA_HOME CUDA_PATH
```

The retained `MAMBA_CPP_BUILD_OVERLAY_DIR` override can select another persistent
directory; it must be writable and contain no spaces. A mere symlink around a
project path is not sufficient if later code calls `Path.resolve()` and recovers
the original spaced path. The overlay fixes the actual paths used by the linker.

## 4. Add a standalone loader

Save the following as **`rwkv7_backend.py` in the new project root**:

```python
import importlib
import os
from pathlib import Path
import sys

import torch
from rwkv7_cuda_paths import configure_cuda_home


def load_rwkv(*, ctx_len=1024, inference_only=False):
    if ctx_len % 16:
        raise ValueError('RWKV context length must be divisible by 16')
    if not torch.cuda.is_available() or not torch.cuda.is_bf16_supported():
        raise RuntimeError('This backend requires CUDA with bf16 support')

    configure_cuda_home()
    os.environ.update({
        'RWKV_MY_TESTING': 'x070',
        'RWKV_KERNEL': '@rwkv3',
        'RWKV_CTXLEN': str(ctx_len),
        'RWKV_HEAD_SIZE': '64',
        'RWKV_HEAD_L2WRAP_CE_CHUNK': '0',
        'RWKV_FLOAT_MODE': 'bf16',
        'RWKV_JIT_ON': '0',
        'RWKV_INFERENCE_ONLY': '1' if inference_only else '0',
    })
    os.environ.setdefault('RWKV_PREFER_CACHED_EXTENSIONS', '1')
    torch.backends.cudnn.enabled = True
    torch.backends.cudnn.benchmark = True
    torch.backends.cuda.matmul.allow_tf32 = True

    train_dir = (Path(__file__).resolve().parent / 'third_party'
                 / 'rwkv-lm' / 'RWKV-v7' / 'train_temp')
    sys.path.insert(0, str(train_dir))
    previous_dir = Path.cwd()
    os.chdir(train_dir)  # The vendored CUDA source paths are relative.
    try:
        return importlib.import_module('src.model')
    finally:
        os.chdir(previous_dir)
```

Set these options **before the first `src.model` import**, and use one RWKV
configuration per Python process. A later environment change does not re-run an
already imported module. In a larger application, avoid collisions with another
package also called `src`; rename/vendor the package imports if necessary.

For this time-series recipe, set the build environment in the WSL run
configuration or shell:

```bash
export TORCH_EXTENSIONS_DIR="$HOME/.cache/torch_extensions/rwkv7-timeseries-py312-torch212-cu130-sm120-v1"
export MAX_JOBS=4
export TORCH_CUDA_ARCH_LIST='12.0'
export RWKV_SKIP_LM_LOSS=1
export RWKV_EXTENSION_VERBOSE=1
mkdir -p "$TORCH_EXTENSIONS_DIR"
```

`12.0` is this RTX 5070's architecture. On another GPU, inspect
`torch.cuda.get_device_capability()` and set the actual architecture. `MAX_JOBS`
limits build parallelism; it does not skip Ninja or speed up already-running
kernels. `TORCH_EXTENSIONS_DIR` is a persistent cache **root**: each extension
gets its own subdirectory. PyTorch documents both controls in its
[extension build API](https://docs.pytorch.org/docs/2.12/cpp_extension.html).

The current Mamba environment leaves `TORCH_EXTENSIONS_DIR` unset and resolves
the operator cache to:

```text
/home/stardustnuke/.cache/torch_extensions/py312_cu130/<extension>/<extension>.so
```

On this same machine, with the same Python/Torch/CUDA ABI, GPU, and copied
operator sources, you can reuse it by instead setting:

```bash
export TORCH_EXTENSIONS_DIR="$HOME/.cache/torch_extensions/py312_cu130"
```

Do not point at the parent `~/.cache/torch_extensions`, since explicit
`TORCH_EXTENSIONS_DIR` replaces the version-specific root. A separate cache for
the new project is easier to maintain; it requires a one-time build.

## 5. Build once, then verify a startup that cannot invoke Ninja

For the time-series adaptation, build the seven backbone extensions in a fresh
Python process:

```bash
RWKV_PREFER_CACHED_EXTENSIONS=0 python -c \
  'from rwkv7_backend import load_rwkv; load_rwkv(); print("RWKV backbone built")'
```

This first command may take several minutes. Subsequent processes should use
the cache shortcut. Verify it with a check that **raises if any extension falls
back to the builder**:

```bash
RWKV_PREFER_CACHED_EXTENSIONS=1 python - <<'PY'
import time
import torch.utils.cpp_extension as cpp_extension

def unexpected_build(*args, **kwargs):
    raise RuntimeError(f'Unexpected extension build/check: {kwargs.get("name")}')

cpp_extension.load = unexpected_build
from rwkv7_backend import load_rwkv
started = time.perf_counter()
load_rwkv()
print(f'RWKV module import: {time.perf_counter() - started:.2f}s; builder was never called')
PY
```

With verbose caching enabled, expect seven `Loaded cached RWKV CUDA extension`
messages. This verifies a bypass, rather than merely hiding Ninja output. It
depends on `RWKV_SKIP_LM_LOSS=1`; for prediction of an LM checkpoint, use
`load_rwkv(inference_only=True)` instead. For full LM training, keep inference
mode off and accept the separate Python loss extension's build check.

Missing cached operator libraries, or an `OSError` while loading them, cause the
normal loader to fall back to compilation. The existing fast path checks for
`<name>.so`, not arbitrary `_v1`/`_v2` filename variants; a fresh process with
the normal build path is the simplest recovery after changing a build.

## 6. Verify forward and backward on continuous time-series inputs

Save this as **`smoke_rwkv7_timeseries.py` in the new project root**. It uses the
trainable RWKV blocks directly, projects four continuous channels to the model
width, and trains against a synthetic next-step target. It does not create a
token embedding or a vocabulary head.

```python
import torch
from torch import nn
from types import SimpleNamespace

from rwkv7_backend import load_rwkv

rwkv = load_rwkv(ctx_len=32, inference_only=False)
args = SimpleNamespace(
    n_embd=128, dim_att=128, dim_ffn=512, n_layer=2,
    head_size=64, my_testing='x070', grad_cp=0,
)


class ForecastSmoke(nn.Module):
    def __init__(self):
        super().__init__()
        self.input = nn.Linear(4, args.n_embd)
        self.blocks = nn.ModuleList([rwkv.Block(args, i)
                                     for i in range(args.n_layer)])
        self.norm = nn.LayerNorm(args.n_embd)
        self.output = nn.Linear(args.n_embd, 4)

    def forward(self, values):
        x = self.input(values)
        v_first = torch.empty_like(x)
        for block in self.blocks:
            x, v_first = block(x, v_first)
        return self.output(self.norm(x))


torch.manual_seed(7)
model = ForecastSmoke().to(device='cuda', dtype=torch.bfloat16).train()
optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)
series = torch.randn(2, 33, 4, device='cuda', dtype=torch.bfloat16)
optimizer.zero_grad(set_to_none=True)
prediction = model(series[:, :-1].contiguous())
loss = (prediction.float() - series[:, 1:].float()).square().mean()
loss.backward()
assert torch.isfinite(loss)
missing = {name for name, p in model.named_parameters()
           if p.requires_grad and p.grad is None}
# Layer zero stores v_first and does not use the value-residual gate.
assert missing <= {'blocks.0.att.v0', 'blocks.0.att.v1', 'blocks.0.att.v2'}
grads = [p.grad for p in model.parameters() if p.grad is not None]
assert all(torch.isfinite(g).all() for g in grads)
assert model.blocks[0].att.output.weight.grad.abs().max() > 0
optimizer.step()
torch.cuda.synchronize()
print(f'PASS: shape={tuple(prediction.shape)}, loss={loss.item():.4f}; forward/backward/update completed')
```

Run it with the same cache/environment settings:

```bash
RWKV_PREFER_CACHED_EXTENSIONS=1 python smoke_rwkv7_timeseries.py
```

For the real foundation model, preserve these backend constraints:

- Activations entering the custom kernels must be **bf16 and contiguous**.
  The original wrapper uses bf16 parameters and avoids an outer autocast context
  because LayerNorm was promoted to fp32 in this Torch build. Convert continuous
  inputs/projected patches accordingly; compute your regression loss in fp32.
- Sequence length must be divisible by **16**; pad and mask your forecasting
  loss as appropriate. For variable-length batches, also account for padding in
  whatever aggregation/head you add.
- Head size is **64** in several Python and CUDA locations, not just a tunable
  argument. Keep the model/attention width equal and divisible by 64 for this
  copied implementation, and use an FFN width divisible by 16.
- Use at least **two layers** with these constructors; initialization computes
  `layer_id / (n_layer - 1)`.
- Set `RWKV_INFERENCE_ONLY=0` for training. Inference mode skips some training
  initialization even though the CUDA operators themselves support backward.
- The smoke model validates wiring and gradients. It is not a proposed complete
  time-series architecture, pretraining objective, or pretrained checkpoint.
  Preserve the appropriate RWKV initialization when designing the real model;
  the full LM wrapper additionally calls `generate_init_weight()`.

These training kernels process a whole sequence and do not expose an external
recurrent-state input/output. Carrying state across streaming chunks or doing
truncated BPTT requires a separate stateful implementation; slicing inputs and
calling this backbone repeatedly resets the sequence computation.

## 7. Rebuild deliberately when the environment changes

Direct `.so` loading bypasses source timestamps, compiler-flag checks, and
PyTorch's build version checks. It can successfully load an outdated library.
Use a new cache directory and run the build step again after changing any of:

- Torch/Python version or ABI, CUDA toolchain, or target GPU architecture;
- the vendored `.cu`/`.cpp` files, compiler options, head size, or kernel variant;
- library locations referenced by the existing binaries, such as a moved venv.

For example, set a new suffix in **every** training/IDE launch configuration:

```bash
export TORCH_EXTENSIONS_DIR="$HOME/.cache/torch_extensions/rwkv7-timeseries-py312-torch212-cu130-sm120-v2"
mkdir -p "$TORCH_EXTENSIONS_DIR"
RWKV_PREFER_CACHED_EXTENSIONS=0 python -c \
  'from rwkv7_backend import load_rwkv; load_rwkv()'
```

Keep the overlay and toolkit/venv locations available for runtime linking. The
cache shortcut does not remove the need for compatible shared libraries.
`RWKV_EXTENSION_VERBOSE=1` prints the exact cached paths and load errors.

If an import appears stuck waiting on an extension lock, inspect the cache's
`lock` file and check for an active compiler/Python process first. Stop the
owning run if necessary. Remove a lock only after confirming its owner has
exited; an actively building process must retain its lock. A new cache directory
also avoids an abandoned cache without touching another process's build.

## 8. Keep startup speed separate from training speed

The direct loader changes startup, not the arithmetic performed by the kernels.
The copied model already uses the `@rwkv3` recurrence kernel and fused time-mix
and channel-mix kernels. `RWKV_JIT_ON=0` does not disable these CUDA extensions.
Keep those fused implementations when extracting the backbone.

For throughput measurements, warm up the model and optimizer first, synchronize
CUDA around timed sections, and exclude imports/compilation, checkpoint I/O,
validation, and model initialization. Keep training and validation batches
within VRAM; reducing the training microbatch does not reduce a separately
configured full validation batch. The [local long-context report](reports/rwkv7_long_context_vram_report.md)
records this distinction on the 12 GB RTX 5070.

Only skip initialization when immediately restoring every required weight from
a complete compatible checkpoint. The prediction helper skips selected default
initializers; it does not eliminate every constructor calculation. Never use
uninitialized weights for training from scratch.

Once the new project passes the cache check and forward/backward smoke test,
record the environment and source hashes there:

```bash
python -m pip freeze > requirements-rwkv7-observed.txt
sha256sum third_party/rwkv-lm/RWKV-v7/train_temp/src/model.py \
  third_party/rwkv-lm/RWKV-v7/train_temp/cuda/* > rwkv7-source-sha256.txt
python -m pip check
# The local CUDA runtime override described above may be reported here.
```

The freeze records installed versions; because of the explicit runtime override,
it is not automatically a conflict-free resolver lockfile. Preserve the ordered
Torch/toolkit install steps above along with it. For long-term reproduction,
archive the matching wheels and copied source in addition to the version list.

## Validation of this guide

On 2026-10-19, the helper-extraction command, time-series loss patch, standalone
loader, and training smoke example were executed from this document in an
isolated copy, using the existing `.venv-wsl` and its compiled operator cache.
All seven libraries loaded directly while the extension builder was replaced
with a function that raises on any call. The GPU forward/backward/optimizer step
passed with output shape `(2, 32, 4)` and finite loss/gradients. All Python
snippets parsed and all Bash blocks passed `bash -n`.

The training-mode module import still took about 131 seconds with the current
mounted-drive environment and its installed training dependencies. Skipping
Ninja does not eliminate that import cost. The fresh-environment installation,
cold compilation, and startup speed on the proposed Linux-filesystem layout
were not retested as part of writing this guide.

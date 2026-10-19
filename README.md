# Basin

Basin is a time-series foundation model project targeting approximately 400
million parameters, built on RWKV-7. This initial repository sets up and validates
the fused CUDA training backbone; the full forecasting architecture, datasets,
and pretraining pipeline are future work.

## Environment

Use Ubuntu 24.04 under WSL2, Python 3.12, and an NVIDIA GPU with bf16 support.
The exact setup and its local RWKV patches are documented in
[the original setup guide](docs/README_RWKV7_FAST_SETUP.md).
The initial installation and GPU checks passed; see
[the validation report](docs/RWKV7_ENVIRONMENT_VALIDATION.md).

From Bash inside Ubuntu-24.04, in this repository:

```bash
bash scripts/setup_rwkv7.sh
bash scripts/validate_rwkv7.sh
```

The first setup copies the verified patched source from the neighboring `Mamba`
project. Set `MAMBA_SOURCE` to change that source location. Once the vendored
files are present, setup no longer requires Mamba.

The virtual environment defaults to
`~/projects/basin-runtime/.venv-wsl` on the Linux filesystem. Set `BASIN_VENV`
consistently during setup and activation to use another location. In PyCharm,
select that environment's `bin/python` using the Ubuntu-24.04 WSL interpreter.
The native Windows `.venv` is not the CUDA training environment.

For subsequent training and smoke-test runs:

```bash
source scripts/activate_rwkv7.sh
python smoke_rwkv7_timeseries.py
```

Activation selects bf16 kernels for the RTX 5070 (`sm_120`), skips the unused LM
cross-entropy extension, and uses a persistent project-specific extension cache.
The validation script builds once, checks a fresh import with the builder
replaced by a function that raises, and runs continuous-valued forward,
backward, and optimizer steps. It records package versions, source hashes, and
dependency-check output.

Create a new extension-cache suffix in `scripts/activate_rwkv7.sh` after changing
the toolchain, GPU architecture, kernel sources, or compiler flags. Direct
library loading intentionally bypasses build freshness checks.

The repository can remain in this Windows checkout. For faster source imports,
clone it into WSL's Linux filesystem and run the same scripts there. The venv
and extension cache already live on the Linux filesystem.

## Backend constraints

Use contiguous bf16 activations, sequence lengths divisible by 16, equal model
and attention widths divisible by 64, FFN widths divisible by 16, and at least
two layers. Compute regression losses in fp32 and preserve training
initialization. These sequence kernels do not expose streaming recurrent state.

The vendored RWKV source retains its upstream Apache-2.0
[license](third_party/rwkv-lm/LICENSE). See
[provenance](docs/RWKV7_PROVENANCE.md) for the upstream revision and local patches.

# Default depth below 400M parameters

Updated 2026-11-02. `configs/config.json` now uses **22 layers** at width 1024,
up from 17 layers. This is the greatest whole layer count below 400,000,000
parameters when the other current architecture settings stay fixed.

| Layers | Total parameters |
| ---: | ---: |
| 17 (previous four-token default) | 307,295,520 |
| **22 (new default)** | **394,914,080** |
| 23 | 412,437,792 |

Each added layer includes both an RWKV temporal block and its cross-variable
attention mixer: 17,523,712 parameters. The count includes the learned patch
projection, role embeddings, all RWKV blocks, every variable mixer, output
normalization, the shared quantile forecast head and all three lead embeddings.
It excludes non-parameter buffers and optimizer state, as usual for parameter
counts. The new default is 5,085,920 parameters below the requested ceiling.

Only model depth changes. Patch size 16, history cap 2048, four direct output
patches (64 timesteps), nine quantiles, width 1024 and 80/16/3/1 lead loss weights
remain. Four-patch validation, sampled holdout horizons, inference graph replay,
training execution settings and output directories retain their policies.
No existing saved weights were modified or automatically expanded.

## Verification

`scripts/verify_model_capacity.py` constructs real `RWKVForecaster` instances
using the actual third-party RWKV backend on PyTorch meta tensors. It counts
every parameter without allocating full model weights in host RAM or VRAM and
checks that the configured depth is below the limit while the next depth is
at or above it. The reproducible report is
[`capacity_400m_verification.json`](capacity_400m_verification.json).

Run inside the configured WSL environment:

```bash
source scripts/activate_rwkv7.sh
python scripts/verify_model_capacity.py
```

This checks architecture size, not training memory, throughput or accuracy.
The previous 307M GPU smoke results remain historical measurements and should
not be presented as measurements of the 22-layer model. The seven focused
four-token policy tests pass with the new default configuration. The full-size
smoke script now follows the current configured architecture and saves to a
depth-specific audit directory, preserving earlier 307M smoke reports.

## Training migration

The existing exact-resume mechanism rejects a changed depth. Archive the
previous default run before starting a new one, regenerate transformed data
and the embedding index (their signatures include model configuration), then
train using the regular entrypoint. See
[the training migration commands](SHORT_HORIZON_TRAINING.md#migration-and-preservation-of-existing-runs).
The screening signature does not depend on depth, so a compatible four-patch
screening report can be reused. Older one-patch screening reports still need
refreshing because the reserved validation tail changed earlier.

The paths remain `results/model/training`, `results/model/holdout_test` and
`results/model/forecasts`; no architecture-specific default folder is added.
Forecasting a previously saved model continues to use that checkpoint's stored
architecture, independently of the new default depth.

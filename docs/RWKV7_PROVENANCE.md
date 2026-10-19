# RWKV-7 provenance

- Upstream: https://github.com/BlinkDL/RWKV-LM
- Starting commit: `952102498e9ed367ea0a59ee64106916d474d30f`.
- Copied subtree: `RWKV-v7/train_temp`, plus the upstream license.
- Local source: neighboring Mamba checkout, as specified by the setup guide.
- Mamba repository HEAD at copy time:
  `ac3355afa18d573a6b5e4b53f395da06b500f456` (with local working-tree changes).
- Source `src/model.py` SHA-256 before the time-series patch:
  `63ca38c31bd9a09f1cd5107a0b7d7213c2e697e1b8b5f6cfa0e0519d53b61a1b`.

The copied model includes local direct loading of cached CUDA operator libraries
and an inference-only import option. Basin applies exactly the additional
`RWKV_SKIP_LM_LOSS` condition from the guide to skip the unused LM cross-entropy
extension while retaining training initialization and autograd. The CUDA path
helpers are extracted from Mamba's `examples/rwkv7_experiment.py` using the
guide's AST extraction recipe.

The full copied CUDA source and upstream notices are retained. Generated
binaries and Python caches are excluded. Post-adaptation hashes are recorded in
`rwkv7-source-sha256.txt` after successful validation.

# CUDA graphs for training: feasibility experiment

This experiment measures Basin's real 1024-wide cross-variable attention module
in training mode, including backward, using its existing compiled math attention.
It compares ordinary compiled execution, PyTorch's `make_graphed_callables`, and
graph execution with gradient storage protected between microbatches.

`experiment.json` contains raw timings and gradient comparisons. Timings cover
four accumulated forward/backward calls, with five warmup updates and 20 measured
updates per variant. They exclude data preparation, RWKV blocks, gradient clipping,
the optimizer and logging, so these are **module timings, not epoch speedups**.
Both variants use the same weights and BF16 inputs. The experiment uses two input
shapes `[batch, variables, patches, width]`: `[1,4,32,1024]` and `[4,4,64,1024]`.
Parameter gradients are compared after one, two and four accumulated microbatches.

## Results, 2026-10-31

Measured on the RTX 5070 with PyTorch 2.12.0+cu130. Each timing includes four
forward/backward microbatches. The protected version includes gradient copies.

| Shape [B,V,T,D] | Existing compiled execution | Graph + gradient protection | Time reduction |
|---|---:|---:|---:|
| [1, 4, 32, 1024] | 5.220 ms | 2.628 ms | 49.7% |
| [4, 4, 64, 1024] | 5.250 ms | 4.593 ms | 12.5% |

All protected parameter gradients matched the reference **exactly** after one,
two and four microbatches for both shapes. The unprotected version matched only
one microbatch and failed accumulation comparisons. The comparison's FP32 TF32
flag was left disabled for both variants (the production RWKV loader enables it),
another reason to validate complete production updates before projecting savings.
Peak CUDA allocator usage stayed below 0.4 GiB. Capture took tens of milliseconds;
compilation and model setup are excluded from the warm timing.

This isolated result did not establish a full-model training speedup. Subsequent
implementation and complete-update measurements are in the
[training integration report](../training_graphs_integration/REPORT.md); that
bounded implementation is now enabled in the default training configuration.

## Why accumulation needs explicit checking

In this installed PyTorch version, the straightforward graph wrapper matched a
single backward but produced different accumulated parameter gradients. Graph
backward returns reusable static buffers. If a parameter's `.grad` retains that
storage, the next replay can overwrite it before the previous gradient has been
preserved. The protected variant clones retained gradients after each backward;
its copying overhead is included in timing. This is an experiment, not an
automatic change to production training.

The same class of issue is documented in
[PyTorch issue 181723](https://github.com/pytorch/pytorch/issues/181723).
[PyTorch's graph documentation](https://docs.pytorch.org/docs/main/notes/cuda.html#cuda-graphs)
describes full and partial forward/backward capture.

## Integration constraints

- Inputs must have stable shapes and memory addresses during replay. Basin has
  varying table lengths and variable counts; a bounded shape cache or shape
  buckets are needed rather than one universal graph.
- Training graphs retain backward intermediates and require substantially more
  memory than inference graphs. A large cache could trigger the shared-memory
  slowdown this project previously encountered.
- The vendored RWKV training recurrence launches on the default CUDA stream.
  Whole-update capture first requires a current-stream-aware forward **and
  backward** kernel. The inference-only kernel has no backward and cannot replace
  this training implementation.
- Gradient accumulation, activation checkpoint recomputation, optimizer updates,
  learning-rate changes and checkpoint resume all need end-to-end verification.
  Correct gradients in an isolated module do not establish those properties.

Run inside the WSL environment:

```bash
source scripts/activate_rwkv7.sh
python scripts/experiment_training_graphs.py
```

The script limits the CUDA allocator to 25% of VRAM, uses four CPU threads and one
compiler worker, writes after each completed shape, and stores run logs under
`logs/*_training_graphs_experiment_*`. `environment.json` identifies the runtime.
No model checkpoint or production training configuration is modified.

# Further training execution experiments — 2026-10-28

The selected change compiles the existing cross-variable attention calculation
for CUDA training. The final repeat comparison measured **21.85 → 25.46 training
windows/second**, a **16.5% throughput increase** and **14.2% reduction in timed
optimizer-update duration**. This is an additional gain over the already batched,
prefetched, selectively checkpointed trainer, not a comparison with the original
one-table trainer. See [machine-readable comparison](comparison.json).

## Scope and measurement

All throughput experiments used the actual **447,482,144-parameter**, 25-layer,
width-1024 model on the RTX 5070, BF16, AdamW, batch size 8, accumulation 1,
length/role bucketing, and the existing 2048 activation-checkpoint token budget
unless the experiment explicitly changed that budget. The context remains 2048
observations / 128 patches of 16 observations.

The cached sample contains 512 real training windows, selected after the whole-
table holdout split. It is the same sample as the previous throughput audit;
no production model/checkpoint or processed data was overwritten. Its 91 batches
include incomplete buckets, variable sequence lengths and 1–8 variables. A
separate stress run repeats the largest sampled corpus window in a full batch.

Final comparisons warmed every sampled batch shape for one complete sweep,
then measured a second 512-window sweep. Runs were sequential, in the order
reference, optimized, optimized, reference. Each update includes preparation,
forward, backward, gradient clipping, finite-value checks and the optimizer.
CUDA is synchronized at timing boundaries. Journals and console writes outside
the update timer, startup/compilation, full-epoch validation and checkpoint I/O
are excluded. These are bounded throughput measurements, **not a measured full
97,045-table epoch**. The initial reference overlapped the tail of CPU regression
testing; use the later repeat pair for the conservative headline comparison.

| Run | Windows/s | Timed 512-window sweep | Peak allocated GPU memory |
| --- | ---: | ---: | ---: |
| Reference, first comparison | 21.17 | 24.19 s | 8.22 GiB |
| Optimized, first comparison | 25.14 | 20.37 s | 8.37 GiB |
| Optimized, repeat | 25.46 | 20.11 s | 8.37 GiB |
| Reference, repeat | 21.85 | 23.43 s | 8.22 GiB |

The optimized stress test, `[B,V,T,D]=[8,8,128,1024]`, completed six measured
updates / 48 windows at 10.67 windows/s, with **8.61 GiB peak allocated** memory.
Allocator reservation is different from allocation: the mixed-shape sweeps
reserved up to 10.87 GiB, versus 10.83 GiB for the reference. The optimization
does not increase batch size or the activation-checkpoint budget.

## Implemented behavior

Code: [variate_attention.py](../../../../src/05_model_training/variate_attention.py),
[train.py](../../../../src/05_model_training/train.py).

The original module projects normalized variables into Q/K/V, performs scaled
dot-product attention independently at each timestamp, applies the output
projection, and adds the residual. The compiled path performs those same
operations and uses exactly the same trainable tensors. It keeps projections
batch-first and uses the math SDPA implementation, compiled by TorchInductor.
For these very short variable axes, this measured faster than the default flash
attention execution. The original
[profile](../training_speed/round2_reference_profile.txt) identified substantial
small-operation dispatch and attention-backward work.

`train.py` automatically enables this through `configs/config.json`:

```json
"compile_variate": true,
"compile_cache_dir": ".cache/torchinductor"
```

These are training execution settings, not architecture/hyperparameter changes.
At the time of the benchmarks, custom configurations without `compile_variate`
retained eager execution. Following the user's request to make compilation the
training default, omission now also enables compilation; explicit `false` remains
the opt-out. `train.py` calls this setup automatically before training.
Set it to `false` to disable compilation. No transform/embed rerun is needed.

Compilation is limited to gradient-enabled training with at most eight variables
and zero attention dropout, the range benchmarked here. Wider tables, evaluation,
forecasting and no-grad calls use the original MultiheadAttention path. CPU
configuration skips compilation. All layers share a dynamically shaped compiled
function; the module hierarchy and state-dict keys stay unchanged. Training and
forecast checkpoints remain loadable without compilation, and the resume
identity deliberately excludes this execution-only flag.

The first encounters with new shapes can take longer while compiling. The
initial exploratory compile trial spent about 21 seconds in its first update
and roughly 6 seconds each on two additional shape variants. Subsequent updates
reuse the compiled code. Cache files default to the repository on **E:**, not
the Windows user cache on C:. Explicit `TORCHINDUCTOR_CACHE_DIR` and
`TRITON_CACHE_DIR` environment overrides take precedence. The experimental and
verification caches together occupied about 85 MiB at the end of testing.

No changes were made to the JSONL schema, per-update progress messages, log
flushing, optimizer update boundaries, data order, training labels, learned
embeddings, loss, full end-of-epoch validation, test holdout, early stopping,
best-weight selection or atomic resumable checkpoints. BF16 attention kernels
can round differently, so numerical equivalence is tolerance-based rather than
a claim of bit-identical training trajectories.

## Other experiments and decisions

The small 12-update trials were screening experiments; their rates should not
be directly compared with the final 512-window measurements.

| Candidate | Evidence | Decision |
| --- | --- | --- |
| Batch-first projection without compilation | `direct_attention.json`, 20.26 windows/s in screening | No convincing standalone gain |
| Reentrant checkpointing | `reentrant.json`, 22.74 windows/s in screening | No convincing gain; keep current checkpoint behavior |
| Uncompiled math attention | `math_attention.json`, 22.88 windows/s in screening | Keep only as part of the compiled implementation |
| cuDNN attention | `logs/throughput_cudnn_attention_*` and its automatic failure journal | Rejected: no kernel for single-variable tables |
| Stop checkpointing variate mixers | `block_only.json` | Small/no gain, higher memory; rejected |
| Larger activation budget, 2560 | `budget2560_sweep.json`, 14.69 windows/s | Rejected: mixed-shape run developed stalls, peak allocation 9.05 GiB |
| Expandable allocator plus budget 2560 | `expandable2560_sweep.json`, 25.53 windows/s | Removed the severe stalls, but little gain and higher allocation; not enabled |
| Force cuBLASLt | `cublaslt_sweep.json`, 24.38 windows/s | No gain over contemporaneous reference sweep, 24.68 windows/s |
| Compiled math attention | `compiled_math_sweep.json`, then final production repeats | Selected after correctness and recovery checks |

The allocator experiment suggests memory fragmentation contributed to the stalls;
this is an inference, not a captured OS paging diagnosis. No global WSL, driver,
allocator or cuBLAS preference was changed for normal training.

## Correctness and recovery

- Full CPU suite: **143 tests, 136 passed, 7 CUDA-only skipped**. Log:
  `logs/20261028_execution_optimizations_tests.log`. The focused CPU tests were
  rerun after adding the wider-table fallback and passed.
- New attention tests compare outputs, input gradients and every parameter
  gradient against the original implementation for 1, 3, 8 and 11 variables.
  Float64 CPU comparisons pass at `atol=rtol=1e-10`.
- CUDA BF16 tests passed for compiled forward/backward, including non-reentrant
  checkpoint recomputation. Relative vector errors for outputs and each tested
  gradient are below 2%; all values are finite. Tests also verify unchanged
  checkpoint keys and eager evaluation/no-grad/wider-table behavior.
- The actual RWKV backend completed a two-epoch interrupted/resumed GPU run
  with compilation enabled. It matched the uninterrupted reference within BF16
  tolerance, restored CUDA AdamW moments and scheduler state, validated every
  fixture table after each epoch, selected best weights, and produced automatic
  JSONL journals. See [recovery result](resume/gpu_verification.json).
  This integration fixture uses the reduced 869,024-parameter model; the
  throughput and memory checks above use all 447M parameters.
- Ruff passed on changed Python files. Bandit passed on changed production
  modules; the verification script intentionally uses assertions.

All benchmark runs produce timestamped text logs and automatic training JSONL
journals under `logs/`. The recovery fixture intentionally records one failed
run before its successful restart. Production model artifacts were untouched.

## Reproduction

Inside the configured WSL environment, from the repository root:

```bash
source scripts/activate_rwkv7.sh
python scripts/experiment_training_execution.py --verify
```

This runs attention correctness tests, the final reference/optimized repeat
comparisons, the maximum-shape stress test and checkpoint recovery sequentially.
It reuses `docs/audits/2026-10-28/training_speed/sample.pt`; the benchmark script
can recreate that bounded sample from the prepared corpus if absent. Benchmark
result JSON is written under this audit directory. Individual experiment flags
are recorded in each result's `arguments` object and exposed by
`scripts/benchmark_training_throughput.py --help`.

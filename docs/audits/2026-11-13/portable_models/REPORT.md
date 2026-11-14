# Portable Basin models: FLA and pure-PyTorch rewrites (2026-11-13)

## Question

Can the trained Basin model (500M parameters, RWKV-7 on custom CUDA kernels) be
rewritten (1) on `flash-linear-attention` (FLA) and (2) entirely in Python, with
the same weights and the same predictions, so that Hugging Face users do not
need CUDA, Linux and a C compiler? And what does that cost in latency?

## Answer

Both rewrites exist, load the exact evaluated weights, and reproduce the
original's forecasts to within the original's own numerical noise. They are not
bit-identical end to end, and cannot be (see "Why not bit-identical").

| | Original | FLA variant | Pure-PyTorch variant |
|---|---|---|---|
| Needs | Linux, NVIDIA GPU, CUDA toolkit (nvcc), C++ compiler, 8 hand-built kernels | Linux, NVIDIA GPU, `pip install flash-linear-attention`, gcc or clang | `torch` + `safetensors` only |
| Runs on CPU | no | no (measured: Triton rejects CPU tensors) | yes |
| Runs on native Windows | no | not as installed (needs `triton-windows`, untested) | yes (measured) |
| Median latency per forecast, RTX 5070 | 7.9 ms (25.7 ms without CUDA-graph replay) | 43.6 ms | 61.9 ms |
| Median latency per forecast, CPU (6 threads) | n/a | n/a | 213 ms |
| Median-forecast difference from original (mean, in context standard deviations) | 0.0032 (its own second code path) | 0.0041 | 0.0037 |
| Holdout MASE on the 100 sampled forecasts | 1.4598 | 1.4602 | 1.4612 |

Only the pure-PyTorch variant meets the stated goal in full. The FLA variant
removes the CUDA toolkit and the hand-built kernels, but FLA imports Triton
unconditionally, and Triton needs a GPU and a C compiler for its launcher stub
(`triton/runtime/build.py` looks for `CC`, gcc or clang). It is the faster of
the two on a GPU.

## Outputs

| What | Where |
|---|---|
| Pure-PyTorch model folder | `results/model/portable/basin-rwkv7-pytorch/` |
| FLA model folder | `results/model/portable/basin-rwkv7-fla/` |
| Source of both (tracked) | `src/08_model_export/` |
| Unit tests (CPU, 7 tests) | `tests/test_model_export.py` |
| Raw result summaries | `docs/audits/2026-11-13/portable_models/results/` |

Each model folder is self-contained and uploadable as is: `model.safetensors`
(1.0 GB, bfloat16 as trained), `config.json` (architecture plus provenance),
two Python files and a README with a usage snippet. `results/` is git-ignored,
so the weights are not committed.

Both variants default to the precision the model was trained and evaluated in:
bfloat16 weights and activations, with float32 inside the same steps where the
training kernels use float32 (state recurrence, gates, group norm, attention
core). `dtype=torch.float32` is available as an option.

## Steps taken

1. **Pinned the weights.** The holdout stage evaluated `best_state_dict` from
   `latest_checkpoint.pt` (optimizer step 848,905). `export_models.py`
   recomputes the holdout's weight fingerprint and refuses to export unless it
   matches `model_used.json`; it matched
   (`c8c323a3…6fca8`). After writing, every tensor is read back and compared
   bit for bit.
2. **Pure-PyTorch model** (`basin_common.py`, `modeling_basin_pytorch.py`).
   Parameter names equal the checkpoint's one to one (1,110 tensors,
   500,056,352 parameters). Each fused CUDA kernel became a small function that
   rounds at the same points as the kernel. The state recurrence is a loop
   over time with two forms of the same equation: five small operations per
   step on CPU, one matrix operation per step on GPU.
3. **FLA model** (`modeling_basin_fla.py`). Uses FLA 0.6.0's own `RWKV7Block`.
   Weights are relabelled to FLA's layout: low-rank matrices transposed,
   low-rank offsets moved to a bias. Three layer-0 tensors that RWKV-7 never
   uses (132,096 values) are dropped. No value is changed.
4. **Reference run.** `reference_forecasts.py` ran the original in WSL on 240
   real holdout requests (stratified over context length, 1 to 6 variables).
   Its medians equal the medians saved by the 2026-11-12 holdout run in all
   15,360 values, so the reference is the evaluated model.
5. **Operation-level test** (`verify_blocks.py`), then **end-to-end test and
   timing** (`verify_ports.py`) on 100 of those requests (40 for CPU).

Two corrections came out of step 5 and are in the final code. The token-shift
kernel rounds once (a fused multiply-add, `torch.addcmul`), not twice. And the
original's variable attention computes its core in float32.

## Evidence that the models are the same

**Per operation, identical inputs** (8 requests, all 28 layers, 20M values per
operation, bfloat16 on the GPU):

| Operation | Bit-identical to the original kernel |
|---|---|
| Tokenizer, patch embedding, forecast head | 100% |
| Token-shift mix, channel mix (FFN) | 100% |
| Learning-rate gate | 100% |
| Key update | 99.9995% |
| Value residual gate | 99.994% |
| Group norm + bonus + output gate | 99.993% |
| State recurrence (vs both original kernels) | 99.960% |
| Variable attention (vs the holdout's compiled path) | 99.988% |
| Whole RWKV block | 99.56% (99.89% within one rounding step) |

For scale: the original's own two attention paths (compiled vs eager) agree in
98.67% of values on the same inputs. The remaining mismatches are float32
summation order inside the recurrence and norms, which PyTorch cannot replicate.

**End to end** (100 forecasts, 64 steps, 9 quantiles; difference of the median
forecast in units of each series' context standard deviation):

| Model, precision, device | Mean | Max | MASE | Mean per-request MASE change |
|---|---|---|---|---|
| Original, second code path (yardstick) | 0.0032 | 0.063 | 1.4602 | 0.0030 |
| Pure PyTorch, bfloat16, GPU | 0.0037 | 0.063 | 1.4612 | 0.0037 |
| FLA, bfloat16, GPU | 0.0041 | 0.039 | 1.4602 | 0.0041 |
| Pure PyTorch, float32, GPU | 0.0064 | 0.071 | 1.4620 | 0.0073 |
| FLA, float32, GPU | 0.0064 | 0.071 | 1.4620 | 0.0073 |
| Pure PyTorch, bfloat16, Windows CPU (40 forecasts) | 0.0044 | 0.047 | 1.5352 vs 1.5363 | 0.0058 |

The original scores 1.4598 on these 100 forecasts. The ports move the MASE by
under 0.1%, in both directions per request (42 to 44 better, 56 to 58 worse).

**Independent cross-check.** In float32 the FLA and pure-PyTorch variants agree
with each other to 2e-7 standard deviations on average (max 4e-5, 80% of values
bit-identical). Two independent implementations of the recurrence giving the
same numbers is strong evidence that both are correct. The pure variant on
Windows CPU and on the Linux GPU agree to 7e-7.

## Why not bit-identical

The model runs in bfloat16, which keeps about three significant digits. A
change in rounding of one value in a few thousand at layer 0 grows to roughly
0.3% of the hidden state by layer 27 (measured on two long requests: 99% of
values identical after layer 0, about 20% after layer 27). Any change in arithmetic order produces this,
including the original's own alternative code path, which is why that path is
the yardstick above. Float32 is further from the original (0.0064) than the
bfloat16 ports because it removes the rounding noise the original has.

## Latency

One forecast = one table at one origin, 64 steps, CPU history in to CPU
forecast out, including tokenisation and transfers; steady state after one
warm-up pass. GPU rows use 100 requests, CPU rows 40.

| Model | Device | Median | Mean | p95 |
|---|---|---|---|---|
| Original, CUDA-graph replay (holdout setting) | RTX 5070 | 7.9 ms | 9.7 ms | 25.2 ms |
| Original, no graph replay | RTX 5070 | 25.7 ms | 25.8 ms | 27.2 ms |
| FLA, bfloat16 | RTX 5070 | 43.6 ms | 52.3 ms | 80.7 ms |
| FLA, float32 | RTX 5070 | 48.0 ms | 55.4 ms | 80.7 ms |
| Pure PyTorch, bfloat16 | RTX 5070 | 61.9 ms | 71.8 ms | 117.4 ms |
| Pure PyTorch, float32 | RTX 5070 | 57.9 ms | 68.2 ms | 129.3 ms |
| Pure PyTorch, bfloat16 | CPU, Windows, 6 threads | 213 ms | 327 ms | 910 ms |
| Pure PyTorch, float32 | CPU, Windows, 6 threads | 302 ms | 458 ms | 1,238 ms |
| Pure PyTorch, bfloat16 | CPU, WSL, 6 threads | 198 ms | 321 ms | 979 ms |
| Pure PyTorch, float32 | CPU, WSL, 6 threads | 287 ms | 460 ms | 1,337 ms |

Reading these:

- The original's 7.9 ms comes from CUDA-graph replay. Without it the original
  takes 25.7 ms, so the FLA variant is 1.7x that and the pure variant 2.4x.
- On the GPU both ports are bound by per-operation launch overhead, not by
  arithmetic: the shortest request (2 tokens) still takes about 40 ms. The pure
  variant adds a Python loop over time steps, so it grows with context length.
- One-off costs: FLA compiles its Triton kernels on first use (first forecast
  7.6 s with a warm Triton cache; 25 to 50 s cold). Weights load in 3 to 11 s on
  the GPU and under 1 s on CPU. Python import time for torch/transformers/FLA
  varied between 3 and 35 s in WSL depending on disk cache and is not included.
- Not measured: batched throughput, other GPUs, CPUs other than this machine's.

### Recurrence rewrite for the GPU (2026-11-14)

The first version of the pure variant ran five small operations per time step
on every device and measured 84.3 ms median (100.0 mean, 187.6 p95) on the GPU.
Writing the step as one matrix product, `S <- S (diag(decay) + a b^T) + v k^T`,
cut that to the 61.9 ms in the table. A second run of the same code measured
72.5 ms, so read the gain as 15 to 25% at the median and about 35% at p95
(long contexts). The remaining time is the roughly 40 ms of per-layer launch
overhead that no change to the recurrence removes. The CPU keeps the
five-operation form because the matrix form does 64 times more arithmetic per
step. The matrix form switches TF32 off for its own products; with TF32 on, only
86% of recurrence outputs matched the original kernel instead of 99.96%.

For comparison, the holdout run on the same GPU measured Chronos-2 (119M
parameters) at 16.2 ms and Timestral (209M) at 43.8 ms median per forecast.

## Try it

`python experiments/portable_models/run_pytorch.py` and
`python experiments/portable_models/run_fla.py` forecast the last 20% of
`data/00_public/airline-passengers.csv` from the first 80%, print error metrics
and show a plot (also saved under `results/model/portable/demo/`). `--data`,
`--column`, `--device` and `--dtype` change the defaults. The FLA script
relaunches itself in WSL when started from Windows.

## Things to know before publishing

- **FLA's chunked mode.** For 64 or more tokens FLA switches to a chunked
  algorithm that is numerically looser (median-forecast difference 0.0053
  instead of 0.0041 overall; about double on long contexts). The FLA variant
  therefore feeds at most 63 tokens at a time through FLA's own state cache,
  which keeps it on the exact recurrent kernel at the same speed.
  `model.exact_recurrence = False` restores FLA's default.
- **FLA's own warning.** FLA prints, per layer, that its RWKV-7 is "potentially
  buggy" and should be cross-checked against BlinkDL's code. This audit is that
  cross-check for the inference path. The loader silences the warning.
- **FLA version.** Tested with `flash-linear-attention==0.6.0`,
  `fla-core==0.6.0`, Triton 3.7.0, torch 2.12.0. FLA was installed in a
  separate directory; the training environment was not modified.
- **Scope.** Inference only (forecast, with rollout beyond 64 steps). No
  training, no recurrent-state reuse across requests, and no CUDA-graph cache.
- **Licence.** The model cards carry no licence line; choose one before upload.
- **Stale status file.** `logs/20261112T082505_134514Z_train_b2979718/status.json`
  still says `running` although that run ended on 2026-11-12 19:28. Nothing was
  touched there.

## Reproduce

```bash
# anywhere, CPU only
python src/08_model_export/export_models.py
python src/08_model_export/build_requests.py
python -m unittest discover -s tests -p test_model_export.py
python src/08_model_export/verify_ports.py --variant pytorch --device cpu --limit 40 --passes 1

# WSL CUDA runtime (source scripts/activate_rwkv7.sh; FLA on PYTHONPATH)
python src/08_model_export/reference_forecasts.py
python src/08_model_export/verify_blocks.py --requests 8
python src/08_model_export/verify_ports.py --variant fla --device cuda --dtype bfloat16 --limit 100
```

Working files (requests, reference outputs, logs) are in `.cache/port_work/`
and can be deleted.

# Training speed, 2026-11-17: faster updates, checkpoints and validation

Date: 2026-11-17. Model: the 221.5M hybrid (768 x 24) of
[HYBRID_220M_2026-11-15](HYBRID_220M_2026-11-15.md), BF16 backbone, one
RTX 5070 (12 GB) under WSL2, PyTorch 2.12.0+cu130.

The aim was to shorten a training run without changing the model it produces.
Everything below was measured on this machine on 2026-11-17. Scripts and raw
results are in `experiments/claude_review/` and
`experiments/claude_review/speedup_2026-11-17/`.

## Summary

| Change | Setting (`training.*`) | Result | Effect on the trained model |
| --- | --- | --- | --- |
| One-pass CUDA kernel for the FP32-master AdamW update | `fused_master_optimizer` (default on) | optimizer step 24.6 ms -> 10.6 ms per update | none: bit-identical |
| Clipping without the multiply-by-one and the slow norm stack | always on | clipping 15.2 ms -> 3.2 ms per update | none: bit-identical |
| Validation panel saved after the first check | `validation_cache` | 6 to 9 minutes of file work per check removed (estimate from a 300-table sample) | none: same windows, same scores |
| Checkpoints written in the background | `background_checkpoints` (default on) | training waits 1-4 s per checkpoint instead of about 75 s | none |
| Checkpoint copy in Azure Blob Storage | `checkpoint_upload` | every checkpoint verified in the container about 2.5 minutes after it is taken | none |
| One log record per 1,000 tables | `log_every_tables` | about 1.3 ms per update less logging on the training thread | none |
| Batches just above the pass limit in one pass | not adopted | 2.7% slower on the real mix of shapes | (rounding only) |

**On the corpus's own mix of batch shapes an update now takes 109.9 ms instead
of 143.7 ms: 72.8 instead of 55.7 tables per second, 31% more.** Peak VRAM is
unchanged (9.15 GiB allocated, 10.2 GiB reserved, allocator limit 10.26 GiB).
This covers the optimizer step, clipping and the training thread's per-update
log lines. The validation, checkpoint and journal savings come on top and are
not in that figure.

How the figure was measured: `shape_mix_benchmark.py` runs the trainer's own
update on all 64 shapes with up to eight variables (99.95% of the training
windows; frequencies from an 882,380-window sample of the patch index),
visited in shuffled order three times. Runs: earlier code 155.3 and 143.7 ms
(the first was the first job after other work on the machine and is taken as
disturbed; a third run without the log lines gave 142.0), current code 109.5
and 110.3 ms.

## 1. Where an update spent its time

`update_phase_breakdown.py` runs the trainer's own update on synthetic batches
(eight tables each) and synchronises the GPU after every phase. Before any
change:

| Shape (tables x variables x patches) | Passes | Update | Forward | Backward | Clipping | Optimizer | Weight average |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 8 x 1 x 16 | 1 | 103.2 ms | 27.7 | 37.9 | 15.2 | 24.8 | 1.3 |
| 8 x 1 x 128 | 1 | 102.0 ms | 28.0 | 40.1 | 15.2 | 24.6 | 1.3 |
| 8 x 3 x 128 | 1 | 156.7 ms | 37.1 | 78.2 | 16.0 | 26.0 | 1.3 |
| 8 x 4 x 128 | 2 | 221.9 ms | 67.3 | 126.4 | 15.8 | 26.0 | 1.4 |
| 8 x 8 x 128 | 3 | 355.6 ms | 104.9 | 217.1 | 16.0 | 26.0 | 1.3 |

After the changes of sections 2 and 3 (`baseline.json`, `current.json`):

| Shape | Update before | Update after | Change | Clipping | Optimizer |
| --- | ---: | ---: | ---: | ---: | ---: |
| 8 x 1 x 16 | 103.2 ms | 71.3 ms | -31% | 3.1 | 10.4 |
| 8 x 1 x 128 | 102.0 ms | 73.2 ms | -28% | 3.0 | 10.6 |
| 8 x 3 x 128 | 156.7 ms | 127.5 ms | -19% | 2.9 | 11.0 |
| 8 x 4 x 128 | 221.9 ms | 191.0 ms | -14% | 3.1 | 11.3 |
| 8 x 8 x 128 | 355.6 ms | 328.5 ms | -8% | 3.0 | 11.1 |

About 30 ms less per update, whatever the batch. The optimizer step is now
bound by memory traffic (28 bytes per parameter and update); forward and
backward are untouched.

Two things stand out in the first table. An update of 128 tokens costs the same as one of 1,024
tokens: the GPU is waiting for the CPU to issue work, not the other way round.
And clipping plus the optimizer step took 40 ms of a 102 ms update, although
neither depends on the batch.

## 2. The optimizer step

`FP32MasterAdamW` keeps float32 master weights and moments next to the BF16
model weights. Its update made three passes over every parameter: cast the
gradient to float32 (in 64 MiB buckets), run PyTorch's fused AdamW on the
masters, copy the masters back to BF16. In Python that is about 830 casts, 830
copies and 13 calls of `AdamW.step`, each of which walks all 869 parameter
tensors.

`master_adamw_kernel.cu` does the same in one pass: it reads the gradient in the
model's dtype, updates master and moments in float32 and writes the rounded
model weight. It needs no staging memory.

- The arithmetic is PyTorch's own `adam_math` template (shipped in
  `ATen/native/cuda/fused_adam_utils.cuh`), instantiated for float32 as
  `torch._fused_adamw_` instantiates it. The bias corrections are computed the
  same way. BF16 -> float32 is exact; float32 -> BF16 uses the same c10 cast as
  `Tensor.copy_`. The kernel is not compiled with fast-math.
- The tensor lists are registered with the extension once
  (`create_plan`); each update passes only the scalars (`run_plan`). Passing
  4,000 tensors from Python on every update cost 3 ms by itself.
- State layout, `state_dict` and checkpoints are unchanged, so a run can move
  between the kernel and the three-pass update, in either direction.
- If the extension cannot be built, training logs a warning and uses the
  three-pass update.

The kernel is built on first use with the toolkit of the WSL environment. That
toolkit pairs nvcc 13.3 with the CUDA 13.0 runtime headers; PyTorch's CUDA
headers include CCCL, which rejects the pairing unless
`CCCL_DISABLE_CTK_COMPATIBILITY_CHECK` is defined. The parity tests below are
the check that the resulting kernel is right.

## 3. Clipping

`torch.nn.utils.clip_grad_norm_` did two unnecessary things on this model:

- It multiplied every gradient by the clipping coefficient even when that was
  exactly 1 (norm below the limit). With float32 input/output layers next to
  the BF16 backbone the multiply also took PyTorch's slow per-tensor route:
  7.4 ms.
- It stacked 830 per-tensor norms one by one, converting each BF16 scalar to
  float32 separately: 6.1 ms.

`gradient_clipping.py` stacks the norms per dtype group (the resulting vector
is identical), and the trainer reads the coefficient in the metrics transfer
it already makes; it scales the gradients only when the coefficient is below
one. The parameter list is cached, which saves walking the module tree
(1.1 ms).

## 4. Proof that training is unchanged

Training on this GPU is not reproducible bit for bit from one run to the next:
two runs of the unchanged trainer with the same seeds drift apart after eight
updates. Two runs can therefore not be compared. Instead:

- `tests/test_master_optimizer.py` (`OnePassKernelTests`, GPU): 60 updates over
  14 tensor shapes (scalars, odd lengths, more than one 65,536-element chunk,
  BF16 and float32 weights, three parameter groups, changing learning rates,
  missing and late gradients, zero gradients, a transposed gradient). Weights,
  masters, moments, step counters and the set of parameters with state are
  compared bit for bit after every update; checkpoints are moved between the two
  implementations and continued.
- `tests/test_gradient_clipping.py` (CPU and GPU): norm and clipped gradients
  equal `clip_grad_norm_` bit for bit, with and without clipping, with mixed
  dtypes, and for norms from 0.9 to 1.1 times the limit.
- `experiments/claude_review/speedup_parity.py` (GPU, full model): trains the
  221.5M model on real forward/backward passes and keeps a shadow copy of all
  weights, masters and moments that receives the same gradients and is updated
  the earlier way. 96 updates, 29 of them clipped (the limit was lowered to the
  median norm for this purpose), 5,202 tensors compared after every update:
  **all identical** (`speedup_2026-11-17/parity.json`; run twice, the second
  time with the final code).
- The whole suite (`python -m unittest discover -s tests`): 474 tests pass on
  CPU, 17 skipped (GPU and live-storage tests, which were run in WSL).

## 5. Validation, checkpoints and logging

### Validation

Every check hashed, opened and sliced each of the 25,000 validation tables
again, on the training thread. Timed on 300 real tables of the current corpus
(scratch script, CPU only):

| Per table | First read | Read again |
| --- | ---: | ---: |
| SHA-256 of the file | 9.6 ms | 3.1 ms |
| `read_parquet` | 6.8 ms | 5.9 ms |
| Five windows (`frame_window`) | 4.2 ms | 4.1 ms |
| Tokenising five requests | 1.7 ms | 1.1 ms |
| **Total** | **22.2 ms** | **14.2 ms** |

For 25,000 tables that is 6 to 9 minutes of file work per check, before the
model runs. Eight worker threads made it slower, not faster (27 ms per table):
the work holds Python's interpreter lock.

With `training.validation_cache` the float32 windows that the first check reads
are also appended to one file, `validation_panel.bin`, with an index. Later
checks map that file and build the same tensors from it; no source table is
opened. The panel is keyed by a SHA-256 over every request (source, source
hash, origin, columns) and the window settings, so a changed panel or a changed
source file builds a new one. Source files are verified against their recorded
hashes when the panel is built.

- Size: 0.039 MiB per table on the sample, so about 1 GiB for 25,000 tables.
  It is mapped, not loaded: the pages stay evictable.
- Same numbers: `tests/test_rolling_training_validation.py` compares every
  window bit for bit and requires identical losses, per-table rows and metrics
  with and without the panel.
- End-to-end run on 300 validation tables (1,500 requests, full-size model):
  the first check took 51 s (it reads the tables and includes one-off
  compilation), the next one 5.4 s.
- At full scale the forward passes (about 3 minutes for 125,000 requests at
  18-21 ms per batch of 16) and tokenising (about 40 s) remain. Expect a check
  of about 4 minutes instead of about 12. This is an estimate from the sample,
  not a measurement of a full check.

Where the panel lives matters. Reading a large file that Windows no longer has
in its cache, through WSL's mount of the Windows drive (`/mnt/e`), ran at
5.4 MiB/s; the same drive read natively gave 106 MiB/s, and WSL's own disk
(which lives on the same hard disk) 110 MiB/s. `validation_cache` therefore
accepts a folder; `configs/config.json` uses `~/.cache/basin/validation_panels`
on WSL's disk.

### Checkpoints

A checkpoint of the 221.5M model is 4.47 GB (weights, best weights, FP32
masters, both moments, weight average). Writing it used to stop training for as
long as the disk needed: 101-108 s per hour for the 8 GB checkpoints of the
500M run.

Now the trainer copies the state to host memory and a background thread writes
the file (`checkpoint_writer.py`). Measured in end-to-end runs on a 3,000-table
copy of the corpus with the full-size model:

| Step | Time |
| --- | ---: |
| Copy of the state to host memory (training waits) | 1.0-3.6 s |
| File written and uploaded, in the background | 124-158 s |
| Update time before / during / after one background write with upload | 110 / 124 / 107 ms |

Training therefore runs about 13% slower for the two to three minutes in which
a checkpoint is written and uploaded (the upload threads share Python's
interpreter lock with the training thread). Per hourly checkpoint that is
about 20 s of training time, against about 75 s for a blocking write of
4.47 GB at the 59 MiB/s measured below: roughly 1.5% of wall-clock time
saved. The 75 s is computed, not measured, for this model size.

Rules: one write at a time; the file replaces the previous checkpoint only when
complete; a failed write stops training; a run that fails or is interrupted
still completes the write in flight; with less free host memory than the state
needs plus 2 GiB, the write blocks as before. The copy needs about 4.2 GiB of
host memory for up to a few minutes per checkpoint.

### Cloud copy

`training.checkpoint_upload` sends every checkpoint to an Azure Blob Storage
container (`checkpoint_upload.py`). Measured with a 512 MiB file:

| | Speed |
| --- | ---: |
| Write to the project drive through WSL (`/mnt/e`) | 59 MiB/s |
| Write to WSL's own disk | 97 MiB/s |
| Upload to the container (16 MiB blocks, 4 at a time) | 28.5 MiB/s |
| Upload (8 at a time) | 34.4 MiB/s |
| Download | 23.8 MiB/s |

So the container is about half as fast as the local disk. It is a second copy,
not a faster place to write; since both happen in the background, neither
costs training time.

The upload is fed the checkpoint's bytes while they are written to disk. A
first version read the finished file back and stalled: that read ran at
5 MiB/s (see the note on `/mnt/e` above) and a 4.5 GB upload took over ten
minutes.

What was tested:

- Unit tests with a fake store (`tests/test_checkpoint_writer.py`,
  `tests/test_timed_training.py`): background and blocking checkpoints hold
  the same state although training moved on during the write; a failed write
  is raised in the training thread and leaves the previous checkpoint; a failed
  upload (at the start, mid-way, at the commit) leaves a complete file and is
  fatal only for the first checkpoint of a run; an upload that hangs is
  abandoned when the next checkpoint is due; an unreachable store stops the
  run before the first update; the journal receives the writer's events; the
  unfinished file of a killed run is removed at the next start.
- Live test against the container (`BASIN_AZURE_LIVE_TEST=1`): several blocks
  up, down, compared; an abandoned upload leaves the committed blob unchanged;
  a second upload replaces it; metadata and listing; clean-up.
- End to end with the real 4.47 GB checkpoint: four uploads in a 12-minute run,
  each verified by size and MD5 in the container. The last one was downloaded
  (SHA-256 equal to the record made while the local file was written), put in
  place of the local file, and training resumed from it at update 3,511.
- Kill test: `kill -9` eight seconds into a background write. The previous
  checkpoint (local and in the container) was intact and loadable; the
  unfinished file stayed behind as `.latest_checkpoint.pt.<id>.tmp`.

Resuming is slow for the same reason as the read-back was: loading the 4.47 GB
checkpoint through `/mnt/e` took 14 minutes. That is unchanged by this work.
At the 110 MiB/s measured for WSL's own disk it would take under a minute
(estimate), which would mean keeping `paths.training` there.

Logging. The earlier trainer wrote, per update, a journal record (0.5 ms,
including process and GPU memory statistics and the ids of the eight tables)
and ten DEBUG lines to the log file on the Windows drive (3.9 ms in total; two
of them from the training thread, eight from the preparation thread). With
`log_every_tables: 1000` the journal and the console get one summary per 1,000
tables, and the per-table, per-pass and per-update lines are TRACE messages,
which the DEBUG log file does not record. The checkpoint and
`loss_history.json` still hold the last 1,000 updates in full.

## 6. Larger single passes (tested, not adopted)

`microbatch_token_limit` and `checkpoint_token_threshold` are both 3,072, so a
batch of 8 tables x 4 variables x 128 patches (4,096 tokens) runs as 6 + 2
tables in two passes. A pass is expensive however small it is (about 65 ms for
forward and backward), so one pass with six of the 24 layers recomputed might
be cheaper. 15.1% of the training windows are in batches above 3,072 tokens;
9.6% are between 3,072 and 4,096.

Shape by shape it looked good (24 updates each, `pass_limit_3072.json`,
`single_pass_4096.json`, `pass_limit_6144.json`):

| Shape | Tokens | Two passes | One pass | Peak allocated |
| --- | ---: | ---: | ---: | ---: |
| 8 x 5 x 80 | 3,200 | 178.8 ms | 132.4 ms | 8.21 -> 8.80 GiB |
| 8 x 4 x 112 | 3,584 | 184.3 ms | 150.2 ms | 8.10 -> 8.85 GiB |
| 8 x 5 x 96 | 3,840 | 182.0 ms | 161.6 ms | 8.34 -> 8.90 GiB |
| 8 x 4 x 128 | 4,096 | 188.4 ms | 173.8 ms | 8.64 -> 8.98 GiB |
| 8 x 5 x 128 | 5,120 | 212.4 ms | out of memory under the allocator limit | |
| 8 x 6 x 128 | 6,144 | 245.3 ms | 262.7 ms | 9.15 -> 9.34 GiB |

Weighted by frequency the four shapes that gain would save about 1.6% of
training time. On the mix of shapes they did not:

| Run (`shape_mix_benchmark.py`) | Update | Tables/s | Peak reserved |
| --- | ---: | ---: | ---: |
| Current, run 1 / run 2 | 109.5 / 110.3 ms | 73.1 / 72.5 | 10.19 GiB |
| One pass up to 4,096 tokens, run 1 / run 2 | 113.5 / 112.3 ms | 70.5 / 71.2 | 10.25 GiB |

The shapes that changed from two passes to one gained 0.6 ms per average
update; all the other shapes together lost 4.2 ms (4.6% slower each on
average). The larger passes push reserved memory to the allocator limit, and
the shapes that follow pay for the allocator having to give memory back and
take it again. One shape (8 x 5 x 96) went from 200 to 274 ms.

So the change is 2.7% slower where it matters and was not adopted; the option
written for the test has been removed. It would also not have been
bit-identical: one pass and two passes give the same gradient up to rounding.

## 7. Not done

- The alternative WKV kernel (`rwkv7_clampw_v3_for_h100_alt`) differs from the
  one in use in three lines: it reads the value vectors from global memory
  where the current kernel first copies them into the block's shared memory.
  The arithmetic is the same. It was not built or measured: the kernel is a
  few percent of GPU time, and updates here are limited by the CPU, so a gain
  would be well under 1%.
- Compiling temporal attention as variate attention is compiled: not measured.

## 8. Reproducing

```bash
source scripts/activate_rwkv7.sh
python experiments/claude_review/update_phase_breakdown.py --label current
python experiments/claude_review/shape_mix_benchmark.py --label current
python experiments/claude_review/shape_mix_benchmark.py --label earlier --earlier-update
python experiments/claude_review/speedup_parity.py --updates 96
python -m unittest discover -s tests -p "test_master_optimizer.py"
python -m unittest discover -s tests -p "test_gradient_clipping.py"
BASIN_AZURE_LIVE_TEST=1 python -m unittest discover -s tests -p "test_checkpoint_writer.py"
```

All of these need the GPU free; the largest (`speedup_parity.py`) peaks at
8.3 GiB allocated and 9.2 GiB reserved, under the trainer's allocator limit.

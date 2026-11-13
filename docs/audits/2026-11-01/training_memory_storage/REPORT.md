# Training graph memory, disk storage and sustained diagnostic

## Investigation

The latest failing production worker is
`logs/20261101T144806_779892Z_train_7885caf8/`. It completed 13 updates and failed
during update 14. CUDA reported an illegal memory access while a post-accumulate
hook cloned a parameter gradient. CUDA errors can be asynchronous, so this does
not prove that the clone itself was the faulty operation. This was not an
allocator out-of-memory exception. The log also shows an older graph profile
being retired halfway through capturing the next profile, as the graph budget
was reached.

The previous code assumed that returning from `backward()` meant all GPU work
was complete before releasing a graph's private memory pool. CUDA execution is
asynchronous, including autograd work on other streams. Pool lifetime therefore
needs explicit protection. The code now synchronizes the device before retiring
or closing cached graphs and evicts older profiles before starting captures for
the new profile, rather than releasing old pools in the middle of that capture
sequence. The existing 1.5 GiB graph-cache budget and 8 GiB process allocator cap
remain in effect. Fatal illegal-memory-access errors are still propagated; the
code does not attempt to continue in a damaged CUDA context.

This removes a concrete lifetime hazard. It is not a sanitizer-level isolation
of the exact failing GPU instruction. The sustained run below provides stronger
coverage than the previous forty-update check.

## Disk findings and changes

Graph replay's private pools consume GPU memory. They are distinct from disk
caches containing compiled kernels. Compiled kernels can accumulate over time;
the paths now explicitly resolve to the project drive before PyTorch imports.

| Storage | Location after this change |
|---|---|
| Windows training TEMP/TMP | `.cache/tmp/windows` under the project on E: |
| WSL training TEMP/TMP/TMPDIR | `.cache/tmp/wsl` under the project on E: |
| TorchInductor | `.cache/torchinductor` on E: |
| Triton | `.cache/torchinductor/triton` on E: |
| Compiler debug output | `.cache/compile_debug` on E: |
| Torch/XDG caches | `.cache/torch`, `.cache/xdg` on E: |
| CUDA driver compute cache | `.cache/cuda_driver` on E:, maximum configured as 1 GiB |
| CUDA GPU dump destination, if dump generation is enabled | `.cache/crash_dumps` on E: |
| WSL Linux filesystem | Existing `E:\WSL\Ubuntu-24.04` |
| Cached RWKV CUDA extensions | Existing Linux home cache, inside that E: WSL filesystem |
| WSL swap | `E:\WSL\swap.vhdx`, maximum configured as 4 GiB |
| Windows pagefile | Existing `E:\pagefile.sys` |

`runtime_storage.py` configures the Python entry point before importing torch;
`scripts/activate_rwkv7.sh` configures the WSL process before Python starts.
The effective Python cache paths are logged at INFO. These changes affect the
training process and its child processes; they do not change the global Windows
TEMP variable for unrelated applications.

`C:\Users\Stardustnuke\.wslconfig` did not previously exist. It now specifies
the E: swap destination and a 4 GiB swap maximum. Its contents are recorded in
`wslconfig.after.txt`; Ubuntu was restarted to apply it. The E: swap file was
verified at a file length of 37,748,736 bytes during the diagnostic, with no
observed swap use. WSL's default host swap location otherwise uses Windows temporary
storage. The RWKV extension cache measured 38 MB, and the old Linux Triton cache
measured about 955 MB; both were already inside the WSL filesystem on E:.

The remaining inspected C: CUDA/graphics caches were small: approximately 2 MB
under LocalAppData/NVIDIA, 191 MB under NVIDIA Corporation, and 406 MB in
D3DSCache. No surviving multi-gigabyte training cache on C: was identified.
Because space had already been cleared, the exact deleted files that filled C:
cannot be established from this investigation. The disk-full event is not proof
that it caused the GPU illegal-address error.

## Diagnostic design

The actual entry point is run with a separate configuration:

```bash
source scripts/activate_rwkv7.sh
MAX_JOBS=1 TORCHINDUCTOR_COMPILE_THREADS=1 python -u \
  src/05_model_training/train.py --config configs/training_memory_diagnostic.json
```

The diagnostic uses the normal full training split, 307M architecture, batch
size eight, physical microbatch maximum four, token limit 1536, prefetch,
compiled attention, graph replay and AdamW. It uses the production epoch-1 seed.
Only diagnostic controls differ: epoch-0 validation is skipped, resume is
disabled, a 660-second training time budget is applied, and final validation is
restricted to 64 training-table final tokens. Outputs go to
`results/diagnostics/training_memory_20261101/`, separate from the default model.
The time budget finishes after an update rather than interrupting backward.
Normal `configs/config.json` training has no time limit and retains full epoch
validation and automatic resume.

Windows disk space and RAM, E: cache sizes and swap-file size, and total GPU
memory/utilization are sampled in `resources.jsonl`. The GPU total includes
desktop and other processes; PyTorch allocator peaks are separately recorded
in training history. The monitor does not measure every operating-system cache
or provide process-specific shared GPU-memory attribution.

## Verification

The focused native suite passed 24 tests (22 passed, two CUDA tests skipped).
It covers storage-path overrides, graph retirement ordering, clean termination
at the diagnostic time limit, epoch/validation/checkpoint behavior, batching,
gradient accumulation and prefetch parity. Applicable pre-commit checks passed.

The actual training script completed with exit code 0. Worker status is
`complete` in `logs/20261101T152314_447575Z_train_248bbb47/status.json`.

| Sustained diagnostic | Result |
|---|---:|
| Timed training phase, monotonic clock | 663.239 seconds (11 min 3 sec) |
| Time between first and final completed updates | 649.902 seconds (10 min 50 sec) |
| Optimizer updates | 347 |
| Training windows processed | 2,775 |
| Successful graph captures | 5,831 |
| Graph replays | 12,098 |
| Profile evictions exercised | 345 |
| Capture/illegal-address/OOM failures | 0 |
| Graph replay disabled by fallback | No |
| Peak PyTorch allocated GPU memory | 6.759 GiB |
| Peak sampled total GPU memory, including other processes | 8,922 MiB |
| Lowest sampled available host RAM | 8.013 GiB |
| Maximum observed decrease in C: free space during monitoring | 16.020 MiB |
| TorchInductor/Triton cache bytes at sampled beginning/end | 159,873,770 / 159,873,770 |
| Sampled TEMP, CUDA-driver, compiler-debug and GPU-dump cache bytes | 0 |

Graph-budget decisions fell back to compiled attention for 51 layer/profile
cases, as designed. There were no capture-failure fallbacks, and replay remained
enabled through the final update. The run then validated all 64 selected final
tokens in 4.216 seconds and saved `model.pt` under the separate diagnostic
directory. Total worker duration including loading, initialization, validation
and saving was 785.215 seconds. The normal epoch-0 baseline was not run.

The 16 MiB C: change is a whole-machine observation, not an attribution to this
process. It shows no multi-gigabyte C: growth during this diagnostic. Approximately
1.26 GB was written to E: for the diagnostic best weights and final model; those
are checkpoint artifacts, not graph/compiler caches.

The complete metric history and output model are under
`results/diagnostics/training_memory_20261101/`. `resources.jsonl` contains 48
disk/RAM/GPU samples. `verification.json` records measured results and source
hashes. The runtime configuration, JSONL training journal and detailed events
are in the timestamped worker log directory above; the console transcript and
focused test log are copied there as well.

This run is an execution and memory check, not an accuracy experiment or a
full-epoch speed benchmark. It does not prove that every future CUDA/driver fault
has been eliminated; it verifies the requested sustained run with the current
configuration and data, across hundreds of graph-cache replacements.

References:

- [PyTorch CUDA graph memory management](https://docs.pytorch.org/docs/2.12/notes/cuda.html#graph-memory-management)
- [PyTorch CUDA stream semantics](https://docs.pytorch.org/docs/2.12/notes/cuda.html#cuda-streams)
- [Microsoft WSL swapFile configuration](https://learn.microsoft.com/windows/wsl/wsl-config)
- [NVIDIA CUDA core-dump configuration](https://docs.nvidia.com/cuda/cuda-gdb/index.html#gpu-coredump-support)

# Embedding run failure, memory diagnosis, and preparation redesign

## What happened to the user's run

The affected run is `logs/20261026T185830_873505Z_embed_3db98ae4`.
Its status file remained `running`; there is no completion or successful artifact
save in its logs. The final log entry is at **2026-10-26 20:45:37 local time**,
while processing `table_30948_evaluation_1295`. The configured `embeddings.pt`
still had its **2026-10-24** modification date and size 292,237,363 bytes before
recovery. That older file was not the result of the attempted run.

Windows System event **2004**, at **20:45:52**, reports a low virtual memory
condition. It identifies `python.exe`, PID 34116, consuming **111,601,561,600
bytes** (111.6 GB, 103.9 GiB). The same event lists PyCharm at about 6 GB. The
event is retained in [windows_memory_event.json](windows_memory_event.json).
This establishes severe memory pressure at the same time the run stopped; no
Python exception or explicit exit code survived, so the exact termination
mechanism cannot be proven from the application log alone.

At investigation start, the machine had approximately 32 GB installed RAM,
18 GiB available physical memory, no active Python embedding process, and
11.9 GB free on C:. The currently allocated Windows pagefile was on E:, not C:.
These are current observations, not a reconstruction of yesterday's pagefile
configuration. There is no evidence that filling C: caused this failure.
Memory pressure and paging explain the system-wide slowdown; other applications
can remain slow as their working sets are restored after the offending job ends.

## Root cause

The old stage loaded all processed windows into memory and accumulated, for
every window, raw batches, prepared features and full **float32, width-1,024
projection outputs** in Python lists. Each snapshot has shape
`[batch, variables, padded_tokens, width]`. With many tables and covariates,
these outputs alone require tens or hundreds of gigabytes. Nothing was written
until the end, so an interrupted run left an older artifact in place.

The projection was initialized randomly and never optimized by `embed.py`.
With the default `training.train_embeddings=true`, the trainer recomputed those
projections anyway. Retaining the entire collection added substantial memory,
CPU and disk cost without supplying useful learned embeddings. Per-tensor debug
statistics also generated approximately 160 MB of logs in the failed run.

## Replacement behavior

The existing stage and command remain available for input/output inspection.
`embed.py` now performs **preparation and indexing only**:

- Read `windows.pt` using memory mapping, without eagerly copying its tensor
  contents into anonymous RAM. Metadata is still loaded; memory does not become
  completely independent of the number of windows.
- Use PyTorch's supported `calculate_storage_offsets` setting while loading.
  This avoids a separate small disk read for every tensor header in modern
  checkpoints. The caller's previous setting is restored afterwards. The pinned
  local Torch 2.12 implementation supports this mode and falls back for older
  layouts that cannot calculate offsets.
- Inspect shapes and token counts without running the embedding projection.
  Write each window's source, split, context shape, input token count, padding
  and expected embedding shape to streamed `windows.jsonl`.
- Save `embeddings.pt` as a **`lazy_patch_index_v1` index**, containing the model
  initialization, processed-data fingerprint, signature, source selection and
  window count. It contains no corpus-sized vector collection or prepared-feature
  collection. The filename remains compatible with existing stage consumers.
- Write the artifact beside its destination on E: and replace it atomically
  after serialization succeeds. Failed saves preserve the previous completed
  index. Temporary files are cleaned up on handled failures; abrupt process
  termination can leave a small identifiable temporary file.
- Limit CPU threads with `embedding.cpu_threads` (default **2**) and log progress
  every `embedding.progress_every` windows (default **1,000**). Record RSS,
  available RAM, output-disk free space, elapsed time, artifact size and estimated
  avoided snapshot size in the manifest.

The stage does not clean caches, alter pagefiles, delete generated data, or write
its temporary artifact to C:.

## Training and embeddings

**The learned patch projection is part of the neural model.** `train.py` already
included its parameters in AdamW and backpropagated the forecasting loss through
it when `training.train_embeddings=true`. This remains the default.

On opening the new index, consumers obtain raw windows backed by the processed
artifact. Each `record['prepared']` access computes causal normalized features
for that window and does not retain them in the record. Normal training transfers
that window's features to the GPU, calls the model's own `encode_tokens`, then
backpropagates through the embedding projection, role embeddings, RWKV layers and
forecast head. There is no separate embedding optimization objective or phase.
If explicitly frozen, the trainer still computes only the current projection
on the GPU, using frozen weights; it does not consume stored random snapshots.

For compatibility with existing inspection code, explicitly requesting
`record['initial_embeddings']` computes a disposable initial projection. It is
not cached or used by the normal training path. Older materialized artifacts
remain readable, but rerunning `embed.py` upgrades them to the index format.

Separating deterministic preparation from training is useful for auditing.
Separately optimizing and then freezing the embeddings would change the model
training scheme; that is not what this pipeline does.

## Validation and limits

New tests in `tests/test_lazy_embeddings.py` verify that preparation never calls
the projection, artifact size remains effectively constant between two and
1,000 windows, lazy features are not retained, gradients reach the projection
and role embeddings, changed processed data is rejected, failed saves preserve
the previous index, and the original CPU thread setting is restored.

The CPU regression suite, full-size CUDA smoke test and supervised real-corpus
recovery all passed. The GPU smoke test uses the regular
447.5M trainer, full per-step validation, final rolling test with both real
baselines, checkpoint reload and identical replay metrics. Its generated
sources/checkpoint are temporary, and retained results are under `gpu_smoke/`.

The transformed file remains a single large archive. Loading its Python metadata
still has a cost, and hashing reads the archive sequentially. This change removes
the large embedding accumulation; it does not claim constant memory for arbitrary
future corpus sizes. Sharded raw-window metadata is a possible later improvement.
Also, validating every training table after every optimizer update remains
expensive on this large cohort; the requested validation policy is unchanged.

## Recorded results

The recovery used the existing main configuration and **the real processed
corpus**, without rerunning transformation or changing the selected source dates.
It completed successfully in
`logs/20261027T182348_440603Z_embed_ba31a88a/status.json`.

| Measurement | Recovered run |
|---|---:|
| Indexed windows | 215,656 |
| Stage duration | 90.24 seconds |
| Peak RSS, supervised process family | 2,119,507,968 bytes (1.97 GiB) |
| Peak Windows private allocation, process family | 6,655,741,952 bytes (6.20 GiB) |
| Index artifact | 17,801,522 bytes (16.98 MiB) |
| Avoided float32 snapshot storage, estimated from shapes | 259.46 GiB |

Peak RSS and private allocation are different measurements; neither should be
silently compared as the same metric as Windows' original virtual-memory event.
The run was supervised with 8 GiB RSS and 12 GiB private-allocation guards, which
were not triggered. It wrote no corpus-sized vectors. At a progress checkpoint,
16.4 GiB of system RAM was still available.

The saved index signature, window count, streamed detail count, successful stage
status and ZIP archive CRC were checked. The retained
[real_corpus_verification.json](real_corpus_verification.json) contains the index
fingerprint, processed fingerprint, manifest and monitored memory results.
`logs/20261027_embed_recovery_console.log` contains the complete recovery output.
The original failed run's files are left untouched for the audit trail.

An initial supervised recovery attempt was stopped deliberately to apply the
sequential-offset reader after observing excessive small archive-header reads.
That attempt did not replace the prior artifact. Its monitor record is retained
as `logs/20261027_embed_memorycheck_first_attempt.json`; it should not be mistaken
for the successful final recovery.

Verification also included:

- **114 regression tests, OK, six CUDA-dependent skips**, with output in
  `logs/20261027_lazy_embeddings_tests.log`.
- Real **447,482,144-parameter CUDA training**, two optimizer updates on nine
  synthetic tables, full validation sweeps and final rolling forecasts on a
  reserved table against actual Prophet and Holt-Winters fits. Reloading the
  checkpoint reproduced the pooled test metrics exactly. Results are retained
  under `gpu_smoke/`; console output is in
  `logs/20261027_lazy_embeddings_gpu.log`.
- Both GPU updates produced nonzero embedding gradients; the projection's skip
  weights changed by L2 norm **0.0119883**, confirming that learning still occurs
  inside training. See `gpu_smoke/embedding_optimization.json`.
- Ruff lint/format checks and Bandit passed on changed Python files.

The main `data/03_embedded/447m_model/embeddings.pt` has already
been rebuilt successfully. There is no need to rerun the expensive transform
stage for this fix. Future `embed.py` calls use the new preparation behavior.
Historical checkpoints referring to a different embedding-artifact fingerprint
retain their provenance checks; they are not silently relabeled as trained on
this new artifact.

# Training graph schedule: capture-inclusive benchmark

2026-11-01. Production 307,307,808-parameter model, RTX 5070 12 GiB, BF16, PyTorch 2.12.0+cu130. Two trials per arm, 1,024 full optimizer updates per trial.

## Selected default and larger-microbatch experiment

Default physical microbatch limit: **8**; replay: **False**. On this sample the selected microbatch-eight execution takes **106.61 ms/update**, **30.39% less time** than the microbatch-four shuffled baseline.

| Microbatch maximum | Replay | Mean ms/update | Trial 1 | Trial 2 | Peak allocated GiB |
|---:|---|---:|---:|---:|---:|
| 8 | False | 106.61 | 106.85 | 106.37 | 6.496 |
| 8 | True | 107.37 | 109.02 | 105.72 | 6.390 |

At microbatch eight, replay changes elapsed update time by **+0.72%** relative to compiled execution without replay (positive means slower). The default uses the faster measured option; inference replay remains enabled.

Optimizer batch size remains eight. The 1,536 variable-token limit still reduces the physical batch for larger inputs, and the 8 GiB allocator cap remains enabled. This experiment uses the same real-table sample and resets initial weights/AdamW for each arm; two paired trials reverse arm order. Raw artifacts and resource monitoring are in `../training_graph_schedule_micro8/`.

The physical-microbatch experiment was run after the microbatch-four experiment in a separate process. The cross-microbatch gain is consequently not an interleaved timing comparison; both used the same GPU, environment, tables, seed and compiled code. It is not a full-epoch throughput or accuracy guarantee.

The microbatch-eight run was interrupted by a usage limit after trial one. The benchmark resumed only the two unfinished arms in a fresh process, after checking identical model/training settings and source batches. Completed arms were retained; the interrupted partial arm was discarded. `settings.json` records the original and resumed log directories. The gap between trials is an additional reason to treat small timing differences cautiously.

## Microbatch-four scheduling experiment

| Execution | Mean ms/update | Trial 1 | Trial 2 | Peak allocated GiB |
|---|---:|---:|---:|---:|
| shuffled_off | 153.16 | 154.76 | 151.56 | 6.504 |
| grouped_off | 154.04 | 154.33 | 153.74 | 6.502 |
| grouped_replay | 151.68 | 152.57 | 150.79 | 6.735 |

Replay saves **1.53%** versus the same grouped schedule without replay. Grouping plus replay saves **0.97%** versus shuffled batches without replay.

## What changed

`make_batches` still shuffles tables within each existing variable-role and length bucket. It now shuffles runs of up to 128 batches instead of every batch separately. Every table occurs once per epoch; table contents, batch membership, temporal ordering, loss weighting, validation and checkpoint frequency are preserved. Optimizer update order changes, so this is not an identical training trajectory or a test of final accuracy.

A new shape needs eight consecutive optimizer updates before capture; accumulation microbatches do not count as additional updates. Already cached profiles replay immediately. Default replay is limited to at most 32 padded patch tokens (512 observations of input); long contexts use compiled attention. Set `training_graph_max_tokens` to null to experiment with all lengths.

The 1.5 GiB graph budget, two-profile bound, 8 GiB allocator limit, main-thread pinning and synchronized graph-pool retirement remain in force. Profiles are separate for physical batch size, variable count, padded patch length and model width, with layer-specific graph storage.

The complete epoch has 14,610 batch groups and 96 logical batch/variable/padded-length combinations, including partial batches. Physical microbatching can introduce further batch sizes. It is not just the number of covariates.

## Shape-level totals (capture included)

| Physical B,V,T,D | Updates/trial | Compiled ms | Replay ms |
|---|---:|---:|---:|
| (1, 7, 48, 1024) | 1 | 84.66 | 108.47 |
| (4, 1, 48, 1024) | 128 | 143.62 | 145.83 |
| (4, 1, 96, 1024) | 128 | 145.77 | 150.28 |
| (4, 1, 128, 1024) | 128 | 145.70 | 145.66 |
| (4, 2, 32, 1024) | 128 | 149.24 | 134.90 |
| (4, 2, 48, 1024) | 128 | 153.47 | 150.31 |
| (4, 3, 96, 1024) | 179 | 162.09 | 162.32 |
| (4, 3, 112, 1024) | 128 | 176.95 | 176.95 |
| (4, 4, 32, 1024) | 75 | 151.60 | 136.30 |
| (4, 7, 48, 1024) | 1 | 176.34 | 230.74 |

## Method, correctness and limits

The sample is the first 1,024 batches of the new epoch-1 schedule, using real admitted training tables. All arms use exactly those same tables and batch memberships. The shuffled arm randomizes their batch order. Every arm resets model parameters and AdamW state, uses the same RNG seed, and starts with an empty graph cache. Trial two reverses arm order. One GPU model is resident at a time.

Timing includes CPU prefetch consumption, pinning, transfer, forward/backward, gradient clipping, optimizer update, all graph captures and evictions, and a final CUDA synchronization. Common compilation is warmed outside every arm. Model/data loading and epoch validation/checkpoint writes are excluded. There is no discarded warmup inside a measured arm. This is a sustained mixed-shape sample, not a full epoch.

Paired loss checks: `[{"repeat": 0, "time_saved_percent": 1.1448613618372239, "max_loss_absolute_difference": 0.058766841888427734, "max_loss_relative_difference": 0.08858762195304391, "mean_loss_relative_difference": 0.005265981168530453}, {"repeat": 1, "time_saved_percent": 1.9194430873331059, "max_loss_absolute_difference": 0.02421271800994873, "max_loss_relative_difference": 0.07732555313221169, "mean_loss_relative_difference": 0.005519803982190987}]`.

Replay-disabled repeats also vary: mean relative loss difference 0.005304, maximum 0.084331. These BF16 full-model runs are not bitwise deterministic, even without replay. Use the dedicated gradient/optimizer regression tests for correctness; this benchmark does not establish identical final accuracy.

`results.json` contains every update's loss, timing and embedding shapes, plus memory/capture counters. `sources.json` identifies the tables; `settings.json` includes the complete epoch shape inventory. Run logs: `/mnt/e/PyCharm/PyCharm Community Edition 2022.2.3/Projects/Basin/logs/20261101T164526_345287Z_training_graph_schedule_0b6e330b`.

## Configuration and continuation

Default `configs/config.json`: `microbatch_size=8`, `shape_run_batches=128`, `training_graph_min_reuse=8`, `training_graph_max_tokens=32`, `training_cuda_graphs=false`. Running `train.py` uses these settings. Replay remains available by setting `training_cuda_graphs=true`; inference replay is unchanged. `shape_run_batches=1` restores globally shuffled batches; the admission rule then prevents repeated capture for isolated shapes. `training_cuda_graphs=false` disables replay while leaving compilation enabled.

Batch ordering is included in checkpoint identity. To continue an existing checkpoint created with the old ordering, set `shape_run_batches=1`; to use the new order from scratch, set `resume=false`. New checkpoints resume normally with the matching setting. No production checkpoint was altered by this benchmark.

## Reproduction

From the activated WSL GPU environment, run `python scripts/benchmark_training_graph_schedule.py --microbatch 4`, then `python scripts/benchmark_training_graph_schedule.py --microbatch 8 --skip-shuffled`. Generate the report with `python scripts/report_training_graph_schedule.py`. Add `--resume` to the benchmark command to retain completed arms after an interruption; model/training settings and source batches must still match. These commands write benchmark artifacts, not production weights.

# Whole-table validation isolation

The previous hourly protocol selected its monitoring panel from training tables
and kept their labels available to the optimizer. That was in-sample monitoring.
The requested correction now reserves whole validation tables before making
training batches; it removes every training window belonging to those sources.

Default selection order:

1. Identify admitted sources and original/coarse/sign-inverted families. Those
   families remain training-only and cannot be validation or test data.
2. Reserve `training.holdout_fraction` of independent sources for test using the
   existing deterministic hash ranking (currently 2.5% in the config).
3. From remaining independent sources, select up to
   `training.validation_sampling.tables` (currently 25,000) with the existing
   separate seeded validation ranking. Require five complete four-token origins
   per table (160 points minimum with patch size 16). Always retain at least one
   training table; a small corpus logs a reduced validation count.
4. Remove all windows of the selected validation sources from training. Remaining
   training tables retain their full labels. Neither validation nor test sources
   enter optimizer updates. Validation still runs hourly, Basin only, with five
   fixed origins per table and 80/16/3/1 lead weights.

The normal hourly entrypoint rejects intersecting training/validation sources
before running updates. Split audit schema 3 stores the three source lists,
counts/fractions and `validation_split_policy=dataset_disjoint_v1`. Metric,
plan and event labels now identify disjoint validation, with `in_sample=false`.
The saved lists make separate holdout runs reject test sources that overlap the
evaluated model's validation sources as well as its training sources.

Checkpoint identity binds the new split policy and validation corpus. The normal
`train.py` preflight rejects old overlapping-policy checkpoints before replacing
split reports. Set `training.resume=false` or a new output directory to start
fresh weights. Reusing old trained weights would not undo earlier validation
exposure. Compatible raw/processed/embedding artifacts can be reused because
the trainer applies the split before optimizer updates. No production model,
checkpoint, data artifact or results were changed during this implementation.

The existing `sampled_training_origins` identifier remains accepted to avoid an
unnecessary artifact rebuild. Its new semantics are always dataset-disjoint.
Explicit legacy `reserved_tail` configurations retain their historical tail
protocol for compatibility; they are not the default hourly protocol.

Regression coverage verifies zero training/validation target-source overlap in
the exact token inventory; partition coverage; matching selected source audits;
unchanged full training labels; deterministic selection; streamed causal windows;
serial/batched validation equivalence; family isolation; rejection of overlapping
training inputs; rejection of old checkpoint policy; independent holdout checks;
and continued mid-epoch optimizer/RNG resume with ordered validation history.

Verification: the full CPU suite completed 308 tests with 12 CUDA-only skips and
no failures. Pre-commit checks passed for the changed files. The split's audit
checksum lookup now uses a source set rather than repeated list membership,
avoiding quadratic work with large corpora. Inventory status distinguishes the
number of reserved validation tables from the five-times-larger forecast count.

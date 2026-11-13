# Training-only augmentation families

Generated coarse copies and sign inversions share the original realization.
Every member of such a family, including the original, is now excluded from
validation and holdout selection. Admitted family members remain training data.
The rule is mandatory for both sampled-origin and legacy reserved-tail splits.

`augmentation_families.py` lists the raw sibling directories once per source
date directory, before deciding evaluation eligibility. The generator's exact
`_coarse_{window}_{offset}` and `_sign_inverted` suffixes identify families.
Suffixes are removed repeatedly to conservatively exclude composed copies too.
This also excludes an original when its copy was rejected by admission screening,
was not selected for processing, or lacks its Parquet/provenance file. Files must
retain their generated directory names and sibling placement; arbitrary renamed
or relocated copies are not detected. This is lineage protection, not numerical
near-duplicate detection of independently generated series.

Only directory entries are read: no series, SCM JSON or per-table provenance
files are opened for this check. There is no persistent cache that could miss
copies added between runs. Date-directory paths are resolved, so relative and
absolute source paths identify the same family.

The holdout fraction applies to independent eligible tables, rounded up while
leaving at least one independent table for training/validation. At least two
independent tables are required. Training contains all admitted augmented
families plus independent tables not assigned to holdout. The validation panel
is sampled only from the independent training tables; full training labels
remain available under the sampled-origin monitoring policy.

`training/dataset_split.json` records the policy version, independent table count,
excluded count and full excluded source list, and holdout fraction denominator.
The progress JSONL reports the count and links the split audit rather than
duplicating the potentially huge exclusion list.

Holdout evaluation also checks raw siblings for older saved splits before
percentage sampling, schema loading and prediction. It drops excluded tables
without drawing replacement test tables or changing saved model weights.
`holdout_test/sample_selection.json` records `augmentation_family_excluded_tables`.
If no independent test tables remain, evaluation fails clearly. Existing result
files are not recomputed until the holdout script is run again.

Checkpoint identity includes this split policy. Existing checkpoints from before
this rule cannot resume as though their original split were compliant. Start a
new training run with `training.resume=false` to establish the new split. This
exclusion alone requires no transform/embed rerun: source paths in existing
artifacts suffice. No production checkpoints or results were modified here.

Regression tests cover originals and both copy types, copies absent from the
admitted records, both validation policies, deterministic reversed input order,
older saved holdout splits, and discovery of newly added copies between runs.

Verification: full CPU suite completed with 282 tests, 12 CUDA-only skips and no
failures. After the final legacy holdout entry-point guard was added, all six
holdout-stage tests passed again. Pre-commit checks passed for all changed files.

# Archived model on the current holdout split

Run from the repository root:

```powershell
uv run python .\src\06_holdout_test\holdout_test.py --config .\configs\holdout_447m_current.json
```

The model weights come from `results/447m_model/training/model.pt` (best epoch
7). The optional `holdout.split_checkpoint` supplies only the saved split and
its raw-source checksums from `results/model/training/model.pt`. Neither
checkpoint is changed. Without this option the existing model-owned split
behavior remains the default. No new split or admission screening is run.

The dedicated configuration enables inference CUDA graphs with eight cached
profiles, uses the archived model architecture and the current benchmark
settings, and selects 2% of the current holdout with seed 7. Comparisons are
naive, Chronos-2 and Timestral, with MASE as the leading metric. Change
`holdout.sample_percent` in this configuration for a different coverage.

Outputs are isolated under
`results/447m_model/holdout_test/current_holdout`. Rerunning this configuration
replaces its own fixed output files. The original archived holdout outputs
and current model's default outputs are preserved. `model_used.json` records
both checkpoint paths and hashes, alongside the evaluated model's best epoch
and step. The model's saved target/covariate policy determines input columns;
the separate split determines sources and expected raw checksums.

The entire supplied holdout split must be disjoint from the evaluated model's
recorded training sources, before sampling. The split checkpoint's own
train/test separation, unique holdout sources and checksum coverage are also
checked. The overlap check uses recorded source paths, consistent with the
existing saved-split protocol; it does not detect renamed duplicate datasets.

Verification on the local real checkpoints: 260 of 12,968 current holdout
tables selected, exactly matching the normal configuration's selected source
list. No recorded training-source overlap was found, and all 260 selected raw
Parquet schemas were readable under the archived model's column policy.
Inference replay is enabled in the configuration. Unit tests cover default
selection, alternate split selection, archived weights/schema policy,
both checkpoint identities and leakage rejection. A full GPU evaluation is
left to the command above; this change does not rerun forecasting or claim new
accuracy or latency measurements. Earlier regression investigation verified
graph replay equivalence for the archived checkpoint.

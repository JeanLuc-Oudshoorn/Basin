# Full-dataset capacity experiment

## Protocol

Run `uv run python src/05_model_training/capacity_test.py` with defaults from
`configs/capacity_test.json`. The script runs on the existing WSL CUDA backend.
It starts from the completed one-epoch, 82-table validated checkpoint, then trains
for **10 additional epochs**, followed by **10 refinement epochs at learning
rate 3e-5**. The refinement uses the first phase's final checkpoint and a fresh
FP32-master optimizer, not an exact Adam-state resume. The original experiment remains intact. The model
still has 301,774,368 parameters, 19 layers, width 960, 16-point patches and a
2,048-observation historical context cap.

Each epoch visits all 82 admitted tables and all 5,888 next-patch transitions.
Tables are grouped by **exact variable/role count** and time padded within batches
of at most 8. Variable dimensions are never padded or flattened across unrelated
tables. Shuffling is seeded independently each epoch. Temporal masks use NaN
labels, so padded positions cannot become training targets. The residual token
projection is optimized jointly with all RWKV and variate layers.

Immediately before the first update and **after every optimizer update**, evaluate:

- all **94,208 observed training target points** across all 82 tables;
- all **1,312 final-patch validation points** across all 82 tables.

Evaluation is exhaustive, not a sampled minibatch or moving average. It uses
`model.eval()` and `torch.no_grad()` and restores training mode afterward. The
validation target is placed only in the label tensor; it is never added to the
input features. Both sweeps use the same newly updated model weights.

The final patches are deliberately monitored repeatedly here. They are validation
for this capacity experiment; they are no longer an untouched test suite for
making claims about generalization. No final-patch values enter gradient updates.

## Objective, precision and measurements

The existing causal, normalized nine-quantile pinball objective is retained.
Loss is averaged over observed target points and quantiles. Batch training and
reported aggregate loss weight observed points equally; the original single-table
smoke run's update schedule effectively gave each table equal update weight.
Both the point-weighted aggregate and mean table-level loss are recorded.

The optimizer uses **FP32 master weights, FP32 AdamW moments, bf16 forward/backward**.
This allows small updates to accumulate even before they exceed a bf16 weight's
rounding interval. The previous smoke optimizer updated bf16 weights and moments
directly. The capacity run uses learning rate `3e-4`, no weight decay, gradient norm
clip 1.0 and activation checkpointing; all values are in the dedicated JSON config.
Gradient clipping operates on FP32 master gradients before the optimizer update;
updated master weights are copied into the computation model.

Raw median-prediction RMSE is also recorded, alongside normalized RMSE computed
using each target patch's causal, historical normalization scale. Raw pooled RMSE
can be dominated by large-unit tables, so it should not replace the normalized
objective as the sole capacity measure. Training includes very short prefixes;
validation uses the full available pre-final history. These are different
forecast difficulty distributions, so training loss need not be below validation
loss. Training metrics measure **teacher-forced
next-patch fit**. They do not demonstrate autonomous rollout reconstruction.

## A demonstrated obstacle to perfect training fit

The test found **two groups of exactly identical first-token feature tensors**
covering **six training tables**, with different normalized next-patch targets.
These are tables whose first right-aligned patch has only one observed point.
Causal normalization subtracts that point and uses scale one, making every observed
normalized value zero. Missing masks, role layout and positions are identical
within each group. Absolute normalization statistics are used to de-normalize
outputs but are not input features to the backbone.

Because the RWKV temporal processing is causal and variate attention mixes only
the current time position, later patches cannot disambiguate that first prediction.
A deterministic backbone must return the same normalized quantiles for identical
prefixes. Thus **a perfect fit of every training target is mathematically impossible
with this representation**, regardless of model parameter count or epoch count.

[input_conflicts.json](input_conflicts.json) preserves source identities, exact
feature hashes and conflicting normalized labels. Considering only these detected
conflicts yields a point-weighted pinball-loss lower bound of **0.00561970** and a
normalized median-RMSE lower bound of **0.834140** for any deterministic point
prediction (the latter minimizes squared error over a shared prediction).
These bounds are partial: there may be additional conflicts not checked by this
first-token diagnostic. Quantile training is also not the same as minimizing
squared median error.

No conflicting example was removed or hidden in the reported losses. A follow-up
representation fix should preserve sufficient causal level/scale information,
then regenerate features and repeat this experiment. Adding table identifiers
would facilitate memorization but would change the foundation-model problem and
would not address transferable representation quality.

## Artifacts and reproducibility

All outputs are under `results/capacity_300m/training/`:

- `evaluation_history.json`: initial sweep and every post-update full evaluation;
- `table_metrics.csv`: every table's loss/RMSE after every update, for both splits;
- `learning_curves.png`: full training/validation loss and normalized RMSE curves;
- `latest_weights.pt`: model-only recovery checkpoint at each epoch boundary;
- `model.pt`: final checkpoint, with pipeline config and embedding hash;
- `summary.json`: initial, final and best observed validation scores;
- `architecture.json`: measured parameter inventory.

Epoch checkpoints do not contain optimizer state, and are not exact optimizer-resume
checkpoints. This diagnostic starts from the explicitly configured initial model;
it does not silently resume from a previous interrupted run. No best-validation
weights are substituted for final weights. The best metric is recorded for
inspection while `model.pt` always represents the final update.

Logs are saved in `logs/capacity_training_console.log` and unique capacity-test run
directories. Full run configuration and source hashes are retained there. Unit
tests check padded-label counts, scalar FP32-master update accumulation, mode
restoration and batched-versus-individual CUDA evaluation. `tests/test_capacity.py`
passed **all three tests on CUDA** (two passed, one skipped in the native CPU run).

## Results

The first phase completed 150 optimizer updates (15 batches per epoch). Full
training pinball loss changed from **0.304446** to **0.296156**; final-patch validation
loss changed from **0.253387** to **0.246506**. The early higher-learning-rate phase
was visibly noisy. A second 10-epoch phase at learning rate `3e-5` checks whether
smaller updates stabilize the fit. It is run with:

```powershell
uv run python src/05_model_training/capacity_test.py --config configs/capacity_refinement.json
```

Its separate artifacts are under `results/capacity_300m_refinement/training/`, and
its full log is `logs/capacity_refinement_console.log`. Both phases retain every
measurement, including initial losses and temporary regressions. The completed combined results and coverage checks are below.

The [complete bf16 prefix scan](bf16_prefix_conflicts.json) additionally checked
every supervised causal prefix, not only the first token. It found the same two
conflicting groups after the actual bf16 feature cast.

## Completed result (finalized 2026-10-25)

Both phases completed: **20 additional epochs, 300 optimizer updates**. These follow
the original one-epoch checkpoint; this is not a fresh initialization run.
Both full-dataset sweeps completed after every update, plus a baseline sweep before
each phase. The combined curve removes the duplicate phase-boundary baseline,
leaving **301 points**; the raw phases preserve all **302 observed sweeps**.

| Metric | Initial | Final |
|---|---:|---:|
| Train Normalized pinball | 0.304446 | 0.262679 |
| Train Normalized median RMSE | 3.946191 | 3.922284 |
| Train Raw pooled median RMSE | 62.101592 | 54.172066 |
| Validation Normalized pinball | 0.253387 | 0.211091 |
| Validation Normalized median RMSE | 0.910469 | 0.814925 |
| Validation Raw pooled median RMSE | 110.647867 | 103.077533 |

Training pinball loss decreased **13.72%**;
validation pinball loss decreased **16.69%**.
**75/82** training tables and **62/82** validation tables improved in pinball loss.
Peak allocated GPU memory was **6.140 GiB**.
The measurements show learning but **do not demonstrate near-perfect memorization**.
Training normalized RMSE remains high, and the identified input collisions alone
already rule out a completely zero-error fit. They do not explain the entire
remaining error. Additional underfitting, optimization and representation limitations
remain to investigate. No failing table or ambiguous prefix was omitted.

The lower-learning-rate phase was substantially steadier than the initial phase;
loss was not monotonic after individual updates. Both final and best-observed
validation values are retained in [summary.json](summary.json), with final weights
always used for the final numbers above. There was no substitution of a better
intermediate checkpoint.

![Combined full-dataset learning curves](learning_curves.png)

The combined source artifacts are in `results/capacity_300m_combined/`:
`evaluation_history.json`, `evaluation_history.csv`, `table_metrics.parquet`,
`summary.json` and `learning_curves.png`. The final model is
`results/capacity_300m_refinement/training/model.pt`.

[evidence.json](evidence.json) records artifact hashes and coverage checks. Every
raw phase has all 82 sources exactly once per split per sweep, 94,208 training
points and 1,312 validation points. Every reported aggregate was independently
recomputed from the persisted table-level CSV and matched within `1e-6`.
[training_schedule.json](training_schedule.json) reconstructs the deterministic
seeded source schedule and verifies every source is visited once per epoch.
The report compiler is saved as `logs/compile_capacity_report.py`.

Both checkpoint files and the source/split/label masks were preserved. Three
capacity tests passed on CUDA, including batching across different kernel-padded
lengths (16 and 32 tokens). Native CPU tests passed two tests and skipped the CUDA
case. Ruff, formatting, Bandit and uv lock checks passed.

Before using more data to pursue a perfect-fit capacity check, preserve enough
causal level/scale information in the embedding representation to disambiguate
currently identical inputs, regenerate features, and repeat the same exhaustive
measurement protocol. The current result supports learning, not the stronger
claim that parameter count alone guarantees memorization of all these targets.

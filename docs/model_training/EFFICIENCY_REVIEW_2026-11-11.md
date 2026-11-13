# Basin efficiency review: where the gap to Chronos-2 / Timestral comes from

Date: 2026-11-11. Independent read-only review. No GPU work was launched, and the
running training job, its configuration and its checkpoints were not touched. All
numbers below come from files already on disk (saved holdout predictions, validation
Parquets, `best_weights.pt` read on CPU, the tail of the live training journal).
Scripts and the intermediate table are in `experiments/claude_review/`.

## Short version

1. **The saved benchmark does not describe the current model.** At the benchmarked
   weights (step 322,507) Basin's four forecast patches were the same 16 values
   repeated four times. That alone is 73% of the pooled MASE gap to Chronos-2 and
   55% of the gap to Timestral. Validation has improved about 20% since.
2. **On the typical case the current model has probably already caught up**, but
   this is an indirect comparison and needs a fresh holdout run to confirm.
3. **The remaining inefficiency is mostly training recipe and compute allocation**,
   not RWKV-7 as such: weight decay is applied to parameters it should not touch,
   the learning-rate schedule is driven by a handful of outlier forecasts, and a
   500M model on a 12 GB card sees each table only about once.
4. **Muon² is not the first thing to try.** At 8 tables per update its fixed
   per-step cost is likely as large as the forward/backward pass itself.

## Evidence

### 1. The benchmarked checkpoint repeated one patch four times

From `results/model/holdout_test/test_predictions.parquet` (5,820 requests, weights
from step 322,507). Distance between each model's own patch *k* and patch 1
forecast, in MASE units, median over requests:

| Series | patch 2 vs 1 | patch 3 vs 1 | patch 4 vs 1 | corr(patch 1, patch 2) |
| --- | ---: | ---: | ---: | ---: |
| Basin | 0.003 | 0.003 | 0.003 | 1.000 (99.9% of requests > 0.95) |
| Chronos-2 | 1.347 | 1.550 | 1.152 | −0.03 |
| Timestral | 1.253 | 1.436 | 1.139 | 0.03 |
| Actual values | 2.424 | 2.744 | 2.659 | −0.05 |

Cause: later leads are produced by adding a lead embedding (initialised at std 0.02,
norm about 0.64) to the hidden state before the final LayerNorm. Under the BF16
update defect those embeddings were 97% frozen, so the offset was negligible. This
is the "learned tendency to repeat the same 16 timestep token" item in
`docs/OPEN_ISSUES.md`.

Contribution to the pooled MASE gap (lead weights 80/16/3/1):

| Forecast patch | Basin | Chronos-2 | Timestral | Share of gap to Chronos-2 | Share of gap to Timestral |
| --- | ---: | ---: | ---: | ---: | ---: |
| 1 (steps 1–16) | 2.163 | 2.100 | 1.995 | 27% | 45% |
| 2 (steps 17–32) | 3.979 | 3.344 | 3.209 | 54% | 41% |
| 3–4 (steps 33–64) | 7.29 / 11.28 | 6.40 / 10.41 | 6.28 / 10.37 | 19% | 13% |

In `best_weights.pt` (step 662,047) the lead embeddings now have norms 59, 83 and
81, so they are no longer inert. Whether the head now produces good later patches
cannot be checked without running the model.

### 2. The old checkpoint also failed to use long history

Same holdout file, median per-request MASE of forecast patch 1 only:

| Context points | Basin | Chronos-2 | Timestral | Basin excess vs Chronos-2 | vs Timestral |
| --- | ---: | ---: | ---: | ---: | ---: |
| ≤ 128 | 1.512 | 1.583 | 1.516 | −4% | 0% |
| 129–256 | 1.259 | 1.168 | 1.136 | +8% | +11% |
| 257–512 | 1.161 | 1.110 | 1.041 | +5% | +12% |
| 513–1024 | 1.167 | 1.044 | 0.987 | +12% | +18% |
| 1025–2048 | 1.229 | 0.986 | 0.928 | +25% | +32% |

With little history all three models tie. The competitors keep improving as
history grows; the old Basin weights did not.

### 3. Validation since the precision repair

Median per-request validation MASE (125,000 forecasts, 80/16/3/1 weights):

| Check | Step | All | ≤ 64 | 65–128 | 129–256 | 257–512 | 513–1024 | 1025–2048 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 4 (benchmarked weights) | 322,507 | 1.537 | 1.820 | 1.780 | 1.547 | 1.447 | 1.442 | 1.444 |
| 11 (latest) | 703,426 | 1.229 | 1.758 | 1.521 | 1.256 | 1.119 | 1.077 | 1.063 |
| Holdout, Chronos-2 | — | 1.312 | 1.821 | 1.586 | 1.361 | 1.210 | 1.140 | 1.058 |
| Holdout, Timestral | — | 1.249 | 1.796 | 1.603 | 1.334 | 1.141 | 1.058 | 0.995 |

The competitor rows are from the 1,184 holdout tables, the Basin rows from the
25,000 validation tables, so this is **not** a like-for-like comparison. What
supports reading them together: at the same weights, Basin's holdout median
(1.539) and validation median (1.537) agree. The pooled mean MASE cannot be
inferred this way because it is dominated by a few extreme requests.

### 4. Weight decay is shrinking parameters it should not touch

`performance.make_optimizer` puts every parameter in one AdamW group with weight
decay 0.01. In `best_weights.pt`:

- All LayerNorm/GroupNorm gains have drifted from 1.0 to a mean of 0.81–0.90
  (final norm 0.897). The decay expected from `exp(-wd · lr · steps)` over the
  roughly 240,000 FP32 updates at 5e-5 is 0.887.
- RWKV's base decay `w0` in layer 13 now spans −7.19 to 2.70; its initial range is
  −8 to 3, and pure decay predicts −7.09. `w0` sets how long each channel
  remembers, and pulling it towards zero shortens memory. For a channel that
  starts at −4 the half-life is about 64 patches; an 11% shrink brings it to
  about 41, and the pull continues for as long as training runs. Gradients can
  counteract it; for the gains they evidently have not.

The upstream RWKV-7 recipe (`configure_optimizers` in the vendored `model.py`)
applies weight decay only to 2-D weight matrices and gives `w0` twice the learning
rate.

### 5. The schedule is steered by five forecasts

The plateau scheduler (patience 1, factor 0.5) and early stopping (patience 3) both
monitor the mean validation loss. Five of 125,000 forecasts make up 54% of that
mean. Checks 9–11:

| Check | Mean loss | Median table loss | Median request MASE | Verdict recorded |
| --- | ---: | ---: | ---: | --- |
| 9 | 1.6976 | 0.1995 | 1.2554 | improved |
| 10 | 1.6956 | 0.1971 | 1.2326 | improved |
| 11 | 1.6964 | 0.1980 | 1.2288 | not improved |

Check 11 differs from check 10 by 0.05% on the mean. The scheduler already has one
bad check recorded: one more halves the learning rate to 2.5e-5, and early
stopping is at 1 of 3.

### 6. Compute, not parameter count, is the binding constraint

- Live journal, steps 722,918–744,890: 0.293 s per update, 26.3 tables/s, median
  7,472 supervised target points per update. One epoch is about 56–60 active hours.
- After 44 active hours the model has seen each of the 5.34M training tables
  about once.
- FP32 master weights and Adam moments hold about 7.5 GiB of the 10.3 GiB
  allocator budget. The remaining activation memory forces an average of two
  physical passes per update (1.7× slower than before the repair, per the epoch-2
  throughput report).

### 7. Things checked and found to be minor or fine

- **BF16 rounding of inputs and outputs.** Confirmed: Basin's normalised median
  forecasts sit exactly on the BF16 grid (median distance 0.0001 grid spacings; a
  random value would be 0.25). The error floor this creates is small: mean 0.012
  MASE on outputs and 0.011 on the last input patch, 0.05 in the smoothest eighth
  of series, above 0.1 for 1% of requests.
- **Batch ordering.** Lag-1 autocorrelation of log update loss is 0.09; median
  same-shape run length is 1. Storage-local shuffling is not producing visibly
  correlated updates.
- **Loss outliers and gradients.** Update loss has median 0.22 and mean 554, but
  the gradient norm has median 0.11 and only 0.7% of updates are clipped. The
  outliers distort monitoring, not the updates.
- **RWKV initialisation.** The constructors use the standalone initialisation
  upstream recommends; only the per-layer `ln_x` gain scaling is skipped.

## Recommended changes, in order

### Before anything else (no retraining)

| # | Change | Why |
| --- | --- | --- |
| A1 | Re-run the holdout benchmark on current weights, with a per-patch and per-context-length breakdown | The saved comparison measures a defect that has since been repaired |
| A2 | Monitor a robust statistic (trimmed or median table loss) for LR decay and early stopping | The current monitor moves on five forecasts; it is one check from halving the LR of the live run |

A1 needs the GPU, so it has to wait until the run is paused or finished. A2 needs
a restart to take effect, so it is a decision for the next safe boundary of the
current run rather than something to do now.

### High likelihood, low cost (next run or a declared resume branch)

| # | Change | Why |
| --- | --- | --- |
| B1 | Optimizer parameter groups: no weight decay on norm gains, biases, embeddings and RWKV's 1-D control parameters; 2× LR on `w0` | Measured shrinkage of gains and of the memory-length parameter; matches the upstream recipe |
| B2 | Give each lead its own output projection (or one head that emits all four patches) instead of an additive offset before LayerNorm; train with flatter lead weights and keep 80/16/3/1 for reporting | Later patches were the largest part of the measured gap and receive 16/3/1% of the gradient through a weak conditioning path |
| B3 | Fixed-budget warmup plus cosine or linear decay; evaluate and export an exponential moving average of the weights | Eight tables per update is noisy; averaging is the cheap way to recover what a larger batch would give |
| B4 | Random sign flip at training time; optionally symmetric averaging at inference | Timestral's default already averages a sign-flipped pass, so the current comparison gives it an ensemble Basin does not have |

### Likely helpful, needs a new run (test at small scale first)

| # | Change | Why |
| --- | --- | --- |
| C1 | Train a model of roughly 120–150M parameters (for example width 768, 16 layers) | Fits without microbatch splitting, so it sees several times more tables per hour; both competitors are in this size range |
| C2 | `asinh` after causal standardisation, for inputs and targets | Compresses the extreme normalised values (inputs are currently clamped only at ±10,000) |
| C3 | Variable mixing every fourth layer with a zero-initialised gate | 118M parameters (23.5% of the model); pure overhead on the roughly 30% of tables with no covariates |
| C4 | Run the first input layer and the last output layer in FP32 | Removes the measured rounding floor; these layers are 2% of parameters |
| C5 | A few temporal attention layers on top of RWKV | Only if the long-history gap in section 2 is still there after A1 |

### Muon²

Not recommended as the next step. Orthogonalising every weight matrix is a fixed
cost per update. My estimate for this model is about 1.3e13 floating-point
operations per update for five Newton–Schulz iterations, against roughly 3e12 for
the forward and backward pass at the current ~900 variable-patch tokens per
update. Large matrix products run more efficiently than small-batch training
steps, so the time penalty is smaller than that ratio, but it could still roughly
double step time unless the logical batch grows to 64 tables or more. It also
leaves RWKV's many non-matrix parameters on AdamW.
B1 and B3 address known defects in the current recipe at no cost and should come
first; Muon or Muon² is then a fair A/B on the small model from C1. The
operation counts are an estimate, not a measurement.

### Data generation

The generator is not what the evidence points at: two models that never saw it
forecast it well, and all three models land within about 12% of each other and
20–28% ahead of the naive forecast, which suggests a large irreducible share.
Two suggestions:

- Measure the headroom. Replaying a saved SCM with fresh noise seeds from a fixed
  origin gives the spread no model can beat; that tells you how much of the
  remaining error is learnable.
- Shift the length distribution towards long tables. With 128 points or fewer of
  history all three models tie; the differences appear above 256 points. 21.5% of
  validation forecasts have 64 points or fewer.

## What this review did not establish

- No change above has been tested; each is a hypothesis backed by a measured
  mechanism, not a measured gain.
- The current head's later-patch quality is unknown until the model is run.
- The parity statement in section 3 compares different table samples.

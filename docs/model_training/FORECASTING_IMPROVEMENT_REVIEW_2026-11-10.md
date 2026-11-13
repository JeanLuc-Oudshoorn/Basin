# Basin forecasting quality: validation, architecture and training review

Date: 2026-11-10. Completed validation checks 0–8 are the analysis snapshot.

## Findings and recommended order

The accuracy gap deserves investigation, but the evidence does **not** establish that RWKV-7 is the wrong backbone. The strongest immediate findings are:

1. The displayed training loss is an online, epoch-to-date average over a changing cohort, not an evaluation of a fixed training panel. Its movement is not directly comparable across validation checkpoints.
2. Check 8 improves typical validation quality appreciably more than the headline mean suggests: median table loss improves **2.16%** and median table MASE **3.11%** over check 7. The mean loss improves only **0.247%** because a few extreme cases dominate its level.
3. The saved competitor evaluation uses Basin weights from **step 322,507**, before the precision repair. It cannot establish the current checkpoint's gap. Its existing gaps are **7.45% to Chronos-2** and **12.37% to Timestral** on pooled MASE, rather than exactly 10% and 15%.
4. There are substantive differences worth testing: robust transformed targets, inference-time symmetry, the forecast decoder, how often variables mix, and supervision/batch composition. Increasing parameter count alone does not address these.
5. Keep the current corrected run as a reference. Improve monitoring first; then use controlled branches for changes to the objective, architecture, or optimizer recipe. Do not combine every proposed change into one experiment.

This review adds a report and a CPU analysis script. It does not change training configuration, running processes, checkpoints, data, or benchmark adapters. No additional GPU evaluation was launched.

## 1. What the training-loss numbers actually mean

In [`timed_training.py`](../../src/05_model_training/timed_training.py), `fit_timed` accumulates:

```text
epoch_loss_sum += update_loss * effective_label_mass
epoch_mass     += effective_label_mass
validation_history.training_loss = epoch_loss_sum / epoch_mass
```

Both accumulators reset when the epoch advances. They persist through resume. Thus check 7's training loss includes every update in the current epoch up to step 531,049; check 8 includes more updates, up to 580,566. They contain different examples **and predictions produced by many earlier model states**. They are neither interval-only losses nor fixed-panel losses evaluated using the checkpoint weights.

The logical batch is eight tables. Updates are weighted by observed label mass, so longer training windows with more supervised anchors have more influence than short ones. In contrast, the validation panel has five fixed, 64-step forecast origins per table. Moreover, training supervises many causal prefixes, including very short prefixes, whereas validation requires enough history and future for its complete four-token requests. Numerical training/validation loss levels are therefore not a simple generalization gap.

[`BatchPlanner`](../../src/05_model_training/batching.py) also keeps shard-local sections together when `storage_locality=true` and groups shape runs of 128 batches. It shuffles within those constraints; it does not draw eight independent tables uniformly from the entire corpus at each update. Similar generators or augmentation families can remain locally concentrated. This is a plausible source of changing difficulty, not a demonstrated data-loader bug.

There is now direct evidence of heavy tails in a **bounded sample of training updates**. In the saved 1,000-update sample, steps 603,888–604,887:

| Statistic | Value |
| --- | ---: |
| Median update loss | 0.2253 |
| Label-mass-weighted mean update loss | 57.4055 |
| Maximum update loss | 88,280.4141 |
| Share of weighted loss from five updates | 96.83% |
| Median gradient norm, measured before clipping | 0.1211 |
| Maximum gradient norm, measured before clipping | 2.8125 |

This establishes concentration in this sample's **loss values**, not that five particular tables dominate the entire training run. Update losses combine multiple tables; the sample cannot identify a full-run table distribution. It also does not establish proportional gradient domination: pinball loss has bounded slope with respect to the normalized quantile prediction, and the trainer clips the global gradient norm at 1.0. A loss of 88,000 does not imply a gradient 88,000 times larger.

### Proposed monitoring correction

- Rename the current field to `online_epoch_to_date_loss` in displays while preserving its meaning in old records.
- Log a label-mass-weighted rolling mean plus median/p90 of recent update losses, with the window explicitly stated. These are operational diagnostics, not model-selection scores.
- Add a small, fixed **training monitoring panel**, evaluated without gradients at the same origins and weighting as validation. Mark it in-sample and keep it separate from the reserved validation/test sources. This makes train-versus-validation comparisons meaningful.
- Log point/label mass, context length, variable count, generator family where available, and clipping frequency. An identical loss formula does not make changing cohort averages identical experiments.

## 2. Check 8 is a broader improvement than the headline suggests

I reconstructed each validation loss from its saved request Parquet, verified the reconstruction against `validation_history.json`, and aggregated five origins within each table before computing robust summaries. The same 25,000 table IDs occur in paired checks.

| Check | Step | Online training loss | Mean validation loss | Median table loss | Trimmed table loss* | Median table MASE |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| 4 | 322,507 | 276.9108 | 1.727374 | 0.233677 | 0.239579 | 1.745566 |
| 5 | 404,768 | 405.7740 | 1.727377 | 0.233567 | 0.239567 | 1.739124 |
| 6 | 478,212 | 378.3252 | 1.729597 | 0.234457 | 0.240442 | 1.751336 |
| 7 | 531,049 | 388.2473 | 1.704223 | 0.205654 | 0.214980 | 1.530941 |
| 8 | 580,566 | 372.2057 | 1.700020 | 0.201204 | 0.211219 | 1.483276 |

*Remove the lowest 5% and highest 5% of table losses and average the central 90%. This is computed for this review, not implemented as the running monitor.*

From check 7 to 8:

- Mean loss improves **0.247%**; median table loss improves **2.164%**; trimmed table loss improves **1.749%**.
- Median table MASE improves **3.113%**. Loss improves on **61.456%** of tables; MASE improves on **62.646%** of the 24,945 tables with defined paired table MASE.
- The median paired MASE change is **−0.02912**. A seed-7, 1,000-replicate table bootstrap gives a descriptive 95% interval **[−0.03066, −0.02770]**. This is the median of paired differences, not the difference between two medians.
- Table MASE p90 improves from **3.5597 to 3.4480**, while p99 worsens from **19.4470 to 19.6445**. Improvement is broad, but not uniform in the tail.
- Five of 125,000 forecasts still contribute **53.93%** of aggregate loss at check 8.

The bootstrap treats tables as sampling units. Shared generator mechanisms can correlate them; this is not a multi-seed confidence statement about future training runs.

From check 5 to 7, both median table loss and median table MASE improve by approximately **12%**. That is consistent with the precision correction helping. However, the learning rate also halves from 1e-4 to 5e-5 at check 6, and training continues on more data. These are observational changes, not a clean causal estimate of the precision fix alone. At check 8 the first epoch is still only **83.69%** complete. Most earlier optimization happened before FP32 masters were introduced.

### Model selection should match the intended performance

Use the trimmed table loss as a proposed primary monitor, with median table MASE, p90/p99 and the original untrimmed means retained. Pinball loss divided by historical standard deviation and MASE divided by historical mean absolute lag-one change emphasize different cases. Lowering one need not minimize the other.

The current configuration still monitors `loss`; only `loss` and `rmse` are currently accepted by the stopping-options validator. Robust monitoring has been discussed but is **not yet implemented**. Changing it requires adding the new metrics and an explicit state migration: establish a new selection baseline, reset metric-specific best/patience state, retain current LR, weights, optimizer state and cursor, and distinguish the new monitor in provenance. Compare available current and saved-best weights under the new score; do not compare a trimmed loss numerically with an old mean-loss best.

## 3. What the holdout comparison establishes

The saved run evaluates 1,184 tables, with 5,820 origin-target requests. All models receive the same capped historical target/past-covariate inputs, the same 64-step horizon, and the same scoring weights. Chronos cross-learning across unrelated requests is disabled. The adapter supplies no observed future covariates.

| Model | Parameters measured by the adapter | Pooled MASE | Basin's relative excess MASE |
| --- | ---: | ---: | ---: |
| Basin | 500,056,352 | 2.698244 | — |
| Chronos-2 | 119,477,664 | 2.511179 | 7.45% |
| Timestral | 209,312,256 | 2.401175 | 12.37% |

Chronos-2 is approximately **120M**, not 109M; **108M** is TimeH2O-v1's published size. Basin is 4.19× the measured Chronos model and 2.39× Timestral. These counts describe capacity, not effective training compute or optimization quality. [Chronos card](https://huggingface.co/amazon/chronos-2), [TimeH2O card](https://huggingface.co/h2oai/timeh2o-v1).

The gap is not solely a few pathological requests: averaging origins within tables, Chronos has lower MASE on **72.23%** of paired valid tables, Timestral on **80.95%**. But this compares the **old, pre-repair Basin best checkpoint at step 322,507**. A fresh comparison is needed before describing the quality of check 8.

| Forecast patch | Basin MASE | Chronos-2 | Timestral |
| --- | ---: | ---: | ---: |
| 1: steps 1–16 | 2.1628 | 2.1002 | 1.9946 |
| 2: steps 17–32 | 3.9786 | 3.3437 | 3.2092 |
| 3: steps 33–48 | 7.2898 | 6.3987 | 6.2784 |
| 4: steps 49–64 | 11.2760 | 10.4106 | 10.3662 |

The second-patch gap is particularly worth testing. However, both training and this holdout score deliberately use 80/16/3/1 weights. The weighting is not a train/test mismatch; changing it is a product-objective experiment, not a bug fix.

### An important inference-default difference

The pinned Timestral package defaults to **symmetric averaging**, forecasting the negated inputs as well and reflecting/averaging quantiles, plus **nonnegative clipping** when all observed target values are nonnegative. Basin's adapter calls `forecast` without overriding either default. Its model weights are FP32 in this evaluation; Basin's compute weights are BF16. The recorded timings include the default forecasting method. These are legitimate product defaults, but they must be recorded explicitly when comparing architectures. [Pinned forecaster source](https://github.com/mistralai/timestral/blob/a534bdbd244d64f3d9d743f46c7bf6fc330ed4ff/src/timestral/forecaster.py).

For an ablation, compare Timestral defaults and its raw single-pass output, and Basin single-pass versus a sign-symmetric version. For a symmetric quantile grid:

```text
Q_sym(tau, x) = [Q(tau, x) - Q(1 - tau, -x)] / 2
```

Negate numeric past covariates consistently. This doubles forecast passes, so report the cost. Apply support constraints only when the forecasting task justifies them: a positive observed history does not guarantee that a generated future cannot become negative. Preserve the official-default baseline as a separate result rather than weakening it to obtain a favorable leaderboard.

## 4. Architectural and training differences worth borrowing

### Basin's actual architecture

The current [`architecture.json`](../../results/model/training/architecture.json) records 28 temporal RWKV-7 blocks, width 1024, patch size 16, and cross-variable attention **after every block**. There are at most 128 history tokens at the 2,048-observation cap. Targets and past covariates use causal prefix standardization, value/mask/time features and role embeddings.

The model is initialized as a forecaster, not from pretrained RWKV language-model weights. Every causal anchor supervises up to four future target patches. All four forecasts are projections of the same final hidden representation, modified by three lead embeddings; later horizons do not receive new backbone processing or attend separately to history.

| Parameter allocation | Parameters | Share |
| --- | ---: | ---: |
| Temporal RWKV blocks | 373,053,440 | 74.60% |
| Cross-variable attention | 117,612,544 | 23.52% |
| Input projection | 4,446,208 | 0.89% |
| Forecast head | 4,935,968 | 0.99% |

The small lead embedding count alone does not prove a bottleneck: the shared nonlinear head has nearly 5M parameters. Conversely, 118M cross-variable parameters do not guarantee better use of covariates.

### Timestral-209M

The model card gives 24 layers, width 768, 32-observation patches, 21 quantiles and a 32,768-observation maximum context. That larger maximum context was **not** available as extra input in our capped evaluation. [Model card](https://huggingface.co/mistralai/Timestral-209M).

Its published design separates observed-context tokens from learned horizon tokens. Those forecast tokens read the permissible history through the transformer, unlike Basin's lead-offset readout. It pretrains on synthetic data at context 8,192, then post-trains on real/synthetic data at 32,768, with normalized-space pinball loss. Public descriptions inspected here do not establish an optimizer/batch-size recipe that we can faithfully reproduce. [Technical report, §§3–4](https://arxiv.org/html/2610.37585v1).

The released layers use RMSNorm, SwiGLU, rotary temporal positions, sigmoid-gated attention and variable mixing every fourth block. These are inspectable design differences, not evidence each one would improve Basin. [Pinned layers](https://github.com/mistralai/timestral/blob/a534bdbd244d64f3d9d743f46c7bf6fc330ed4ff/src/timestral/layers.py).

Two especially concrete implementation differences are causal standardization followed by **asinh**, inverted with sinh, and a **direct median with cumulative positive quantile increments**, rather than sorting nine unconstrained outputs as Basin does. The scaler computes running statistics in FP64. [Scaler](https://github.com/mistralai/timestral/blob/a534bdbd244d64f3d9d743f46c7bf6fc330ed4ff/src/timestral/scaler.py), [model/head](https://github.com/mistralai/timestral/blob/a534bdbd244d64f3d9d743f46c7bf6fc330ed4ff/src/timestral/model.py).

### TimeH2O-v1

Its model card specifies 108M parameters, nine quantiles and up to 16,384 context observations. [TimeH2O model card](https://huggingface.co/h2oai/timeh2o-v1).

The public description reports a causal transformer with group attention every fourth block, flip equivariance, patch sizes 2/4/8/16 with multi-resolution inference, a frozen univariate backbone followed by covariate-adapter training, sequence packing, varied masking, and a lightweight next-patch auxiliary head. It also reports quantile-spread-based loss weighting to emphasize more predictable values. [Publisher architecture/training description](https://h2o.ai/blog/2026/timeh2o-foundation-models).

Basin already supervises next-patch predictions at every causal anchor; it is not missing all dense supervision. The differences to test are the distribution of supervised contexts/horizons, information learned from covariates, and padding/compute efficiency. The exact spread-weight formula, stabilization/detachment, and optimizer settings are not specified sufficiently in that public description. An unconstrained inverse-predicted-spread weight can reward inflated intervals; copying that idea without validating its gradients is unsafe statistically.

### Chronos-2

Chronos-2 adds asinh to historical standardization, uses temporal and group attention, and decodes future patch embeddings directly. Its training includes real and synthetic data and structured synthetic multivariate relationships. These overlap substantially with the principles of Basin's synthetic-data approach; the external models are not necessarily learning an entirely unrelated class of processes. [Chronos-2 technical report](https://arxiv.org/html/2510.15821v1), [training-data overview](https://huggingface.co/amazon/chronos-2).

## 5. Prioritized experiments

These are proposed experiments, not measured improvements. Use a fixed development panel from reserved validation sources for selection; keep a separate final test panel for confirmation. Start with 1,000–2,000 stratified development tables to keep competitor evaluation practical, then confirm winners on the complete reserved panel. Do not repeatedly tune on the existing test leaderboard.

| Priority | Experiment | Why it is relevant | Main acceptance evidence |
| --- | --- | --- | --- |
| P0 | Robust monitor + fixed training panel | Removes the demonstrated ambiguity in progress/selection | Correct weighted arithmetic; stable panel; tail metrics retained |
| P0 | Updated checkpoint benchmark and explicit inference defaults | Existing leaderboard predates the repair | Same tables/origins; paired table differences; latency and options recorded |
| P1 | Asinh-space inputs, targets and quantile outputs | Very large future/history scale ratios remain after standardization | Better trimmed loss/MASE and calibration; original-unit tail scores retained |
| P1 | Median-anchored monotone quantile head | Gives the median a stable output channel | Median error and interval coverage improve together |
| P1 | Richer horizon decoder | Later horizons currently share one compressed representation | Per-patch MASE gains without worsening the first patch |
| P1 | Sparse/gated variable mixing and target-only control | Nearly a quarter of parameters currently mix variables | Informative-covariate gain; univariate quality; equal-compute comparison |
| P1 | Batch/sampling and optimizer parameter groups | Eight related tables/update may be a poor optimization mix | Stable matched-budget curves across seeds |
| P2 | Smaller transformer and hybrid controls | Directly tests the backbone question | Same preprocessing/data/objective and both equal-compute/equal-size comparisons |
| P2 | Multi-resolution and richer data/covariate supervision | Tests representation and information coverage | Gains by generator family, context length and variable count |

### P1a. Transform extremes without introducing future information

Basin already standardizes errors. A fixed unit-dependent standard-deviation floor would reintroduce a problem previously removed. Instead, compare a separate branch using:

```text
z = asinh((y - historical_mean) / historical_std)
y_quantile = historical_mean + historical_std * sinh(z_quantile)
```

Inputs and supervised outputs must use the same causal frame at each forecast anchor. Apply quantile loss in transformed space; invert for raw-unit metrics and MASE. Keep missing masks and complete-horizon rules unchanged. An input-only transform would leave the output objective's extreme scale intact. This is a representation/objective change, requiring retraining or an explicit adaptation branch, not a silent preprocessing change to the active checkpoint.

Monotonic transforms preserve ordering of quantiles, but finite models, inverse-transform overflow, loss weighting, and calibration still require validation. The benefit is a testable conditioning hypothesis, not a guarantee of lower MASE. Remember that untransformed pinball gradients are already bounded; enormous scalar losses alone do not prove exploding updates.

### P1b. Readout and useful supervision

Compare the existing head with (a) a direct median plus positive interval increments and (b) a modest lead-conditioned decoder that can read history states. A few temporal attention/query blocks after RWKV are a useful intermediate design: they preserve the backbone while allowing each forecast lead a different history readout. Keep the 80/16/3/1 objective fixed initially to isolate the decoder. A later horizon-weight experiment must state that it changes the short-term product preference.

Also test a small auxiliary weight for causal next-patch prediction of observed covariates. Currently covariates are inputs only, with no direct forecast supervision. Richly generated covariates therefore consume compute while receiving gradients only through target usefulness. Auxiliary covariate targets may improve representation, or may waste capacity on irrelevant/noisy series; compare both. Future covariate labels must remain labels and never enter forecast inputs.

### P1c. Reallocate variable-mixing capacity

Compare a new model with variable mixing every fourth layer, a gated residual mix initialized near zero, and an independently trained target-only control. Simply removing trained variable mixing at inference is not a fair ablation. Retaining seven of the existing 28 full-width mixers would save approximately **88.21M** parameters; it would still leave about **411.85M**, not magically create a 200M model. Test smaller width/depth separately if compute efficiency is the goal.

Past-only variables here remain causal. The current unrestricted cross-variable mixing is not, by itself, future leakage. If known-future covariates are added later, explicit role-aware masks are necessary; role embeddings alone do not enforce causality.

### P1d. Optimizer and example balance

Current production AdamW uses FP32 masters/moments, BF16 model gradients, LR now 5e-5, beta=(0.9, 0.999), epsilon=1e-8 and weight decay 0.01 applied through one parameter group. The original RWKV-7 language-model recipe instead specifies beta2=0.99, selective decay for linear/embedding weights, a 2× LR group for base decay `w0`, and specialized initialization. Those settings motivate controlled tuning; its epsilon=1e-18 is not a safe forecasting default to copy without numerical tests. [RWKV-7, Appendix E](https://arxiv.org/html/2503.14456v1#A5).

The vendored constructors already initialize time-mixing output and channel-mixing value projections to zero and initialize several control tensors explicitly. I did not find evidence that Basin accidentally omits all RWKV initialization. The added cross-variable blocks have their own initialization and deserve separate residual-scale tests.

First compare selective no-decay for normalization/bias/control parameters, beta2=0.99 versus 0.999, and a modest LR sweep. Then compare logical batches 8/32/64 using accumulation rather than requiring larger physical microbatches. Gradient clipping and point-weighted accumulation must stay at the logical-update boundary.

Interleave a bounded number of shards/families to improve diversity without abandoning storage locality. Compare current point-weighted training with per-table-balanced weighting on a pilot: long tables currently contribute more anchors. Measure actual supervised target patches and padding, rather than treating all covariate tokens or augmentation copies as independent training examples. The current token inventory is explicitly `not_computed`; do not derive a universal data-sufficiency claim from an unverified token estimate.

These changes affect optimization identity and usually require a declared continuation branch or new run. Do not reset moments/LR or change batch composition silently in the reference run. Muon and FP64 are lower priorities than fixing measurement and testing the known recipe differences; no inspected competitor source establishes that either is the missing ingredient.

### P2. Sampling, data and multiple resolutions

The corpus admission artifact records screening skipped and all 5,483,212 sources admitted. Being generated by the same mechanism as test data is helpful, but does not ensure the model has learned its conditional forecasting distribution. Hidden drivers, noisy jumps, short histories, related copies and limited effective independent examples remain relevant. Broader external pretraining can learn generic trend/seasonal/causal structures that transfer to these draws.

Construct a small development battery with separate trend, seasonal, regime-change, flat-to-active, noisy, univariate, informative-covariate and irrelevant-covariate cases. Include low-noise/oracle-generator conditional forecasts where the generator makes those available; this distinguishes avoidable model error from uncertainty that no history-only forecaster can remove. Do not screen the final test set with a competitor to make it easier.

Test patch sizes 8/16/32 before implementing a shared multi-resolution model. At the current cap, patch size 8 needs 256 tokens, not 128: account for kernel limits, training cost and altered context distribution. Multi-resolution inference and sequence packing are not free accuracy gains. Packing in an RWKV implementation must reset WKV state and time-shift state at each series boundary and prevent variable mixing between unrelated series. Concatenating tables without these safeguards leaks information.

## 6. Does recurrence make RWKV-7 unsuitable?

There is a plausible inductive-bias trade-off. A transformer can make a separate content-dependent read from stored history for each forecast query. A recurrent backbone compresses history into finite state; patterns needing precise retrieval of a distant matching phase may be harder. Basin then imposes an additional readout restriction by projecting multiple horizons from one hidden representation. This is a hypothesis to test, not proof of an inherent RWKV failure.

At only **128 historical patch tokens**, the quadratic temporal-attention term is much less daunting than a comparison based on 2,048 raw observations suggests. A smaller transformer is therefore a practical control. Benchmark actual wall time and memory; theoretical linear scaling does not guarantee the fastest implementation at these sizes or variable counts.

Three relevant research results prevent a categorical conclusion:

- **TiRex** is a competitive time-series foundation model built on xLSTM, demonstrating that recurrence can work well for zero-shot forecasting. It does not establish that RWKV and xLSTM are interchangeable. [Paper](https://arxiv.org/html/2505.23719v1).
- **BlackGoose Rimer** reports favorable RWKV-7-derived forecasting results against Timer, but uses a small modified model and a limited benchmark/setup. Its headline gains are not a prediction for Basin versus modern multivariate probabilistic foundation models. [Preprint](https://arxiv.org/html/2503.06121v1).
- **HART** studies degradation when RWKV state advances through future placeholder tokens, and reports benefits from hybrid attention/RWKV blocks. Its limitations acknowledge that the mechanism is not completely isolated. **Basin's current direct four-patch head does not run future placeholders through the backbone**, so that mechanism does not directly diagnose this run. It does motivate a carefully matched hybrid experiment. [Preprint, §§3, 6](https://arxiv.org/html/2610.12883v1).

For the decisive experiment, train a modest common-size RWKV, a transformer and a hybrid on the same fixed corpus, causal scaler, masks, quantiles, decoder, supervision, optimizer precision and validation panel. Compare both equal parameter count and equal wall-clock/target-label budget, using multiple seeds. Only then increase scale. Changing backbone, scaler, head and training data together would not answer whether RWKV itself is responsible.

## 7. Concrete next steps

1. Continue the corrected reference run while adding monitoring in a reviewed change: fixed in-sample panel, recent-update summaries, per-table robust validation and tail reporting. Keep the current mean metrics for continuity.
2. Evaluate a saved post-repair checkpoint on a predeclared development panel against frozen baseline configurations. Record inference defaults and paired per-table improvements. Confirm a final winner once on held-out test sources.
3. Run the inexpensive symmetry/defaults ablation and a small head comparison before committing to a complete new 500M training run.
4. Run a small factorial pilot: current scaler versus transformed scaler, and current versus sparse/gated variable mixing. Advance changes that improve typical quality without unacceptable tail/calibration regressions.
5. Use that evidence for an RWKV/transformer/hybrid comparison at a smaller common scale. Budget by useful supervision and actual compute, not parameter count alone.

The present evidence supports **better monitoring and controlled recipe/representation experiments**. It does not justify concluding that 500M parameters should automatically beat pretrained smaller models, that full FP64 is needed, or that replacing RWKV alone will close the gap.

## Reproduction and evidence

Run from the repository root:

```powershell
.\.venv\Scripts\python.exe scripts/analyze_forecasting_improvements.py
```

The script reads completed evaluations through check 8 and a snapshot of the latest saved 1,000 updates. It checks duplicate request keys, stable paired table IDs and forecast-origin panels, and exact reconstruction of aggregate validation loss, computes paired table bootstrap summaries, and writes [`forecasting_review_20261110/evidence.json`](forecasting_review_20261110/evidence.json). That file includes input hashes, the holdout's saved checkpoint identity and measured benchmark parameter counts. Rerunning during training can change the recent-update snapshot; the hashes identify this report's snapshot. It does not load weights, scan the full corpus or launch CUDA.

Local code reviewed: [timed training](../../src/05_model_training/timed_training.py), [batch planning](../../src/05_model_training/batching.py), [causal normalization](../../src/04_creating_embeddings/embed.py), [training/readout](../../src/05_model_training/train.py), [four-lead loss](../../src/05_model_training/multi_horizon.py), [validation](../../src/05_model_training/rolling_validation.py), [master optimizer](../../src/05_model_training/master_optimizer.py), [benchmark adapters](../../src/06_holdout_test/benchmarks.py), [vendored RWKV blocks](../../third_party/rwkv-lm/RWKV-v7/train_temp/src/model.py).

External sources were accessed on 2026-11-10. Timestral code links pin the dependency commit `a534bdbd244d64f3d9d743f46c7bf6fc330ed4ff`; its evaluated weights use revision `9874d252ab0aae16f7f7c999f98f98d6ad7d8365`. Model cards and publisher benchmark claims describe their own evaluations, not verified gains on Basin. Public inference code does not establish an undisclosed full training recipe.

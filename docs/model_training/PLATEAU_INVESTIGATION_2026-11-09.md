# Validation plateau and training precision investigation

November 9, 2026. Read-only investigation of the active run. No training code,
configuration, checkpoint, split or optimizer state was changed; no CUDA work
was launched. Small CPU probes and saved validation Parquets were used.

## Conclusion

Checks 4–5 confirm a practical plateau in the medians, not just the means.
There is also a concrete training precision defect to investigate before
interpreting this as exhaustion of RWKV's capacity: parameters and AdamW
moment estimates are stored in BF16, without FP32 master weights or a rounding
residual. Small updates can disappear at every optimizer step. Real checkpoint
comparisons show normalization scales remaining exactly one, with nonzero
optimizer momentum. This establishes a plausible mechanism and affected
parameters; it does not quantify its contribution to forecasting error.

The current model is **500,056,352 parameters, 28 layers, width 1024**, rather
than 397M. This was checked from the active checkpoint, not inferred from an
older architecture document. The source-screening selection artifact also
confirms that screening was disabled for this corpus.

## The medians

For each check, first average the five forecast-origin metrics within each
table, then take the median across 25,000 tables. Each origin already uses
the 80/16/3/1 percent horizon weights. This is distinct from taking the median
over 125,000 individual forecast requests.

| Check | Active hours | Table median loss | Table median MASE | Request median loss | Request median MASE |
| --- | --- | --- | --- | --- | --- |
| 0 | 0 | 0.500930 | 4.828234 | 0.438174 | 3.396360 |
| 1 | 4 | 0.236846 | 1.771442 | 0.201738 | 1.561022 |
| 2 | 8 | 0.234348 | 1.741001 | 0.199517 | 1.531597 |
| 3 | 12 | 0.233856 | 1.739426 | 0.198995 | 1.530122 |
| 4 | 16 | 0.233677 | 1.745566 | 0.199021 | 1.537206 |
| 5 | 20 | 0.233567 | 1.739124 | 0.198958 | 1.530892 |

Between checks 3 and 5, table median loss improves approximately 0.124%, and
table median MASE improves approximately 0.017%. Request median MASE is slightly
worse. At check 5 versus check 4, 50.61% of matched requests improve on loss
and 50.10% improve on MASE. Table MASE p90 is 3.8361, 3.9088 and 3.8826 at
checks 3, 4 and 5. There is no convincing broad recovery in this interval.
These are descriptive results, not confidence intervals or a proof of a floor.

The latest completed validation here is check 5, at 3,115,639 processed training
tables (58.35% of the 5,339,817 table epoch). Training was further along when
the user asked; that later percentage should not be attributed to check 5.

Outputs: [raw summaries](plateau_diagnostics_20261109/validation_medians.json)
and [interactive median progression](plateau_diagnostics_20261109/validation_medians.html).

## A reproducible precision problem

`train.py` constructs the trainable model with `.to(dtype=torch.bfloat16)`.
`performance.make_optimizer` passes those parameters directly to fused AdamW.
The saved checkpoint confirms BF16 `exp_avg` and `exp_avg_sq`; there is no
separate FP32 parameter copy. Optimizer internal arithmetic does not preserve
an update after the result has been rounded back into its BF16 parameter.

Near 1.0, the preceding BF16 representable number is 0.99609375. A proposed
decrease around 0.0001 rounds back to 1.0. Repeating this does not accumulate
the missing decreases when no higher-precision copy or residual retains them.

A CPU fused AdamW probe uses the production learning rate 0.0001, weight decay
0.01, default betas and epsilon, and a constant gradient of 0.1 for 1,000 steps:

| Initial value | BF16 result | FP32 result |
| --- | --- | --- |
| 1.0 | 1.0 | 0.899020 |
| 0.5 | 0.5 | 0.399557 |
| about 0.02 | -0.062500 | -0.079969 |

This is a rounding demonstration, not a full GPU training comparison. Parameters
near zero can still learn, and not all BF16 updates vanish. BF16 optimizer
moments introduce an additional numerical limitation, separate from parameter
rounding. The probe does not separately measure that limitation's accuracy cost.

The best checkpoint at step 322,507 and latest checkpoint inspected at step
465,162 show:

| Parameter family | Elements compared | Exactly unchanged |
| --- | --- | --- |
| Normalization weights | 116,736 (all matching weights) | 100% |
| RWKV base decay `w0` | 28,672 (all) | 99.72% |
| RWKV key scaling `k_k` | 28,672 (all) | 100% |
| Forecast lead embeddings | 3,072 (all) | 97.43% |
| Dense weight sample | 948,224 (strided) | 22.99% |

Every normalization weight inspected is exactly 1.0, its initialization value.
Representative layer-0 normalization, decay and key-scaling tensors have nonzero
first-moment entries throughout, as does the lead embedding. Thus a missing
optimizer/gradient connection is not the only possible explanation for their
lack of movement. Parameter order was matched to optimizer IDs and all 1,110
moment tensor shapes were verified before checking these values. Equality at
two snapshots does not prove a value never moved between them.

Raw evidence: [checkpoint comparison](plateau_diagnostics_20261109/checkpoint_precision_comparison.json),
[optimizer snapshot](plateau_diagnostics_20261109/optimizer_snapshot.json),
[CPU probe](plateau_diagnostics_20261109/precision_probe.json).

This changes the priority of the earlier suggestion to lower the learning rate:
smaller updates can make this rounding issue worse. Correct update accumulation
should be tested before using an LR reduction to conclude that learning is over.

Potential remedies include FP32 master weights and optimizer moments with BF16
forward/backward, or a carefully verified stochastic-rounding/compensated-update
optimizer. Keeping the relatively small normalization and RWKV control tensors
in FP32 is a useful intermediate ablation. BF16 matrix kernels may need explicit
casts; simply casting arbitrary parameters is not a complete implementation.
Full master-weight/moment storage would add several GB versus the current
BF16 setup, so physical microbatch and peak memory must be rebenchmarked on
the 12 GB GPU. No such change was made to this running job.

## Other hypotheses, ranked below the observed precision issue

1. **Data predictability and diversity.** Millions of draws from a procedural
   generator need not contain millions of distinct learnable rules. Common
   patterns can be learned early; stochastic shocks and latent regime changes
   may remain difficult regardless of model size. The prior investigation found
   nearly flat histories followed by enormous relative scale changes. This is
   evidence about particular samples, not proof that all residual error is noise.
2. **Screening is off in this corpus.**
   `data/02_processed/model/validation_selection.json` explicitly records
   `screening_skipped=true`, `reason=validation.enabled=false`, and all 5,483,212
   sources admitted. One cannot assume the current data passed Chronos
   predictability screening. Enabling a screen is a different corpus experiment;
   it should not silently alter the running job or cherry-pick validation.
3. **Learning recipe and batch composition.** The current optimizer has one
   group, LR 1e-4, betas (0.9, 0.999), weight decay 0.01, and no warmup in this
   plateau schedule. Eight tables per update is a small number of independent
   tasks despite many correlated patch labels. Storage-local shuffling preserves
   I/O efficiency but can group related generator outputs. These choices need
   controlled tests, not blind borrowing of language-model hyperparameters.
4. **Representation/readout limitations.** Covariate mixing and a small
   lead-conditioned head deserve ablations, but the aggregate plateau does not
   identify either as the cause. Per-horizon and per-family errors are needed.

The [RWKV-7 paper](https://arxiv.org/html/2503.14456v1) cautions about numerical
precision and reports a specialized training recipe, including a separate
learning-rate multiplier for base decay. Basin's single-group optimizer is
therefore not equivalent to that recipe. Its published LM settings are not
validated replacements for this forecasting task.

## What the supplied language-model PDF actually establishes

Read all 13 pages of `docs/linear_attention_models_experiment_report.pdf`.
It reports approximately **145M parameters**, not a 400M run. Page 4 identifies
**Mamba-3** as the strongest early-plateau case; RWKV-7 still improves by roughly
0.05–0.06 smoothed NLL over the final interval discussed there.

RWKV-7 beats the transformer in that experiment: test NLL 3.6405 versus 4.0250
and perplexity 38.11 versus 55.98, with less measured training time. The report
also notes one seed, recipe differences and a parameter-matched rather than
compute-matched comparison. It supports investigating training details, not a
conclusion that linear/recurrent models inherently saturate before transformers.

The PDF mentions BF16 parameters/autocast. It does not establish whether the
other project retains FP32 master weights; that must be checked in its code.
If a separate 400M experiment exists, it was not part of this supplied PDF.

## Token counts and Chinchilla

The [Chinchilla study](https://arxiv.org/abs/2203.15556) concerns compute-optimal
transformer language-model training. It is not a minimum token count before
which a training curve is forbidden to flatten. Its language-model heuristic
of about 20 tokens per parameter would mean roughly 10 billion tokens for a
500M model, but should not be treated as a prescription for Basin.

A continuous 16-observation patch is not equivalent to a wordpiece token.
Covariate patches consume model compute but are not additional independently
supervised target tokens. Adjacent causal training targets and augmentation
copies are correlated; noise and repeated generator families affect useful
information content. Distinguish temporal patches, target patches, all-variable
patches, supervised scalar observations, and padding. None establishes a direct
equivalence with language-model data scaling.

`token_inventory_status.json` currently says `not_computed` / no matching cached
inventory. The proposed 500M tokens per epoch is an estimate, not a verified
count from this run. No multi-million-table inventory scan was launched.

## Covariate attention, the head, and a transformer alternative

Basin uses temporal RWKV blocks followed by **self-attention across variables**
at each patch index. The existing term cross-attention refers to cross-variable
mixing; it is not a separate encoder-decoder attention stack. There is no code
evidence here that this operation is broken. The
[Chronos-2 paper](https://arxiv.org/html/2510.15821v1) also combines temporal and
group attention, so mixing variables through attention is not inherently an
obstacle. It additionally applies an inverse-hyperbolic-sine transformation to
standardized values to control outliers, a relevant alternative to test.

The current four-token head adds **3 x 1024 = 3,072 lead parameters**, not 15,000;
15 x 1024 = 15,360 belonged to the older 16-token configuration. The lead
vectors condition a shared nonlinear head of about **4.94M parameters**.
Their small count alone does not establish a bottleneck. The 80% next-patch
weight also means a poor distant-lead head is not, by itself, a compelling
explanation for a universal short-term plateau.

Useful architecture tests after precision is corrected: identical data with
univariate versus informative covariates; covariate dropout/gated mixing; shared
head versus separate lead projections or a richer lead-conditioned MLP; and
matched temporal transformer blocks. Do not drop trained variable mixing at
inference and call that a fair architecture ablation.

A transformer **could** improve quality; this evidence does not predict a win.
At a 2048-observation context Basin has only 128 temporal patch tokens, so the
quadratic attention term is much smaller than a comparison based on 2048 raw
observations suggests. Compare both parameter-matched and equal-wall-time
training, with identical scaling, objective, precision and validation sources.

## Muon squared

[Muon²](https://arxiv.org/html/2604.09967v1) adds adaptive second-moment
preconditioning before Muon's matrix orthogonalization. Its reported GPT/LLaMA
experiments show gains over Muon variants with fewer orthogonalization iterations;
they do not establish a gain for RWKV forecasting or over a tuned Basin AdamW.
Muon²-F factorizes the second moment to reduce storage. Fewer orthogonalization
iterations is not the same as that percentage reduction in total training time.

Treat it as an optimizer ablation after precision is fixed, with separately
tuned learning rates and measured memory/time. The ordinary
[PyTorch Muon](https://docs.pytorch.org/docs/main/generated/torch.optim.Muon.html)
is not Muon². Its guidance applies Muon to hidden 2D matrices and AdamW to
other parameter types. Important normalization and RWKV control vectors would
still need a precision-safe update path. Replacing the optimizer alone would
not establish that the observed BF16 rounding problem is solved.

## Recommended experiment order

1. Verify precision-safe updates on a small fixed learnable subset and confirm
   normalization/control parameters move. Test identical BF16-compute runs
   with and without high-precision update accumulation, same initial checkpoint,
   data order and update budget. Account for increased peak VRAM.
2. Compare validation medians, per-horizon errors, tail contributions, and a
   fixed training diagnostic subset. A memorization test must distinguish
   fitting observed labels from forecasting intrinsically random future noise.
3. Test LR/grouping, task-batch size, and robustness to generated regime changes.
   Consider monotone compression such as asinh with correctly inverted quantiles
   and original-unit metrics. This changes the training objective and requires
   a controlled run, not a display-only transformation.
4. Then compare heads, variable mixing, Muon² and a temporal transformer, changing
   one factor at a time. Preserve an untouched holdout for final evaluation.

We have not run these GPU ablations because training is ongoing. The existing
checkpoint contains learned information; this investigation does not establish
that it must be discarded or that a corrected continuation cannot help.

## Reproduction

CPU-only summaries and the tiny fused AdamW probe, with outputs outside the run:

```powershell
uv run python scripts/analyze_validation_plateau.py --output docs/model_training/plateau_diagnostics_20261109 --precision-probe
```

The script reads the latest completed validation history; a later invocation
may include additional checks. Plotly output embeds its JavaScript for offline
viewing. PDF text was extracted with pypdf installed only in `.cache/pdf_reader`;
project dependencies and lockfiles were not changed.

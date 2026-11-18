"""Write the forecast regression audit from saved matched experiments."""

import hashlib
import json
from pathlib import Path

import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "docs/audits/2026-11-02/forecast_regression"


def read(name):
    return json.loads((OUT / name).read_text())


def comparisons_plot():
    frame = pd.read_parquet(OUT / "447m_model_on_model.parquet")
    frame["extra_error"] = (
        (frame.rwkv - frame.actual).abs() - (frame.compiled_native - frame.actual).abs()
    ) / frame.context_scale
    ranked = (
        frame[frame.context_points.ge(128)]
        .groupby(["source", "origin"], sort=False)
        .extra_error.mean()
        .sort_values(ascending=False)
    )
    selected, seen = [], set()
    for source, origin in ranked.index:
        if source not in seen:
            selected.append((source, int(origin)))
            seen.add(source)
        if len(selected) == 4:
            break
    fig = make_subplots(
        rows=4,
        cols=1,
        subplot_titles=[f"{Path(s).parent.name} — origin {o}" for s, o in selected],
        vertical_spacing=0.07,
    )
    methods = dict(
        rwkv=("307M", "#d62728"),
        compiled_native=("447M", "#1f77b4"),
        chronos=("Chronos-2", "#2ca02c"),
        timestral=("Timestral", "#9467bd"),
        naive=("Naive", "#7f7f7f"),
    )
    for index, (source, origin) in enumerate(selected, 1):
        raw = pd.read_parquet(ROOT / source)
        rows = frame[frame.source.eq(source) & frame.origin.eq(origin)]
        start = max(0, origin - 128)
        fig.add_trace(
            go.Scatter(
                x=list(range(start, origin + len(rows))),
                y=raw.outcome.iloc[start : origin + len(rows)],
                name="Actual",
                line=dict(color="black"),
                showlegend=index == 1,
            ),
            row=index,
            col=1,
        )
        for name, (label, color) in methods.items():
            fig.add_trace(
                go.Scatter(
                    x=rows.time_index,
                    y=rows[name],
                    name=label,
                    line=dict(color=color, dash="dash"),
                    showlegend=index == 1,
                ),
                row=index,
                col=1,
            )
        fig.add_vline(x=origin, line_dash="dot", line_color="gray", row=index, col=1)
    fig.update_layout(
        height=1300,
        title="Diagnostic examples selected for larger 307M errors; not a random sample",
        template="plotly_white",
        hovermode="x unified",
    )
    fig.write_html(OUT / "matched_forecast_examples.html", include_plotlyjs=True)
    return selected


def outlier_plot(rows):
    cases = pd.read_parquet(OUT / "twenty_percent_worst_origins.parquet").head(2)
    fig = make_subplots(
        rows=2,
        cols=1,
        subplot_titles=[f"Nearly constant history — origin {o}" for o in cases.origin],
    )
    for index, case in enumerate(cases.itertuples(), 1):
        raw = pd.read_parquet(ROOT / case.source)
        start = max(0, case.origin - 64)
        fig.add_trace(
            go.Scatter(
                x=list(range(start, case.origin + 16)),
                y=raw.outcome.iloc[start : case.origin + 16],
                name="Actual",
                line=dict(color="black"),
                showlegend=index == 1,
            ),
            row=index,
            col=1,
        )
        for name, label, color in (
            ("model", "307M", "#d62728"),
            ("447m_model", "447M", "#1f77b4"),
        ):
            entry = next(
                r
                for r in rows
                if r["model"] == name
                and r["source"] == case.source
                and r["origin"] == case.origin
            )
            fig.add_trace(
                go.Scatter(
                    x=list(range(case.origin, case.origin + 16)),
                    y=entry["prediction"],
                    name=label,
                    line=dict(color=color, dash="dash"),
                    showlegend=index == 1,
                ),
                row=index,
                col=1,
            )
        fig.add_vline(x=case.origin, line_dash="dot", row=index, col=1)
    fig.update_layout(
        height=760,
        title="The two largest MASE contributors: inspect absolute units as well as scaled error",
        template="plotly_white",
        hovermode="x unified",
    )
    fig.write_html(OUT / "near_constant_outliers.html", include_plotlyjs=True)


def main():
    matrix = {
        (model, cohort): read(f"{model}_on_{cohort}.json")
        for model in ("model", "447m_model")
        for cohort in ("model", "447m_model")
    }
    twenty = read("twenty_percent_diagnostics.json")
    read("training_loss_diagnostics.json")
    exported = read("export_verification.json")
    intervention = read("objective_ablation_results.json")
    outliers = read("outlier_checkpoint_comparison.json")
    parity = []
    requests = 0
    for (model, cohort), result in matrix.items():
        frame = pd.read_parquet(OUT / f"{model}_on_{cohort}.parquet")
        cases = frame.groupby(["source", "origin"]).ngroups
        requests += cases * 4
        row = dict(
            model=model,
            cohort=cohort,
            cases=cases,
            points=len(frame),
            replay_vs_compiled_max=float(
                abs(frame.fast_replay - frame.compiled_native).max()
            ),
            custom_kernel_vs_compiled_max=float(
                abs(frame.fast_no_replay - frame.compiled_native).max()
            ),
        )
        if model == "447m_model" and cohort == "447m_model":
            row["reproduced_historical_max_error"] = float(
                abs(frame.compiled_native - frame.rwkv).max()
            )
        parity.append(row)
    if not all(
        r["replay_vs_compiled_max"] == 0 and r["custom_kernel_vs_compiled_max"] == 0
        for r in parity
    ):
        raise RuntimeError("Inference path predictions differ")
    if parity[-1]["reproduced_historical_max_error"] != 0:
        raise RuntimeError("Archived predictions were not reproduced")
    if any(e["mismatched_tensors"] for e in exported.values()):
        raise RuntimeError("Exported model differs from best weights")
    examples = comparisons_plot()
    outlier_plot(outliers)
    new = matrix["model", "model"]["metrics"]
    old = matrix["447m_model", "model"]["metrics"]
    lines = [
        "# Investigation: 307M forecasting accuracy regression",
        "",
        "2026-11-02. Read-only checkpoint investigation plus a disposable controlled fine-tuning experiment. Production weights, configuration and holdout outputs were not modified.",
        "",
        "## Findings",
        "",
        "There are two separate problems: a genuine learned-model accuracy regression on the matched typical-case sample, and extreme MASE sensitivity to nearly constant histories. CUDA graph replay is not the cause in the matched tests. The archived 447M model remains competitive on the same new 2% sample where the 307M model is substantially worse. However, the old model fails even more severely on some new near-constant cases outside that sample; it is not universally better on the new distribution.",
        "",
        "The change was broader than reducing 25 layers to 17. The new model also trains a 16-patch auxiliary forecasting task with 50% loss weight, uses different training data/screening, changes batch ordering, and exports its best epoch after four passes instead of seven. The existing checkpoints alone cannot assign a unique causal percentage to model size versus those training changes.",
        "",
        "The attached `previous_holdout.txt` is the **new 307M model's 20% holdout run**, not the competitive historical 447M run. Its values match the recovered 307M journal. The actual archived 447M result is in `results/447m_model/holdout_test/test_metrics.json`.",
        "",
        "## 1. Same new tables, same origins, same labels",
        "",
        "260 tables from 2026-10-31; 1,295 forecast origins; 18,851 scored observations, of which 18,707 have defined MASE. Competitor forecasts are reused from the saved run, so every row is compared against the exact same observations.",
        "",
        "| Model / execution | MASE | RMSE | MAE |",
        "|---|---:|---:|---:|",
    ]
    for label, values in (
        ("307M native, original attention", new["native"]),
        ("307M compiled, no replay", new["compiled_native"]),
        ("307M fast kernel, no replay", new["fast_no_replay"]),
        ("307M fast kernel + replay", new["fast_replay"]),
        ("447M compiled", old["compiled_native"]),
        ("Chronos-2", new["chronos"]),
        ("Timestral", new["timestral"]),
        ("Naive", new["naive"]),
    ):
        lines.append(
            f"| {label} | {values['mase']:.6f} | {values['rmse']:.4f} | {values['mae']:.4f} |"
        )
    mase_gap = 100 * (
        new["compiled_native"]["mase"] / old["compiled_native"]["mase"] - 1
    )
    rmse_gap = 100 * (
        new["compiled_native"]["rmse"] / old["compiled_native"]["rmse"] - 1
    )
    lines += [
        "",
        f"On these identical cases the new model's MASE is {mase_gap:.1f}% higher and RMSE {rmse_gap:.1f}% higher than the archived model's. This is a real regression. It persists with replay disabled and with the original RWKV forward path. Raw RMSE across the two different historical datasets is not a matched comparison.",
        "",
        "## 2. Inference and checkpoint checks",
        "",
        "Four execution variants were checked for each model and cohort: original RWKV + original variable attention, original RWKV + compiled variable attention, the forward-only recurrence without graph replay, and that recurrence with replay. Both models used an eight-profile inference cache, including cache saturation/fallback, different table lengths and covariate counts.",
        "",
        f"Across **{requests:,} forecast executions**, all scored median predictions from replay, the forward-only path without replay, and the standard compiled path match exactly. Original attention introduces small BF16 differences, but its aggregate scores show essentially the same regression. This does not prove bitwise equivalence for every possible input or every unrecorded quantile; it does directly rule out replay as the explanation for the observed MASE/RMSE gap.",
        "",
        "Both exported checkpoints match every tensor in their corresponding `best_weights.pt`: selected epochs 4 (307M) and 7 (447M). Embeddings were jointly trained; the new run records nonzero embedding gradients at all 73,050 updates. The wrong checkpoint, frozen embeddings and dropped variable attention do not explain the result.",
        "",
        "The old raw 2026-10-25 directory is absent. Historical inputs were recovered from `data/02_processed/447m_model/windows.pt`. It contains full target history and covariates only up to the reserved final 16 observations. **204 final partial-patch origins requiring unavailable covariates were excluded**, without imputing them. The comparison below therefore uses 872 recoverable origins / 13,952 observations across 216 old tables, not the complete old run.",
        "",
        "| Model on the recoverable old cases | MASE | RMSE |",
        "|---|---:|---:|",
    ]
    for label, values in (
        ("307M", matrix["model", "447m_model"]["metrics"]["compiled_native"]),
        (
            "447M current execution",
            matrix["447m_model", "447m_model"]["metrics"]["compiled_native"],
        ),
        (
            "447M saved historical predictions",
            matrix["447m_model", "447m_model"]["metrics"]["rwkv"],
        ),
    ):
        lines.append(f"| {label} | {values['mase']:.6f} | {values['rmse']:.4f} |")
    lines += [
        "",
        "The archived checkpoint reproduces its historical predictions exactly on these recovered inputs. The 307M model is worse on both old and new tables, so the new evaluation cohort alone is insufficient to explain the regression.",
        "",
        "## 3. Why the 20% headline is alarming",
        "",
        "The complete 20% result was recovered from `logs/rwkv7_holdout_test_20261102_054242_743315Z_pid3696_a6cf0231.jsonl`; it had been overwritten in the shared output directory by the subsequent 2% run. It contains 2,594 tables and 186,640 scored observations.",
        "",
        f"The largest origin contributes **{100 * twenty['top1_share']:.1f}%** of total RWKV scaled absolute error; the largest five contribute **{100 * twenty['top5_share']:.1f}%**. These are genuine scored errors, amplified by nearly zero historical naive errors, not a JSON aggregation bug.",
        "",
        "| Diagnostic exclusion only | RWKV MASE | Naive | Chronos-2 | Timestral |",
        "|---|---:|---:|---:|---:|",
    ]
    for n in (0, 1, 5, 10):
        row = twenty["sensitivity_drop_origins"][str(n)]
        lines.append(
            f"| Drop top {n} RWKV contributors | {row['rwkv']:.4f} | {row['naive']:.4f} | {row['chronos']:.4f} | {row['timestral']:.4f} |"
        )
    lines += [
        "",
        "These exclusions are sensitivity diagnostics, **not replacement scores**. The original unfiltered scores remain the valid record of the configured metric. The 2% and 20% samples are different populations; the latter includes rare pathological cases missing from the former.",
        "",
        "For `table_285516885900_019943`, origin 144 has historical naive MAE about `5.82e-7`; a future change produces absolute errors around 0.15 for all models, yielding MASE near 250,000. At origin 32 on the same table, historical naive MAE is `4.88e-7`. RWKV's absolute MAE is only 0.0603 in original units, but becomes MASE 123,720; naive MAE is approximately `1.16e-5`. This second case is a real near-constant-series forecasting weakness, not solely common unpredictable future movement.",
        "",
        "`mase_scale()` correctly uses historical lag-one naive MAE and excludes exact-zero denominators. Nonzero denominators close to zero remain extremely influential. In addition, `tokenize_next()` replaces historical standard deviations at or below `1e-5` with **1.0**. On those prefixes a modest learned normalized residual can become much larger than the tiny data variation. This normalization rule predates the new model, but the new data makes the corner case visible. Changing it only at inference would change the trained input/output representation and must be evaluated carefully.",
        "",
        "The MASE definition and historical-naive scaling follow [Forecasting: Principles and Practice, scaled errors](https://otexts.com/fpp3/toolbox.html#scaled-errors). Our near-zero sensitivity findings follow from the formula and the measured prefixes. MASE 1 refers to historical one-step naive error; it does not force a 16-step out-of-sample carry-forward baseline to score 1.",
        "",
        "### The old model also fails the new pathological cases",
        "",
        "Both original frozen checkpoints were run on the 20 worst 307M origins. At origin 32 of `table_285516885900_019943`, the 447M model has absolute MAE **1.1692** and MASE **2,398,415**, versus 307M MAE **0.0603** and MASE **123,720**. At origin 144 both are poor: 447M MASE **276,352**, 307M **284,785**. The 307M outlier predictions reproduce the saved 20% journal metrics, confirming that the same cases were reconstructed.",
        "",
        "The 20 measured origins' nonnegative scaled errors, divided by **all 185,344 MASE-eligible observations** in the 20% run, already contribute **231.763** to the 447M model's hypothetical full-20% MASE. Therefore its complete-20% MASE would be **at least 231.763**, even if it predicted every remaining point perfectly. This is a lower bound, not a measured complete 447M 20% evaluation. For comparison, those same 20 origins contribute 36.079 of the new model's measured 38.214.",
        "",
        "Consequently, reverting to the 447M checkpoint would not fix the catastrophic 20% mean MASE. The changed data distribution plus the normalization/scaling corner case is sufficient to produce an enormous headline for the old model too. The ordinary 2% comparison and the near-constant-tail problem must be reported separately; do not extrapolate the favorable old-model ranking in that small sample to the whole new holdout.",
        "",
        "## 4. Training changes and evidence",
        "",
        "| Item | Archived 447M run | New 307M run |",
        "|---|---|---|",
        "| Capacity | 447,482,144; 25 layers | 307,307,808; 17 layers |",
        "| Width / patch / history cap | 1024 / 16 / 2048 | Same |",
        "| Training data | 2026-10-25 | 2026-10-31, broader generator |",
        "| Admission | Prophet RMSE <= naive RMSE, through explicit smoke-test MAE aliases | Covariate-aware Chronos-2 Small MAE <= 1.15 × naive MAE |",
        "| Admitted / candidates | 107,828 / 150,000 (71.9%) | 129,671 / 150,000 (86.4%) |",
        "| Best epoch / update | 7 / 85,127 | 4 / 58,440 |",
        "| Supervision | Next patch only | 50% next-patch + 50% sampled longer-lead pinball |",
        "| Output training | One patch | Leads up to 16 patches (256 observations), shared head with lead offsets |",
        "| Physical microbatch / optimizer batch | Up to 4 / 8 | Up to 8 / 8, same 1536 variable-token cap |",
        "| Shuffling | All batches shuffled | Shuffled blocks up to 128 similar batches |",
        "| Training CUDA replay | Not used | Disabled for the successful run |",
        "| Learning rate reached | 1e-4 | 1e-4, never reduced |",
        "",
        "The old screening report explicitly records `actual_scoring_metric=rmse`, `metric_alias=rmse_as_mae`, `smoke_test_only=true`. It must not be described as a true Prophet MAE experiment. The new checkpoint records **Chronos-2 Small**, not Chronos-Bolt; its recorded tolerance is **15%**, not 10%. [The official model card](https://huggingface.co/autogluon/chronos-2-small) identifies this small model as approximately 28M parameters. The benchmark's Chronos-2 is the larger approximately 120M model. These are confirmed provenance differences, not configuration changes made by this investigation.",
        "",
        "The extra forecasting objective was added with the direct multi-patch head. It changes what the backbone and shared decoder learn, despite leaving the first-patch output shape unchanged. For stochastic future covariates or sharp regime changes, predicting distant patches is a different task from predicting the next patch. Its effect must be ablated rather than attributed to inference compilation.",
        "",
        "New training loss is not comparable to the old next-patch-only loss. In epoch one, the new median update loss is 0.421, but the mean is 12.45; the 99th percentile is 381.6 and maximum 26,111.9. The largest 1% of updates account for 69.9% of logged point-weighted loss, versus 11.5% in the old run. Pinball gradients with respect to predicted normalized values are bounded, so this loss concentration does **not** imply those cases contribute 69.9% of parameter gradients. It does show that the loss average is dominated by a very different, heavy-tailed target distribution.",
        "",
        "Validation monitors only the final next patch, whereas training includes early, shorter prefixes and the added long-lead objective. The new best validation pinball loss is 0.1909, versus 0.1288 on the old, different training cohort; those numbers cannot isolate capacity loss.",
        "",
        "The next-patch gap grows with context on the matched new sample: above 1024 history points, MASE is 2.017 for 307M versus 1.204 for 447M. At the first 32-point origin it is 2.273 versus 2.108. The regression is not restricted to newly introduced near-constant starts.",
        "",
        "Early stopping and scheduling currently conflict: `ReduceLROnPlateau(patience=1)` needs two bad validations to reduce LR, but early stopping terminates after the first bad epoch. Both recorded runs therefore finish without any LR reduction. The new model also has about 31% fewer optimizer updates at its selected best checkpoint, alongside its smaller capacity and changed supervision.",
        "",
        "## 5. Controlled short intervention",
        "",
        "Both arms start from the exact same best 307M state, use fresh AdamW at `1e-5`, the same seed, the same 1,024 shuffled batches from 8,192 **training-only** tables, the same clipping and physical microbatch limits, and no training/inference graph replay. The final validation patch is excluded from these training inputs. Only the auxiliary loss weight differs between arms. Production model files are never saved or replaced.",
        "",
        "| Checkpoint / auxiliary weight | New-cohort MASE | New RMSE | Old-subset MASE | Old RMSE |",
        "|---|---:|---:|---:|---:|",
    ]
    lines.append(
        f"| Original 307M | {new['compiled_native']['mase']:.6f} | {new['compiled_native']['rmse']:.4f} | {matrix['model', '447m_model']['metrics']['compiled_native']['mase']:.6f} | {matrix['model', '447m_model']['metrics']['compiled_native']['rmse']:.4f} |"
    )
    for arm in intervention:
        a, b = arm["scores"]["model"], arm["scores"]["447m_model"]
        lines.append(
            f"| Fine-tune weight {arm['weight']} | {a['mase']:.6f} | {a['rmse']:.4f} | {b['mase']:.6f} | {b['rmse']:.4f} |"
        )
    control = next(r for r in intervention if r["weight"] == 0.5)["scores"]["model"]
    no_aux = next(r for r in intervention if r["weight"] == 0.0)["scores"]["model"]
    lines += [
        "",
        f"During this continuation, removing auxiliary weighting improves new-cohort MASE by "
        f"**{100 * (1 - no_aux['mase'] / control['mase']):.2f}%** and RMSE by "
        f"**{100 * (1 - no_aux['rmse'] / control['rmse']):.2f}%** relative to the control arm. "
        "That is positive causal evidence that the 50% auxiliary objective contributes to the next-patch tradeoff. "
        "It does not quantify its from-scratch effect or close the full gap to 447M.",
    ]
    lines += [
        "",
        "This is a short continuation experiment, not a controlled from-scratch retraining or a model-size ablation. Both arms also change learning rate, reset optimizer state and use globally shuffled batches relative to the production run. Their difference isolates auxiliary weighting **during this continuation**; improvement over the original checkpoint cannot be assigned uniquely to LR, renewed optimization or shuffling. It does not establish which objective would win after full training. Test results are diagnostic and must not become an unreported hyperparameter-selection loop.",
        "",
        "## 6. Causal conclusion and next actions",
        "",
        "**Confirmed:** the large 20% headline is driven by near-zero MASE denominators and real near-constant-series forecast errors; those new pathological cases also defeat the old model, sometimes more severely. Separately, the 307M learned weights genuinely underperform on the matched 2% sample and recoverable old cases. Replay and the inference-only recurrence are exonerated for the observed scores. Simply reverting to 447M or generating more of the same data is not a demonstrated remedy for both problems.",
        "",
        "**Not identified separately:** the share caused by parameter reduction, new training distribution/screening, auxiliary supervision, changed ordering, or shorter optimization. These were changed together. No honest inference-only experiment can determine their independent training effects. The short intervention is evidence about continuation, not proof that size is harmless or that the auxiliary task alone caused the regression.",
        "",
        "Recommended next baseline: keep the inference speedups, freeze the raw tables/split/metric origins, and train a 307M **next-patch-only** model with the old supervision and ordinary batch shuffling. Give LR reduction at least one epoch to work before stopping. Compare its learning curves and the frozen 447M checkpoint on the same scorecard. Then add the long-horizon objective, new screening and blocked ordering one at a time; use multiple seeds before attributing small differences to model size. Restore/test an old-data 307M baseline from archived processed tensors if isolating the size change is the immediate priority.",
        "",
        "Keep standard MASE unmodified in the audit, but also show per-table medians, percentiles, near-constant-prefix counts, absolute errors and the largest scaled-error contributors. Do not silently discard difficult tables or epsilon-clip denominators and continue calling that the same MASE. A regularized variant would need a separate name and consistent treatment across all models. Review constant-prefix normalization on a dedicated test suite before retraining; do not switch its representation only for inference.",
        "",
        "Preserve each holdout run in a timestamped directory or snapshot its final metrics. The current shared output path overwrites the previous sample's aggregate file; the JSONL journal was essential to recover the 20% experiment.",
        "",
        "## Plots and evidence",
        "",
        "- [Matched forecast examples](matched_forecast_examples.html): four deliberately selected regressions, not a representative random sample.",
        "- [Near-constant outliers](near_constant_outliers.html): both original trained checkpoints on the largest contributors, in absolute units.",
        "- `*_on_*.json` / `.parquet`: matched metrics and every scored median prediction for all four execution variants.",
        "- `*_equivalence.parquet`: per-origin quantile differences; old-cohort experiments additionally record full quantile comparisons against the compiled path.",
        "- `recovered_20_percent_metrics.json`, `twenty_percent_diagnostics.json`, `twenty_percent_worst_origins.parquet`: recovered full 20% result and sensitivity analysis.",
        "- `objective_ablation_settings.json`, `objective_ablation_results.json`, `objective_weight_*`: exact training-only sample and disposable intervention results.",
        "- `export_verification.json`, `*_checkpoint_metadata.json`: export equality and saved run configurations.",
        "- `verification.json`: source hashes, executed forecast counts and numerical checks.",
        "",
        "Reproduction scripts: `scripts/recover_regression_inputs.py`, `scripts/investigate_forecast_regression.py --model model` (and `--model 447m_model`), `scripts/extract_holdout_regression_evidence.py`, `scripts/ablate_regression_objective.py`, `scripts/inspect_regression_outliers.py`, then `scripts/report_forecast_regression.py`. GPU scripts require the activated WSL environment; run GPU jobs sequentially.",
        "",
    ]
    (OUT / "REPORT.md").write_text("\n".join(lines), encoding="utf-8", newline="\n")
    sources = [
        "src/05_model_training/train.py",
        "src/05_model_training/fast_inference.py",
        "src/05_model_training/recurrent.py",
        "src/05_model_training/recurrent_kernel.cu",
        "src/05_model_training/multi_horizon.py",
        "src/05_model_training/epoch_training.py",
        "src/04_patch_indexing/index_patches.py",
        "src/06_holdout_test/holdout.py",
        "configs/config.json",
        "scripts/investigate_forecast_regression.py",
        "scripts/ablate_regression_objective.py",
        "scripts/report_forecast_regression.py",
    ]
    checkpoints = {}
    for name in ("model", "447m_model"):
        path = ROOT / "results" / name / "training/model.pt"
        with path.open("rb") as handle:
            digest = hashlib.file_digest(handle, "sha256").hexdigest()
        recorded = json.loads(
            (ROOT / "results" / name / "holdout_test/model_used.json").read_text()
        )["checkpoint_sha256"]
        if digest != recorded:
            raise RuntimeError(
                "Checkpoint differs from the saved holdout model identity"
            )
        checkpoints[name] = dict(
            path=str(path), sha256=digest, matches_saved_holdout_identity=True
        )
    checks = dict(
        forecast_executions=requests,
        parity=parity,
        exports_equal=True,
        production_checkpoints_modified=False,
        checkpoints=checkpoints,
        outlier_contribution_lower_bounds={
            name: sum(r["mase"] * r["points"] for r in outliers if r["model"] == name)
            / twenty["pooled"]["rwkv"]["mase_observations"]
            for name in ("model", "447m_model")
        },
        source_sha256={
            s: hashlib.sha256((ROOT / s).read_bytes()).hexdigest() for s in sources
        },
        selected_examples=examples,
    )
    (OUT / "verification.json").write_text(
        json.dumps(checks, indent=2) + "\n", newline="\n"
    )
    print("Report complete; verified", requests, "forecast executions")


if __name__ == "__main__":
    main()

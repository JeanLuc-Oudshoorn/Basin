"""Summarize the completed, isolated November 8 training execution experiments."""

import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DIRECTORY = ROOT / "docs/time_benchmark/training_tuning/20261108T180449_072290Z"


def read(path):
    return json.loads(path.read_text(encoding="utf-8"))


def main():
    real = {
        name: read(DIRECTORY / "real" / f"{name}.json")
        for name in (
            "real_current",
            "real_current_repeat",
            "real_less_recomputation",
            "real_larger_budget",
            "real_full_microbatch",
            "real_margin_budget",
        )
    }
    stress = read(DIRECTORY / "margin_all_shapes_stress.json")
    gradients = read(DIRECTORY / "gradient_margin.json")
    if not stress.get("completed") or not gradients.get("passed"):
        raise RuntimeError("Selected-policy verification is incomplete")
    if any(row["oom_retries"] for row in stress["updates"]):
        raise RuntimeError(
            "Selected policy still needs OOM recovery during the stress test"
        )
    baseline = (
        real["real_current"]["windows_per_second"]
        + real["real_current_repeat"]["windows_per_second"]
    ) / 2
    selected = real["real_margin_budget"]["windows_per_second"]
    decision = {
        "parameters": stress["parameters"],
        "reference_tables_per_second": baseline,
        "selected_tables_per_second": selected,
        "throughput_gain_percent": 100 * (selected / baseline - 1),
        "selection": {
            key: stress["training"][key]
            for key in ("microbatch_token_limit", "checkpoint_token_threshold")
        },
        "validation_interval_seconds": stress["training"][
            "validation_interval_seconds"
        ],
        "checkpoint_interval_seconds": stress["training"][
            "checkpoint_interval_seconds"
        ],
        "stress_updates": len(stress["updates"]),
        "stress_case_count": stress["case_count"],
        "stress_peak_allocated_gib": max(
            r["peak_allocated_gib"] for r in stress["updates"]
        ),
        "stress_peak_reserved_gib": max(
            r["peak_reserved_gib"] for r in stress["updates"]
        ),
        "stress_oom_retries": sum(r["oom_retries"] for r in stress["updates"]),
        "gradient_comparison_passed": gradients["passed"],
    }
    (DIRECTORY / "decision.json").write_text(
        json.dumps(decision, indent=2) + "\n", encoding="utf-8", newline="\n"
    )
    rows = "\n".join(
        f"| {name.removeprefix('real_')} | {r['windows_per_second']:.2f} | {r['peak_gib']:.2f} |"
        for name, r in real.items()
    )
    content = f"""# Training throughput tuning — November 8, 2026

The selected default is a **3072 variable-patch-token microbatch limit** with
**1792 activation-checkpoint budget**. On 32 recorded training batches,
throughput improved from a reference average of **{baseline:.2f} to {selected:.2f}
tables/second ({decision["throughput_gain_percent"]:.1f}%)**. This is a measured bounded-sample gain,
not a 2× result or a measured complete-epoch speedup.

## Method and results

All experiments use the current **500,056,352-parameter, 28-layer, width-1024**
model on the RTX 5070, BF16, compiled variate attention, AdamW and the existing
80/16/3/1 loss. The model architecture, seed, data and logical optimizer groups
are unchanged. GPU experiments run sequentially while production training is
stopped. No production model/checkpoint is written. Inference replay and the
default-disabled training replay are unchanged.

The initial five-profile comparison repeats its reference and exercises six
synthetic anchors (256–2048 observations, 1–6 variables), 180 updates/profile,
discarding three warm calls per case. It indicates approximately 10–15% gains
for some larger activation/microbatch policies. Raw [anchor results](training_tuning/20261108T180449_072290Z/summary.json)
are not weighted by the actual corpus distribution.

The stronger practical check reuses **252 distinct tables in 32 actual optimizer
batches**, sampled at equal spacing from a bounded tail of the stopped run's
journal. Their original batch membership/order and chronology are preserved,
including partial batches. Raw processed records are recovered from verified
shard locators, matching IDs and sources; the 2.5 MB tensor sample stays on E:
under `.cache/training_tuning/live_sample.pt`. There is no new train/validation
split and no use of held-out tables. Each run warms a complete sweep, measures
a second, and synchronizes CUDA around update timings. Timings include feature
preparation, forward/backward, clipping, finite checks and optimizer execution;
startup/compilation, validation and checkpoint I/O are excluded.

| Recorded-data profile | Tables/second | Peak live GiB |
| --- | ---: | ---: |
{rows}

References differ by about 3%, so the selected gain ranges from roughly 6.5%
against the faster reference to 9.8% against the slower one. We report their
average, rather than the most favorable comparison. The sample is bounded and
does not establish an exact new epoch ETA. At a comparable workload mix, an
approximately 40-hour training pass would remain on the order of 37 hours;
periodic validation/checkpoint pauses are additional.

## Why this policy

The old 1536-token/1024-budget policy saves VRAM by splitting more batches and
recomputing more layers. The selected token cap permits, for example, four
2048-observation/six-variable tables per physical microbatch instead of two.
Eight tables still contribute to each optimizer update. The larger checkpoint
budget retains more intermediates; lowering it slightly from the exploratory
2048 setting provides memory margin without surrendering most of the gain.

The most aggressive full-eight-table setting looked better on synthetic anchors
but improved the recorded sample by only about 4%. The 3072/2048 setting reached
44.45 tables/s, but a subsequent 48-shape sweep needed an OOM retry at 1792
observations/six variables (update 90). It completed safely via existing recovery,
but was **not selected**. Retaining a conservative margin is preferable to a
small headline advantage that depends on recovery. The existing OOM recovery
remains enabled as a safeguard for external memory pressure/new workloads.

The [profile](training_tuning/20261108T180449_072290Z/real/real_current_profile.txt)
points to dense matrix multiplications and many small kernel launches as major
costs. Its CPU timings are instrumentation-affected and its operator/kernel
attributions overlap; do not sum table rows or interpret them as an unprofiled
CPU/GPU idle fraction. Post-update `allocated≈3.8 GiB` is not the activation
peak. Bigger logical batches or new fused training kernels could offer other
gains, but would need separate convergence/backend verification and are not
silently introduced here.

## Safety checks

- **{len(stress["updates"])} consecutive selected-policy updates**, spanning all 48 kernel
  buckets (8 padded patch lengths × 6 variable counts), completed with **zero
  OOM retries and finite loss/gradients**. Peak live memory was
  **{decision["stress_peak_allocated_gib"]:.2f} GiB**, peak reservation **{decision["stress_peak_reserved_gib"]:.2f} GiB**.
  The allocator stayed within its approximately 10.2 GiB budget selected from
  an 11 GiB request after desktop/headroom checks. Reservation includes reusable
  cache. These checks cannot guarantee behavior under arbitrary future external
  GPU allocations or multi-day training; recovery remains available.
- At identical initial weights, full-model GPU gradient comparisons cover a
  recorded batch and maximum 2048-observation/five-covariate inputs. Both pass;
  see [loss and gradient errors](training_tuning/20261108T180449_072290Z/gradient_margin.json).
  Changing physical execution can change BF16 rounding/accumulation order;
  bitwise equivalence is not claimed. The objective and optimizer grouping stay
  the same. No production weights were modified by these gradient comparisons.
- Full CPU suite: **328 tests, no failures, 13 skips** (CUDA-only/Windows symlink
  limitations). Resume tests verify cursor, RNG, optimizer and validation
  history, including a cadence change after an interrupted validation. Focused
  checks are rerun after updating the default runtime settings.

## Cadence, resume and default ETA

The user's **four-hour validation interval** (14400 seconds) is preserved.
**Hourly checkpoints** (3600 seconds) are unchanged. Resume now permits cadence
changes without rejecting the checkpoint. If the interval changes, a saved
pending validation is recalculated from cumulative active training time and the
last completed validation; unchanged-cadence interrupted checks still rerun.

The actual production checkpoint was inspected read-only: **epoch 1, step 17270,
next batch 17270, active training 3600.10 seconds**, with a one-hour validation
pending. Under four-hour validation, it resumes training from that cursor and
waits for the new cumulative four-hour deadline. Baseline validation is retained.
Scheduler/early-stopping patience are still counts of validation checks, so
their time span increases with the less frequent cadence.

Default INFO progress every 100 updates includes completion percentage and
**epoch training ETA**, followed by checkpoint/validation countdowns and memory
statistics. This estimate excludes future validation/checkpoint pauses and can
fluctuate with table shapes. Preparation-wait/model-update timing diagnostics
are also available by default. No progress watcher is required.

Restart normally with `uv run python .\\src\\05_model_training\\train.py`.
`resume="auto"` restores the existing weights, optimizer and cursor. Runtime
execution/cadence changes do not require transform/embed to run again.

## Reproduction and raw evidence

- [Selected decision](training_tuning/20261108T180449_072290Z/decision.json)
- [All-shape selected-policy stress](training_tuning/20261108T180449_072290Z/margin_all_shapes_stress.json)
- [Recorded-data runs](training_tuning/20261108T180449_072290Z/real/)
- [Rejected 2048-budget shape stress](training_tuning/20261108T180449_072290Z/all_shapes_stress.json)
- [Full-eight-table experiments](training_tuning/20261108T181815_518577Z/summary.json)

Scripts: `prepare_live_training_sample.py`, `tune_training_throughput.py`,
`benchmark_training_throughput.py --preserve-sample-batches`,
`verify_training_memory.py --all-shapes`, `verify_training_tuning_gradients.py`.
All benchmark configs, logs and raw results stay under the project on E:.
The report generator refuses incomplete or retry-dependent selected-policy
stress results. Historical investigations remain available under `docs/audits`.
"""
    (ROOT / "docs/time_benchmark/TRAINING_THROUGHPUT_2026-11-08.md").write_text(
        content, encoding="utf-8", newline="\n"
    )
    print(json.dumps(decision, indent=2))


if __name__ == "__main__":
    main()

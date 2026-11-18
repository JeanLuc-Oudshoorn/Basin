"""Render the saved inference experiment, retaining incomplete-run provenance."""

import json
import statistics
from pathlib import Path

import pandas as pd
import plotly.express as px

ROOT = Path(__file__).resolve().parents[1]
DIRECTORY = ROOT / "docs/time_benchmark/fast_inference"


def main():
    rows = json.loads((DIRECTORY / "experiment.json").read_text())
    summaries = []
    for row in rows:
        if "times_ms" not in row:
            continue
        summaries.append(
            {
                **{
                    k: row[k]
                    for k in (
                        "context",
                        "horizon",
                        "mode",
                        "cold_seconds",
                        "max_absolute_error",
                    )
                },
                "median_ms": statistics.median(row["times_ms"]),
                "min_ms": min(row["times_ms"]),
                "max_ms": max(row["times_ms"]),
                "repeats": len(row["times_ms"]),
            }
        )
    frame = pd.DataFrame(summaries)
    frame.to_csv(DIRECTORY / "summary.csv", index=False)
    chart = px.bar(
        frame,
        x="horizon",
        y="median_ms",
        color="mode",
        facet_col="context",
        barmode="group",
        log_y=True,
        title="Basin 307M: warm request latency (direct leads untrained)",
    )
    chart.write_html(DIRECTORY / "latency.html", include_plotlyjs="cdn")
    table = []
    for (context, horizon), group in frame.groupby(["context", "horizon"]):
        values = group.set_index("mode").median_ms.to_dict()
        if "direct_graphs" not in values:
            continue
        table.append(
            f"| {context} | {horizon} | {values['reference']:.2f} | "
            f"{values['graphs']:.2f} | {values['direct_graphs']:.2f} | "
            f"{values['reference'] / values['direct_graphs']:.1f}x |"
        )
    text = (
        """# Basin 307M inference experiments

2026-10-31. The default configuration now uses **307,307,808 parameters**:
17 RWKV-7 layers, width 1024, 16-point input patches and 16 forecast leads.
The direct head returns up to 256 timesteps from one encoded context.

## Saved measurements

Median end-to-end latency in milliseconds. One target and three historical
covariates; one request at a time; RTX 5070; CUDA; BF16; four CPU threads.
Every saved row contains 20 measured requests after a first request/warmup.
Measurements include CPU input preparation, transfers and CPU quantile results,
with CUDA synchronization. Model loading is excluded. Capture/first-use time is
reported separately as `cold_seconds` in the raw results.

| Context | Forecast steps | Previous patch rollout | Graph patch rollout | Direct + graphs | Speedup vs previous |
|---:|---:|---:|---:|---:|---:|
"""
        + "\n".join(table)
        + """

The direct head's added lead vectors are **untrained**. These results establish
execution speed and output shape, not forecast accuracy. The experiment uses a
17-layer view of the trained 25-layer checkpoint, retaining the original width
and weights. A newly trained 307M model is needed for an accuracy comparison.

Graph replay had zero maximum absolute error against its corresponding eager
reference in all saved graph rows, including multi-patch rollouts. Direct-graph
results are compared against eager direct decoding, not against autoregressive
forecasts: direct decoding is a new predictive architecture.

## What was implemented

- A shared projection conditioned on learned forecast-lead vectors. It adds only
  15,360 parameters and removes repeated backbone calls for horizons up to 256.
- The original next-patch training loss plus sampled longer-lead training targets,
  with missing/padded/out-of-range labels masked and the final validation patch
  excluded by the existing train/validation split.
- CUDA graph replay in holdout and forecasting, with a bounded eight-shape cache.
  Once full, uncached shapes run eagerly, avoiding repeated capture overhead.
- A current-stream-aware inference recurrence; the original training kernel and
  its backward implementation are retained.
- An experimental recurrent rollout with per-request states and resets when the
  2048-point window slides. It is not the default.

## Recurrent experiment and limitations

Recurrent BF16 execution differed from full-context execution: the largest saved
absolute forecast difference was about 0.085 on these synthetic inputs. Changing
GEMM shapes and prefix execution can alter rounding. This option needs accuracy
evaluation before promotion. The recorded later recurrent cases also encountered
the two-entry experimental graph-cache limit and used eager fallback. They are
not a clean measurement of fully warmed recurrent decoding. The experiment script
now clears caches between modes to prevent that confound on future runs.

Recurrent reuse preserves the prefix only while it fits in the context window.
At a window slide the normalization statistics and time features change, requiring
a fresh prefill. Direct decoding avoids this problem within each 256-point block.

This is a focused latency experiment, not a new simultaneous comparison with
Chronos/Timestral. The earlier comparator benchmark remains in the parent folder.
Results vary with variable count, hardware, precision, cold compilation and shape.

## Completion and verification record

The last benchmark rerun was interrupted during the final 2048-context / 256-step
case. Only completed rows in `experiment.json` are included above; missing rows
are not estimated. Earlier runs also exercised that case, but the table is based
strictly on the retained raw measurements. The planned three-update 307M GPU
training smoke did not produce a completion artifact and is not claimed as passed.

The CPU suite completed with 193 tests, eight skipped, before the final additional
GPU parity test. It included direct-head gradient checks, one-backbone-pass output,
future-label isolation, missing-label masking and legacy configuration defaults.
See `verification.json` for the final targeted checks, including GPU availability.

## Files and use

- [Raw requests](experiment.json), [summary CSV](summary.csv),
  [interactive latency plot](latency.html), [environment](environment.json).
- [Architecture, training objective and migration](../../model_training/FAST_INFERENCE_307M.md).
- `configs/config.json`: new defaults and separate `model` paths.
- `configs/legacy_447m.json`: prior settings and checkpoint paths.
- `scripts/experiment_fast_inference.py`: reproduce all modes; caps CUDA memory at
  4 GiB and compilation at one worker, writes results after each completed case,
  and saves detailed run logs under `logs`.

To train the new architecture, run `transform.py`, `index_patches.py`, then `train.py`
with the default config. Strict artifact signatures require rebuilding the new
processed data and patch index. Existing 447M weights cannot be resumed into 17 layers.
After training, normal holdout and forecasting stages use the new model and graph
settings. Horizons above 256 continue in blocks, with a final-window context cap.
"""
    )
    (DIRECTORY / "REPORT.md").write_text(text, encoding="utf-8", newline="\n")


if __name__ == "__main__":
    main()

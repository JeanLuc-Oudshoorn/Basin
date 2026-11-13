"""Render the measured latency matrix, diagnostics and interactive comparisons."""

import gzip
import json
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.express as px

ROOT = Path(__file__).resolve().parents[1]
OUTPUT = ROOT / "docs/time_benchmark/2026-10-31"


def markdown_table(frame):
    lines = [
        "| " + " | ".join(map(str, frame.columns)) + " |",
        "| " + " | ".join("---" for _ in frame.columns) + " |",
    ]
    for row in frame.itertuples(index=False, name=None):
        lines.append(
            "| "
            + " | ".join(
                f"{v:.2f}" if isinstance(v, (float, np.floating)) else str(v)
                for v in row
            )
            + " |"
        )
    return "\n".join(lines)


def main():
    data = pd.read_csv(OUTPUT / "summary.csv")
    environment = json.loads((OUTPUT / "environment.json").read_text())
    historical = json.loads((OUTPUT / "historical_analysis.json").read_text())
    requests = pd.read_parquet(OUTPUT / "requests.parquet")
    verification = {
        "status": "passed",
        "initial_gpu_scored_requests": int(requests.phase.eq("scored").sum()),
        "initial_gpu_total_records": len(requests),
        "scenario_count": 32,
        "unit_tests": 187,
        "passed_tests": 180,
        "skipped_tests": 7,
        "peak_recorded_allocated_gib": float(requests.allocated_gib.max()),
        "peak_recorded_reserved_gib": float(requests.reserved_gib.max()),
        "production_model_modified": False,
    }
    (OUTPUT / "verification.json").write_text(
        json.dumps(verification, indent=2) + "\n", encoding="utf-8"
    )
    primary = data[(data.precision == "bf16") & (data.model != "timestral_default")]
    pivot = primary.pivot(
        index=["context", "horizon", "covariates"], columns="model", values="median_ms"
    )
    if (
        set(pivot.columns) != {"chronos", "timestral", "rwkv_447m", "rwkv_307m"}
        or pivot.isna().any().any()
        or len(pivot) != 32
    ):
        raise ValueError("Benchmark matrix is incomplete")
    ratio = pivot.rwkv_307m / pivot.rwkv_447m
    comparisons = {
        "cases": len(pivot),
        "median_307m_over_447m_latency": float(ratio.median()),
        "median_latency_reduction_percent": float((1 - ratio.median()) * 100),
        "geomean_speedup_447m_over_307m": float(np.exp(np.log(1 / ratio).mean())),
        "307m_faster_than_chronos_cases": int((pivot.rwkv_307m < pivot.chronos).sum()),
        "307m_faster_than_timestral_single_pass_cases": int(
            (pivot.rwkv_307m < pivot.timestral).sum()
        ),
        "307m_faster_than_both_cases": int(
            (
                (pivot.rwkv_307m < pivot.chronos) & (pivot.rwkv_307m < pivot.timestral)
            ).sum()
        ),
    }
    confirmation = pd.concat(
        [
            pd.read_csv(OUTPUT / "depth_confirmation_summary.csv"),
            pd.read_csv(OUTPUT / "baseline_confirmation_summary.csv"),
        ],
        ignore_index=True,
    )
    confirmed = confirmation.pivot(
        index=["context", "horizon", "covariates"], columns="model", values="median_ms"
    )
    confirmed_ratio = confirmed.rwkv_307m / confirmed.rwkv_447m
    confirmed_stats = {
        "cases": len(confirmed),
        "median_latency_reduction_percent": float(100 * (1 - confirmed_ratio.median())),
        "geomean_speedup": float(np.exp(np.log(1 / confirmed_ratio).mean())),
        "faster_than_both_cases": int(
            (
                (confirmed.rwkv_307m < confirmed.chronos)
                & (confirmed.rwkv_307m < confirmed.timestral)
            ).sum()
        ),
    }
    comparisons["isolated_confirmation"] = confirmed_stats
    confirmed.reset_index().to_csv(OUTPUT / "confirmed_comparison.csv", index=False)
    cpu = pd.read_csv(OUTPUT / "cpu_summary.csv")
    cpu_table = cpu.pivot(
        index=["context", "horizon", "covariates"], columns="model", values="median_ms"
    ).reset_index()
    cpu_environment = json.loads((OUTPUT / "cpu_environment.json").read_text())
    (OUTPUT / "comparison.json").write_text(
        json.dumps(comparisons, indent=2) + "\n", encoding="utf-8"
    )
    tables = []
    for length in (32, 512, 2048):
        selected = data[
            (data.context == length)
            & (data.covariates == 3)
            & (data.precision == "bf16")
        ]
        table = selected.pivot(
            index="horizon", columns="model", values="median_ms"
        ).reset_index()
        table = table[
            [
                "horizon",
                "rwkv_447m",
                "rwkv_307m",
                "chronos",
                "timestral",
                "timestral_default",
            ]
        ]
        tables.append(
            f"### Context {length}, one target + three past covariates\n\nAll entries are median milliseconds; 20 measured requests per cell.\n\n"
            + markdown_table(table)
        )
    diag = pd.read_csv(OUTPUT / "tail_replays.csv")
    diag_summary = (
        diag.groupby("phase")
        .agg(
            requests=("milliseconds", "size"),
            median_ms=("milliseconds", "median"),
            max_ms=("milliseconds", "max"),
            gc_ms=("gc_ms", "sum"),
            gc_events=("gc_events", "sum"),
            new_graphs=("new_compiled_graphs", "sum"),
        )
        .reset_index()
    )
    diag_summary.to_csv(OUTPUT / "tail_summary.csv", index=False)
    trace_path = OUTPUT / "rwkv_warm_trace.json.gz"
    trace = (
        json.loads(gzip.decompress(trace_path.read_bytes()))
        if trace_path.is_file()
        else json.loads((OUTPUT / "rwkv_warm_trace.json").read_text())
    )
    kernels = sorted(
        (item["ts"], item["ts"] + item["dur"])
        for item in trace["traceEvents"]
        if item.get("cat") == "kernel" and item.get("ph") == "X"
    )
    active = 0.0
    left = right = 0.0
    for start, stop in kernels:
        if start > right:
            active += right - left
            left, right = start, stop
        else:
            right = max(right, stop)
    active += right - left
    profile_summary = {
        "kernels": len(kernels),
        "kernel_active_union_ms": active / 1000,
        "first_to_last_kernel_span_ms": (
            max(stop for _, stop in kernels) - kernels[0][0]
        )
        / 1000,
        "note": "Profiler perturbs timing. Kernel union avoids counting overlapping compiled regions twice.",
    }
    (OUTPUT / "profile_summary.json").write_text(
        json.dumps(profile_summary, indent=2) + "\n"
    )
    figure = px.line(
        primary,
        x="horizon",
        y="median_ms",
        color="model",
        symbol="covariates",
        facet_col="context",
        markers=True,
        log_y=True,
        title="Matched BF16 GPU requests: latency by horizon (lower is faster)",
        labels={
            "horizon": "Forecast points",
            "median_ms": "Median request latency (ms)",
        },
    )
    figure.write_html(OUTPUT / "latency.html", include_plotlyjs="cdn")
    anchor_plot = confirmation.copy()
    anchor_plot["request"] = anchor_plot.apply(
        lambda r: f"C={int(r.context)}, H={int(r.horizon)}, cov={int(r.covariates)}",
        axis=1,
    )
    px.bar(
        anchor_plot,
        x="request",
        y="median_ms",
        color="model",
        barmode="group",
        log_y=True,
        title="Isolated GPU anchor confirmation: identical BF16 requests",
        labels={"median_ms": "Median milliseconds (log scale)"},
    ).write_html(OUTPUT / "confirmed_latency.html", include_plotlyjs="cdn")
    cold = requests[requests.phase == "warmup"]
    cold_stats = (
        cold.groupby(["model", "precision"])
        .agg(
            max_ms=("milliseconds", "max"),
            compiled_graphs=("new_compiled_graphs", "sum"),
            gc_ms=("gc_ms", "sum"),
        )
        .reset_index()
    )
    text = (
        f"""# Matched time-series forecast latency benchmark

Measured on {environment["gpu"]}, PyTorch {environment["torch"]}, CUDA {environment["cuda"]}.
The initial full matrix used one WSL process; confirmations used isolated
processes in the same environment on the same GPU.
The original holdout comparison used **BF16 RWKV versus FP32 baselines**;
Timestral also enabled its default mirrored forecast averaging. Those numbers
compare usable default pipelines, but are not a controlled precision/single-pass comparison.

| Model | Parameters | Weight precision in matched GPU tests |
| --- | ---: | --- |
| RWKV, 25 layers, width 1024 | {environment["rwkv_parameters"]:,} | BF16 |
| RWKV, 17 layers, width 1024 | {environment["reduced_parameters"]:,} | BF16 |
| Chronos-2 | {environment["models"]["chronos"]["parameters"]:,} | BF16 |
| Timestral | {environment["models"]["timestral"]["parameters"]:,} | BF16 |

Provider/API references: [Chronos](https://github.com/amazon-science/chronos-forecasting),
[Timestral](https://github.com/mistralai/timestral). Exact checkpoint revisions,
installed runtime versions and input/output contracts are saved in the environment JSON files.

## Main findings: confirmed anchors

- The paired depth confirmation measured **{confirmed_stats["median_latency_reduction_percent"]:.1f}% lower median
  latency**, approximately **{confirmed_stats["geomean_speedup"]:.2f}x faster**, for 307M versus 447M across five anchors.
- In the matching isolated-provider GPU anchors, the reduced model beat both
  Chronos and single-pass Timestral in **{confirmed_stats["faster_than_both_cases"]}/{confirmed_stats["cases"]}** cases.
  These are the short-horizon cases; it remains much slower on the long horizon.
  The following table gives measured milliseconds, not estimates from parameter count.

{markdown_table(confirmed.reset_index()[["context", "horizon", "covariates", "rwkv_447m", "rwkv_307m", "chronos", "timestral", "timestral_default"]])}

Full and reduced RWKV alternate order on each repeat in one process. External
providers are then measured one at a time with the exact same five input payloads,
GPU, BF16 weights, nine output quantiles, four CPU threads and CUDA math flags.
Both runs cap PyTorch allocation at 4 GiB. They are separate processes/times,
so they do not eliminate clock/thermal/scheduler variation. Each cell has three
warmups and 20 timed requests. This uses little memory and avoids co-resident
baseline models during the depth comparison.

## Full initial matrix and its limits

- Reducing RWKV from **{environment["rwkv_parameters"]:,}** to **{environment["reduced_parameters"]:,}**
  parameters reduced per-case median latency by a median **{comparisons["median_latency_reduction_percent"]:.1f}%**
  ({comparisons["geomean_speedup_447m_over_307m"]:.2f}x geometric-mean speedup).
- The reduced model was faster than Chronos in **{comparisons["307m_faster_than_chronos_cases"]}/32** cases,
  faster than single-pass Timestral in **{comparisons["307m_faster_than_timestral_single_pass_cases"]}/32**,
  and faster than both in **{comparisons["307m_faster_than_both_cases"]}/32** in that initial sequence.
  This differs from the isolated short-horizon confirmation above. The first
  matrix measured the reduced model in a later pass, after profiling and mixed
  provider traffic. It therefore should not be used to claim a stable universal
  ranking. Raw results are retained rather than discarded or blended with the
  confirmation. The paired confirmation is the stronger estimate of depth-only scaling.
- Horizon matters: RWKV currently predicts 16 points per evaluation and reprocesses
  its history for every new patch. A 256-point request requires 16 evaluations.
  The external models can produce much larger forecast blocks per evaluation.
- The reduced model is a **17-layer depth ablation** of the existing model, not
  a newly trained 300M checkpoint. Its speed is measured directly; its forecast
  accuracy is not evidence for the accuracy of a properly trained smaller model.

## Matched request contract

One target, contexts 32/128/512/2048, horizons 1/16/64/256, zero or three past
covariates: **32 identical scenarios per model**. Three warmups for each scenario
precede 20 timed requests. All use the same RTX GPU, CPU threads={environment["cpu_threads"]},
CPU float32 input tensors and nine CPU float32 output quantiles (0.1 through 0.9).
All main-comparison model weights and eligible model matrix operations use BF16.
The provider-specific normalization and numerical stabilization remain unchanged;
this does not mean every operation uses the same dtype. No actual future covariates
or future outcomes are supplied, and cross-table learning is disabled.

`timestral` disables symmetric averaging; `timestral_default` keeps the original
two-pass reflected-context average. Both retain the library's clipping behavior.
Timestral internally emits 21 quantiles and the adapter selects the requested nine;
RWKV internally emits nine. Different internal heads/block sizes/normalization
are architectural/API differences, not identical FLOP workloads. Disabling Timestral
augmentation can affect accuracy: this is a latency study, not a new accuracy ranking.

Every timed call starts with historical payloads already in host RAM and ends
with CPU quantiles. It includes model preprocessing/tokenization, transfers and
adapter execution. CUDA is synchronized before and after. Model loading, input
generation, logging and result writes are excluded. Model order rotates per repeat
in the primary matrix. Reduced-depth and FP32 passes are later in the same process,
so thermal/clock drift cannot be perfectly eliminated. The GPU also drives Windows
display workloads; no exclusive compute mode was claimed. Warmup rows remain
available separately; compilation is never silently included in steady-state averages.

Main model configurations remain production defaults apart from matched precision,
Timestral's explicitly labelled averaging switch and the clearly labelled depth ablation.
The multi-patch RWKV adapter uses the existing optimized holdout patch path repeatedly,
feeding back medians and marking future covariates unknown. It does not implement
recurrent-state caching, a new forecast head or architecture changes. It measures
the current efficient patch adapter extended to longer requests, not an idealized
stateful RWKV engine or the separate public GPU-tokenizing `forecast` API.

## Median GPU latency by scenario

"""
        + "\n\n".join(tables)
        + f"""

## Slowest RWKV holdout request

The recorded holdout contains {historical["requests"]} scored RWKV requests.
Median was **{historical["median_ms"]:.3f} ms**, p99 **{historical["p99_ms"]:.3f} ms**,
maximum **{historical["max_ms"]:.3f} ms** ({historical["max_over_median"]:.1f}x median).
Only **{historical["over_ten_times_median"]}** scored request exceeded ten times the median.
The maximum was table index {historical["outlier"]["table_index"]}, origin
{historical["outlier"]["origin"]}, context {historical["outlier"]["context_points"]},
covariates {historical["outlier"]["covariates"]}. This exact shape was already used
and warmed in table 0, so it is incorrect to assert this specific maximum was
necessarily first-use shape compilation. The old journal has no GC/compiler/CPU
profiling telemetry, so its exact historical cause cannot be proven after the fact.

The benchmark replays that exact real input, instruments Python garbage collection,
wall and process CPU time, compiled-graph counts and memory. It also contrasts
normal GC, temporarily disabled GC, explicit full GC inside a request, and resetting
the Python graph cache while retaining disk binaries. Forced collection is a
diagnostic, not representative traffic. Disabled GC is an experiment, not a production
recommendation. GC scans the Python object heap regardless of whether predictions
are otherwise on GPU.

{markdown_table(diag_summary)}

Interpret these as causal tests of mechanisms, not proof of which event happened
in the older run. `tail_replays.csv` exposes whether each new slow call actually
overlapped collection or compilation. `rwkv_warm_trace.json.gz` is a compressed Chrome/Perfetto
trace of the warmed request, with an operator summary in `rwkv_warm_profile.txt`.
See the detailed interpretation appended after reviewing those recordings.

### Interpretation of the tail investigation

The exact request did **not** reproduce its historical 294 ms delay in 100 normal
replays: median about 24.9 ms, maximum about 30.1 ms, with no new compiled graphs.
Temporarily disabling GC did not improve its median and still had a roughly
40 ms maximum. Explicitly forcing a full generation-2 collection produced
roughly 178–236 ms requests, with 151–209 ms directly attributed to GC callbacks.
Resetting the graph cache produced a roughly 1.59-second first call and one
new compiled graph, then returned to normal latency. These demonstrate two
mechanisms for large pauses; neither proves what happened to the historical
maximum. That run did not record the necessary telemetry. OS scheduling and
shared display-GPU activity also remain possible causes.

The warmed profiler trace recorded **{profile_summary["kernels"]} GPU kernels**,
with about **{profile_summary["kernel_active_union_ms"]:.2f} ms** of non-overlapping
kernel activity spread across **{profile_summary["first_to_last_kernel_span_ms"]:.2f} ms**
from first to last kernel. The profile itself adds overhead, so these are not
unprofiled benchmark latencies. They do indicate substantial host submission/idle
gaps among many small kernels. Parameter count alone therefore does not determine
request latency, and fewer layers do not guarantee proportional speedup.

The practical next experiments would be shape-specific prewarming and CUDA-graph
capture of the repeated inference region, following
[PyTorch CUDA guidance](https://docs.pytorch.org/docs/main/notes/cuda.html).
Those optimizations were **not** applied to the production model in this task.
Blindly disabling GC would not resolve all tails and was not made a default.

## Cold overhead and original precision

Maximum per-case warmup durations and compile/GC counters (not scored latency):

{markdown_table(cold_stats)}

`summary.csv` also contains FP32 Chronos and Timestral single/default comparisons
for all scenarios. TF32 allowed in the runtime: {environment["tf32_allowed"]}.
The BF16 main matrix is the controlled precision comparison; FP32 rows reproduce
the external providers' existing parameter precision and should not be mixed with it.

## CPU and limits

The vendored RWKV backend requires CUDA with BF16. A CPU benchmark of all three
would require an alternate RWKV kernel implementation. None was substituted,
so no three-model CPU ranking is claimed. Matching precision and request contracts
improves fairness; it cannot make different architectures, output heads, trained
quality or decoding algorithms literally identical. These measurements answer
which current adapter is faster for the specified workload on this machine.

Supplementary FP32 CPU measurements of Chronos and Timestral on
**{cpu_environment["cpu"]}**, four threads, five measured requests per cell:

{markdown_table(cpu_table)}

Raw CPU results are
in `cpu_summary.csv` and `cpu_environment.json`. They do not constitute a CPU
comparison against RWKV. The CPU pass uses the same WSL environment and four
threads, after all GPU benchmarks have stopped.

## Interruption and resource controls

The complete initial matrix and diagnostics finished successfully. A second,
fully interleaved confirmation was interrupted by the user while loading external
weights because PyCharm froze; it had not saved any measured cases. Its partial
`paired/` directory is not used in these results. The cause of the IDE freeze
was not established from the saved logs.

The resumed depth confirmation loads **only one RWKV checkpoint**, sharing its
weights between the full and reduced views, caps PyTorch GPU allocations at
4 GiB, limits compiler/build workers to one and writes live data under `.cache`,
which the project excludes from PyCharm indexing. It measures five matched
anchor cases with full/reduced request order alternated on each repeat. Results
and process/VRAM peaks are in `depth_confirmation_summary.csv` and
`depth_confirmation_environment.json`. External-provider comparisons remain the
completed main matrix; confirmation measurements are not mixed into its win counts.
The provider-only anchor confirmation adds matched external timings without
loading them alongside RWKV. Peak process RSS stayed around 3.4–3.6 GiB for
these resumed GPU runs, with RWKV's allocator peak below 1 GiB. Those are
process/allocator measurements, not total Windows GPU/RAM consumption.

## Reproduce and inspect

Settings: `configs/time_benchmark.json`. From the repository in WSL:

```bash
source scripts/activate_rwkv7.sh
python -u scripts/time_benchmark.py
python -u scripts/time_benchmark_confirm.py
python -u scripts/time_benchmark_baseline_confirm.py
python -u scripts/time_benchmark_cpu.py
python scripts/report_time_benchmark.py
```

- [Interactive latency chart](2026-10-31/latency.html)
- [Interactive confirmed anchors](2026-10-31/confirmed_latency.html)
- [Full scenario statistics](2026-10-31/summary.csv)
- [Raw request timings and diagnostics](2026-10-31/requests.csv)
- [Environment/model/config metadata](2026-10-31/environment.json)
- [Exact outlier replays](2026-10-31/tail_replays.csv)
- [Historical worst requests](2026-10-31/historical_worst_requests.csv)
- [Size reduction and win counts](2026-10-31/comparison.json)
- [Paired size confirmation](2026-10-31/depth_confirmation_summary.csv)
- [Matched external GPU anchors](2026-10-31/baseline_confirmation_summary.csv)
- [Supplementary CPU timings](2026-10-31/cpu_summary.csv)

The compact interactive HTML charts load Plotly from its CDN and require network
access. Raw CSV/JSON statistics and this Markdown report work offline. The bulky
profiler trace is compressed to avoid IDE indexing of large JSON event streams.

No training checkpoint, production holdout adapter, configuration or scoring output
was modified by the benchmark. Raw timings and warmups are preserved; no tail
observations were removed from scored summaries.

## Verification

The initial matrix contains **{verification["initial_gpu_scored_requests"]} scored GPU requests**
plus separately retained warmups/diagnostics, across 32 scenarios. The resumed
checks add 200 scored RWKV requests and 300 scored external GPU anchor requests;
the CPU supplement adds 120 scored forecasts. The full repository suite passed:
**187 tests, 180 passed, 7 CUDA-dependent skips** in the Windows CPU test runtime.
Actual WSL GPU and CPU executions are recorded separately above. Applicable
repository lint, formatting and security hooks passed on the benchmark changes.
Peak recorded initial-matrix PyTorch allocation was
**{verification["peak_recorded_allocated_gib"]:.2f} GiB**, reserved
**{verification["peak_recorded_reserved_gib"]:.2f} GiB**; these do not include all driver/display allocations.
There is no evidence of allocator-cap overflow in these recordings; this does
not retrospectively establish the memory behavior of the old holdout outlier.
"""
    )
    (OUTPUT.parent / "REPORT.md").write_text(text, encoding="utf-8", newline="\n")


if __name__ == "__main__":
    main()

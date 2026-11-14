"""Measure which mechanisms and signal shapes the production generator emits.

CPU-only and read-only with respect to data/: tables are sampled in memory with
the settings of generate_operational_data.py and never written to disk.

Usage: python generator_coverage.py [num_tables] [workers]
"""

import json
import sys
from collections import Counter
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src" / "01_data_generation"))

LAG_OPS = ("first_order_lag", "erlang", "diffusion")
TEMPORAL = (
    "first_order_lag",
    "delay",
    "integrator",
    "hysteresis",
    "rate_limiter",
    "erlang",
    "diffusion",
    "resonance",
    "regime_switch",
    "pid",
    "volatility_coupling",
)


def production_generator():
    """Build the generator exactly as the runner does, without generating."""
    import generate_operational_data as runner
    from scm_generator import OperationalSCMGenerator

    captured = {}

    def capture(self, num_tables):
        captured["generator"] = self
        return [Path("unused") / "unused"]

    original = OperationalSCMGenerator.generate
    OperationalSCMGenerator.generate = capture
    try:
        runner.main()
    finally:
        OperationalSCMGenerator.generate = original
    return captured["generator"]


def signal_statistics(y):
    """Shape descriptors that need no knowledge of the generating model."""
    y = np.asarray(y, dtype=float)
    missing_share = float(np.isnan(y).mean())
    y = y[np.isfinite(y)]  # Shape statistics describe the observed readings.
    n = len(y)
    spread = float(np.std(y))
    diff = np.diff(y)
    equal = diff == 0
    longest, run = 0, 0
    for flag in equal:
        run = run + 1 if flag else 0
        longest = max(longest, run)
    if spread < 1e-12:
        return {"constant": True, "flat_run": n, "rows": n, "missing": missing_share}
    rho1 = float(np.corrcoef(y[:-1], y[1:])[0, 1])
    smooth = pd.Series(y).rolling(5, center=True, min_periods=1).median().to_numpy()
    time = np.arange(n)
    detrended = y - np.polyval(np.polyfit(time, y, 1), time)
    power = np.abs(np.fft.rfft(detrended)) ** 2
    power[:3] = 0  # Fewer than three cycles in the window is trend, not a season.
    total = power.sum()
    peak = int(np.argmax(power))
    peak_share = float(power[max(0, peak - 1) : peak + 2].sum() / total) if total else 0
    low, high = float(y.min()), float(y.max())
    return {
        "constant": False,
        "rows": n,
        "missing": missing_share,
        "flat_run": longest + 1,
        "flat_share": float(equal.mean()),
        "unique": int(len(np.unique(y))),
        "zero_share": float((y == 0).mean()),
        "at_max_share": float((y == high).mean()),
        "at_min_share": float((y == low).mean()),
        "monotone": bool((diff >= 0).all() or (diff <= 0).all()),
        "roughness": float(np.sqrt(max(0.0, 1 - rho1))),
        "hf_noise": float(np.std(y - smooth) / spread),
        "peak_share": peak_share,
        "peak_period": float(n / peak) if peak else float("inf"),
    }


def model_flags(model):
    nodes = {n["name"]: n for n in model["nodes"]}
    outcome = model["outcome"]
    parents_of = {name: [e["parent"] for e in n["edges"]] for name, n in nodes.items()}
    ancestors, frontier = set(), [e["parent"] for e in outcome["edges"]]
    while frontier:
        name = frontier.pop()
        if name in ancestors or name not in nodes:
            continue
        ancestors.add(name)
        frontier.extend(parents_of[name])

    def step_source(name):
        node = nodes.get(name)
        if node is None:
            return False
        return node["kind"] in ("markov", "bernoulli", "switch") or node.get(
            "family"
        ) in ("state_machine", "process_run")

    def ops(edge):
        return [i["op"] for i in edge["program"]]

    relevant = [("outcome", e) for e in outcome["edges"]] + [
        (name, e) for name in ancestors for e in nodes[name]["edges"]
    ]
    flags = Counter()
    for target, edge in relevant:
        for op in set(ops(edge)):
            if op in TEMPORAL:
                flags[f"op_{op}"] = 1
        if step_source(edge["parent"]):
            flags["step_source_reaches_outcome"] = 1
            if set(ops(edge)) & set(LAG_OPS):
                flags["settling_after_step"] = 1
            if "resonance" in ops(edge):
                flags["underdamped_step"] = 1
        if edge["parent"].startswith("outcome_lag_"):
            flags["feedback_loop"] = 1
            if "hysteresis" in ops(edge):
                flags["relay_in_feedback_loop"] = 1
    slow = [
        nodes[name]["persistence"]
        for name in ancestors
        if nodes[name]["kind"] == "process"
        and any(step_source(p) for p in parents_of[name])
    ]
    flags["step_into_slow_state_tau_ge_20"] = int(any(p >= 0.951 for p in slow))
    operation = outcome.get("operation", {})
    trend = outcome.get("trend", {})
    observation = model.get("observation", {})
    for artifact in observation.get("artifacts", {}).get("outcome", []):
        flags[f"artifact_{artifact['kind']}"] = 1
    for missing in observation.get("missing", []):
        flags[f"missing_{missing['paradigm']}"] = 1
        if "outcome" in missing["columns"]:
            flags["missing_in_outcome"] = 1
    for edge in outcome["edges"]:
        if edge["program"][0].get("damping") and len(edge["program"]) == 1:
            flags["outcome_second_order_edge"] = 1
    flags.update(
        {
            f"plant_{model.get('plant', {}).get('family', 'none')}": 1,
            f"noise_profile_{model.get('noise_profile', {}).get('name', 'legacy')}": 1,
            "prehistory": int("recorded_rows" in model),
            "row_gaps": int("row_gaps" in model),
            "redundant_sensor": int(bool(observation.get("redundant"))),
            "outcome_any_artifact": int("outcome" in observation.get("artifacts", {})),
            "startup_transient": int("stop_tau" in operation),
            "slow_state_covariate": int(
                any("forcing_gain" in nodes[name] for name in ancestors)
            ),
            "calendar_gate": int(any("gate" in s for s in outcome["seasons"])),
            "outcome_ringing": int("ringing" in outcome),
            "outcome_event_shock": int(bool(outcome["event_parent"])),
            "outcome_quantized": int(outcome["quantization_step"] > 0),
            "outcome_sensor_bounds": int("sensor" in outcome),
            "outcome_volatility": int("volatility" in outcome),
            "outcome_has_season": int(bool(outcome["seasons"])),
            "univariate": int(model["univariate"]),
            f"operation_{operation.get('mode', 'none')}": 1,
            f"trend_{trend.get('kind', 'none')}": 1,
            f"seasons_{len(outcome['seasons'])}": 1,
            f"visible_{min(3, model['visibility']['visible_count'])}plus": 1,
        }
    )
    return dict(flags)


def run_chunk(arguments):
    seeds, start = arguments
    generator = production_generator()
    records = []
    for offset, seed in enumerate(seeds):
        generator.rng = np.random.default_rng(int(seed))
        model, frame = generator._sample_accepted_model((start + offset) % 2 == 0)
        records.append(
            {
                "flags": model_flags(model),
                "stats": signal_statistics(frame["outcome"].to_numpy()),
                "attempts": model["quality"]["sampling_attempts"],
            }
        )
    return records


def real_windows(limit_rows=60000, window=2048):
    """Windows of real operational columns already stored in data/00_public."""
    public = ROOT / "data" / "00_public"
    sources = {
        "energydata_complete.csv": None,
        "time_series_15min_singleindex.csv": limit_rows,
        "anomaly-free.csv": None,
        "Metro_PT3_(AirCompressor).csv": 400000,
        "BATADAL_dataset03.csv": None,
        "ETTh1.csv": None,
        "Kelmarsh_era5.csv": limit_rows,
        "ec2_cpu_utilization_24ae8d.csv": None,
        "ec2_cpu_utilization_fe7f93.csv": None,
        "elb_request_count_8c0756.csv": None,
        "TravelTime_387.csv": None,
    }
    rows = []
    for name, nrows in sources.items():
        path = public / name
        if not path.exists():
            continue
        frame = pd.read_csv(path, nrows=nrows, sep=None, engine="python")
        frame = frame.select_dtypes("number")
        for column in frame:
            # Documented random columns, row counters and attack labels.
            if column in ("rv1", "rv2", "ATT_FLAG") or column.startswith("Unnamed"):
                continue
            values = frame[column].to_numpy(dtype=float)
            for begin in range(0, len(values) - window + 1, window):
                part = values[begin : begin + window]
                if np.isnan(part).any() or len(np.unique(part)) <= 2:
                    continue  # Gaps, and binary status flags (never a target).
                stats = signal_statistics(part)
                stats.update(source=name, column=column)
                rows.append(stats)
    return pd.DataFrame(rows)


def describe(table, label):
    live = table[~table["constant"]]
    summary = {
        "windows": int(len(table)),
        "rows_mean": float(table["rows"].mean()),
        "rows_at_least_2048": float((table["rows"] >= 2048).mean()),
        "any_missing": float((table["missing"] > 0).mean()),
        "constant": float(table["constant"].mean()),
        "flat_run_ge_16": float((table["flat_run"] >= 16).mean()),
        "flat_share_ge_0.5": float((live["flat_share"] >= 0.5).mean()),
        "unique_le_12": float((live["unique"] <= 12).mean()),
        "zero_share_ge_0.1": float((live["zero_share"] >= 0.1).mean()),
        "pinned_at_extreme_ge_0.02": float(
            ((live["at_max_share"] >= 0.02) | (live["at_min_share"] >= 0.02)).mean()
        ),
        "monotone": float(live["monotone"].mean()),
        "hf_noise_lt_0.02": float((live["hf_noise"] < 0.02).mean()),
        "hf_noise_lt_0.05": float((live["hf_noise"] < 0.05).mean()),
        "peak_share_ge_0.3": float((live["peak_share"] >= 0.3).mean()),
        "peak_share_lt_0.1": float((live["peak_share"] < 0.1).mean()),
    }
    for name in ("hf_noise", "roughness", "peak_share"):
        summary[f"{name}_quantiles_5_25_50_75_95"] = [
            round(float(v), 4)
            for v in live[name].quantile([0.05, 0.25, 0.5, 0.75, 0.95])
        ]
    print(f"\n== {label} ==")
    print(json.dumps(summary, indent=2))
    return summary


def main():
    count = int(sys.argv[1]) if len(sys.argv) > 1 else 200
    workers = int(sys.argv[2]) if len(sys.argv) > 2 else 4
    seeds = np.random.default_rng(20261113).integers(0, 2**63, size=count)
    chunk = max(1, count // (workers * 4))
    jobs = [(seeds[i : i + chunk], i) for i in range(0, count, chunk)]
    if workers == 1:
        results = [run_chunk(job) for job in jobs]
    else:
        with ProcessPoolExecutor(max_workers=workers) as pool:
            results = list(pool.map(run_chunk, jobs))
    records = [record for group in results for record in group]
    flags = Counter()
    for record in records:
        flags.update(record["flags"])
    print(f"tables={len(records)}")
    print("== mechanism frequency, share of tables ==")
    for name, value in sorted(flags.items()):
        print(f"{name:40s} {value / len(records):7.4f}")
    print("mean sampling attempts", np.mean([r["attempts"] for r in records]))
    synthetic = pd.DataFrame([r["stats"] for r in records])
    out = Path(__file__).with_name("generator_coverage_synthetic.parquet")
    synthetic.to_parquet(out, index=False)
    describe(synthetic, "synthetic outcome")
    describe(synthetic[synthetic["rows"] >= 1024], "synthetic outcome, rows>=1024")
    real = real_windows()
    if len(real):
        real.to_parquet(
            Path(__file__).with_name("generator_coverage_real.parquet"), index=False
        )
        describe(real, "real windows (data/00_public)")
        for source, group in real.groupby("source"):
            describe(group, f"real: {source}")


if __name__ == "__main__":
    main()

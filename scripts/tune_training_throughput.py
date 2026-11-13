"""Compare physical execution settings on an idle GPU; never edit the default run."""

import argparse
import json
import os
import statistics
import subprocess  # nosec B404 - fixed local benchmark interpreter/script
import sys
from datetime import datetime, timezone
from pathlib import Path

import psutil

ROOT = Path(__file__).resolve().parents[1]


def active_trainers():
    result = []
    for process in psutil.process_iter(["pid", "cmdline"]):
        try:
            arguments = process.info["cmdline"] or []
            if process.pid != psutil.Process().pid and any(
                Path(arg).name == "train.py" for arg in arguments
            ):
                result.append(process.pid)
        except (psutil.NoSuchProcess, psutil.AccessDenied):
            continue
    return result


def summarize(payload, warmup):
    cases = {}
    for row in payload["updates"]:
        cases.setdefault((row["timesteps"], row["variables"]), []).append(row)
    measurements = []
    for (length, variables), rows in sorted(cases.items()):
        warm = rows[warmup:]
        if not warm:
            raise ValueError("Not enough measured updates after per-case warmup")
        measurements.append(
            {
                "timesteps": length,
                "variables": variables,
                "mean_ms": statistics.mean(r["seconds"] for r in warm) * 1000,
                "median_ms": statistics.median(r["seconds"] for r in warm) * 1000,
                "mean_physical_microbatches": statistics.mean(
                    r["microbatches"] for r in warm
                ),
            }
        )
    return {
        "completed": payload.get("completed", False),
        "equal_case_mean_ms": statistics.mean(c["mean_ms"] for c in measurements),
        "peak_allocated_gib": max(r["peak_allocated_gib"] for r in payload["updates"]),
        "peak_reserved_gib": max(r["peak_reserved_gib"] for r in payload["updates"]),
        "oom_retries": sum(r["oom_retries"] for r in payload["updates"]),
        "cases": measurements,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--spec", default="configs/training_throughput_tuning.json")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    if not args.dry_run:
        if os.name == "nt":
            parser.error(
                "Run benchmarks inside the activated WSL CUDA environment; --dry-run also works on Windows"
            )
        trainers = active_trainers()
        if trainers:
            parser.error(
                f"Training is active (PIDs {trainers}); pause after a saved checkpoint first"
            )
    spec = json.loads((ROOT / args.spec).read_text(encoding="utf-8"))
    config = json.loads((ROOT / spec["source_config"]).read_text(encoding="utf-8"))
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S_%fZ")
    directory = ROOT / "docs/time_benchmark/training_tuning" / stamp
    directory.mkdir(parents=True)
    report = {
        "dry_run": args.dry_run,
        "source_config": spec["source_config"],
        "scope": "same eight-table logical optimizer batches; equal-weight synthetic cases, not a corpus epoch speedup estimate",
        "profiles": [],
    }
    for profile in spec["profiles"]:
        name = profile["name"]
        if Path(name).name != name or not name:
            raise ValueError("Profile name must be one path component")
        candidate = json.loads(json.dumps(config))
        candidate["training"].update({k: v for k, v in profile.items() if k != "name"})
        candidate["training"]["resume"] = False  # Diagnostic never loads a checkpoint.
        config_path = directory / f"{name}_config.json"
        config_path.write_text(json.dumps(candidate, indent=2) + "\n", encoding="utf-8")
        output = directory / f"{name}.json"
        command = [
            sys.executable,
            "-u",
            str(ROOT / "scripts/verify_training_memory.py"),
            "--config",
            str(config_path),
            "--updates",
            str(spec["updates_per_profile"]),
            "--output",
            str(output),
        ]
        row = {"name": name, "settings": profile, "command": command}
        if not args.dry_run:
            # Check again at each profile boundary in case training was restarted.
            if active_trainers():
                raise RuntimeError(
                    "Training restarted; no further GPU benchmarks will launch"
                )
            print(f"Benchmarking {name}; log={directory / (name + '.log')}", flush=True)
            with (directory / f"{name}.log").open("w", encoding="utf-8") as stream:
                result = subprocess.run(
                    command, cwd=ROOT, stdout=stream, stderr=subprocess.STDOUT
                )  # nosec B603
            row["exit_code"] = result.returncode
            if result.returncode == 0:
                row.update(
                    summarize(
                        json.loads(output.read_text()), spec["warmup_updates_per_case"]
                    )
                )
        report["profiles"].append(row)
        (directory / "summary.json").write_text(
            json.dumps(report, indent=2) + "\n", encoding="utf-8"
        )
    print(
        f"{'Prepared plan' if args.dry_run else 'Benchmark results'}: {directory / 'summary.json'}",
        flush=True,
    )


if __name__ == "__main__":
    main()

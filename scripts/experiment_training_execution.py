"""Run isolated, sequential execution experiments on the cached training sample."""

import argparse
import subprocess  # nosec B404
import sys
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--verify", action="store_true")
    args = parser.parse_args()
    common = [
        sys.executable,
        "-u",
        "scripts/benchmark_training_throughput.py",
        "--batch-size",
        "8",
        "--accumulation",
        "1",
        "--fused",
        "--bucket",
        "--sample",
        "docs/audits/2026-10-28/training_speed/sample.pt",
        "--output-dir",
        "docs/audits/2026-10-28/training_speed_round2",
    ]
    cases = {
        "reference_sweep": ["--checkpoint-threshold", "2048", "--sweep"],
        "budget2560_sweep": ["--checkpoint-threshold", "2560", "--sweep"],
        "compiled_math_sweep": [
            "--checkpoint-threshold",
            "2048",
            "--sweep",
            "--warmup-sweep",
            "--direct-attention",
            "--attention-backend",
            "math",
            "--compile-attention",
        ],
    }
    if args.verify:
        commands = [
            (
                "attention_tests",
                [
                    sys.executable,
                    "-m",
                    "unittest",
                    "discover",
                    "-s",
                    "tests",
                    "-p",
                    "test_variate_attention.py",
                ],
            ),
        ]
        cases = {
            "reference_warm": ["--sweep", "--warmup-sweep"],
            "production_warm": ["--sweep", "--warmup-sweep", "--production-compile"],
            "production_repeat": ["--sweep", "--warmup-sweep", "--production-compile"],
            "reference_repeat": ["--sweep", "--warmup-sweep"],
            "production_worst": [
                "--worst-case",
                "--updates",
                "6",
                "--production-compile",
            ],
        }
        commands.extend(
            (name, common + ["--case", name, "--checkpoint-threshold", "2048"] + flags)
            for name, flags in cases.items()
        )
        commands.append(
            (
                "compiled_resume",
                [
                    sys.executable,
                    "-u",
                    "scripts/verify_epoch_resume.py",
                    "--compile-variate",
                    "--output-dir",
                    "docs/audits/2026-10-28/training_speed_round2/resume",
                ],
            )
        )
    else:
        commands = [
            (name, common + ["--case", name] + flags) for name, flags in cases.items()
        ]
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    for name, command in commands:
        log = ROOT / "logs" / f"execution_{name}_{stamp}.log"
        print(f"Starting {name}: {log}", flush=True)
        with log.open("w") as stream:
            subprocess.run(
                command, cwd=ROOT, check=True, stdout=stream, stderr=subprocess.STDOUT
            )  # nosec B603
        print(f"Passed {name}", flush=True)


if __name__ == "__main__":
    main()

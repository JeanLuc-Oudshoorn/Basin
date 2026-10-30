"""Display active training progress without loading tensors, using CUDA or restarting training."""
# ruff: noqa: E402

import argparse
import json
import sys
from pathlib import Path
from time import sleep

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src/05_model_training"))
from training_progress import journal_progress


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default=str(ROOT / "configs/config.json"))
    parser.add_argument("--journal", type=Path)
    parser.add_argument("--follow", action="store_true")
    parser.add_argument("--interval", type=float, default=5.0)
    args = parser.parse_args()
    if args.interval <= 0:
        parser.error("interval must be positive")
    config_path = Path(args.config)
    if not config_path.is_absolute():
        config_path = ROOT / config_path
    config = json.loads(config_path.read_text(encoding="utf-8"))
    plan_path = ROOT / config["paths"]["training"] / "training_plan.json"
    if not plan_path.is_file():
        parser.error(
            f"Training plan not found: {plan_path}; wait for model initialization"
        )
    plan = json.loads(plan_path.read_text(encoding="utf-8"))
    journals = list((ROOT / config["paths"]["logs"]).glob("rwkv7_train_*.jsonl"))
    if args.journal is None and not journals:
        parser.error("No training journals found")
    journal = args.journal or max(journals, key=lambda p: p.stat().st_mtime)
    print(f"Reading {journal}", flush=True)
    try:
        while True:
            progress = journal_progress(journal, plan)
            print(
                f"{progress['message']} | event={progress['latest_event']}", flush=True
            )
            if not args.follow or progress["latest_event"] in (
                "run_complete",
                "run_error",
            ):
                break
            sleep(args.interval)
    except KeyboardInterrupt:
        pass  # Only the watcher exits; the training worker is untouched.


if __name__ == "__main__":
    main()

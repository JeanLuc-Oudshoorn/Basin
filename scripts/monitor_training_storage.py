"""Sample Windows disk/RAM and WSL GPU use alongside a bounded training run."""

import argparse
import json
import shutil
import subprocess  # nosec B404 - fixed local read-only GPU query
import time
from datetime import datetime, timezone
from pathlib import Path

import psutil


def directory_bytes(path):
    total = 0
    for item in path.rglob("*"):
        try:
            if item.is_file():
                total += item.stat().st_size
        except OSError:
            continue
    return total


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--status", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[1]
    args.output.parent.mkdir(parents=True, exist_ok=True)
    cache = root / ".cache"
    with args.output.open("w", encoding="utf-8") as handle:
        for index in range(100):
            gpu = subprocess.run(  # nosec B603 - fixed executable and arguments
                [
                    "C:/Windows/System32/wsl.exe",
                    "-d",
                    "Ubuntu-24.04",
                    "--",
                    "nvidia-smi",
                    "--query-gpu=memory.used,utilization.gpu",
                    "--format=csv,noheader,nounits",
                ],
                capture_output=True,
                text=True,
                timeout=15,
                check=False,
            )
            ram = psutil.virtual_memory()
            record = dict(
                utc=datetime.now(timezone.utc).isoformat(),
                c_free_bytes=shutil.disk_usage("C:/").free,
                e_free_bytes=shutil.disk_usage("E:/").free,
                host_available_ram_bytes=ram.available,
                gpu_query=gpu.stdout.strip(),
                gpu_query_error=gpu.stderr.strip() if gpu.returncode else None,
            )
            if index % 4 == 0:
                record["cache_bytes"] = {
                    name: directory_bytes(cache / name)
                    for name in (
                        "torchinductor",
                        "tmp",
                        "cuda_driver",
                        "compile_debug",
                        "crash_dumps",
                    )
                }
                swap = Path("E:/WSL/swap.vhdx")
                record["swap_file_bytes"] = (
                    swap.stat().st_size if swap.exists() else None
                )
            handle.write(json.dumps(record) + "\n")
            handle.flush()
            if args.status.exists():
                status = json.loads(args.status.read_text(encoding="utf-8"))
                if status.get("status") in (
                    "complete",
                    "completed",
                    "failed",
                    "success",
                ):
                    break
            time.sleep(15)


if __name__ == "__main__":
    main()

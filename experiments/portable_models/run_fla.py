"""Run the flash-linear-attention Basin model on a CSV series (default: airline passengers).

    python experiments/portable_models/run_fla.py

FLA needs Triton, so this runs on the GPU inside WSL. Started from Windows, the
script relaunches itself there and then opens the saved plot.
"""

import os
import shlex
import subprocess  # nosec B404
import sys
from pathlib import Path

from forecast_demo import MODELS, ROOT, parse, run

# FLA is installed beside, not inside, the training environment (see the audit report).
FLA_SITE = "$HOME/projects/basin-runtime/fla-site"


def relaunch_in_wsl(args):
    script = Path(__file__).resolve().relative_to(ROOT).as_posix()
    arguments = [a for a in sys.argv[1:] if a != "--no-show"]
    data = Path(args.data).resolve()
    if "--data" in arguments and data.is_relative_to(ROOT):
        arguments[arguments.index("--data") + 1] = data.relative_to(ROOT).as_posix()
    command = (
        "source scripts/activate_rwkv7.sh && "
        f'PYTHONPATH="{FLA_SITE}" python {shlex.quote(script)} --no-show '
        + " ".join(shlex.quote(a) for a in arguments)
    )
    # Fixed executable; arguments are shell-quoted.
    subprocess.run(  # nosec B603
        ["C:/Windows/System32/wsl.exe", "--cd", str(ROOT), "bash", "-lc", command],
        check=True,
    )
    plot = MODELS / "demo" / f"fla_{Path(args.data).stem}.png"
    if not args.no_show and plot.is_file():
        os.startfile(plot)  # nosec B606 - opens the PNG this script just wrote


if __name__ == "__main__":
    options = parse("fla")
    if os.name == "nt":
        relaunch_in_wsl(options)
    else:
        run(options)

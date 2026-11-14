"""Run the pure-PyTorch Basin model on a CSV series (default: airline passengers).

    python experiments/portable_models/run_pytorch.py
    python experiments/portable_models/run_pytorch.py --data data/00_public/monthly-sunspots.csv --device cpu

Works in the Windows .venv (CPU) and in the WSL runtime (GPU).
"""

from forecast_demo import parse, run

if __name__ == "__main__":
    run(parse("pytorch"))

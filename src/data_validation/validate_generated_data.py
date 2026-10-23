"""Edit settings below, then run this file directly (no command-line arguments)."""

import logging
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
INPUT_DIR = PROJECT_ROOT / "data" / "generated" / "2026-10-21"
OUTPUT_DIR = PROJECT_ROOT / "results" / "data_validation"
TRAIN_FRACTION = 0.8
SEED = 42
# known_future evaluates conditional forecasts using actual test covariates.
# forecast fits separate Prophet models to training covariates and extrapolates
# them. Neither mode supplies test outcomes or recorded outcome-lag columns.
COVARIATE_MODE = "known_future"
STEP_DURATION = "1D"  # Synthetic samples have no calendar; map one step to one day.
# Prophet's automatic weekly/yearly seasonality is the default. Optional custom
# periods below are in timesteps, shared across models; no SCM metadata is read.
SEASONAL_PERIODS = ()
PROPHET_OPTIONS = {
    "daily_seasonality": False,
    "weekly_seasonality": "auto",
    "yearly_seasonality": "auto",
    "seasonality_mode": "additive",
    "uncertainty_samples": 0,
}

sys.path.insert(0, str(Path(__file__).resolve().parent))
from prophet_validator import ProphetValidator


def main():
    # Give CmdStan a root handler before it installs its verbose default logger.
    logging.basicConfig(level=logging.WARNING)
    logging.getLogger("cmdstanpy").setLevel(logging.WARNING)
    validator = ProphetValidator(
        INPUT_DIR,
        OUTPUT_DIR,
        train_fraction=TRAIN_FRACTION,
        covariate_mode=COVARIATE_MODE,
        step_duration=STEP_DURATION,
        seed=SEED,
        prophet_options=PROPHET_OPTIONS,
        seasonal_periods=SEASONAL_PERIODS,
    )
    results, directory = validator.run()
    print(f"Saved {len(results)} evaluations to {directory}")
    for variant in ("univariate", "multivariate"):
        wins = results[f"{variant}_beats_baseline"].dropna()
        print(
            f"{variant}: {int(wins.sum())}/{len(wins)} successful evaluations beat baseline"
        )
    print(f"Tables with errors: {(results['status'] != 'ok').sum()}")


if __name__ == "__main__":
    main()

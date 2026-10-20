"""Edit the settings below, then run this file directly or with IPython %run."""

from pathlib import Path
import sys


# User settings (one fresh structural causal model per table).
NUM_TABLES = 20
SEED = 42  # Use None for a fresh random seed each execution.
OUTPUT_DIR = Path(__file__).resolve().parents[2] / "data" / "generated"
MIN_ROWS = 20
MAX_ROWS = 2056  # Inclusive.
MIN_COVARIATES = 3  # Includes a binary event source; other nodes are compound processes.
MAX_COVARIATES = 10
MAX_LAG = 192  # Also bounds delayed covariate effects, in samples.
MAX_LAG_COUNT = 6  # Lag 1 is always present; all selected outcome lags are visible.
HIDDEN_COVARIATE_PROBABILITY = 0.5
EDGE_PROBABILITY = 0.18  # Extra edges beyond randomly selected causal scaffolds.
DIRECT_OUTCOME_PROBABILITY = 0.3  # Upstream causes can act through mediators.
DELAYED_EDGE_PROBABILITY = 0.3
DRIFT_PROBABILITY = 0.5
RANDOM_WALK_PROBABILITY = 0.3
MAX_EFFECT_COMPONENTS = 8  # Expression DAG size per edge; increase for more complexity.
# Relative family weights, applied to outcome noise AND covariate innovations.
# Unlisted families keep weight 1. Defaults give the three preferred types 1/6 each.
# Covariates without incoming edges can thus start from either process.
NOISE_WEIGHTS = {"perlin": 4.0, "student_t": 4.0, "pink": 4.0}
# Optional overrides of OperationalSCMGenerator.DEFAULT_PRIORS. These ranges
# control the distributions of model parameters, not per-observation resampling.
PRIOR_RANGES = {
    "student_df": (2.2, 35.0),
    "edge_strength": (0.005, 3.0),  # Signed, log-uniform magnitude.
    "drift_magnitude": (0.0005, 0.04),
    "mixture_separation": (0.5, 3.5),
    "perlin_frequency": (0.005, 0.2),  # Lattice units per time step.
    "perlin_octaves": (1, 6),
    "perlin_persistence": (0.25, 0.8),  # Amplitude multiplier between octaves.
    "perlin_lacunarity": (1.5, 3.0),  # Frequency multiplier between octaves.
    "pink_exponent": (0.8, 1.2),  # Power spectrum ~ 1/f**exponent; 1 is pink.
    "pink_filter_length": (256, 2048),  # Finite memory; longer retains slower variation.
    "inverse_square_floor": (0.01, 0.3),  # Avoid the singularity in 1/x² effects.
}
BURN_IN = 256  # Unrecorded history for warm-up and valid initial lag columns.
SAVE_SCM = True
SAVE_PLOT = True
SAVE_DAG = True  # Causal structure with hidden nodes and explicit history proxies.
PLOT_COVARIATES = True  # False plots only the outcome.


sys.path.insert(0, str(Path(__file__).resolve().parent))
from scm_generator import OperationalSCMGenerator


def main():
    generator = OperationalSCMGenerator(
        OUTPUT_DIR, seed=SEED, min_rows=MIN_ROWS, max_rows=MAX_ROWS,
        min_covariates=MIN_COVARIATES, max_covariates=MAX_COVARIATES,
        max_lag=MAX_LAG, max_lag_count=MAX_LAG_COUNT,
        hidden_probability=HIDDEN_COVARIATE_PROBABILITY,
        edge_probability=EDGE_PROBABILITY, burn_in=BURN_IN,
        direct_probability=DIRECT_OUTCOME_PROBABILITY,
        delayed_probability=DELAYED_EDGE_PROBABILITY,
        drift_probability=DRIFT_PROBABILITY, walk_probability=RANDOM_WALK_PROBABILITY,
        max_effect_components=MAX_EFFECT_COMPONENTS, prior_ranges=PRIOR_RANGES,
        noise_weights=NOISE_WEIGHTS,
        save_scm=SAVE_SCM, save_plot=SAVE_PLOT, plot_covariates=PLOT_COVARIATES,
        save_dag=SAVE_DAG,
    )
    paths = generator.generate(NUM_TABLES)
    print(f"Saved {len(paths)} tables under {paths[0].parent}")


if __name__ == "__main__":
    main()

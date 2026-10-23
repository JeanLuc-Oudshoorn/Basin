"""Edit the settings below, then run this file directly or with IPython %run."""

import sys
from pathlib import Path

# User settings (one fresh structural causal model per table).
NUM_TABLES = 100
SEED = 1  # Use None for a fresh random seed each execution.
OUTPUT_DIR = Path(__file__).resolve().parents[2] / "data" / "generated"
MIN_ROWS = 20
MAX_ROWS = 2056  # Inclusive.
MIN_COVARIATES = (
    3  # Includes a binary event source; other nodes are compound processes.
)
MAX_COVARIATES = 10
MAX_LAG = 192  # Maximum outcome-history lag.
MAX_LAG_COUNT = 6  # Lag 1 is always present; all selected outcome lags are visible.
SEASONAL_LAG_PROBABILITY = (
    0.8  # If selected: 1/2/3 seasonal anchors with weights 80/16/4%.
)
LONG_LAG_PROBABILITY = (
    0.20  # One additional unstructured lag, subject to MAX_LAG_COUNT.
)
MAX_COVARIATE_LAG = (
    24  # Maximum delay for covariate edges, including events and switches.
)
INSTANTANEOUS_PROBABILITY = 0.10  # Otherwise there is at least one timestep of delay.
COVARIATE_DELAY_SCALE = 0.75  # Exponential scale; positive lag = 1 + floor(draw).
# Additional outcome lags in 2..7: zero/one/two with probabilities 15/75/10%.
VARIANCE_DYNAMICS_PROBABILITY = 0.37  # Noise envelope can change even with a flat mean.
HIDDEN_COVARIATE_PROBABILITY = 0.4
EDGE_PROBABILITY = 0.18  # Extra edges beyond randomly selected causal scaffolds.
DIRECT_OUTCOME_PROBABILITY = 0.65  # At least two process parents when available.
OUTCOME_FEEDBACK_PROBABILITY = 0.15  # Historical outcome -> covariate edge probability.
DRIFT_PROBABILITY = (
    0.5  # Covariate drift within trending profiles, not a second outcome-trend draw.
)
RANDOM_WALK_PROBABILITY = 0.15
TREND_FREE_FRACTION = (
    0.5  # Batch quota, screened using outcome linear-trend R² <= 0.15.
)
# Remaining tables MUST have a discernible trend (trend-shape R² >= 0.4).
# They include rising/falling, accelerating/decelerating, and permanent shock breaks.
RESIDUAL_HISTORY_PROBABILITY = (
    0.08  # Per selected outcome lag; residuals supplement full values.
)
MULTIPLICATIVE_TREND_PROBABILITY = (
    0.25  # Among trending proposals (~15% overall at a 50/50 split).
)
MULTIPLICATIVE_CONSTANT_NOISE_PROBABILITY = (
    0.05  # Among multiplicative trends; fixed direct noise scale.
)
MULTIPLICATIVE_CONSTANT_SEASONALITY_PROBABILITY = (
    0.05  # Independent rare draw; no seasonal growth.
)
SWITCHING_PROBABILITY = 0.15  # Binary shutdown/restart or ramp-reset node.
MAX_EFFECT_COMPONENTS = 8  # Expression DAG size per edge; increase for more complexity.
FOURIER_PROBABILITY = 0.7  # Per seasonal component, for covariates AND outcome.
MAX_FOURIER_HARMONICS = 16  # Also capped by period and maximum clock speed.
SEASONAL_SHAPE_DRIFT_PROBABILITY = 0.25  # Smoothly evolve between two sampled shapes.
# Relative shape weights; use zero to disable a family. Unlisted families keep
# defaults. Existing amplitude, cyclic timing and alternating peaks still apply.
FOURIER_WEIGHTS = {
    "sine": 1,
    "pulse": 2,
    "sawtooth": 1,
    "triangle": 1,
    "full_rectified": 1,
    "half_rectified": 1,
    "parabolic": 1,
    "multi_peak": 2,
    "random_spectrum": 2,
}
# Relative family weights, applied to outcome noise AND covariate innovations.
# Unlisted families (including brown/red noise) keep weight 1.
# Defaults give Perlin, Student-t, and pink noise 16% each.
# Covariates without incoming edges can thus start from either process.
NOISE_WEIGHTS = {"perlin": 2.0, "student_t": 4.0, "pink": 4.0}
# Optional overrides of OperationalSCMGenerator.DEFAULT_PRIORS. These ranges
# control the distributions of model parameters, not per-observation resampling.
PRIOR_RANGES = {
    "student_df": (2.1, 35.0),
    "edge_strength": (0.005, 3.0),  # Signed, log-uniform magnitude.
    "drift_magnitude": (0.0005, 0.06),
    "mixture_separation": (0.5, 3.5),
    "perlin_frequency": (0.005, 0.2),  # Lattice units per time step.
    "perlin_octaves": (1, 6),
    "perlin_persistence": (0.25, 0.8),  # Amplitude multiplier between octaves.
    "perlin_lacunarity": (1.5, 3.0),  # Frequency multiplier between octaves.
    "pink_exponent": (0.8, 1.2),  # Power spectrum ~ 1/f**exponent; 1 is pink.
    "pink_filter_length": (
        256,
        2048,
    ),  # Finite memory; longer retains slower variation.
    "inverse_square_floor": (0.01, 0.3),  # Avoid the singularity in 1/x² effects.
    "brown_step_scale": (
        0.01,
        0.15,
    ),  # Brown/red noise: Gaussian random-walk step size.
    "damping_rate": (0.05, 2.0),  # Exponential decay per unit of edge input.
    "oscillation_frequency": (0.3, 4.0),  # Radians per unit of edge input.
    "ringing_decay": (0.002, 0.1),  # Event-response exponential decay per time step.
    "ringing_period": (6.0, 128.0),  # Event-response oscillation period in samples.
    "multiplicative_log_change": (
        0.7,
        2.5,
    ),  # Total |log(end/start)|; mostly growth, sometimes decay.
    "relative_noise_scale": (
        0.025,
        0.10,
    ),  # Multiplicative series: initial noise SD relative to level.
    "relative_seasonal_amplitude": (0.04, 0.18),
    "variance_growth_factor": (1.5, 5.0),  # End/start SD factor, not variance factor.
    "recent_lag_decay": (
        0.5,
        1.2,
    ),  # Exponential falloff in recent-lag selection probability.
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
        OUTPUT_DIR,
        seed=SEED,
        min_rows=MIN_ROWS,
        max_rows=MAX_ROWS,
        min_covariates=MIN_COVARIATES,
        max_covariates=MAX_COVARIATES,
        max_lag=MAX_LAG,
        max_lag_count=MAX_LAG_COUNT,
        hidden_probability=HIDDEN_COVARIATE_PROBABILITY,
        edge_probability=EDGE_PROBABILITY,
        burn_in=BURN_IN,
        direct_probability=DIRECT_OUTCOME_PROBABILITY,
        delayed_probability=OUTCOME_FEEDBACK_PROBABILITY,
        drift_probability=DRIFT_PROBABILITY,
        walk_probability=RANDOM_WALK_PROBABILITY,
        trend_free_fraction=TREND_FREE_FRACTION,
        residual_history_probability=RESIDUAL_HISTORY_PROBABILITY,
        multiplicative_trend_probability=MULTIPLICATIVE_TREND_PROBABILITY,
        multiplicative_constant_noise_probability=MULTIPLICATIVE_CONSTANT_NOISE_PROBABILITY,
        multiplicative_constant_seasonality_probability=MULTIPLICATIVE_CONSTANT_SEASONALITY_PROBABILITY,
        variance_dynamics_probability=VARIANCE_DYNAMICS_PROBABILITY,
        seasonal_lag_probability=SEASONAL_LAG_PROBABILITY,
        long_lag_probability=LONG_LAG_PROBABILITY,
        instantaneous_probability=INSTANTANEOUS_PROBABILITY,
        covariate_delay_scale=COVARIATE_DELAY_SCALE,
        max_covariate_lag=MAX_COVARIATE_LAG,
        switching_probability=SWITCHING_PROBABILITY,
        max_effect_components=MAX_EFFECT_COMPONENTS,
        prior_ranges=PRIOR_RANGES,
        fourier_probability=FOURIER_PROBABILITY,
        max_fourier_harmonics=MAX_FOURIER_HARMONICS,
        seasonal_shape_drift_probability=SEASONAL_SHAPE_DRIFT_PROBABILITY,
        fourier_weights=FOURIER_WEIGHTS,
        noise_weights=NOISE_WEIGHTS,
        save_scm=SAVE_SCM,
        save_plot=SAVE_PLOT,
        plot_covariates=PLOT_COVARIATES,
        save_dag=SAVE_DAG,
    )
    paths = generator.generate(NUM_TABLES)
    print(f"Saved {len(paths)} tables under {paths[0].parent}")


if __name__ == "__main__":
    main()

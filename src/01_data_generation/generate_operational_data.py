"""Edit the settings below, then run this file directly or with IPython %run."""

import sys
from pathlib import Path

# User settings (one fresh structural causal model per table).
NUM_TABLES = 6_500_000
SEED = 7  # Use None for a fresh random seed each execution.
OUTPUT_DIR = Path(__file__).resolve().parents[2] / "data" / "01_generated"
MIN_ROWS = 112
MAX_ROWS = 2056  # Inclusive.
# Real series are usually longer than the model's context, so the model sees
# exactly the context length. Give that length a point mass instead of leaving
# it one value among ~2000; the rest stay uniform between MIN_ROWS and MAX_ROWS.
FULL_LENGTH_PROBABILITY = 0.5
# Among full-length tables: simulate this many EARLIER rows and discard them, so
# the table is the tail of a longer history (mid-life trends, running cycles).
PREHISTORY_PROBABILITY = 0.5
MAX_PREHISTORY_ROWS = 2048  # Uniform 64..this; costs generation time, not disk.
MIN_COVARIATES = (
    3  # Includes a binary event source; other nodes are compound processes.
)

MAX_COVARIATES = 6
MAX_LAG = 512  # Maximum outcome-history lag.
MAX_LAG_COUNT = 4  # Internal SCM dependencies only; no lag columns are exported.
SEASONAL_LAG_PROBABILITY = (
    0.6  # If selected: 1/2/3 seasonal anchors with weights 80/16/4%.
)
LONG_LAG_PROBABILITY = (
    0.035  # One additional unstructured lag, subject to MAX_LAG_COUNT.
)
MAX_COVARIATE_LAG = (
    96  # Maximum delay for covariate edges, including events and switches.
)
INSTANTANEOUS_PROBABILITY = 0.02  # Otherwise there is at least one timestep of delay.
COVARIATE_DELAY_SCALE = 0.75  # Exponential scale; positive lag = 1 + floor(draw).
# Additional outcome lags in 2..7: zero/one/two with probabilities 15/75/10%.
VARIANCE_DYNAMICS_PROBABILITY = 0.35  # Noise envelope can change even with a flat mean.
VISIBLE_COVARIATE_LAMBDA = 1.3  # Poisson count, capped at available covariates.
EDGE_PROBABILITY = 0.1  # Extra edges beyond randomly selected causal scaffolds.
DIRECT_OUTCOME_PROBABILITY = 0.65  # At least two process parents when available.
OUTCOME_FEEDBACK_PROBABILITY = 0.15  # Historical outcome -> covariate edge probability.
DRIFT_PROBABILITY = (
    0.5  # Covariate drift within trending profiles, not a second outcome-trend draw.
)
RANDOM_WALK_PROBABILITY = 0.08
NOISE_FREE_COVARIATE_PROBABILITY = (
    0.35  # Per process covariate; deterministic given its parents.
)
TREND_FREE_FRACTION = (
    0.7  # Proposal share; incidental moderate trends are accepted as trending.
)
# Trending proposals qualify by linear or trend-shape R-squared > 0.15.
# They include rising/falling, accelerating/decelerating, and permanent shock breaks.
RESIDUAL_HISTORY_PROBABILITY = (
    0.05  # Per selected outcome lag; residuals supplement full values.
)
MULTIPLICATIVE_TREND_PROBABILITY = 0.22  # Among trending proposals
MULTIPLICATIVE_CONSTANT_NOISE_PROBABILITY = (
    0.05  # Among multiplicative trends; fixed direct noise scale.
)
MULTIPLICATIVE_CONSTANT_SEASONALITY_PROBABILITY = (
    0.05  # Independent rare draw; no seasonal growth.
)
SWITCHING_PROBABILITY = 0.14  # Binary shutdown/restart or ramp-reset node.
PROCESS_EPISODE_PROBABILITY = (
    0.025  # One sustained run with idle start/end and gradual startup/release.
)
PROCESS_MULTILEVEL_PROBABILITY = 0.55  # Among episodes: extra operating setpoints.
PROCESS_PRIMING_PROBABILITY = 0.30  # Small deterministic jitters before startup.
FORMULA_COVARIATE_PROBABILITY = (
    0.08  # Eligible extra slots: clock formulas plus optional noise.
)
DISCRETE_COVARIATE_PROBABILITY = (
    0.25  # Additional source slots after the first continuous process.
)
SENSOR_BOUNDS_PROBABILITY = (
    0.05  # Hard physical bounds, per continuous covariate and outcome; measured units.
)
DISCRETE_WEIGHTS = {
    "bernoulli": 0.35,
    "binary_markov": 0.35,
    "categorical_markov": 0.30,
}
SWITCH_PROFILE_WEIGHTS = {"alternating": 0.4, "long_dwell": 0.5, "single_episode": 0.1}
MAX_EFFECT_COMPONENTS = 6  # Expression DAG size per edge; increase for more complexity.
COMPOSITE_EFFECT_PROBABILITY = 0.25  # Otherwise one operator per edge.
ANGLE_COVARIATE_PROBABILITY = (
    0.05  # Wrapped radians; sine/cosine/tangent maps require angular parents.
)
# Relative weights, not percentages. Unlisted operators retain tier defaults:
# often=8, sometimes=2, rare=0.15. Set a weight to zero to disable an operator.
# Static and temporal operators share this prior. See effect_mechanisms.py.
# second_order (settling with overshoot/ringing around the new level) is opt-in.
EFFECT_WEIGHTS = {"second_order": 2.0}  # Example: {"affine": 16, "tangent": 0}.
PARENT_COMBINATION_WEIGHTS = {}  # Override sum/product/ratio/min/max/soft_min/gate/etc.
NODE_LINK_WEIGHTS = {}  # identity/saturation/softplus/exponential/sigmoid/logarithm.
PAIRWISE_INTERACTION_PROBABILITY = 0.85  # Most selected interactions use a random pair.
ADDITIVE_INTERACTION_PROBABILITY = (
    0.25  # Product-like maps may supplement the sum instead of replacing a group.
)
JOINT_NUMERIC_LIMIT = 1e6  # Overflow guard, independent of physical saturation links.
FOURIER_PROBABILITY = 0.7  # Per seasonal component, for covariates AND outcome.
MAX_FOURIER_HARMONICS = 16  # Also capped by period and maximum clock speed.
SEASONAL_SHAPE_DRIFT_PROBABILITY = 0.15  # Smoothly evolve between two sampled shapes.
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
    "effect_tau": (1.0, 200.0),  # Edge lag time constants, log-uniform (was 1..30).
}

# --- Realism settings (all default to OFF inside OperationalSCMGenerator, so
# --- older seeds and saved SCMs replay unchanged; set a value to 0 to disable).
# Seasonality. Share of tables whose outcome has clock-driven cycles at all. In
# the others the outcome has no seasonal term or seasonal AR anchor, and each
# covariate keeps its cycles only with the second probability below.
SEASONAL_TABLE_PROBABILITY = 0.4
NONSEASONAL_COVARIATE_SEASON_PROBABILITY = 0.15
FORCED_SEASON_PROBABILITY = 0.85  # Seasonal tables: add a cycle if none was drawn.
CALENDAR_GATE_PROBABILITY = 0.15  # Per season: reduced "rest day" cycles (5+2 etc.).
# Slow mean-reverting wander of the outcome itself (time constant 8-600 samples).
# It is what a cycle-free signal mostly consists of; cycles can ride on it too.
NONSEASONAL_WANDER_PROBABILITY = 0.7
SEASONAL_WANDER_PROBABILITY = 0.25
# Noise budget per table. clean: no white floor, innovations x0.003-0.03;
# low: floor 0.004-0.05; moderate: the previous fixed floor 0.15-0.45;
# noisy: floor 0.6-2 (a sensor that is mostly noise). None restores the old floor.
NOISE_PROFILE_WEIGHTS = {"clean": 0.45, "low": 0.28, "moderate": 0.19, "noisy": 0.08}
# Time scales and settling.
SLOW_DYNAMICS_PROBABILITY = 0.25  # Per process covariate: time constant 20-400.
TRANSPORT_DELAY_PROBABILITY = 0.10  # Per edge: dead time 4..MAX_COVARIATE_LAG.
STEP_RESPONSE_PROBABILITY = 0.5  # Edges from switches/levels: clean lag/overshoot.
STARTUP_TRANSIENT_PROBABILITY = 0.6  # Shutdown tables: coast-down and inrush.
# Trends. Among trending tables: wear, onset, commissioning, maintenance sawtooth
# and level-shift curves instead of the linear/broken/multiplicative kinds.
LIFECYCLE_TREND_PROBABILITY = 0.35
# Closed-loop units, among trend-free tables. The classic outcome becomes the
# unit's demand; the recorded outcome is its level/pressure/controller output.
# Budget influential covariate sources; weak secondary drivers remain possible.
MAX_DOMINANT_DRIVERS = 2  # None disables the excitation budget.
PERSISTENT_HIDDEN_PROBABILITY = 0.8
SHARED_STATE_PROBABILITY = 0.2  # One categorical mode drives multiple channels.
PLANT_PROBABILITY = 0.85
# A totalizer (running total of a flow) only counts up, so it is drawn for
# TRENDING tables instead, with probability PLANT_PROBABILITY x its weight share.
PLANT_WEIGHTS = {
    "relay": 0.4,
    "pid": 0.25,
    "batch": 0.2,
    "step_response": 0.15,
    "totalizer": 0.1,
    "stick_slip": 0.08,
    "interlock": 0.08,
    "coupled_storage": 0.10,
    "rotating_machine": 0.10,
    # Patterns of recorded operational data (operational_patterns.py). They
    # take the place of a unit: the SCM outcome is their demand.
    "held_value": 1.0,
    "daily_load": 1.2,
    "daylight": 0.3,
    "duty_cycle": 0.4,
    "random_walk": 0.4,
    "smooth_drift": 0.7,
    "counts": 0.5,
    "staircase": 0.4,
    "oscillation": 0.2,
    "resource_use": 0.5,
}
# Recording layer, applied after simulation (see measurement_layer.py).
REDUNDANT_SENSOR_PROBABILITY = 0.12  # Per table: 1-2 extra sensors on a signal.
MEASUREMENT_PROBABILITY = 0.2  # 20% per continuous column: 1-3 sensor artifacts.
ARTIFACT_WEIGHTS = {}  # noise/quantize/clip/deadband/hold/stuck/spikes/dropout/...
MISSINGNESS_PROBABILITY = 0.15  # Per table: one missing-value mechanism (NaN).
# mcar: independent of all values. mar: depends on an observed column, the clock
# or the previous reading. structural: periodic gaps, slower sampling, late
# start, early end, whole-logger outage, out-of-range values not reported.
MISSINGNESS_WEIGHTS = {"mcar": 0.35, "mar": 0.25, "structural": 0.40}
ROW_GAP_PROBABILITY = 0.04  # Rows absent from the file: the series jumps.
# Stored resolution, per continuous column: a fixed number of distinct levels over the signal's range (measurement_layer.RESOLUTION_LEVELS).
RESOLUTION_PROBABILITY = 0.6
# Per continuous column: the reading follows the process with a response time of 0.7 to 15 points (a damped transmitter, or a logger that stores interval means).
DAMPING_PROBABILITY = 0.3
# Among the nodes whose noise changes size: the share in which calm and busy stretches come and go (the others grow, cycle or change once).
VOLATILITY_BURST_SHARE = 0.5
# Among the tables with a trend: the share in which an operational pattern rides on the trend (a growing load, a drifting setpoint).
TRENDING_PATTERN_PROBABILITY = 0.5
# Cycle lengths of the causal model's outcome follow the calendar at common
# sampling rates (a day of 24, 48, 96, 144 or 288 points most often, a week of
# 7 or 168), instead of six lengths with equal chances.
CALENDAR_PERIODS = True
# Per cycle of a season of the causal model: every repetition gets its own size (lognormal, spread 0.05 to 0.6).
CYCLE_JITTER_PROBABILITY = 0.5
# Resample tables whose outcome never moves (or moves by less than float32 can
# hold). Flat STRETCHES inside a table (shutdown, stuck sensor, clipping) stay.
REJECT_CONSTANT_OUTCOME = True
BURN_IN = 128  # Unrecorded history for warm-up and valid initial lag columns.
SAVE_SCM = False
SAVE_PLOT = False
SAVE_DAG = False  # Causal structure with hidden nodes and explicit history proxies.
PLOT_COVARIATES = False  # False plots only the outcome.
RENDER_WORKERS = (
    None  # Automatically use up to four plot processes; 1 renders serially.
)
# None picks half the logical processors (at most eight): six on this machine.
# All twelve measured 105 instead of 76 tables/s, with identical tables; use
# None to keep the machine responsive. 1 keeps the old serial RNG stream.
GENERATION_WORKERS = 10
GENERATION_BATCH_SIZE = 16  # Tables per task; bounded queues avoid large memory spikes.
PROGRESS_INTERVAL = 1.0  # Seconds between elapsed/rate/ETA console updates.
COARSE_COPY_PROBABILITY = 0.03  # Extra thinned copy, only for originals with >512 rows.
AGGREGATE_COPY_PROBABILITY = (
    0.03  # One causal min/max/mean copy; windows 2-5, >512 rows.
)
SIGN_INVERSION_PROBABILITY = 0.05  # Extra copy negating all values except time_index.


sys.path.insert(0, str(Path(__file__).resolve().parent))
from scm_generator import OperationalSCMGenerator


def generator_settings():
    """The settings above as OperationalSCMGenerator keyword arguments.

    Other tools (the annealing-set builder) use this to draw tables from
    exactly the distribution the training corpus was generated with.
    """
    return dict(
        seed=SEED,
        min_rows=MIN_ROWS,
        max_rows=MAX_ROWS,
        min_covariates=MIN_COVARIATES,
        max_covariates=MAX_COVARIATES,
        max_lag=MAX_LAG,
        max_lag_count=MAX_LAG_COUNT,
        visible_covariate_lambda=VISIBLE_COVARIATE_LAMBDA,
        edge_probability=EDGE_PROBABILITY,
        burn_in=BURN_IN,
        direct_probability=DIRECT_OUTCOME_PROBABILITY,
        delayed_probability=OUTCOME_FEEDBACK_PROBABILITY,
        drift_probability=DRIFT_PROBABILITY,
        walk_probability=RANDOM_WALK_PROBABILITY,
        noise_free_covariate_probability=NOISE_FREE_COVARIATE_PROBABILITY,
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
        process_episode_probability=PROCESS_EPISODE_PROBABILITY,
        process_multilevel_probability=PROCESS_MULTILEVEL_PROBABILITY,
        process_priming_probability=PROCESS_PRIMING_PROBABILITY,
        formula_covariate_probability=FORMULA_COVARIATE_PROBABILITY,
        discrete_covariate_probability=DISCRETE_COVARIATE_PROBABILITY,
        sensor_bounds_probability=SENSOR_BOUNDS_PROBABILITY,
        discrete_weights=DISCRETE_WEIGHTS,
        switch_profile_weights=SWITCH_PROFILE_WEIGHTS,
        max_effect_components=MAX_EFFECT_COMPONENTS,
        composite_effect_probability=COMPOSITE_EFFECT_PROBABILITY,
        angle_covariate_probability=ANGLE_COVARIATE_PROBABILITY,
        effect_weights=EFFECT_WEIGHTS,
        parent_combination_weights=PARENT_COMBINATION_WEIGHTS,
        node_link_weights=NODE_LINK_WEIGHTS,
        pairwise_interaction_probability=PAIRWISE_INTERACTION_PROBABILITY,
        additive_interaction_probability=ADDITIVE_INTERACTION_PROBABILITY,
        joint_numeric_limit=JOINT_NUMERIC_LIMIT,
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
        render_workers=RENDER_WORKERS,
        generation_workers=GENERATION_WORKERS,
        generation_batch_size=GENERATION_BATCH_SIZE,
        progress_interval=PROGRESS_INTERVAL,
        coarse_copy_probability=COARSE_COPY_PROBABILITY,
        aggregate_copy_probability=AGGREGATE_COPY_PROBABILITY,
        sign_inversion_probability=SIGN_INVERSION_PROBABILITY,
        full_length_probability=FULL_LENGTH_PROBABILITY,
        prehistory_probability=PREHISTORY_PROBABILITY,
        max_prehistory_rows=MAX_PREHISTORY_ROWS,
        row_gap_probability=ROW_GAP_PROBABILITY,
        seasonal_table_probability=SEASONAL_TABLE_PROBABILITY,
        nonseasonal_covariate_season_probability=NONSEASONAL_COVARIATE_SEASON_PROBABILITY,
        forced_season_probability=FORCED_SEASON_PROBABILITY,
        noise_profile_weights=NOISE_PROFILE_WEIGHTS,
        slow_dynamics_probability=SLOW_DYNAMICS_PROBABILITY,
        transport_delay_probability=TRANSPORT_DELAY_PROBABILITY,
        step_response_probability=STEP_RESPONSE_PROBABILITY,
        lifecycle_trend_probability=LIFECYCLE_TREND_PROBABILITY,
        calendar_gate_probability=CALENDAR_GATE_PROBABILITY,
        plant_probability=PLANT_PROBABILITY,
        plant_weights=PLANT_WEIGHTS,
        max_dominant_drivers=MAX_DOMINANT_DRIVERS,
        persistent_hidden_probability=PERSISTENT_HIDDEN_PROBABILITY,
        shared_state_probability=SHARED_STATE_PROBABILITY,
        startup_transient_probability=STARTUP_TRANSIENT_PROBABILITY,
        redundant_sensor_probability=REDUNDANT_SENSOR_PROBABILITY,
        measurement_probability=MEASUREMENT_PROBABILITY,
        artifact_weights=ARTIFACT_WEIGHTS,
        missingness_probability=MISSINGNESS_PROBABILITY,
        missingness_weights=MISSINGNESS_WEIGHTS,
        nonseasonal_wander_probability=NONSEASONAL_WANDER_PROBABILITY,
        seasonal_wander_probability=SEASONAL_WANDER_PROBABILITY,
        resolution_probability=RESOLUTION_PROBABILITY,
        damping_probability=DAMPING_PROBABILITY,
        volatility_burst_share=VOLATILITY_BURST_SHARE,
        trending_pattern_probability=TRENDING_PATTERN_PROBABILITY,
        calendar_periods=CALENDAR_PERIODS,
        cycle_jitter_probability=CYCLE_JITTER_PROBABILITY,
        reject_constant_outcome=REJECT_CONSTANT_OUTCOME,
    )


def main():
    generator = OperationalSCMGenerator(OUTPUT_DIR, **generator_settings())
    paths = generator.generate(NUM_TABLES)
    print(f"Saved {len(paths)} tables under {paths[0].parent}")


if __name__ == "__main__":
    main()

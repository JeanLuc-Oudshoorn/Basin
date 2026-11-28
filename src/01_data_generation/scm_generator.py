"""Composable dynamic SCMs for diverse synthetic operational time series.

Sampling ranges are priors, not claims about a particular industry. All sampled
parameters are saved. Evaluation resamples innovations, never the equations.
"""

import copy
import json
import math
import os
import sys
import uuid
from collections import deque
from concurrent.futures import FIRST_COMPLETED, ProcessPoolExecutor, wait
from datetime import date
from multiprocessing import get_context
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.signal import lfilter

try:
    from .generation_runtime import (
        GenerationProgress,
        aggregate_frame,
        coarsen_frame,
        generate_batch,
        generate_worker_batch,
        initialize_worker,
        invert_frame,
    )
except ImportError:
    from generation_runtime import (
        GenerationProgress,
        aggregate_frame,
        coarsen_frame,
        generate_batch,
        generate_worker_batch,
        initialize_worker,
        invert_frame,
    )

try:
    from .process_mechanisms import (
        DISCRETE_KINDS,
        EXOGENOUS_KINDS,
        FORMULA_FAMILIES,
        discrete_path,
        episode_envelope,
        formula_path,
        sensor_reading,
    )
except ImportError:
    from process_mechanisms import (
        DISCRETE_KINDS,
        EXOGENOUS_KINDS,
        FORMULA_FAMILIES,
        discrete_path,
        episode_envelope,
        formula_path,
        sensor_reading,
    )

try:
    from .effect_mechanisms import (
        STATIC_OPERATORS,
        TEMPORAL_OPERATORS,
        bind_node_link,
        combine_parents,
        compile_program,
        compile_steps,
        prepare_parent_combination,
        run_steps,
    )
except ImportError:
    from effect_mechanisms import (
        STATIC_OPERATORS,
        TEMPORAL_OPERATORS,
        bind_node_link,
        combine_parents,
        compile_program,
        compile_steps,
        prepare_parent_combination,
        run_steps,
    )

try:
    from .measurement_layer import (
        DEFAULT_ARTIFACT_WEIGHTS,
        DEFAULT_MISSINGNESS_WEIGHTS,
        apply_observation,
        recorded_window,
        sample_observation,
        sample_row_gaps,
    )
    from .plant_mechanisms import (
        DEFAULT_PLANT_WEIGHTS,
        LIFECYCLE_KINDS,
        lifecycle_path,
        physical_dag_view,
        sample_lifecycle,
        sample_plant,
        simulate_plant,
    )
except ImportError:
    from measurement_layer import (
        DEFAULT_ARTIFACT_WEIGHTS,
        DEFAULT_MISSINGNESS_WEIGHTS,
        apply_observation,
        recorded_window,
        sample_observation,
        sample_row_gaps,
    )
    from plant_mechanisms import (
        DEFAULT_PLANT_WEIGHTS,
        LIFECYCLE_KINDS,
        lifecycle_path,
        physical_dag_view,
        sample_lifecycle,
        sample_plant,
        simulate_plant,
    )


def _render_plot_job(generator, model, frame, directory):
    """Independent rendering job; workers never sample SCMs or observations."""
    if generator.save_plot:
        generator._plot(frame, directory / "series.png")
    if generator.save_dag:
        generator._plot_dag(model, directory / "dag.png")


class OperationalSCMGenerator:
    # Override these ranges with prior_ranges. Distribution identities and
    # numerical safeguards, unlike model parameters, remain fixed in evaluation.
    DEFAULT_PRIORS = {
        "noise_scale": (0.02, 0.6),
        "noise_location": (-0.3, 0.3),
        "student_df": (2.1, 35.0),
        "beta_shape": (0.3, 8.0),
        "weibull_shape": (0.5, 5.0),
        "poisson_rate": (0.2, 40.0),
        "lognormal_sigma": (0.15, 1.3),
        "quantization_step": (0.01, 0.8),
        "mixture_separation": (0.5, 3.5),
        "component_scale": (0.3, 1.6),
        "contamination_probability": (0.01, 0.15),
        "outlier_scale": (3.0, 12.0),
        "edge_strength": (0.005, 3.0),
        "input_scale": (0.5, 20.0),
        "effect_gain": (0.2, 2.0),
        "effect_offset": (-2.0, 2.0),
        "effect_limit": (2.0, 12.0),
        "persistence": (-0.7, 0.97),
        "seasonal_period": (6.0, 512.0),
        "seasonal_amplitude": (0.05, 3.0),
        "drift_magnitude": (0.0005, 0.06),
        "walk_scale": (0.001, 0.12),
        "slope_walk_scale": (0.00001, 0.0005),
        "event_probability": (0.002, 0.05),
        "decay": (0.75, 0.998),
        "shock_magnitude": (0.1, 5.0),
        "ar_mass": (0.1, 0.94),
        "baseline": (10.0, 500.0),
        "outcome_scale": (0.5, 20.0),
        "perlin_frequency": (0.005, 0.2),
        "perlin_offset": (0.1, 250.0),
        "perlin_persistence": (0.25, 0.8),
        "perlin_lacunarity": (1.5, 3.0),
        "perlin_octaves": (1, 6),
        "pink_exponent": (0.8, 1.2),
        "pink_filter_length": (256, 2048),
        "inverse_square_floor": (0.01, 0.3),
        "brown_step_scale": (0.01, 0.15),
        "damping_rate": (0.05, 2.0),
        "oscillation_frequency": (0.3, 4.0),
        "ringing_decay": (0.002, 0.1),
        "ringing_period": (6.0, 128.0),
        "multiplicative_log_change": (0.7, 2.5),
        "relative_noise_scale": (0.025, 0.10),
        "relative_seasonal_amplitude": (0.04, 0.18),
        "variance_growth_factor": (1.5, 5.0),
        "recent_lag_decay": (0.5, 1.2),
        "effect_k": (0.2, 4.0),
        "effect_width": (0.2, 3.0),
        "effect_threshold": (0.05, 2.0),
        "effect_shape": (0.5, 4.0),
        "effect_leak": (0.01, 0.25),
        "effect_floor": (0.01, 0.2),
        "effect_tau": (1.0, 30.0),
        "effect_rate": (0.02, 1.5),
        "effect_pid_i": (0.01, 0.5),
        "effect_pid_d": (0.01, 0.3),
        "effect_kernel_spread": (1.2, 3.0),
        "effect_delay_steps": (1, 8),
        "effect_kernel_stages": (2, 5),
        "joint_strength": (0.2, 4.0),
        "joint_mix_concentration": (0.3, 3.0),
        "joint_power": (0.2, 1.8),
        "joint_mlp_weight_scale": (0.3, 1.5),
        "joint_mlp_width": (3, 12),
        "binary_probability": (0.03, 0.97),
        "markov_state_count": (3, 6),
        "markov_dwell_fraction": (0.02, 0.5),
        "markov_concentration": (0.2, 2.0),
        "markov_level_scale": (0.5, 10.0),
        "episode_start_fraction": (0.08, 0.35),
        "episode_stop_fraction": (0.58, 0.82),
        "episode_ramp_fraction": (0.05, 0.18),
        "episode_amplitude": (2.0, 12.0),
        "episode_detail_fraction": (0.05, 0.35),
        "episode_noise_fraction": (0.005, 0.08),
        "episode_idle_noise_fraction": (0.0001, 0.003),
        "process_idle_fraction": (0.0, 0.8),
        "process_ramp_fraction": (0.005, 0.5),
        "process_active_fraction": (0.08, 0.8),
        "process_level_count": (2, 5),
        "process_level": (0.2, 1.6),
        "process_transition_fraction": (0.005, 0.15),
        "process_curvature": (0.4, 4.0),
        "process_priming_fraction": (0.005, 0.12),
        "formula_amplitude": (0.3, 6.0),
        "formula_offset": (-2.0, 2.0),
        "formula_power": (0.35, 3.5),
        "formula_center": (0.1, 0.9),
        "formula_width_fraction": (0.02, 0.35),
        "formula_period_fraction": (0.04, 0.7),
        "formula_decay": (0.2, 8.0),
        "formula_chirp": (-0.8, 3.0),
        "formula_noise_fraction": (0.005, 0.15),
        "sensor_bound_distance": (0.2, 1.5),
        "switch_long_on_fraction": (0.1, 0.7),
        "switch_long_off_fraction": (0.2, 1.2),
    }
    DEFAULT_DISCRETE_WEIGHTS = {
        "bernoulli": 0.35,
        "binary_markov": 0.35,
        "categorical_markov": 0.30,
    }
    DEFAULT_SWITCH_PROFILE_WEIGHTS = {
        "alternating": 0.4,
        "long_dwell": 0.5,
        "single_episode": 0.1,
    }
    BASE_NOISES = (
        "normal",
        "student_t",
        "laplace",
        "uniform",
        "skewed",
        "poisson",
        "beta",
        "weibull",
        "quantization",
        "perlin",
        "pink",
        "brown",
    )
    MIXTURES = (
        "student_t_mixture",
        "normal_mixture",
        "heterogeneous_mixture",
        "contamination",
    )
    NOISES = BASE_NOISES + MIXTURES
    # Relative weights, not percentages: the three preferred families receive four
    # times the probability of each other family (16% each with these defaults).
    DEFAULT_NOISE_WEIGHTS = {
        name: (4.0 if name in ("perlin", "student_t", "pink") else 1.0)
        for name in NOISES
    }
    DEFAULT_FOURIER_WEIGHTS = {
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
    OPERATORS = (
        (
            "affine",
            "sine",
            "cosine",
            "damped_sine",
            "damped_cosine",
            "tangent",
            "sigmoid",
            "exponential",
            "power",
            "inverse_square",
            "log",
            "relu",
            "mlp",
            "add",
            "multiply",
            "gate",
        )
        + STATIC_OPERATORS
        + TEMPORAL_OPERATORS
    )
    DEFAULT_EFFECT_WEIGHTS = dict.fromkeys(OPERATORS, 2.0)
    DEFAULT_EFFECT_WEIGHTS.update(
        dict.fromkeys(
            (
                "affine",
                "saturation",
                "power",
                "exponential",
                "arrhenius",
                "antoine",
                "relu",
                "deadband",
                "clip",
                "knee",
                "leaky_relu",
                "log",
                "decreasing_quadratic",
                "quadratic_friction",
                "stefan_boltzmann",
                "first_order_lag",
                "delay",
                "integrator",
                "min",
                "max",
                "soft_min",
                "multiply",
                "ratio",
                "add",
            ),
            8.0,
        )
    )
    DEFAULT_EFFECT_WEIGHTS.update(
        dict.fromkeys(
            (
                "sine",
                "cosine",
                "tangent",
                "damped_sine",
                "damped_cosine",
                "inverse_square",
                "swiglu",
                "mlp",
            ),
            0.15,
        )
    )
    # Opt-in (appended last, weight zero): earlier seeds keep their draws.
    DEFAULT_EFFECT_WEIGHTS["second_order"] = 0.0
    DEFAULT_PARENT_COMBINATION_WEIGHTS = {
        "sum": 16,
        "product": 4,
        "saddle": 0.2,
        "ratio": 3,
        "min": 3,
        "max": 3,
        "soft_min": 3,
        "gate": 1,
        "exponent_gate": 0.5,
        "threshold": 1,
        "convex_mix": 3,
        "mlp": 0.5,
        "log_sum": 1,
        "regime_switch": 1,
        "volatility_coupling": 1,
        "swiglu": 0.15,
    }
    NOISE_PROFILES = ("clean", "low", "moderate", "noisy")
    DEFAULT_NODE_LINK_WEIGHTS = {
        "identity": 16,
        "saturation": 4,
        "softplus": 2,
        "exponential": 1,
        "sigmoid": 1,
        "logarithm": 1,
    }

    def __init__(
        self,
        output_dir,
        *,
        seed=42,
        min_rows=20,
        max_rows=2056,
        min_covariates=8,
        max_covariates=18,
        max_lag=192,
        max_lag_count=6,
        visible_covariate_lambda=1.3,
        edge_probability=0.18,
        direct_probability=0.65,
        delayed_probability=0.15,
        drift_probability=0.5,
        walk_probability=0.15,
        noise_free_covariate_probability=0.10,
        trend_free_fraction=0.5,
        residual_history_probability=0.08,
        switching_probability=0.3,
        process_episode_probability=0.02,
        process_multilevel_probability=0.55,
        process_priming_probability=0.30,
        formula_covariate_probability=0.08,
        discrete_covariate_probability=0.25,
        sensor_bounds_probability=0.01,
        discrete_weights=None,
        switch_profile_weights=None,
        multiplicative_trend_probability=0.1,
        multiplicative_constant_noise_probability=0.05,
        multiplicative_constant_seasonality_probability=0.05,
        variance_dynamics_probability=0.55,
        seasonal_lag_probability=0.8,
        long_lag_probability=0.20,
        instantaneous_probability=0.10,
        covariate_delay_scale=0.75,
        max_covariate_lag=24,
        fourier_probability=0.85,
        max_fourier_harmonics=16,
        seasonal_shape_drift_probability=0.25,
        fourier_weights=None,
        max_effect_components=8,
        composite_effect_probability=0.25,
        angle_covariate_probability=0.05,
        effect_weights=None,
        parent_combination_weights=None,
        node_link_weights=None,
        pairwise_interaction_probability=0.85,
        additive_interaction_probability=0.35,
        joint_numeric_limit=1e6,
        prior_ranges=None,
        noise_weights=None,
        burn_in=256,
        save_scm=True,
        save_plot=True,
        plot_covariates=True,
        save_dag=True,
        render_workers=None,
        generation_workers=1,
        generation_batch_size=16,
        progress_interval=1.0,
        coarse_copy_probability=0.03,
        aggregate_copy_probability=0.0,
        sign_inversion_probability=0.03,
        full_length_probability=0.0,
        prehistory_probability=0.0,
        max_prehistory_rows=2048,
        row_gap_probability=0.0,
        seasonal_table_probability=1.0,
        nonseasonal_covariate_season_probability=0.15,
        forced_season_probability=0.85,
        noise_profile_weights=None,
        slow_dynamics_probability=0.0,
        transport_delay_probability=0.0,
        step_response_probability=0.0,
        lifecycle_trend_probability=0.0,
        calendar_gate_probability=0.0,
        plant_probability=0.0,
        plant_weights=None,
        max_dominant_drivers=None,
        persistent_hidden_probability=0.0,
        shared_state_probability=0.0,
        startup_transient_probability=0.0,
        redundant_sensor_probability=0.0,
        measurement_probability=0.0,
        artifact_weights=None,
        missingness_probability=0.0,
        missingness_weights=None,
        nonseasonal_wander_probability=0.0,
        seasonal_wander_probability=0.0,
        resolution_probability=0.0,
        damping_probability=0.0,
        volatility_burst_share=0.0,
        reject_constant_outcome=False,
    ):
        settings = locals().copy()
        self._auto_render_workers = render_workers is None
        if render_workers is None:
            render_workers = min(4, max(1, (os.cpu_count() or 2) // 2))
        if (
            isinstance(render_workers, bool)
            or not isinstance(render_workers, int)
            or render_workers < 1
        ):
            raise ValueError("render_workers must be a positive integer or None")
        self.render_workers = render_workers
        self._seeded_generation = generation_workers != 1
        if generation_workers is None:
            generation_workers = min(8, max(1, (os.cpu_count() or 2) // 2))
        for name, value in (
            ("generation_workers", generation_workers),
            ("generation_batch_size", generation_batch_size),
        ):
            if isinstance(value, bool) or not isinstance(value, int) or value < 1:
                raise ValueError(f"{name} must be a positive integer")
            setattr(self, name, value)
        if not np.isfinite(progress_interval) or progress_interval <= 0:
            raise ValueError("progress_interval must be finite and positive")
        self.progress_interval = float(progress_interval)
        for name in (
            "min_rows",
            "max_rows",
            "min_covariates",
            "max_covariates",
            "max_lag",
            "max_lag_count",
            "max_covariate_lag",
            "burn_in",
            "max_effect_components",
            "max_fourier_harmonics",
        ):
            value = settings[name]
            if isinstance(value, bool) or not isinstance(value, int) or value < 1:
                raise ValueError(f"{name} must be a positive integer")
            setattr(self, name, value)
        if min_rows > max_rows or not 3 <= min_covariates <= max_covariates:
            raise ValueError(
                "Invalid row/covariate bounds; at least 3 covariates required"
            )
        for name in (
            "edge_probability",
            "direct_probability",
            "delayed_probability",
            "drift_probability",
            "walk_probability",
            "noise_free_covariate_probability",
            "trend_free_fraction",
            "residual_history_probability",
            "switching_probability",
            "process_episode_probability",
            "process_multilevel_probability",
            "process_priming_probability",
            "formula_covariate_probability",
            "discrete_covariate_probability",
            "sensor_bounds_probability",
            "coarse_copy_probability",
            "aggregate_copy_probability",
            "sign_inversion_probability",
            "multiplicative_trend_probability",
            "variance_dynamics_probability",
            "multiplicative_constant_noise_probability",
            "multiplicative_constant_seasonality_probability",
            "seasonal_lag_probability",
            "long_lag_probability",
            "instantaneous_probability",
            "fourier_probability",
            "seasonal_shape_drift_probability",
            "composite_effect_probability",
            "angle_covariate_probability",
            "pairwise_interaction_probability",
            "additive_interaction_probability",
            "full_length_probability",
            "prehistory_probability",
            "row_gap_probability",
            "seasonal_table_probability",
            "nonseasonal_covariate_season_probability",
            "forced_season_probability",
            "slow_dynamics_probability",
            "transport_delay_probability",
            "step_response_probability",
            "lifecycle_trend_probability",
            "calendar_gate_probability",
            "plant_probability",
            "persistent_hidden_probability",
            "shared_state_probability",
            "startup_transient_probability",
            "redundant_sensor_probability",
            "measurement_probability",
            "missingness_probability",
            "nonseasonal_wander_probability",
            "seasonal_wander_probability",
            "resolution_probability",
            "damping_probability",
            "volatility_burst_share",
        ):
            if not 0 <= settings[name] <= 1:
                raise ValueError(f"{name} must lie in [0, 1]")
            setattr(self, name, settings[name])
        if (
            isinstance(visible_covariate_lambda, bool)
            or not np.isfinite(visible_covariate_lambda)
            or visible_covariate_lambda < 0
        ):
            raise ValueError("visible_covariate_lambda must be finite and nonnegative")
        self.visible_covariate_lambda = float(visible_covariate_lambda)
        if max_dominant_drivers is not None and (
            isinstance(max_dominant_drivers, bool)
            or not isinstance(max_dominant_drivers, int)
            or max_dominant_drivers < 1
        ):
            raise ValueError("max_dominant_drivers must be None or a positive integer")
        self.max_dominant_drivers = max_dominant_drivers
        if not np.isfinite(covariate_delay_scale) or covariate_delay_scale <= 0:
            raise ValueError("covariate_delay_scale must be finite and positive")
        self.covariate_delay_scale = float(covariate_delay_scale)
        if not np.isfinite(joint_numeric_limit) or joint_numeric_limit <= 0:
            raise ValueError("joint_numeric_limit must be finite and positive")
        self.joint_numeric_limit = float(joint_numeric_limit)
        for attribute, defaults, overrides in (
            ("effect_weights", self.DEFAULT_EFFECT_WEIGHTS, effect_weights),
            (
                "parent_combination_weights",
                self.DEFAULT_PARENT_COMBINATION_WEIGHTS,
                parent_combination_weights,
            ),
            ("node_link_weights", self.DEFAULT_NODE_LINK_WEIGHTS, node_link_weights),
            ("discrete_weights", self.DEFAULT_DISCRETE_WEIGHTS, discrete_weights),
            (
                "switch_profile_weights",
                self.DEFAULT_SWITCH_PROFILE_WEIGHTS,
                switch_profile_weights,
            ),
        ):
            weights = dict(defaults)
            for name, weight in (overrides or {}).items():
                if name not in weights or not np.isfinite(weight) or weight < 0:
                    raise ValueError(f"Invalid {attribute} entry: {name}")
                weights[name] = float(weight)
            if sum(weights.values()) <= 0:
                raise ValueError(f"At least one {attribute} entry must be positive")
            setattr(self, attribute, weights)
        self.fourier_weights = dict(self.DEFAULT_FOURIER_WEIGHTS)
        for name, weight in (fourier_weights or {}).items():
            if (
                name not in self.fourier_weights
                or not np.isfinite(weight)
                or weight < 0
            ):
                raise ValueError(f"Invalid Fourier family weight: {name}")
            self.fourier_weights[name] = float(weight)
        if sum(self.fourier_weights.values()) <= 0:
            raise ValueError("At least one Fourier family must have positive weight")
        self.priors = dict(self.DEFAULT_PRIORS)
        for name, bounds in (prior_ranges or {}).items():
            if name not in self.priors:
                raise ValueError(f"Unknown prior: {name}")
            if (
                len(bounds) != 2
                or not np.isfinite(bounds).all()
                or bounds[0] > bounds[1]
            ):
                raise ValueError(f"Invalid bounds for {name}")
            self.priors[name] = tuple(bounds)
        signed = {
            "noise_location",
            "effect_offset",
            "persistence",
            "baseline",
            "formula_offset",
            "formula_chirp",
        }
        for name, (low, high) in self.priors.items():
            if name == "process_idle_fraction" and 0 <= low <= high < 1:
                continue
            if name not in signed and low <= 0:
                raise ValueError(f"{name} bounds must be positive")
        if self.priors["student_df"][0] <= 2:
            raise ValueError(
                "student_df must exceed 2 for finite variance normalization"
            )
        for name in (
            "decay",
            "ar_mass",
            "event_probability",
            "contamination_probability",
            "effect_floor",
            "effect_leak",
        ):
            if self.priors[name][1] >= 1:
                raise ValueError(f"{name} must be below 1")
        for name in (
            "effect_delay_steps",
            "effect_kernel_stages",
            "joint_mlp_width",
            "markov_state_count",
            "process_level_count",
        ):
            low, high = self.priors[name]
            if int(low) != low or int(high) != high or high > 1024:
                raise ValueError(f"{name} must have integer bounds within [1, 1024]")
        if self.priors["markov_state_count"][0] < 3:
            raise ValueError(
                "markov_state_count must be at least three; binary chains have their own family"
            )
        for name in (
            "binary_probability",
            "episode_start_fraction",
            "episode_stop_fraction",
            "episode_ramp_fraction",
            "episode_detail_fraction",
        ):
            if self.priors[name][1] >= 1:
                raise ValueError(f"{name} must be below one")
        if (
            self.priors["episode_start_fraction"][1]
            >= self.priors["episode_stop_fraction"][0]
        ):
            raise ValueError("Episode start bounds must precede stop bounds")
        low, high = self.priors["perlin_octaves"]
        if int(low) != low or int(high) != high or low < 1 or high > 16:
            raise ValueError("perlin_octaves must have integer bounds within [1, 16]")
        if self.priors["perlin_persistence"][1] > 1:
            raise ValueError("perlin_persistence must be at most 1")
        low, high = self.priors["pink_filter_length"]
        if int(low) != low or int(high) != high or low < 2 or high > 65536:
            raise ValueError(
                "pink_filter_length must have integer bounds within [2, 65536]"
            )
        if self.priors["pink_exponent"][1] >= 2:
            raise ValueError("pink_exponent must be below 2")
        if self.priors["multiplicative_log_change"][1] > 20:
            raise ValueError(
                "multiplicative_log_change must be at most 20 to bound exponential growth"
            )
        if not -1 < self.priors["persistence"][0] <= self.priors["persistence"][1] < 1:
            raise ValueError("persistence must be within (-1, 1)")
        self.output_dir, self.rng = Path(output_dir), np.random.default_rng(seed)
        self.noise_weights = dict(self.DEFAULT_NOISE_WEIGHTS)
        for name, weight in (noise_weights or {}).items():
            if name not in self.NOISES or not np.isfinite(weight) or weight < 0:
                raise ValueError(f"Invalid noise weight: {name}={weight}")
            self.noise_weights[name] = float(weight)
        if not any(self.noise_weights.values()):
            raise ValueError("At least one noise family must have positive weight")
        # Heterogeneous mixtures draw only base families. If all base weights
        # are zero, that mixture cannot be sampled and must also be disabled.
        if self.noise_weights["heterogeneous_mixture"] and not any(
            self.noise_weights[k] for k in self.BASE_NOISES
        ):
            raise ValueError(
                "Heterogeneous mixtures require a positive base-family weight"
            )
        self.burn_in = max(burn_in, max_lag, max_covariate_lag)
        self.save_scm, self.save_plot = save_scm, save_plot
        self.plot_covariates, self.save_dag = plot_covariates, save_dag
        if (
            isinstance(max_prehistory_rows, bool)
            or not isinstance(max_prehistory_rows, int)
            or max_prehistory_rows < 64
        ):
            raise ValueError("max_prehistory_rows must be an integer of at least 64")
        self.max_prehistory_rows = max_prehistory_rows
        self.reject_constant_outcome = bool(reject_constant_outcome)
        for attribute, defaults, overrides in (
            ("plant_weights", DEFAULT_PLANT_WEIGHTS, plant_weights),
            ("artifact_weights", DEFAULT_ARTIFACT_WEIGHTS, artifact_weights),
            ("missingness_weights", DEFAULT_MISSINGNESS_WEIGHTS, missingness_weights),
        ):
            weights = dict(defaults)
            for name, weight in (overrides or {}).items():
                if name not in weights or not np.isfinite(weight) or weight < 0:
                    raise ValueError(f"Invalid {attribute} entry: {name}")
                weights[name] = float(weight)
            if sum(weights.values()) <= 0:
                raise ValueError(f"At least one {attribute} entry must be positive")
            setattr(self, attribute, weights)
        # None keeps the original outcome noise floor, U(0.15, 0.45).
        self.noise_profile_weights = None
        if noise_profile_weights is not None:
            weights = dict.fromkeys(self.NOISE_PROFILES, 0.0)
            for name, weight in noise_profile_weights.items():
                if name not in weights or not np.isfinite(weight) or weight < 0:
                    raise ValueError(f"Invalid noise_profile_weights entry: {name}")
                weights[name] = float(weight)
            if sum(weights.values()) <= 0:
                raise ValueError("At least one noise profile must be positive")
            self.noise_profile_weights = weights

    def _draw(self, name, *, log=False, signed=False):
        low, high = self.priors[name]
        value = (
            np.exp(self.rng.uniform(np.log(low), np.log(high)))
            if log
            else self.rng.uniform(low, high)
        )
        return float(value * (self.rng.choice([-1, 1]) if signed else 1))

    def _log_uniform(self, low, high):
        return float(np.exp(self.rng.uniform(np.log(low), np.log(high))))

    def _sample_noise_profile(self):
        """One noise budget per table: most real sensors are far cleaner than
        the original fixed floor, a few are almost pure noise."""
        r = self.rng
        names = list(self.noise_profile_weights)
        weights = np.array(list(self.noise_profile_weights.values()), dtype=float)
        name = str(r.choice(names, p=weights / weights.sum()))
        if name == "clean":
            white, factor = 0.0, self._log_uniform(0.003, 0.03)
        elif name == "low":
            white, factor = self._log_uniform(0.004, 0.05), self._log_uniform(0.03, 0.3)
        elif name == "moderate":
            white, factor = float(r.uniform(0.15, 0.45)), 1.0
        else:
            white, factor = float(r.uniform(0.6, 2.0)), float(r.uniform(1.0, 3.0))
        return {"name": name, "white_noise_scale": white, "innovation_factor": factor}

    def _response_program(self):
        """One clean settling operator: lag, cascade, or second-order overshoot."""
        r = self.rng
        return [
            {
                "op": str(
                    r.choice(
                        ["first_order_lag", "erlang", "second_order"],
                        p=[0.45, 0.2, 0.35],
                    )
                ),
                "a": 0,
                "b": 0,
                "gain": 1.0,
                "offset": 0.0,
                "limit": 12.0,
                # Read by every stateful operator; unused by these three.
                "k": 1.0,
                "width": 1.0,
                "threshold": 1.0,
                "floor": 0.05,
                "tau": self._log_uniform(3, 250),
                "stages": int(r.integers(2, 5)),
                "spread": 1.5,
                "period": self._log_uniform(8, 300),
                "damping": self._log_uniform(0.04, 0.9),
            }
        ]

    def _sample_noise(self, kind=None):
        """Separate shape, location, and scale so tail changes need not change SNR."""
        r = self.rng
        kind = self._choose_noise(self.NOISES) if kind is None else kind
        kind = "brown" if kind == "red" else kind
        spec = {
            "kind": kind,
            "location": self._draw("noise_location"),
            "scale": self._draw("noise_scale", log=True),
        }
        if kind in self.MIXTURES:
            count = (
                2
                if kind in ("student_t_mixture", "contamination")
                else int(r.integers(2, 5))
            )
            families = {
                "student_t_mixture": ["student_t"] * count,
                "normal_mixture": ["normal"] * count,
                "contamination": ["normal", "student_t"],
            }
            chosen = (
                families[kind]
                if kind in families
                else [self._choose_noise(self.BASE_NOISES) for _ in range(count)]
            )
            components = [self._sample_noise(k) for k in chosen]
            separation = self._draw("mixture_separation")
            # Shifted components allow overlapping modes, rather than only a
            # zero-centered scale mixture. Avoid empirical sample recentering.
            for i, component in enumerate(components):
                component.update(
                    location=float((i - (count - 1) / 2) * separation),
                    scale=self._draw("component_scale", log=True),
                )
            weights = r.dirichlet(r.uniform(0.5, 3.0, size=count)).tolist()
            if kind == "contamination":
                probability = self._draw("contamination_probability")
                weights = [1 - probability, probability]
                components[1]["scale"] = self._draw("outlier_scale", log=True)
            spec.update(components=components, weights=weights)
        elif kind == "student_t":
            spec["df"] = self._draw("student_df", log=True)
        elif kind == "pink":
            low, high = self.priors["pink_filter_length"]
            spec.update(
                seed=int(r.integers(0, 2**63)),
                exponent=self._draw("pink_exponent"),
                filter_length=int(r.integers(int(low), int(high) + 1)),
            )
        elif kind == "brown":
            spec.update(
                seed=int(r.integers(0, 2**63)),
                step_scale=self._draw("brown_step_scale", log=True),
                initial_value=self._draw("effect_offset"),
            )
        elif kind == "perlin":
            low, high = self.priors["perlin_octaves"]
            spec.update(
                seed=int(r.integers(0, 2**63)),
                frequency=self._draw("perlin_frequency", log=True),
                offset=self._draw("perlin_offset"),
                octaves=int(r.integers(int(low), int(high) + 1)),
                persistence=self._draw("perlin_persistence"),
                lacunarity=self._draw("perlin_lacunarity"),
            )
        elif kind == "beta":
            spec.update(
                alpha=self._draw("beta_shape", log=True),
                beta=self._draw("beta_shape", log=True),
            )
        elif kind == "poisson":
            spec["rate"] = self._draw("poisson_rate", log=True)
        elif kind == "weibull":
            spec["shape"] = self._draw("weibull_shape", log=True)
        elif kind == "skewed":
            spec["sigma"] = self._draw("lognormal_sigma")
        elif kind == "quantization":
            spec.update(
                step=self._draw("quantization_step", log=True),
                offset=self._draw("effect_offset"),
                latent_scale=self._draw("component_scale"),
            )
        elif kind not in self.BASE_NOISES:
            raise ValueError(f"Unknown noise family: {kind}")
        return spec

    def _choose_noise(self, families):
        weights = np.array([self.noise_weights[k] for k in families], dtype=float)
        return str(self.rng.choice(families, p=weights / weights.sum()))

    @staticmethod
    def _perlin(spec, size):
        """Seeded 1D gradient Perlin noise, with optional fractal octaves.

        NumPy keeps this portable without a compiled 'noise' dependency. The
        quintic fade is the Perlin interpolation kernel, not a sampled parameter.
        Lattice gradients, phase, wavelength and octave structure all vary.
        Never normalize using sample statistics: prefixes must stay identical
        when a longer sequence is requested, including inside a mixture.
        """
        rng = np.random.default_rng(spec["seed"])
        permutation = rng.permutation(256)
        gradients = rng.uniform(-1, 1, size=256)
        positions = spec["offset"] + np.arange(size) * spec["frequency"]
        result = np.zeros(size)
        amplitude, total_amplitude = 1.0, 0.0
        for _ in range(spec["octaves"]):
            lattice = np.floor(positions).astype(np.int64)
            fraction = positions - lattice
            fade = fraction**3 * (fraction * (fraction * 6 - 15) + 10)
            left = gradients[permutation[lattice % 256]] * fraction
            right = gradients[permutation[(lattice + 1) % 256]] * (fraction - 1)
            result += amplitude * (left + fade * (right - left))
            total_amplitude += amplitude
            amplitude *= spec["persistence"]
            positions *= spec["lacunarity"]
        # A fixed amplitude convention preserves temporal structure and phase;
        # Perlin is a correlated random process, not a unit-variance IID law.
        return 2 * result / total_amplitude

    @classmethod
    def _contains_correlated_noise(cls, spec):
        return spec["kind"] in ("perlin", "pink", "brown") or any(
            cls._contains_correlated_noise(c) for c in spec.get("components", [])
        )

    @staticmethod
    def _pink(spec, size):
        """Approximate 1/f**exponent noise by causal fractional integration.

        The filter (1-L)**(-exponent/2) colors white innovations. Truncation
        bounds memory and flattens the spectrum below roughly 1/filter_length.
        Saved random prehistory avoids a zero-state startup transient. Unlike
        whole-sequence FFT synthesis, this also keeps prefixes independent of
        the requested sequence length. Normalize by theoretical filter energy,
        never by the sample's mean or standard deviation.
        """
        if size == 0:
            return np.empty(0)
        taps = spec["filter_length"]
        d = spec["exponent"] / 2
        kernel = np.ones(taps)
        kernel[1:] = np.cumprod((np.arange(1, taps) - 1 + d) / np.arange(1, taps))
        kernel /= np.linalg.norm(kernel)
        white = np.random.default_rng(spec["seed"]).normal(size=size + taps - 1)
        return np.convolve(white, kernel, mode="valid")

    @classmethod
    def _noise(cls, rng, spec, size):
        """Population-normalized base laws, followed by saved location/scale.

        sqrt(3), for example, is a distribution identity, not a model parameter.
        Mixtures retain between-component variance and modes. Quantization errors
        retain their step size. Poisson/beta/Weibull are centered innovations,
        not a claim that the resulting telemetry itself is a count/proportion.
        """
        kind = spec["kind"]
        if kind in cls.MIXTURES:
            labels = rng.choice(len(spec["components"]), size=size, p=spec["weights"])
            result = np.empty(size)
            for i, component in enumerate(spec["components"]):
                mask = labels == i
                if cls._contains_correlated_noise(spec):
                    # Evaluate on the ORIGINAL time axis before selecting.
                    # Compressing to the selected samples would speed up/slur
                    # correlated trajectories whenever mixture labels switch.
                    result[mask] = cls._noise(rng, component, size)[mask]
                else:
                    # Preserve replay of existing v2 non-Perlin mixtures.
                    result[mask] = cls._noise(rng, component, int(mask.sum()))
        elif kind == "perlin":
            result = cls._perlin(spec, size)
        elif kind == "pink":
            result = cls._pink(spec, size)
        elif kind == "brown":
            # Brown/red noise is integrated white noise (~1/f²), not IID draws.
            # Preserve accumulated variation and time alignment; no sample-wise
            # recentering or normalization, which would change earlier values.
            steps = np.random.default_rng(spec["seed"]).normal(
                0, spec["step_scale"], size
            )
            result = spec["initial_value"] + np.cumsum(steps)
        elif kind == "normal":
            result = rng.normal(size=size)
        elif kind == "student_t":
            df = spec["df"]
            result = rng.standard_t(df, size=size) / np.sqrt(df / (df - 2))
        elif kind == "laplace":
            result = rng.laplace(scale=1 / np.sqrt(2), size=size)
        elif kind == "uniform":
            result = rng.uniform(-np.sqrt(3), np.sqrt(3), size=size)
        elif kind == "skewed":
            variance = spec["sigma"] ** 2
            result = (
                rng.lognormal(0, spec["sigma"], size) - np.exp(variance / 2)
            ) / np.sqrt(np.expm1(variance) * np.exp(variance))
        elif kind == "poisson":
            result = (rng.poisson(spec["rate"], size) - spec["rate"]) / np.sqrt(
                spec["rate"]
            )
        elif kind == "beta":
            a, b = spec["alpha"], spec["beta"]
            result = (rng.beta(a, b, size) - a / (a + b)) / np.sqrt(
                a * b / ((a + b) ** 2 * (a + b + 1))
            )
        elif kind == "weibull":
            k = spec["shape"]
            mean = math.gamma(1 + 1 / k)
            result = (rng.weibull(k, size) - mean) / np.sqrt(
                math.gamma(1 + 2 / k) - mean**2
            )
        elif kind == "quantization":
            latent = rng.normal(spec["offset"], spec["latent_scale"], size)
            result = cls._quantize(latent, spec["step"], spec["offset"]) - latent
        else:
            raise ValueError(f"Unknown noise family: {kind}")
        if "memory_tau" in spec:
            # Unit-DC causal filtering prefers inferable, persistent hidden
            # disturbances. Compiled filtering adds no Python loop per row.
            alpha = -math.expm1(-1 / spec["memory_tau"])
            result = lfilter([alpha], [1, alpha - 1], result)
        return spec["location"] + spec["scale"] * result

    @staticmethod
    def _quantize(value, step, offset):
        return offset + step * np.round((value - offset) / step) if step else value

    def _expression(self, input_domain=None):
        """Build an expression DAG with reusable intermediate results.

        Evaluation in instruction order is dynamic programming: shared results
        are computed once. Each instruction consumes the previous result, plus
        an optional older result, so the program contains no dead instructions.
        """
        r, program = self.rng, []
        count = (
            int(r.integers(2, self.max_effect_components + 1))
            if self.max_effect_components > 1
            and r.random() < self.composite_effect_probability
            else 1
        )
        names = list(self.effect_weights)
        weights = np.array(list(self.effect_weights.values()), dtype=float)
        if input_domain is not None and input_domain != "angle":
            # Static trig of an arbitrary load/pressure is rarely meaningful.
            # Direct inspection can omit context; sampled SCM edges supply it.
            for op in ("sine", "cosine", "tangent"):
                weights[names.index(op)] = 0
        if weights.sum() <= 0:
            raise ValueError(f"No enabled effects for input domain {input_domain!r}")
        for i in range(count):
            op = str(r.choice(names, p=weights / weights.sum()))
            instruction = {
                "op": op,
                "a": i,
                "b": int(r.integers(0, i + 1)),
                "gain": self._draw("effect_gain", signed=True),
                "offset": self._draw("effect_offset"),
                "limit": self._draw("effect_limit"),
            }
            if op == "power":
                instruction["power"] = float(r.uniform(0.4, 3.0))
            if op in ("damped_sine", "damped_cosine"):
                instruction.update(
                    decay=self._draw("damping_rate", log=True),
                    frequency=self._draw("oscillation_frequency", log=True),
                    phase=float(r.uniform(-np.pi, np.pi)),
                )
            if op == "inverse_square":
                instruction["floor"] = self._draw("inverse_square_floor", log=True)
            if op == "tangent":
                instruction["angle_limit"] = float(r.uniform(0.7, 1.45))
            if op == "mlp":
                width = int(r.integers(3, 13))
                instruction.update(
                    w=r.normal(size=width).tolist(),
                    bias=r.normal(0, r.uniform(0.2, 2), size=width).tolist(),
                    v=(r.normal(size=width) / np.sqrt(width)).tolist(),
                )
            if op in STATIC_OPERATORS + TEMPORAL_OPERATORS:
                # Sample the shape and temporal parameters once per equation,
                # never while evaluating observations. Ranges are configurable.
                instruction.update(
                    k=self._draw("effect_k", log=True),
                    width=self._draw("effect_width", log=True),
                    threshold=self._draw("effect_threshold", log=True),
                    shape=self._draw("effect_shape"),
                    center=self._draw("effect_offset"),
                    level=self._draw("effect_gain"),
                    leak=self._draw("effect_leak"),
                    floor=self._draw("effect_floor", log=True),
                    tau=self._draw("effect_tau", log=True),
                    rise_rate=self._draw("effect_rate", log=True),
                    fall_rate=self._draw("effect_rate", log=True),
                    steps=int(
                        r.integers(
                            int(self.priors["effect_delay_steps"][0]),
                            int(self.priors["effect_delay_steps"][1]) + 1,
                        )
                    ),
                    stages=int(
                        r.integers(
                            int(self.priors["effect_kernel_stages"][0]),
                            int(self.priors["effect_kernel_stages"][1]) + 1,
                        )
                    ),
                    spread=self._draw("effect_kernel_spread"),
                    period=self._draw("ringing_period", log=True),
                    high_gain=self._draw("effect_gain"),
                    ki=self._draw("effect_pid_i", log=True),
                    kd=self._draw("effect_pid_d", log=True),
                )
            if op == "second_order":
                instruction["damping"] = self._log_uniform(0.04, 1.5)
            program.append(instruction)
        return program

    @staticmethod
    def _evaluate(program, value, state=None):
        # One state dictionary per edge: it keeps the program bound to its
        # parameters, with independent memory per temporal instruction.
        # Omitting state evaluates a fresh one-step response for inspection.
        run = None if state is None else state.get("run")
        if run is None:
            run = compile_program(program)
            if state is not None:
                state["run"] = run
        return run(value)

    def _edge(self, parent, lag=0, input_domain=None):
        return {
            "parent": parent,
            "lag": lag,
            "weight": self._draw("edge_strength", log=True, signed=True),
            "input_scale": self._draw("input_scale", log=True),
            "input_offset": self._draw("effect_offset"),
            "output_offset": self._draw("effect_offset"),
            "program": self._expression(input_domain),
        }

    def _sample_parent_combination(self, parent_count):
        """Sample operator, participating parents and roles independently."""
        if parent_count < 2:
            raise ValueError("A joint map requires at least two parents")
        r = self.rng
        modes = list(self.parent_combination_weights)
        weights = np.array(list(self.parent_combination_weights.values()), float)
        # A log-sum requires an exponential link to form a power law. Honor an
        # explicit decision to disable exponential links by excluding that pair.
        if not self.node_link_weights["exponential"]:
            weights[modes.index("log_sum")] = 0
        if weights.sum() <= 0:
            raise ValueError("No compatible parent combination and node link enabled")
        mode = str(r.choice(modes, p=weights / weights.sum()))
        if mode == "log_sum":
            count = parent_count
        elif mode == "mlp" and parent_count >= 3:
            count = int(r.integers(3, min(5, parent_count) + 1))
        elif parent_count == 2 or r.random() < self.pairwise_interaction_probability:
            count = 2
        else:
            count = int(r.integers(3, min(5, parent_count) + 1))
        spec = {
            "version": 2,
            "mode": mode,
            "indices": r.choice(parent_count, count, replace=False).tolist(),
            "application": "replace",
            "scale": self._draw("input_scale", log=True),
            "floor": self._draw("effect_floor"),
            "strength": self._draw("joint_strength", log=True),
            "positive_map": str(r.choice(["softplus", "exp"], p=[0.85, 0.15])),
            "threshold": self._draw("effect_offset"),
            "temperature": self._draw("effect_width", log=True),
            "low_gain": self._draw("effect_leak"),
            "high_gain": self._draw("joint_power"),
            "numeric_limit": self.joint_numeric_limit,
        }
        if (
            mode
            in ("product", "saddle", "ratio", "mlp", "volatility_coupling", "swiglu")
            and r.random() < self.additive_interaction_probability
        ):
            spec["application"] = "additive"
        if mode == "convex_mix":
            concentration = self._draw("joint_mix_concentration")
            spec["weights"] = r.dirichlet(np.full(count, concentration)).tolist()
        if mode == "mlp":
            low, high = self.priors["joint_mlp_width"]
            width = int(r.integers(int(low), int(high) + 1))
            magnitude = self._draw("joint_mlp_weight_scale")
            spec.update(
                w=(r.normal(size=(width, count)) * magnitude / np.sqrt(count)).tolist(),
                bias=r.normal(0, magnitude, width).tolist(),
                v=(r.normal(size=width) / np.sqrt(width)).tolist(),
            )
        if mode == "log_sum":
            spec["powers"] = [self._draw("joint_power") for _ in range(count)]
        return spec

    def _sample_node_link(self, force_exponential=False):
        modes = list(self.node_link_weights)
        weights = np.array(list(self.node_link_weights.values()), float)
        mode = (
            "exponential"
            if force_exponential
            else str(self.rng.choice(modes, p=weights / weights.sum()))
        )
        return {
            "mode": mode,
            "scale": 1.0 if force_exponential else self._draw("input_scale", log=True),
            "input_offset": 0.0 if force_exponential else self._draw("effect_offset"),
            "output_offset": 0.0 if force_exponential else self._draw("effect_offset"),
            "gain": self._draw("effect_gain"),
            "capacity": self._draw("effect_limit"),
            "numeric_limit": self.joint_numeric_limit,
        }

    def _sample_discrete(self, name, rows, family=None):
        r = self.rng
        families = list(self.discrete_weights)
        weights = np.array(list(self.discrete_weights.values()), float)
        family = family or str(r.choice(families, p=weights / weights.sum()))
        node = {
            "name": name,
            "kind": "bernoulli" if family == "bernoulli" else "markov",
            "visible": False,  # Visibility is assigned jointly after sampling the SCM.
            "edges": [],
            "family": family,
        }
        if family == "bernoulli":
            node["probability"] = self._draw("binary_probability")
            return node
        low, high = self.priors["markov_state_count"]
        count = (
            2 if family == "binary_markov" else int(r.integers(int(low), int(high) + 1))
        )
        values = np.arange(count, dtype=float)
        if count > 2 and r.random() < 0.5:
            values = np.sort(
                r.uniform(-1, 1, count) * self._draw("markov_level_scale", log=True)
            )
        transition = np.zeros((count, count))
        for state in range(count):
            # Independently sampled dwell times and exit destinations create
            # asymmetric, persistent, non-periodic operating modes.
            dwell = max(2.0, rows * self._draw("markov_dwell_fraction", log=True))
            exit_probability = 1 / dwell
            destinations = np.arange(count) != state
            transition[state, destinations] = (
                r.dirichlet(np.full(count - 1, self._draw("markov_concentration")))
                * exit_probability
            )
            transition[state, state] = 1 - exit_probability
        node.update(
            values=values.tolist(),
            transition=transition.tolist(),
            initial_state=int(r.integers(count)),
        )
        return node

    def _sample_switch(self, rows, force_episode=False):
        r = self.rng
        profiles = list(self.switch_profile_weights)
        weights = np.array(list(self.switch_profile_weights.values()), float)
        profile = (
            "single_episode"
            if force_episode
            else str(r.choice(profiles, p=weights / weights.sum()))
        )
        node = {
            "name": "covariate_01",
            "kind": "switch",
            "profile": profile,
            "visible": False,
            "edges": [],
            "initial_on": bool(r.integers(2)),
        }
        if profile == "single_episode":
            node.update(
                start=int(rows * self._draw("episode_start_fraction")),
                stop=int(rows * self._draw("episode_stop_fraction")),
                initial_on=False,
            )
        elif profile == "long_dwell":
            for role in ("on", "off"):
                duration = max(
                    2, int(rows * self._draw(f"switch_long_{role}_fraction", log=True))
                )
                node[f"{role}_duration"] = [
                    max(1, duration // 2),
                    max(2, int(1.5 * duration)),
                ]
        else:
            node.update(
                on_duration=[20, max(30, min(rows // 2, 200))],
                off_duration=[5, max(10, min(rows // 5, 80))],
            )
        return node

    def _sample_process_driver(self, name, rows):
        """A sampled exogenous machine program, rather than a binary on/off clock.

        Independent duration weights allow different idle tails and rise/fall
        ratios. Rescale the whole program BEFORE simulation, never the values.
        Internal level changes are actual setpoint commands with their own ramps.
        """
        r = self.rng
        before = (
            self._draw("process_idle_fraction")
            if r.random() < 0.75
            else r.uniform(0, 0.015)
        )
        after = (
            self._draw("process_idle_fraction")
            if r.random() < 0.75
            else r.uniform(0, 0.015)
        )
        weights = np.array(
            [
                before,
                self._draw("process_ramp_fraction", log=True),
                self._draw("process_active_fraction"),
                self._draw("process_ramp_fraction", log=True),
                after,
            ]
        )
        # Reserve at least one observation for rise, active operation and release.
        lengths = np.floor(weights / weights.sum() * (rows - 3)).astype(int)
        lengths[[1, 2, 3]] += 1
        lengths[2] += rows - int(lengths.sum())
        idle, rise, active, fall, tail = map(int, lengths)
        start, stop = idle, idle + rise + active

        def command(time, target, steps):
            return {
                "start": int(time),
                "target": float(target),
                "steps": int(steps),
                "shape": str(
                    r.choice(["linear", "smoothstep", "cosine", "power", "release"])
                ),
                "curvature": self._draw("process_curvature", log=True),
            }

        transitions = [command(start, 1.0, rise)]
        if active > 3 and r.random() < self.process_multilevel_probability:
            low, high = self.priors["process_level_count"]
            levels = min(active - 1, int(r.integers(int(low), int(high) + 1)))
            changes = sorted(
                r.choice(np.arange(start + rise, stop), size=levels - 1, replace=False)
            )
            for time in changes:
                steps = max(
                    1, int(rows * self._draw("process_transition_fraction", log=True))
                )
                transitions.append(command(time, self._draw("process_level"), steps))
        transitions.append(command(stop, 0.0, fall))
        priming = []
        if start >= 4 and r.random() < self.process_priming_probability:
            for _ in range(int(r.integers(1, 5))):
                duration = int(r.integers(2, max(3, min(start, max(3, rows // 20)))))
                priming.append(
                    {
                        "start": int(r.integers(max(1, start - duration + 1))),
                        "duration": duration,
                        "amplitude": self._draw("process_priming_fraction"),
                    }
                )
        return {
            "name": name,
            "kind": "formula",
            "family": "process_run",
            "visible": False,
            "edges": [],
            "start": start,
            "stop": stop,
            "idle_after": tail,
            "initial_level": 0.0,
            "transitions": transitions,
            "priming": priming,
            "amplitude": 1.0,
            "offset": 0.0,
            "noise_free": True,
            "noise": {"kind": "normal", "location": 0.0, "scale": 0.0},
        }

    def _sample_formula(self, name, rows):
        """Formulaic source plus optional saved innovations, usable in any DAG."""
        r = self.rng
        family = str(r.choice(FORMULA_FAMILIES))
        if family == "state_machine":
            node = self._sample_process_driver(name, rows)
            node["family"] = family
        else:
            node = {
                "name": name,
                "kind": "formula",
                "family": family,
                "edges": [],
                "visible": False,
                "parameters": {
                    "duration": rows,
                    "period": max(
                        4.0, rows * self._draw("formula_period_fraction", log=True)
                    ),
                    "phase": float(r.uniform(-np.pi, np.pi)),
                    "power": self._draw("formula_power"),
                    "center": self._draw("formula_center"),
                    "width": self._draw("formula_width_fraction", log=True),
                    "decay": self._draw("formula_decay", log=True),
                    "chirp": self._draw("formula_chirp"),
                },
            }
        node.update(
            amplitude=self._draw("formula_amplitude", log=True, signed=True),
            offset=self._draw("formula_offset"),
            noise_free=bool(r.random() < self.noise_free_covariate_probability),
        )
        if family == "chirp":
            p = node["parameters"]
            # Keep the fastest recorded carrier below Nyquist, as for Fourier seasons.
            p["period"] = max(p["period"], 2.25 * (1 + max(0.0, p["chirp"])))
        noise = self._sample_noise()
        noise.update(
            location=0.0,
            scale=0.0
            if node["noise_free"]
            else abs(node["amplitude"])
            * self._draw("formula_noise_fraction", log=True),
        )
        node["noise"] = noise
        return node

    def _sample_episode(self, outcome, command, rows):
        lag = min(self._sample_covariate_lag(), max(0, rows - command["stop"] - 2))
        tail = max(1, int(0.04 * rows))
        rise = min(
            int(rows * self._draw("episode_ramp_fraction")),
            command["stop"] - command["start"],
        )
        fall = min(
            int(rows * self._draw("episode_ramp_fraction")),
            rows - command["stop"] - lag - tail,
        )
        if command.get("kind") == "formula":
            lag = min(lag, command["idle_after"])
            rise = command["transitions"][0]["steps"]
            fall = command["transitions"][-1]["steps"]
        amplitude = outcome["scale"] * self._draw("episode_amplitude", log=True)
        return {
            "schema_version": 2,
            "mode": "episode",
            "activity_source": "direct"
            if command.get("kind") == "formula"
            else "binary_response",
            "parent": command["name"],
            "lag": lag,
            "idle_value": 0.0 if self.rng.random() < 0.85 else outcome["baseline"],
            "amplitude": amplitude,
            "rise_steps": max(1, rise),
            "fall_steps": max(1, fall),
            "ramp_shape": str(self.rng.choice(["smoothstep", "cosine", "linear"])),
            "detail_fraction": self._draw("episode_detail_fraction"),
            "noise_fraction": self._draw("episode_noise_fraction", log=True)
            if self.rng.random() < 0.85
            else 0.0,
            "seasonal_fraction": self._draw("episode_detail_fraction"),
            "detail_scale": max(1.0, abs(outcome["baseline"] / outcome["scale"])),
            "idle_noise_scale": amplitude
            * self._draw("episode_idle_noise_fraction", log=True)
            if self.rng.random() < 0.5
            else 0.0,
        }

    def _sample_sensor(self, spec, is_outcome):
        operation = spec.get("operation", {})
        if operation.get("mode") == "episode":
            center = operation["idle_value"] + operation["amplitude"] / 2
            span = operation["amplitude"] / 2
        else:
            center = spec["baseline"] if is_outcome else spec["intercept"]
            span = max(
                1.0,
                sum(s["amplitude"] for s in spec["seasons"]) + spec["noise"]["scale"],
            )
            if is_outcome:
                span *= spec["scale"]
        mode = str(self.rng.choice(["floor", "ceiling", "both"], p=[0.35, 0.35, 0.30]))
        return {
            "floor": center - span * self._draw("sensor_bound_distance")
            if mode != "ceiling"
            else None,
            "ceiling": center + span * self._draw("sensor_bound_distance")
            if mode != "floor"
            else None,
        }

    @classmethod
    def _effect(cls, edge, value, state=None):
        value = (value - edge["input_offset"]) / edge["input_scale"]
        return edge["weight"] * (
            cls._evaluate(edge["program"], value, state) + edge["output_offset"]
        )

    @staticmethod
    def _bind_effect(edge):
        """effect(value) of one edge in one simulation, with its own memory.

        Same arithmetic as `_effect`; the saved parameters are read once
        instead of on every row.
        """
        input_offset, input_scale = edge["input_offset"], edge["input_scale"]
        weight, output_offset = edge["weight"], edge["output_offset"]
        steps = compile_steps(edge["program"])
        if len(steps) == 1 and steps[0][1:] == (0, 0):
            step = steps[0][0]  # Most edges: one instruction on the edge input.

            def effect(value):
                value = (value - input_offset) / input_scale
                return weight * (step(value, value) + output_offset)

            return effect
        run = run_steps(steps)

        def effect(value):
            return weight * (run((value - input_offset) / input_scale) + output_offset)

        return effect

    def _dynamics(self, event_name):
        r = self.rng
        drifting = r.random() < self.drift_probability
        return {
            "noise": self._sample_noise(),
            "persistence": self._draw("persistence"),
            "intercept": self._draw("effect_offset"),
            "seasons": [
                {
                    "period": self._draw("seasonal_period", log=True),
                    "amplitude": self._draw("seasonal_amplitude", log=True),
                    "phase": float(r.uniform(-np.pi, np.pi)),
                    "modulation": float(r.uniform(0, 0.8)),
                    "modulation_period": self._draw("seasonal_period", log=True),
                }
                for _ in range(int(r.integers(0, 4)))
            ],
            "slope": self._draw("drift_magnitude", log=True, signed=True)
            if drifting
            else 0.0,
            "slope_walk": self._draw("slope_walk_scale", log=True) if drifting else 0.0,
            "walk_scale": self._draw("walk_scale", log=True)
            if r.random() < self.walk_probability
            else 0.0,
            "event_parent": event_name if r.random() < 0.7 else None,
            "event_mode": str(r.choice(["reverse", "reset", "accelerate"])),
            "event_slope": self._draw("drift_magnitude", log=True, signed=True),
            "level_jump": self._draw("shock_magnitude", log=True, signed=True)
            if r.random() < 0.5
            else 0.0,
            "shock_size": self._draw("shock_magnitude", log=True, signed=True),
            "shock_decay": self._draw("decay"),
            "heteroskedasticity": float(r.uniform(0, 1.5)),
            "quantization_step": self._draw("quantization_step", log=True)
            if r.random() < 0.25
            else 0.0,
            "quantization_offset": self._draw("effect_offset"),
        }

    def _sample_lags(self, rows, seasonal_table=True):
        """Draw independent short, seasonal and unstructured history groups.

        Capacity is a ceiling, not a target to fill with compulsory recent lags.
        Reserve the unstructured slot first so seasonal choices do not dilute
        its configured probability. Small user-specified caps still take priority.
        """
        r = self.rng
        capacity = min(self.max_lag_count, self.max_lag)
        roles = {1: "recent"}
        candidates = [
            p for p in (8, 12, 24, 48, 96, 168) if p <= min(self.max_lag, rows // 3)
        ]
        distant = [p for p in range(8, self.max_lag + 1) if p not in candidates]
        if capacity > 1 and distant and r.random() < self.long_lag_probability:
            roles[int(r.choice(distant))] = "rare_long"
        recent = np.arange(2, min(7, self.max_lag) + 1)
        # Zero/one/two extra short lags: one is overwhelmingly the usual case.
        short_count = min(
            int(r.choice([0, 1, 2], p=[0.15, 0.75, 0.10])),
            len(recent),
            capacity - len(roles),
        )
        if short_count:
            weights = np.exp(-0.25 * self._draw("recent_lag_decay") * (recent - 2))
            roles.update(
                {
                    int(p): "recent"
                    for p in r.choice(
                        recent, short_count, replace=False, p=weights / weights.sum()
                    )
                }
            )
        seasonal = []
        if (
            seasonal_table
            and candidates
            and len(roles) < capacity
            and r.random() < self.seasonal_lag_probability
        ):
            number = min(
                capacity - len(roles),
                len(candidates),
                int(r.choice([1, 2, 3], p=[0.80, 0.16, 0.04])),
            )
            seasonal = sorted(r.choice(candidates, number, replace=False).tolist())
            roles.update({p: "seasonal" for p in seasonal})
        lags = sorted(roles)
        return lags, [roles[p] for p in lags], seasonal

    def _sample_covariate_lag(self):
        """A small mass at zero, otherwise a truncated discretized exponential.

        Inverse-CDF sampling avoids piling an exponential tail onto the maximum
        lag. Positive delays use 1+floor(X): one timestep has the most mass.
        """
        if (
            self.transport_delay_probability
            and self.max_covariate_lag > 4
            and self.rng.random() < self.transport_delay_probability
        ):
            # Transport/dead time: the exponential law almost never exceeds 3.
            return int(self._log_uniform(4, self.max_covariate_lag + 1))
        if self.rng.random() < self.instantaneous_probability:
            return 0
        mass = -np.expm1(-self.max_covariate_lag / self.covariate_delay_scale)
        value = -self.covariate_delay_scale * np.log1p(-self.rng.random() * mass)
        return min(self.max_covariate_lag, 1 + int(value))

    def _lag_weights(self, lags, roles):
        # Strong seasonal anchors coexist with decaying short-range influence.
        r = self.rng
        priorities = [
            r.uniform(0.5, 1.0)
            if role == "seasonal"
            else r.uniform(0.03, 0.1)
            if role == "rare_long"
            else np.exp(-0.5 * (lag - 1)) * r.uniform(0.85, 1.15)
            for lag, role in zip(lags, roles)
        ]
        weights = np.asarray(priorities)
        return (weights / weights.sum() * r.uniform(0.1, 0.45)).tolist()

    def _sample_visibility(self, nodes, count=None):
        """Cap a Poisson count at available sources; choose an unbiased subset.

        Capping (rather than rejection sampling) retains the zero-count mass.
        Visibility is independent of source type, order and causal importance.
        An explicit count is useful for controlled comparisons of the same SCM.
        """
        if count is None:
            count = int(self.rng.poisson(self.visible_covariate_lambda))
        elif isinstance(count, bool) or not isinstance(count, int) or count < 0:
            raise ValueError("visible_covariate_count must be a nonnegative integer")
        visible = set(
            self.rng.choice(len(nodes), size=min(count, len(nodes)), replace=False)
        )
        for index, node in enumerate(nodes):
            node["visible"] = index in visible
        return count

    def sample_model(self, trend_free=None, *, visible_covariate_count=None):
        """Sample causal equations, then choose which sources are recorded.

        Hiding every source creates a strictly univariate observation without
        removing latent mechanisms, oscillations, or changes in operating state.
        The visible count is Poisson distributed, capped at the available sources;
        there is no univariate batch quota or per-source Bernoulli hiding draw.
        """
        r = self.rng
        rows = int(r.integers(self.min_rows, self.max_rows + 1))
        # Below, `rows` is the SIMULATED span. Only the final `recorded` rows are
        # kept, like a long-running series cut to the model's context length.
        recorded, row_gaps = rows, None
        if self.full_length_probability and r.random() < self.full_length_probability:
            recorded = rows = self.max_rows
            if self.prehistory_probability and r.random() < self.prehistory_probability:
                rows += int(r.integers(64, self.max_prehistory_rows + 1))
        if self.row_gap_probability and r.random() < self.row_gap_probability:
            row_gaps = sample_row_gaps(r, recorded)
            rows += row_gaps["total"]
        seasonal_table = bool(
            self.seasonal_table_probability >= 1
            or r.random() < self.seasonal_table_probability
        )
        noise_profile = (
            self._sample_noise_profile() if self.noise_profile_weights else None
        )
        trend_free = (
            bool(r.random() < self.trend_free_fraction)
            if trend_free is None
            else bool(trend_free)
        )
        count = int(r.integers(self.min_covariates, self.max_covariates + 1))
        lags, lag_roles, seasonal_lags = self._sample_lags(rows, seasonal_table)
        event = "covariate_00"
        nodes = [
            {
                "name": event,
                "kind": "event",
                "edges": [],
                "visible": False,
                "probability": self._draw("event_probability"),
            }
        ]
        # Branching/direct structures dominate. Short mediated paths remain
        # possible, but a full covariate chain is no longer a scaffold.
        topology = str(
            r.choice(["sparse", "fork", "diamond", "dense"], p=[0.35, 0.35, 0.25, 0.05])
        )
        switch = None
        process_episode = bool(r.random() < self.process_episode_probability)
        if process_episode or r.random() < self.switching_probability:
            switch = "covariate_01"
            nodes.append(
                self._sample_process_driver(switch, rows)
                if process_episode
                else self._sample_switch(rows)
            )

        shared_mode = None
        if (
            self.shared_state_probability
            and count - len(nodes) >= 2
            and r.random() < self.shared_state_probability
        ):
            mode = self._sample_discrete(
                f"covariate_{len(nodes):02d}", rows, family="categorical_markov"
            )
            # Integer codes denote mutually exclusive modes, not independent
            # flags. Multiple downstream equations read this same state.
            mode["values"] = list(range(len(mode["values"])))
            mode["shared_mode"] = True
            nodes.append(mode)
            shared_mode = mode["name"]

        def sample_edge(parent, lag=0):
            source = next((n for n in nodes if n["name"] == parent), {})
            edge = self._edge(parent, lag, input_domain=source.get("domain", "real"))
            if (
                self.step_response_probability
                and (
                    source.get("kind") in ("switch", "markov")
                    or source.get("family") in ("state_machine", "process_run")
                )
                and r.random() < self.step_response_probability
            ):
                # A level change arrives as a clean settling response instead of
                # passing through a random static map that hides its shape.
                edge.update(
                    program=self._response_program(),
                    input_scale=float(
                        max(1.0, np.abs(source.get("values", [1])).max())
                    ),
                    input_offset=0.0,
                    output_offset=0.0,
                    weight=float(r.choice([-1, 1]) * r.uniform(0.8, 4.0)),
                )
            return edge

        for i in range(len(nodes), count):
            # Retain at least one continuous process. Additional discrete sources
            # are genuine SCM inputs and may directly or indirectly drive Y.
            if (
                any(n["kind"] == "process" for n in nodes)
                and r.random() < self.formula_covariate_probability
            ):
                nodes.append(self._sample_formula(f"covariate_{i:02d}", rows))
                continue
            if (
                any(n["kind"] == "process" for n in nodes)
                and r.random() < self.discrete_covariate_probability
            ):
                nodes.append(self._sample_discrete(f"covariate_{i:02d}", rows))
                continue
            node = {
                "name": f"covariate_{i:02d}",
                "kind": "process",
                "domain": "angle"
                if r.random() < self.angle_covariate_probability
                else "real",
                "visible": False,
                "edges": [],
                **self._dynamics(event),
            }
            if (
                not seasonal_table
                and r.random() >= self.nonseasonal_covariate_season_probability
            ):
                node["seasons"] = []
            if (
                self.slow_dynamics_probability
                and r.random() < self.slow_dynamics_probability
            ):
                # A slow first-order state with unit static gain: steps from its
                # parents arrive as long exponential settling curves.
                persistence = float(np.exp(-1 / self._log_uniform(20, 400)))
                node.update(
                    persistence=persistence,
                    forcing_gain=1 - persistence,
                    innovation_gain=float(np.sqrt(1 - persistence**2)),
                )
            probability = self.edge_probability * (1.8 if topology == "dense" else 1)
            for parent in nodes:
                if r.random() < probability:
                    lag = self._sample_covariate_lag()
                    node["edges"].append(sample_edge(parent["name"], lag))
            # Scaffolds allow deep mediation, shared causes and colliders.
            # Random extra edges retain diversity; visibility is sampled later.
            forced = []
            processes = [n for n in nodes if n["kind"] == "process"]
            if topology == "fork" and len(processes) in (1, 2):
                forced = [processes[0]["name"]]
            if topology == "diamond":
                if len(processes) in (1, 2):
                    forced = [processes[0]["name"]]
                elif len(processes) == 3:
                    forced = [n["name"] for n in processes[1:3]]
            if shared_mode and len(processes) < 2:
                forced.append(shared_mode)
            for parent in forced:
                if not any(e["parent"] == parent for e in node["edges"]):
                    node["edges"].append(
                        sample_edge(parent, self._sample_covariate_lag())
                    )
            if r.random() < self.delayed_probability:
                node["edges"].append(sample_edge(f"outcome_lag_{int(r.choice(lags))}"))
            nodes.append(node)
        processes = [n for n in nodes if n["kind"] == "process"]
        predictors = [
            n
            for n in nodes
            if n["kind"] in ("process", "formula") or n["kind"] in DISCRETE_KINDS
        ]
        selected = [n for n in predictors if r.random() < self.direct_probability]
        # At least two independent inputs when available, selected without a
        # special preference for the last node in topological order.
        minimum = min(2, len(predictors))
        remaining = [n for n in predictors if n not in selected]
        while len(selected) < minimum:
            selected.append(remaining.pop(int(r.integers(len(remaining)))))
        if shared_mode:
            mode_node = next(n for n in nodes if n["name"] == shared_mode)
            if mode_node not in selected:
                selected.append(mode_node)
        outcome = {
            **self._dynamics(event),
            "edges": [
                sample_edge(n["name"], self._sample_covariate_lag()) for n in selected
            ],
            "baseline": self._draw("baseline"),
            "scale": self._draw("outcome_scale", log=True),
            "ar_weights": [0.0] * len(lags),
            "outcome_ar_weights": self._lag_weights(lags, lag_roles),
        }
        if not seasonal_table:
            outcome["seasons"] = []
        # Use the same periods in the structural seasonal equation and the AR
        # anchors. They are not unrelated, uniformly sampled distant lags.
        for index, period in enumerate(seasonal_lags):
            if index >= len(outcome["seasons"]):
                outcome["seasons"].append(
                    {
                        "amplitude": self._draw("seasonal_amplitude", log=True),
                        "phase": float(r.uniform(-np.pi, np.pi)),
                        "modulation": float(r.uniform(0, 0.5)),
                        "modulation_period": float(rows),
                    }
                )
            outcome["seasons"][index]["period"] = float(period)
        residual_support = r.random(len(lags)) < self.residual_history_probability
        if residual_support.any():
            # Residual lag support is a subset of explicit outcome lag support.
            # A small total mass keeps combined feedback below unity.
            weights = np.zeros(len(lags))
            weights[residual_support] = r.dirichlet(
                np.ones(residual_support.sum())
            ) * r.uniform(0.02, 0.12)
            outcome["ar_weights"] = weights.tolist()
        for node in processes + [outcome]:
            node["event_lag"] = (
                self._sample_covariate_lag() if node["event_parent"] else 0
            )
            # Joint maps explicitly combine DIFFERENT parent effects. Existing
            # saved SCMs without this field keep their original sum/interaction.
            node["interaction"] = 0.0
            if len(node["edges"]) >= 2:
                node["parent_combination"] = self._sample_parent_combination(
                    len(node["edges"])
                )
            if node["edges"]:
                node["node_link"] = self._sample_node_link(
                    node.get("parent_combination", {}).get("mode") == "log_sum"
                )
            self._revise_dynamics(
                node, rows, trend_free, node is outcome, seasonal_table, noise_profile
            )
            if r.random() < 0.25:
                # A causal impulse response supplies actual time-domain decay,
                # even when the edge input does not monotonically represent time.
                node["ringing"] = {
                    "parent": event,
                    "lag": self._sample_covariate_lag(),
                    "decay": self._draw("ringing_decay", log=True),
                    "period": self._draw("ringing_period", log=True),
                    "amplitude": self._draw("shock_magnitude", log=True, signed=True),
                    "phase": float(r.uniform(-np.pi, np.pi)),
                    "wave": str(r.choice(["sine", "cosine"])),
                }
            if (
                node is not outcome
                and r.random() < self.noise_free_covariate_probability
            ):
                # A deterministic structural equation has no private innovation,
                # random walk, measurement error or auxiliary random shock input.
                # Explicit causal parents can still transmit their own variation.
                # Zero-scale normal uses the existing replay schema: saved models
                # do not require a new noise family or simulator special case.
                node["noise_free"] = True
                node["noise"] = {"kind": "normal", "location": 0.0, "scale": 0.0}
                for key in (
                    "walk_scale",
                    "slope_walk",
                    "white_noise_scale",
                    "heteroskedasticity",
                    "quantization_step",
                ):
                    node[key] = 0.0
                node["event_parent"] = None
                node["event_lag"] = 0
                node.pop("ringing", None)
                node.pop("volatility", None)
        if process_episode:
            outcome["operation"] = self._sample_episode(outcome, nodes[1], rows)
        elif switch:
            outcome["operation"] = {
                "schema_version": 2,
                "parent": switch,
                "mode": str(r.choice(["shutdown", "reset"], p=[0.7, 0.3])),
                "lag": self._sample_covariate_lag(),
                "flat_value": 0.0 if r.random() < 0.7 else self._draw("baseline"),
                "restart_rate": float(r.uniform(0.03, 0.3)),
                "ramp_rate": float(r.uniform(0.01, 0.08) * r.choice([-1, 1])),
            }
            if (
                self.startup_transient_probability
                and r.random() < self.startup_transient_probability
            ):
                # Coast-down instead of an instant stop; inrush on restart.
                outcome["operation"].update(
                    stop_tau=0.0 if r.random() < 0.4 else self._log_uniform(2, 80),
                    restart_shape=str(r.choice(["linear", "exponential"])),
                    restart_overshoot=0.0
                    if r.random() < 0.5
                    else float(r.uniform(0.15, 2.5)),
                    overshoot_decay=float(r.uniform(0.4, 0.95)),
                )
        if not trend_free:
            # The trending batch profile must not undergo a second Bernoulli
            # drift draw. Size its change relative to local variation, and start
            # accumulation at t=0 so warm-up cannot consume the trend budget.
            activity = max(
                1.0,
                sum(s["amplitude"] for s in outcome["seasons"])
                + outcome["noise"]["scale"],
            )
            kind = str(
                r.choice(
                    ["linear", "accelerating", "decelerating", "broken"],
                    p=[0.3, 0.15, 0.15, 0.4],
                )
            )
            if r.random() < self.multiplicative_trend_probability:
                kind = "multiplicative"
            trend = {
                "kind": kind,
                "initial_slope": float(
                    r.choice([-1, 1]) * r.uniform(4, 12) * activity / rows
                ),
                "event_parent": None,
                "event_time": None,
                "slope_multiplier": 1.0,
            }
            if (
                self.lifecycle_trend_probability
                and r.random() < self.lifecycle_trend_probability
            ):
                # Wear, commissioning and maintenance curves, same change budget.
                trend = sample_lifecycle(
                    r, rows, float(r.choice([-1, 1]) * r.uniform(4, 12) * activity)
                )
                kind = trend["kind"]
            if kind == "multiplicative":
                # A latent positive level feeds its own next value: L[t+1] =
                # L[t]*exp(log_rate). The total log change is horizon-scaled,
                # so shorter contexts do not grow less and long ones cannot
                # accidentally overflow. Most cases grow; some decay.
                trend.update(
                    initial_level=max(
                        activity * 4, outcome["baseline"] / outcome["scale"]
                    ),
                    log_rate=float(
                        self._draw("multiplicative_log_change")
                        * r.choice([-1, 1], p=[0.2, 0.8])
                        / rows
                    ),
                )
                outcome["multiplicative_scaling"] = {
                    "noise_elasticity": float(
                        r.uniform(1.0, 1.4)
                        if r.random() < 0.85
                        else r.uniform(0.7, 1.0)
                    ),
                    "seasonal_elasticity": float(r.uniform(0.95, 1.3)),
                    "seasonal_base_scale": max(
                        1.0,
                        trend["initial_level"]
                        * self._draw("relative_seasonal_amplitude")
                        / max(
                            0.1,
                            np.sqrt(
                                sum(s["amplitude"] ** 2 for s in outcome["seasons"])
                            ),
                        ),
                    ),
                }
                outcome["white_noise_scale"] = max(
                    outcome["white_noise_scale"],
                    trend["initial_level"]
                    * self._draw("relative_noise_scale")
                    * (
                        min(1.0, noise_profile["innovation_factor"])
                        if noise_profile
                        else 1
                    ),
                )
                # Draw these independently: constant noise can coexist with
                # growing seasons, and constant seasonal amplitude with growing
                # noise. Most multiplicative series retain level-scaled variation.
                if r.random() < self.multiplicative_constant_noise_probability:
                    outcome["multiplicative_scaling"]["noise_elasticity"] = 0.0
                    outcome["heteroskedasticity"] = 0.0
                    outcome.pop("volatility", None)
                    # This fixes direct innovation scaling, not total observed
                    # variance: stochastic covariates and feedback still act.
                if r.random() < self.multiplicative_constant_seasonality_probability:
                    outcome["multiplicative_scaling"]["seasonal_elasticity"] = 0.0
                    for season in outcome["seasons"]:
                        # Suppress separate exponential amplitude growth too.
                        # Bounded modulation and shape/timing changes remain.
                        season["growth"] = 0.0
            if kind == "broken":
                # A visible/hidden binary shock is forced at the saved change
                # point. Its slope change is latched permanently; later pulses
                # can still produce transient shocks but cannot undo the break.
                trend.update(
                    event_parent=event,
                    event_time=int(r.uniform(0.25, 0.65) * rows),
                    event_lag=self._sample_covariate_lag(),
                    slope_multiplier=float(r.choice([-1, 1]) * r.uniform(1.5, 3.5)),
                )
                nodes[0]["forced_event_times"] = [
                    trend["event_time"] - trend["event_lag"]
                ]
            outcome["trend"] = trend
            for key in ("slope", "slope_walk", "event_slope", "level_jump"):
                outcome[key] = 0.0
        for spec in processes + [outcome]:
            if self.rng.random() < self.sensor_bounds_probability:
                spec["sensor"] = self._sample_sensor(spec, spec is outcome)
        # Draw observation randomness first so changing ONLY visibility cannot
        # change the simulated process, even if subset sampling consumes RNG,
        # unless the explicitly visibility-aware hidden-persistence policy is on.
        observation_seed = int(r.integers(0, 2**63))
        sampled_visible_count = self._sample_visibility(nodes, visible_covariate_count)
        driver_policy = self._budget_drivers(nodes, outcome, observation_seed)
        model = {
            "version": 3,
            "observation_seed": observation_seed,
            "rows": rows,
            "trend_free": trend_free,
            "univariate": not any(node["visible"] for node in nodes),
            "driver_policy": driver_policy,
            "visibility": {
                "distribution": "poisson"
                if visible_covariate_count is None
                else "fixed",
                "lambda": self.visible_covariate_lambda,
                "sampled_count": sampled_visible_count,
                "visible_count": sum(node["visible"] for node in nodes),
            },
            "burn_in": self.burn_in,
            "lags": lags,
            "lag_roles": lag_roles,
            "seasonal_lags": seasonal_lags,
            "nodes": nodes,
            "outcome": outcome,
            "topology": topology,
            "prior_ranges": self.priors.copy(),
            "noise_weights": self.noise_weights.copy(),
            "effect_weights": self.effect_weights.copy(),
            "parent_combination_weights": self.parent_combination_weights.copy(),
            "node_link_weights": self.node_link_weights.copy(),
            "pairwise_interaction_probability": self.pairwise_interaction_probability,
            "additive_interaction_probability": self.additive_interaction_probability,
            "composite_effect_probability": self.composite_effect_probability,
            "process_episode": process_episode,
            "discrete_weights": self.discrete_weights.copy(),
            "switch_profile_weights": self.switch_profile_weights.copy(),
            "equations": {
                "process": "x=quantize(intercept+level+seasonal+shock+residual); residual=persistence*previous_residual+forcing+heteroskedastic_noise",
                "level": "slope+=slope_walk*N(0,1); on event reverse/reset/add event_slope; level+=slope+walk_scale*N(0,1)+event*level_jump",
                "shock": "shock=shock_decay*previous_shock+event*shock_size",
                "seasonal": "sum(amplitude*(1+modulation*sin(2*pi*t/modulation_period))*sin(2*pi*t/period+phase))",
                "forcing": "node_link(selected-parent joint map plus remaining additive edge effects); optional additive joint contribution; log_sum plus exponential link gives positive multi-parent power laws; innovations remain outside link; legacy schemas retain original maps",
                "noise": "saved noise draw*(1+heteroskedasticity*abs(tanh(previous_residual)))",
                "variance": "center innovations by their distribution mean; scale centered innovations and white noise by time-varying volatility and optional (latent_level/initial_level)**noise_elasticity; retain innovation mean",
                "multiplicative_seasonality": "seasonal * seasonal_base_scale * (latent_level/initial_level)**seasonal_elasticity",
                "outcome": "residual includes optional residual AR plus outcome_ar_weights times full normalized historical outcome; physical outcome=baseline+scale*x; operation overrides output during shutdown or adds a ramp reset on binary transitions",
                "trend": "optional uncapped outcome trend starts at t=0; linear/accelerating/decelerating rates, a permanent event-triggered slope multiplier, or multiplicative level L[t+1]=L[t]*exp(log_rate) contributing L[t]-initial_level; ordinary shock decay does not restore a changed slope",
                "ringing": "state[t]=exp(-decay+i*2*pi/period)*state[t-1]+amplitude*event[t]*exp(i*phase); add imag(state) for sine or real(state) for cosine",
                "seasonal_v3": "integrated phase with smoothly varying frequency; saved Fourier sine/cosine expansion (or legacy harmonic wave) times alternating-cycle envelope and optional exponential amplitude growth",
                "fourier": "sum_k(a_k(t)*cos(k*phase)+b_k(t)*sin(k*phase)); coefficients optionally interpolate smoothly between saved start/end spectra; no DC term",
                "quantization": "offset+step*round((x-offset)/step), bypassed when step=0",
                "initialization": "States zero except saved slopes; t starts at -burn_in; discard t<0. Outcome-lag edge parents use full normalized historical x internally; exported columns contain only current outcome, time index and visible current covariates.",
                "graph": "Topological contemporaneous edges plus positive temporal lags. Past values are separate vertices when time is unrolled; innovations are independent.",
                "episode": "causal finite-rise/fall command envelope; physical y=idle+amplitude*envelope*(1+detail_fraction*tanh(latent/detail_scale)+noise_fraction*tanh(standardized_white_error)+seasonal_fraction*tanh(seasonal))+bounded idle innovation; written to causal history before later rows",
                "formula_source": "saved clock formula or causal setpoint response plus saved innovation law; no empirical centering or output correction",
                "sensor": "optional hard floor/ceiling in recorded units; measured values, including clipping, enter causal history",
                "discrete": "exogenous Bernoulli or saved Markov transition matrix over binary/multi-level values; no added continuous noise",
                "plant": "optional closed-loop unit driven by the standardized outcome as relative demand; its chosen channel replaces the recorded outcome; other channels may be exported",
                "observation": "optional recording layer after simulation: redundant sensors, removed rows, sensor artifacts sized by the clean column's span, then NaN missingness (MCAR, MAR, structural); never enters causal history",
            },
        }
        return self._sample_recording(
            model, recorded, row_gaps, seasonal_table, noise_profile
        )

    def _budget_drivers(self, nodes, outcome, seed):
        """Budget independent inputs at sampling time, without screening tables.

        This is an excitation budget, not a guarantee of realized importance:
        nonlinear gates can amplify a weak input. Feedback and all causal edges
        remain. The target's own innovation is retained as ordinary measurement/
        process uncertainty; event nodes remain rare impulses rather than being
        turned into sustained binary states.
        """
        if self.max_dominant_drivers is None and not self.persistent_hidden_probability:
            return {}
        r = np.random.default_rng([seed, 0xD21])
        candidates = [n for n in nodes if n["kind"] not in ("event", "switch")]
        selected = candidates
        if self.max_dominant_drivers is not None:
            indices = r.choice(
                len(candidates),
                min(len(candidates), self.max_dominant_drivers),
                replace=False,
            )
            selected = [candidates[i] for i in indices]
        strong = {n["name"] for n in selected}
        gains = {}
        for node in candidates:
            gain = 1.0 if node["name"] in strong else float(r.uniform(0.03, 0.15))
            gains[node["name"]] = gain
            if gain < 1:
                # Budget every outgoing edge together. This includes categorical
                # and formula inputs, not just Gaussian innovation sources.
                for target in nodes + [outcome]:
                    for edge in target.get("edges", []):
                        if edge["parent"] == node["name"]:
                            edge["weight"] *= gain
            if node["visible"] or r.random() >= self.persistent_hidden_probability:
                continue
            if node["kind"] == "bernoulli":
                probability = node.pop("probability")
                exit_rate = float(r.uniform(0.01, 0.08))
                node.update(
                    kind="markov",
                    family="binary_markov",
                    values=[0, 1],
                    transition=[
                        [1 - exit_rate * probability, exit_rate * probability],
                        [
                            exit_rate * (1 - probability),
                            1 - exit_rate * (1 - probability),
                        ],
                    ],
                    initial_state=int(r.random() < probability),
                )
            elif node["kind"] == "markov":
                matrix = np.asarray(node["transition"])
                retain = float(r.uniform(0.5, 0.9))
                node["transition"] = (
                    (1 - retain) * matrix + retain * np.eye(len(matrix))
                ).tolist()
            elif "noise" in node and node["noise"]["scale"]:
                node["noise"]["memory_tau"] = float(r.uniform(8, 80))
                # White sensor jitter need not become a dominant hidden input.
                node["white_noise_scale"] = node.get("white_noise_scale", 0) * 0.2
        return {
            "max_dominant_drivers": self.max_dominant_drivers,
            "dominant_sources": sorted(strong),
            "outgoing_gains": gains,
            "persistent_hidden_probability": self.persistent_hidden_probability,
        }

    def _sample_recording(self, model, recorded, row_gaps, seasonal_table, profile):
        """Closed-loop unit and recording layer. Absent keys replay as before."""
        r, nodes = self.rng, model["nodes"]
        if not seasonal_table:
            model["seasonal_table"] = False
        if profile:
            model["noise_profile"] = profile
        if recorded != model["rows"]:
            model["recorded_rows"] = recorded
        if row_gaps:
            model["row_gaps"] = row_gaps
        next_index = len(nodes)
        # Regulated units hold a level, so they belong to the trend-free profile;
        # a totalizer only ever counts up, so it belongs to the trending one.
        if self.plant_probability and r.random() < self.plant_probability:
            weights = {
                family: weight
                for family, weight in self.plant_weights.items()
                if (family == "totalizer") != model["trend_free"]
            }
            share = sum(weights.values()) / sum(self.plant_weights.values())
            if share > 0 and (model["trend_free"] or r.random() < share):
                model["plant"] = sample_plant(r, model["rows"], weights, next_index)
                next_index += len(model["plant"]["exports"])
        columns = [{"name": "outcome", "discrete": False}]
        columns += [
            {"name": n["name"], "discrete": n["kind"] not in ("process", "formula")}
            for n in nodes
            if n["visible"]
        ]
        columns += [
            {"name": e["name"], "discrete": e["channel"] in ("status", "mode")}
            for e in model.get("plant", {}).get("exports", [])
        ]
        if (
            self.redundant_sensor_probability
            or self.measurement_probability
            or self.missingness_probability
            or self.resolution_probability
            or self.damping_probability
        ):
            observation = sample_observation(
                r,
                columns,
                {
                    "next_index": next_index,
                    "redundant_sensor_probability": self.redundant_sensor_probability,
                    "measurement_probability": self.measurement_probability,
                    "resolution_probability": self.resolution_probability,
                    "damping_probability": self.damping_probability,
                    "artifact_weights": self.artifact_weights,
                    "missingness_probability": self.missingness_probability,
                    "missingness_weights": self.missingness_weights,
                },
            )
            if any(observation.values()):
                model["observation"] = observation
        if model.get("plant") or model.get("observation"):
            model["univariate"] = len(columns) == 1 and not model.get(
                "observation", {}
            ).get("redundant")
        return model

    def _revise_dynamics(
        self,
        node,
        rows,
        trend_free,
        is_outcome,
        seasonal_table=True,
        noise_profile=None,
    ):
        """Budget cumulative changes against the observation horizon.

        Disabling only slope is insufficient: hidden random walks, permanent
        jumps, and event-induced slopes would still transmit trends downstream.
        """
        r = self.rng
        if r.random() < self.variance_dynamics_probability:
            mode = str(r.choice(["growth", "cyclic", "regime"], p=[0.6, 0.2, 0.2]))
            if r.random() < self.volatility_burst_share:
                mode = "bursts"
            node["volatility"] = {
                "mode": mode,
                "duration": rows,
                "log_change": float(
                    np.log(self._draw("variance_growth_factor"))
                    * (1 if r.random() < 0.85 else -1)
                ),
                "amplitude": float(r.uniform(0.3, 0.8)),
                "cycles": float(r.uniform(1, 4)),
                "phase": float(r.uniform(-np.pi, np.pi)),
                "change_fraction": float(r.uniform(0.25, 0.75)),
            }
            if mode == "bursts":
                # Calm and busy stretches that come and go, not one change.
                node["volatility"].update(
                    burst_tau=self._log_uniform(5, 300),
                    burst_depth=float(r.uniform(0.4, 1.3)),
                    seed=int(r.integers(0, 2**63)),
                )
        if trend_free:
            for key in (
                "slope",
                "slope_walk",
                "walk_scale",
                "event_slope",
                "level_jump",
            ):
                node[key] = 0.0
        else:
            node["slope"] = float(np.sign(node["slope"]) * r.uniform(0.5, 3) / rows)
            node["event_slope"] = (
                float(r.uniform(-2, 2) / rows) if node["slope"] else 0.0
            )
            node["slope_walk"] *= 0.05
            node["level_jump"] *= 0.15
            node["event_mode"] = str(r.choice(["reverse", "reset"]))
        node["level_limit"] = float(r.uniform(2, 6))
        if noise_profile and node["noise"]["kind"] not in ("perlin", "brown"):
            # Smooth wander (Perlin/brown) is process behaviour, not sensor hiss.
            node["noise"]["scale"] *= noise_profile["innovation_factor"]
        if noise_profile:
            # A resolution fixed for unit-scale variation would flatten a quiet
            # signal to a single level; coarse sensors are a recording artifact.
            node["quantization_step"] *= min(1.0, noise_profile["innovation_factor"])
        if is_outcome:
            # A visible stochastic floor prevents seasonal/slow components from
            # producing nearly deterministic ramps. It is a causal innovation,
            # not noise added after seeing or normalizing the finished series.
            node["white_noise_scale"] = float(r.uniform(0.15, 0.45))
            if noise_profile:
                node["white_noise_scale"] = noise_profile["white_noise_scale"]
            chance = (
                self.seasonal_wander_probability
                if seasonal_table
                else self.nonseasonal_wander_probability
            )
            if chance and r.random() < chance:
                # Mean-reverting load wander: slow, bounded, not a trend. Without
                # it a cycle-free outcome is little more than its noise floor.
                node["wander"] = {
                    "tau": self._log_uniform(8, 600),
                    "scale": self._log_uniform(0.3, 3.0),
                    "seed": int(r.integers(0, 2**63)),
                }
            if (
                seasonal_table
                and not node["seasons"]
                and r.random() < self.forced_season_probability
            ):
                node["seasons"] = [
                    {
                        "period": min(
                            rows / 3, self._draw("seasonal_period", log=True)
                        ),
                        "amplitude": 1.0,
                        "phase": float(r.uniform(-np.pi, np.pi)),
                        "modulation": 0.2,
                        "modulation_period": float(rows),
                    }
                ]
        for season in node["seasons"]:
            # Limit periods relative to the context so a partial cycle is less
            # often mistaken for a straight trend. Harmonics supply peaked,
            # asymmetric and multi-peak cycles instead of only sine humps.
            season["period"] = float(min(season["period"], max(6, rows / 3)))
            if is_outcome:
                season["amplitude"] = max(0.6, season["amplitude"])
            season["harmonics"] = [
                {
                    "order": k,
                    "weight": float(r.uniform(0.1, 0.8) / np.sqrt(k)),
                    "phase": float(r.uniform(-np.pi, np.pi)),
                }
                for k in range(2, int(r.integers(3, 6)))
            ]
            season["alternation"] = (
                float(r.uniform(0.15, 0.55)) if r.random() < 0.4 else 0.0
            )
            season["frequency_depth"] = (
                float(r.uniform(0.05, 0.35)) if r.random() < 0.45 else 0.0
            )
            season["frequency_period"] = float(r.uniform(2, 6) * season["period"])
            season["growth"] = (
                float(r.uniform(0.3, 1.1) / rows)
                if not trend_free and r.random() < 0.4
                else 0.0
            )
            if r.random() < self.fourier_probability:
                season["fourier"] = self._sample_fourier(season, rows)
                # The new expansion replaces, rather than doubles, the older
                # sine-plus-harmonics waveform. Legacy specs still replay below.
                season.pop("harmonics", None)
            if (
                self.calendar_gate_probability
                and r.random() < self.calendar_gate_probability
            ):
                # Working days versus rest days: some whole cycles run reduced.
                cycles = int(r.choice([5, 7, 7, 7, 10, 14]))
                season["gate"] = {
                    "cycles": cycles,
                    "off": int(r.integers(1, max(2, cycles // 2))),
                    "shift": int(r.integers(cycles)),
                    "depth": float(r.uniform(0.3, 1.0)),
                    "level": float(r.uniform(-0.8, 0.3)) if r.random() < 0.5 else 0.0,
                }

    def _sample_fourier(self, season, rows):
        """Project operational templates onto a finite sine/cosine basis.

        Template parameters vary per season, so a family is a shape prior, not
        a repeated canned waveform. No observations enter this construction.
        """
        r = self.rng
        families = list(self.fourier_weights)
        probabilities = np.array(list(self.fourier_weights.values()), dtype=float)
        family = str(r.choice(families, p=probabilities / probabilities.sum()))
        # Frequency warping can speed up the clock. Keep the fastest carrier
        # harmonic below Nyquist, with headroom for smooth modulation sidebands.
        limit = max(
            1,
            int(
                np.floor(
                    0.45
                    * season["period"]
                    / (1 + abs(season.get("frequency_depth", 0)))
                )
            ),
        )
        count = min(self.max_fourier_harmonics, limit)
        count = int(r.integers(min(3, count), count + 1))
        orders = np.arange(1, count + 1)
        grid = np.arange(2048) / 2048
        smoothing = float(r.uniform(0.0, 3.0))

        def coefficients():
            duty = float(r.uniform(0.12, 0.88))
            if family == "sine":
                template = np.sin(2 * np.pi * grid)
            elif family == "pulse":
                template = (grid < duty).astype(float)
            elif family == "sawtooth":
                template = float(r.choice([-1, 1])) * grid
            elif family == "triangle":
                template = np.where(grid < duty, grid / duty, (1 - grid) / (1 - duty))
            elif family == "full_rectified":
                # One rectified lobe per declared period; avoid silently
                # halving the cycle associated with an outcome seasonal lag.
                template = np.sin(np.pi * grid) ** float(r.uniform(0.5, 3))
            elif family == "half_rectified":
                template = np.maximum(0, np.sin(2 * np.pi * grid)) ** float(
                    r.uniform(0.5, 3)
                )
            elif family == "parabolic":
                template = abs(2 * grid - 1) ** float(r.uniform(1.3, 4))
            elif family == "multi_peak":
                template = np.zeros_like(grid)
                for _ in range(int(r.integers(2, 5))):
                    distance = (grid - r.uniform(0, 1) + 0.5) % 1 - 0.5
                    template += r.uniform(0.3, 1.5) * np.exp(
                        -0.5 * (distance / r.uniform(0.03, 0.15)) ** 2
                    )
            else:
                template = None
            if template is None:
                # Smooth spectra dominate, while shallow decay retains occasional
                # jagged load profiles. Both sine and cosine coefficients vary.
                decay = float(r.uniform(0.8, 2.5))
                a, b = r.normal(size=(2, count)) / orders**decay
            else:
                # Subtracting DC separates cyclic load from intercept/trend.
                centered = template - template.mean()
                spectrum = np.fft.rfft(centered) / len(grid)
                a, b = 2 * spectrum[orders].real, -2 * spectrum[orders].imag
                # Small shape perturbations avoid exact textbook repetitions.
                perturbation = r.uniform(0, 0.12) * np.sqrt(np.sum(a * a + b * b))
                a += r.normal(size=count) * perturbation / orders**1.5
                b += r.normal(size=count) * perturbation / orders**1.5
            taper = np.exp(-smoothing * (orders / (count + 1)) ** 2)
            a, b = a * taper, b * taper
            # Parseval normalization keeps variance comparable across families
            # and harmonic counts: unit expansion has RMS 1/sqrt(2), like sine.
            norm = max(1e-12, float(np.sqrt(np.sum(a * a + b * b))))
            return {"cos": (a / norm).tolist(), "sin": (b / norm).tolist()}

        spec = {
            "family": family,
            "orders": orders.tolist(),
            "start": coefficients(),
            "smoothing": smoothing,
        }
        if r.random() < self.seasonal_shape_drift_probability:
            spec.update(
                end=coefficients(),
                transition_center=float(r.uniform(0.25, 0.75) * rows),
                transition_width=float(r.uniform(0.15, 0.4) * rows),
            )
        return spec

    @staticmethod
    def trend_score(frame):
        """Linear-trend R²: explicit, scale-independent screening criterion.

        This measures discernible linear trend, not formal stationarity; curved
        cycles, variance evolution and operational shutdowns remain possible.
        """
        values = frame["outcome"].to_numpy(dtype=float)
        keep = np.isfinite(values)  # Recorded tables may contain missing values.
        if keep.sum() < 3 or np.std(values[keep]) < 1e-12:
            return 0.0
        return float(np.corrcoef(np.arange(len(values))[keep], values[keep])[0, 1] ** 2)

    @staticmethod
    def trend_strength(frame, model):
        """Variance explained by the specified low-dimensional trend shape.

        Include a hinge at a saved shock time: a strong reversal may have almost
        zero global linear slope. This assesses the observed series, not merely
        the presence of nonzero parameters. No output values are modified.
        """
        y = frame["outcome"].to_numpy()
        # With unrecorded prehistory the window starts `offset` rows into the
        # simulated span; every shape below is evaluated on that original clock.
        offset = max(0, model["rows"] - len(y)) if "recorded_rows" in model else 0
        total = offset + len(y)
        index = np.arange(offset, total)
        t = index.astype(float) / max(1, total - 1)
        columns = [np.ones(len(y)), t]
        trend = model["outcome"].get("trend", {})
        if trend.get("kind") == "multiplicative":
            # Fit in observation units without logging possibly signed/noisy
            # observations. The known exponential shape is a structural prior,
            # not a curve selected by searching over the 01_generated outcome.
            columns = [
                np.ones(len(y)),
                np.expm1(trend["log_rate"] * (index + 1)),
            ]
        if trend.get("kind") in LIFECYCLE_KINDS:
            columns.append(lifecycle_path(trend, index))
        if trend.get("kind") in ("accelerating", "decelerating"):
            columns.append(t**2)
        if trend.get("event_time") is not None:
            columns.append(np.maximum(0, t - trend["event_time"] / max(1, total - 1)))
        operation = model["outcome"].get("operation", {})
        if operation.get("mode") == "episode":
            # Process humps are structured nonstationary shapes even when both
            # ends return to idle. Fit the saved causal envelope for screening.
            source = next(n for n in model["nodes"] if n["name"] == operation["parent"])
            clock = np.arange(-model["burn_in"], total)
            if operation.get("activity_source") == "direct":
                activity = formula_path(source, clock - operation.get("lag", 0))
            else:
                command = (clock >= source["start"] + operation.get("lag", 0)) & (
                    clock < source["stop"] + operation.get("lag", 0)
                )
                activity = episode_envelope(operation, command)
            columns.append(activity[model["burn_in"] + offset :])
        design = np.column_stack(columns)
        fitted = design @ np.linalg.lstsq(design, y, rcond=None)[0]
        total = np.sum((y - y.mean()) ** 2)
        return (
            float(max(0, 1 - np.sum((y - fitted) ** 2) / total))
            if total > 1e-20
            else 0.0
        )

    @staticmethod
    def _trend_path(trend, time, events):
        """Integrate a persistent slope state; warm-up leaves this state at zero."""
        if trend["kind"] in LIFECYCLE_KINDS:
            return lifecycle_path(trend, time)
        values = np.zeros(len(time))
        level, changed = 0.0, False
        horizon = max(1, int(time[-1]) + 1)
        if trend["kind"] == "multiplicative":
            level = trend["initial_level"]
            multiplier = np.exp(trend["log_rate"])
            for i, t in enumerate(time):
                if t >= 0:
                    level *= multiplier
                    values[i] = level - trend["initial_level"]
            return values
        for i, t in enumerate(time):
            if t < 0:
                continue
            rate = trend["initial_slope"]
            if trend["kind"] == "accelerating":
                rate *= 0.3 + 1.4 * t / horizon
            elif trend["kind"] == "decelerating":
                rate *= 1.7 - 1.4 * t / horizon
            elif trend["kind"] == "broken":
                changed |= bool(t >= trend["event_time"] and events[i])
                if changed:
                    rate *= trend["slope_multiplier"]
            level += rate
            values[i] = level
        return values

    def simulate(self, model):
        transform = model.get("observation_transform")
        if transform:
            if transform.get("kind") != "sign_inversion":
                raise ValueError("Unsupported observation transform")
            source = model.copy()
            source.pop("observation_transform")
            # Transform only the observation view; negating inputs inside the
            # equations would resimulate a different causal realization.
            return invert_frame(self.simulate(source))
        selection = model.get("observation_selection")
        if selection:
            # The DAG still runs at its ORIGINAL clock rate. Only observations
            # are thinned; skipping latent updates would change the process.
            source = model.copy()
            source.pop("observation_selection")
            source["rows"] = selection.get("simulated_rows", selection["source_rows"])
            original = self.simulate(source)
            if selection["kind"] == "trailing_aggregate":
                frame = aggregate_frame(
                    original, selection["window_size"], selection["statistic"]
                )
            else:
                frame = coarsen_frame(
                    original, selection["window_size"], selection["offset"]
                )
            if len(frame) != model["rows"]:
                raise ValueError("Coarse observation count disagrees with saved SCM")
            return frame
        if model["version"] == 1:
            try:
                from .legacy_scm_generator import OperationalSCMGenerator as Legacy
            except ImportError:
                from legacy_scm_generator import OperationalSCMGenerator as Legacy
            # Dispatch on a legacy instance so its noise/effect evaluators also
            # use the original schema, preserving existing dataset replay.
            frame = Legacy(self.output_dir).simulate(model)
            return frame.drop(
                columns=[c for c in frame if c.startswith("outcome_lag_")]
            )
        if model["version"] not in (2, 3):
            raise ValueError("Unsupported SCM schema version")
        return self._observe(self._simulate_process(model), model)

    @staticmethod
    def _observe(frame, model):
        """Recording layer: what was logged, given what the process did."""
        if not any(k in model for k in ("observation", "recorded_rows", "row_gaps")):
            return frame
        return apply_observation(
            frame,
            model.get("observation", {}),
            model.get("recorded_rows"),
            model.get("row_gaps"),
            model["observation_seed"],
        )

    def _simulate_process(self, model):
        """Clean process values over the full simulated span, before recording."""
        # Validate before evaluating any equations. A positive delay may refer
        # to any node's past; a zero delay must refer to an earlier current node.
        self.graph_spec(model)
        r = np.random.default_rng(model["observation_seed"])
        rows, warmup, lags = model["rows"], model["burn_in"], model["lags"]
        length = rows + warmup
        names = [n["name"] for n in model["nodes"]] + ["outcome"]
        specs = model["nodes"] + [model["outcome"]]
        # Precompute innovations once. Visibility never affects random draws.
        history = {name: np.zeros(length) for name in names}
        residuals = {name: np.zeros(length) for name in names}
        active_nodes = [
            (name, spec)
            for name, spec in zip(names, specs)
            if spec.get("kind") not in EXOGENOUS_KINDS
        ]
        # Resolve source buffers, lag aliases and fixed joint-map arrays once.
        # Keep per-row arithmetic and random-draw order identical to replay.
        edge_inputs, parent_maps = {}, {}
        for name, spec in active_nodes:
            sources = []
            for edge in spec["edges"]:
                parent, lag = edge["parent"], edge["lag"]
                if parent.startswith("outcome_lag_"):
                    lag += int(parent.rsplit("_", 1)[1])
                    parent = "outcome"
                sources.append((self._bind_effect(edge), history[parent], lag))
            edge_inputs[name] = sources
            parent_maps[name] = prepare_parent_combination(
                spec.get("parent_combination"), len(sources)
            )
        residual_ar = list(zip(model["outcome"]["ar_weights"], lags))
        outcome_ar = list(zip(model["outcome"].get("outcome_ar_weights", []), lags))
        randoms, season_values, white_errors = {}, {}, {}
        noise_scales, noise_means = {}, {}
        time = np.arange(-warmup, rows)
        for name, spec in zip(names, specs):
            if spec.get("kind") == "event":
                history[name][:] = r.binomial(1, spec["probability"], length)
                for event_time in spec.get("forced_event_times", []):
                    history[name][warmup + event_time] = 1
            elif spec.get("kind") == "switch":
                if spec.get("profile") == "single_episode":
                    history[name][:] = (time >= spec["start"]) & (time < spec["stop"])
                    continue
                # Semi-Markov dwell times create contiguous outages rather than
                # isolated Bernoulli blips. The node remains binary and can be hidden.
                position, on = 0, spec["initial_on"]
                while position < length:
                    low, high = spec["on_duration"] if on else spec["off_duration"]
                    duration = int(r.integers(low, high + 1))
                    history[name][position : position + duration] = int(on)
                    position += duration
                    on = not on
            elif spec.get("kind") == "formula":
                history[name][:] = formula_path(spec, time)
                if spec["noise"]["scale"]:
                    history[name] += self._noise(r, spec["noise"], length)
            elif spec.get("kind") in DISCRETE_KINDS:
                history[name][:] = discrete_path(spec, r, length)
            else:
                randoms[name] = (
                    self._noise(r, spec["noise"], length),
                    r.normal(size=length),
                    r.normal(size=length),
                )
                if model["version"] == 3:
                    season_values[name] = self._seasonal(spec["seasons"], time)
                    white_errors[name] = r.normal(
                        0, spec.get("white_noise_scale", 0), length
                    )
                if spec.get("volatility") or spec.get("multiplicative_scaling"):
                    noise_scales[name], seasonal_scale = self._component_scales(
                        spec, time
                    )
                    noise_means[name] = self._noise_mean(spec["noise"])
                    season_values[name] *= seasonal_scale

        trend = model["outcome"].get("trend")
        ringing_values = {
            name: self._ringing(
                spec["ringing"],
                self._delay_values(
                    history[spec["ringing"]["parent"]], spec["ringing"].get("lag", 0)
                ),
            )
            for name, spec in zip(names, specs)
            if spec.get("ringing")
        }
        trend_values = (
            self._trend_path(
                trend,
                time,
                self._delay_values(
                    history[trend["event_parent"]], trend.get("event_lag", 0)
                )
                if trend["event_parent"]
                else np.zeros(length),
            )
            if trend
            else np.zeros(length)
        )
        recovery, ramp, previous_on, surge = 1.0, 0.0, True, 0.0
        wander_values = (
            self._wander(model["outcome"]["wander"], length)
            if model["outcome"].get("wander")
            else None
        )
        envelopes = {
            name: (
                self._delay_values(
                    history[spec["operation"]["parent"]],
                    spec["operation"].get("lag", 0),
                )
                if spec["operation"].get("activity_source") == "direct"
                else episode_envelope(
                    spec["operation"],
                    self._delay_values(
                        history[spec["operation"]["parent"]],
                        spec["operation"].get("lag", 0),
                    ),
                )
            )
            for name, spec in active_nodes
            if spec.get("operation", {}).get("mode") == "episode"
        }
        outcome_operation = model["outcome"].get("operation", {})
        measured_outcome = (
            np.empty(length)
            if model["outcome"].get("sensor")
            or outcome_operation.get("schema_version") == 2
            else None
        )

        version = model["version"]

        def node_step(name, spec):
            """Per-row update of one node, with buffers and constants bound once.

            Resolving them here, rather than in a dozen dictionaries per row,
            changes no arithmetic or its order. Buffers and dynamic state
            remain private to this simulation; the saved SCM is never changed.
            """
            is_outcome = name == "outcome"
            noise, walk, slope_walk = randoms[name]
            level, slope, shock = 0.0, spec["slope"], 0.0  # shock decays
            node_residuals, node_history = residuals[name], history[name]
            node_seasons, node_white = season_values.get(name), white_errors.get(name)
            node_noise_scale = noise_scales.get(name)
            noise_mean = noise_means.get(name)
            inputs, parent_map = edge_inputs[name], parent_maps[name]
            node_ringing, envelope = ringing_values.get(name), envelopes.get(name)
            event_source = (
                history[spec["event_parent"]] if spec["event_parent"] else None
            )
            event_lag = spec.get("event_lag", 0)
            event_mode, event_slope = spec.get("event_mode"), spec.get("event_slope")
            slope_walk_scale, walk_scale = spec["slope_walk"], spec["walk_scale"]
            level_jump = spec["level_jump"]
            limited = "level_limit" in spec
            level_high = spec["level_limit"] if limited else None
            level_low = -level_high if limited else None
            shock_decay, shock_size = spec["shock_decay"], spec["shock_size"]
            has_combination = "parent_combination" in spec
            combination = spec.get("parent_combination")
            interaction = None if has_combination else spec["interaction"]
            link = spec.get("node_link")
            link_active = link is not None and link["mode"] != "identity"
            node_link = bind_node_link(link)
            persistence = spec.get("persistence")
            has_gain = "forcing_gain" in spec
            forcing_gain = spec.get("forcing_gain")
            innovation_gain = spec.get("innovation_gain")
            heteroskedasticity, intercept = (
                spec["heteroskedasticity"],
                spec["intercept"],
            )
            quantization_step = spec["quantization_step"]
            quantization_offset = spec.get("quantization_offset")
            operation = spec.get("operation")
            if operation:
                operation_source = history[operation["parent"]]
                operation_lag = operation.get("lag", 0)
            angle = spec.get("domain") == "angle"
            sensor = spec.get("sensor")
            measured = is_outcome and measured_outcome is not None
            if version == 3:
                # Plain floats are the same IEEE doubles as the array elements;
                # sums and products of them skip NumPy's scalar dispatch. The
                # seasonal term stays a NumPy scalar, so `value` below does too.
                noise, walk, slope_walk = (
                    noise.tolist(),
                    walk.tolist(),
                    slope_walk.tolist(),
                )
                node_white = node_white.tolist()
                if node_noise_scale is not None:
                    node_noise_scale = node_noise_scale.tolist()
            previous = 0.0

            def step(i):
                nonlocal level, slope, shock, previous
                nonlocal recovery, ramp, previous_on, surge
                reading = None
                event = (
                    (event_source[i - event_lag] if i >= event_lag else 0.0)
                    if event_source is not None
                    else 0
                )
                slope += slope_walk_scale * slope_walk[i]
                if event:
                    slope = (
                        -slope
                        if event_mode == "reverse"
                        else (
                            event_slope
                            if event_mode == "reset"
                            else slope + event_slope
                        )
                    )
                level += slope + walk_scale * walk[i] + event * level_jump
                if limited:
                    level = float(
                        level_low
                        if level < level_low
                        else level_high
                        if level > level_high
                        else level
                    )
                shock = shock_decay * shock + event * shock_size
                if version == 3:
                    seasonal = node_seasons[i]
                else:
                    t = i - warmup
                    seasonal = sum(
                        s["amplitude"]
                        * (
                            1
                            + s["modulation"]
                            * np.sin(2 * np.pi * t / s["modulation_period"])
                        )
                        * np.sin(2 * np.pi * t / s["period"] + s["phase"])
                        for s in spec["seasons"]
                    )
                effects = [
                    effect(source[i - lag] if i >= lag else 0.0)
                    for effect, source, lag in inputs
                ]
                if has_combination:
                    forcing = combine_parents(effects, combination, parent_map)
                else:
                    forcing = sum(effects)
                    if interaction:
                        forcing += (
                            interaction * np.tanh(effects[0]) * np.tanh(effects[1])
                        )
                if link_active:
                    forcing = node_link(forcing)
                if is_outcome:
                    # Lists, not a manual loop: sum() keeps its own arithmetic.
                    ar = sum(
                        [
                            w * node_residuals[i - lag]
                            for w, lag in residual_ar
                            if i >= lag
                        ]
                    )
                    ar += sum(
                        [
                            w * (node_history[i - lag] if i >= lag else 0.0)
                            for w, lag in outcome_ar
                        ]
                    )
                else:
                    ar = persistence * previous
                innovation = noise[i]
                if node_noise_scale is not None:
                    # Scale deviations, not the distribution's location. Thus
                    # changing variance need not also change the expected mean.
                    innovation = (
                        noise_mean + (innovation - noise_mean) * node_noise_scale[i]
                    )
                if has_gain:
                    forcing *= forcing_gain
                    innovation *= innovation_gain
                # `previous` is this node's residual one row earlier. A zero
                # coefficient multiplies the innovation by exactly one, unless
                # that residual is NaN (the only value unequal to itself).
                residual = (
                    ar
                    + forcing
                    + (
                        innovation * (1 + heteroskedasticity * abs(np.tanh(previous)))
                        if heteroskedasticity or previous != previous
                        else innovation
                    )
                )
                if version == 3:
                    residual += node_white[i] * (
                        node_noise_scale[i] if node_noise_scale is not None else 1
                    )
                previous = residual
                if is_outcome:
                    node_residuals[i] = residual  # Only its lags are read back.
                value = intercept + level + shock + seasonal + residual
                if node_ringing is not None:
                    value += node_ringing[i]
                if is_outcome:
                    value += trend_values[i]
                    if wander_values is not None:
                        value += wander_values[i]
                if quantization_step:
                    # rint is what np.round does for zero decimals.
                    value = quantization_offset + quantization_step * np.rint(
                        (value - quantization_offset) / quantization_step
                    )
                if operation:
                    on = bool(
                        operation_source[i - operation_lag]
                        if i >= operation_lag
                        else 0.0
                    )
                    if operation["mode"] == "episode":
                        activity = envelope[i]
                        # Preserve visible active-process variation even if a
                        # large latent offset saturates the parent-detail map.
                        # These are the existing causal innovations/seasons,
                        # expressed as saved fractions of process amplitude.
                        white = (
                            node_white[i] / max(spec.get("white_noise_scale", 0), 1e-12)
                            if node_white is not None
                            else 0.0
                        )
                        texture = operation.get("noise_fraction", 0) * np.tanh(
                            white
                        ) + operation.get("seasonal_fraction", 0) * np.tanh(seasonal)
                        reading = (
                            operation["idle_value"]
                            + operation["amplitude"]
                            * activity
                            * (
                                1
                                + texture
                                + operation["detail_fraction"]
                                * np.tanh(value / operation["detail_scale"])
                            )
                            + operation["idle_noise_scale"] * np.tanh(innovation)
                        )
                        value = (reading - spec["baseline"]) / spec["scale"]
                    elif operation["mode"] == "shutdown":
                        coast = operation.get("stop_tau", 0.0)
                        if not on:
                            # Instant stop, or an exponential coast/cool-down.
                            recovery = recovery * math.exp(-1 / coast) if coast else 0.0
                        elif operation.get("restart_shape") == "exponential":
                            recovery += operation["restart_rate"] * (1 - recovery)
                        else:
                            recovery = min(1.0, recovery + operation["restart_rate"])
                        if on and not previous_on:
                            surge = operation.get("restart_overshoot", 0.0)
                        flat = (operation["flat_value"] - spec["baseline"]) / spec[
                            "scale"
                        ]
                        # Inrush: the running level is briefly exceeded on restart.
                        value = flat + recovery * (1 + surge) * (value - flat)
                        surge *= operation.get("overshoot_decay", 0.0)
                        if not on and not coast:
                            reading = operation["flat_value"]
                    else:
                        # Reset accumulated ramp on each binary transition:
                        # opposite slopes on the two states yield repeating
                        # sawtooth-like charging/discharging cycles.
                        if on != previous_on:
                            ramp = 0.0
                        ramp += operation["ramp_rate"] * (1 if on else -1)
                        value += ramp
                    previous_on = on
                if angle:
                    value = float(np.arctan2(np.sin(value), np.cos(value)))
                if measured:
                    reading = (
                        spec["baseline"] + spec["scale"] * value
                        if reading is None
                        else reading
                    )
                    reading = sensor_reading(reading, sensor) if sensor else reading
                    measured_outcome[i] = reading
                    # Store the measured value before later rows read its lags.
                    value = (reading - spec["baseline"]) / spec["scale"]
                elif sensor:
                    value = sensor_reading(value, sensor)
                node_history[i] = value

            return step

        # Nodes share the restart/ramp state above, as they always have.
        steps = [node_step(name, spec) for name, spec in active_nodes]
        for i in range(length):
            for step in steps:
                step(i)
        outcome = model["outcome"]
        physical = (
            measured_outcome
            if measured_outcome is not None
            else outcome["baseline"] + outcome["scale"] * history["outcome"]
        )
        operation = outcome.get("operation")
        if measured_outcome is None and operation and operation["mode"] == "shutdown":
            physical[history[operation["parent"]] == 0] = operation["flat_value"]
        plant = model.get("plant")
        if plant:
            # The SCM outcome becomes the unit's demand; the unit's own channel
            # (level, pressure, controller output ...) is what gets recorded.
            channels = simulate_plant(plant, history["outcome"], time)
            physical = (
                outcome["baseline"]
                + outcome["scale"]
                * plant["output_gain"]
                * channels[plant["outcome_channel"]]
            )
        data = {"time_index": np.arange(rows), "outcome": physical[warmup:]}
        for node in model["nodes"]:
            if node["visible"]:
                data[node["name"]] = history[node["name"]][warmup:]
        if plant:
            for export in plant["exports"]:
                data[export["name"]] = channels[export["channel"]][warmup:]
        frame = pd.DataFrame(data)
        if not np.isfinite(frame.to_numpy()).all():
            raise FloatingPointError(
                "Non-finite SCM observations; inspect sampled parameters"
            )
        return frame

    @staticmethod
    def _wander(spec, length):
        """Stationary first-order Gauss-Markov path with a saved time constant."""
        rng = np.random.default_rng(spec["seed"])
        persistence = math.exp(-1 / spec["tau"])
        # Start inside the stationary law. Drawn first, so a longer request
        # extends the same path instead of producing a different one.
        value = rng.normal(0, spec["scale"])
        steps = rng.normal(0, spec["scale"] * math.sqrt(1 - persistence**2), length)
        result = np.empty(length)
        for index in range(length):
            value = persistence * value + steps[index]
            result[index] = value
        return result

    @staticmethod
    def _delay_values(values, lag):
        """Shift a precomputed exogenous path without wrapping future values."""
        if not lag:
            return values
        delayed = np.zeros_like(values)
        if lag < len(values):
            delayed[lag:] = values[:-lag]
        return delayed

    @staticmethod
    def _component_scales(spec, time):
        """Structural envelopes, evaluated without looking at observed values.

        Separate noise and seasonal elasticities permit faster/slower growth.
        Independent volatility applies even with a flat or linear mean. Working
        with the latent multiplicative level avoids a noisy scale-feedback loop.
        """
        noise_scale, seasonal_scale = np.ones(len(time)), np.ones(len(time))
        volatility = spec.get("volatility")
        if volatility:
            progress = np.maximum(time, 0) / volatility["duration"]
            mode = volatility["mode"]
            if mode == "growth":
                log_scale = volatility["log_change"] * progress
            elif mode == "cyclic":
                log_scale = volatility["amplitude"] * (
                    np.sin(
                        2 * np.pi * volatility["cycles"] * progress
                        + volatility["phase"]
                    )
                    - np.sin(volatility["phase"])
                )
            elif mode == "regime":
                log_scale = np.where(
                    progress >= volatility["change_fraction"],
                    volatility["log_change"],
                    0,
                )
            elif mode == "bursts":
                persistence = math.exp(-1 / volatility["burst_tau"])
                burst_rng = np.random.default_rng(volatility["seed"])
                shocks = burst_rng.normal(0, math.sqrt(1 - persistence**2), len(time))
                shocks[0] = burst_rng.normal()
                log_scale = volatility["burst_depth"] * lfilter(
                    [1.0], [1.0, -persistence], shocks
                )
            else:
                raise ValueError(f"Unknown volatility mode: {mode}")
            noise_scale *= np.exp(np.clip(log_scale, -4, 4))
        scaling = spec.get("multiplicative_scaling")
        if scaling:
            log_ratio = spec["trend"]["log_rate"] * np.maximum(time + 1, 0)
            noise_scale *= np.exp(scaling["noise_elasticity"] * log_ratio)
            seasonal_scale *= scaling["seasonal_base_scale"] * np.exp(
                scaling["seasonal_elasticity"] * log_ratio
            )
        return noise_scale, seasonal_scale

    @classmethod
    def _noise_mean(cls, spec):
        """Population location, so variance modulation does not scale a bias."""
        if spec["kind"] in cls.MIXTURES:
            base_mean = sum(
                w * cls._noise_mean(c)
                for w, c in zip(spec["weights"], spec["components"])
            )
        elif spec["kind"] == "brown":
            base_mean = spec["initial_value"]
        else:
            base_mean = 0.0
        return spec["location"] + spec["scale"] * base_mean

    @staticmethod
    def _ringing(spec, events):
        """Stable superposition of damped sinusoidal responses to event pulses.

        The complex recurrence computes the convolution in O(n), carrying both
        phase and exponential decay. A new pulse adds energy without resetting
        an existing oscillation; no future observations enter the calculation.
        """
        pole = np.exp(complex(-spec["decay"], 2 * np.pi / spec["period"]))
        impulse = spec["amplitude"] * np.exp(1j * spec["phase"])
        state = 0j
        result = np.empty(len(events))
        for i, event in enumerate(events):
            state = pole * state + impulse * event
            result[i] = state.imag if spec["wave"] == "sine" else state.real
        return result

    @staticmethod
    def _seasonal(seasons, time):
        result = np.zeros(len(time))
        for s in seasons:
            period = s.get("frequency_period", s["period"])
            # Integrate instantaneous frequency rather than substituting a
            # changing period into sin(t/period), which produces phase jumps.
            warped_time = time + s.get("frequency_depth", 0) * period / (2 * np.pi) * (
                1 - np.cos(2 * np.pi * time / period)
            )
            phase = 2 * np.pi * warped_time / s["period"] + s["phase"]
            if "fourier" in s:
                fourier = s["fourier"]
                mix = (
                    (
                        0.5
                        * (
                            1
                            + np.tanh(
                                (time - fourier["transition_center"])
                                / fourier["transition_width"]
                            )
                        )
                    )
                    if "end" in fourier
                    else 0.0
                )
                wave = np.zeros(len(time))
                start, end = fourier["start"], fourier.get("end", fourier["start"])
                for index, order in enumerate(fourier["orders"]):
                    a = start["cos"][index] * (1 - mix) + end["cos"][index] * mix
                    b = start["sin"][index] * (1 - mix) + end["sin"][index] * mix
                    wave += a * np.cos(order * phase) + b * np.sin(order * phase)
            else:
                # Preserve the exact arithmetic of previously saved SCMs.
                wave = np.sin(phase)
                harmonics = s.get("harmonics", [])
                for h in harmonics:
                    wave += h["weight"] * np.sin(h["order"] * phase + h["phase"])
                wave /= np.sqrt(1 + sum(h["weight"] ** 2 for h in harmonics))
            envelope = 1 + s["modulation"] * np.sin(
                2 * np.pi * time / s["modulation_period"]
            )
            envelope *= 1 + s.get("alternation", 0) * np.cos(phase / 2)
            envelope *= np.exp(np.clip(s.get("growth", 0) * time, -3, 3))
            gate = s.get("gate")
            if gate:
                cycle = np.floor(phase / (2 * np.pi)) + gate["shift"]
                resting = cycle % gate["cycles"] < gate["off"]
                envelope = np.where(resting, envelope * (1 - gate["depth"]), envelope)
                result += s["amplitude"] * gate["level"] * resting
            result += s["amplitude"] * envelope * wave
        return result

    @staticmethod
    def graph_spec(model):
        """Validated one-step DAG: every edge ends at t and starts at t or earlier.

        Historical proxies are source snapshots, never updated by current
        covariates. Internal state[t-1] is distinct from the current process[t].
        """
        names = [n["name"] for n in model["nodes"]] + ["outcome"]
        if len(set(names)) != len(names):
            raise ValueError("SCM node names must be unique")
        if any(
            isinstance(lag, bool) or not isinstance(lag, (int, np.integer)) or lag < 1
            for lag in model["lags"]
        ):
            raise ValueError("Outcome history lags must be positive integers")
        order = {name: index for index, name in enumerate(names)}
        vertices = {
            n["name"]: {"visible": n["visible"], "kind": n["kind"], "time_offset": 0}
            for n in model["nodes"]
        }
        vertices["outcome"] = {"visible": True, "kind": "outcome", "time_offset": 0}
        edges = []

        def connect(parent, target, lag, label):
            if (
                isinstance(lag, bool)
                or not isinstance(lag, (int, np.integer))
                or lag < 0
            ):
                raise ValueError("Causal delays must be nonnegative integers")
            internal = parent == f"{target} state" or parent == "outcome residual"
            if parent not in order and not internal:
                raise ValueError(f"Unknown causal parent: {parent}")
            if lag == 0 and (internal or order[parent] >= order[target]):
                raise ValueError(
                    f"Contemporaneous parent must precede target: {parent} -> {target}"
                )
            source = f"{parent}[t-{lag}]" if lag else parent
            vertices.setdefault(
                source,
                {
                    "visible": vertices.get(parent, {}).get("visible", True),
                    "kind": "history",
                    "time_offset": -int(lag),
                    "variable": parent,
                },
            )
            edges.append(
                {"source": source, "target": target, "label": label, "lag": lag}
            )

        for name, spec in [(n["name"], n) for n in model["nodes"]] + [
            ("outcome", model["outcome"])
        ]:
            if spec.get("kind") in DISCRETE_KINDS + ("formula",) and spec["edges"]:
                raise ValueError("Discrete and formula source nodes must be exogenous")
            for edge in spec["edges"]:
                parent, lag = edge["parent"], edge["lag"]
                if parent.startswith("outcome_lag_"):
                    alias_lag = int(parent.rsplit("_", 1)[1])
                    if (
                        alias_lag < 1
                        or isinstance(lag, bool)
                        or not isinstance(lag, (int, np.integer))
                        or lag < 0
                    ):
                        raise ValueError(
                            "Outcome history requires positive lags and nonnegative delays"
                        )
                    lag += alias_lag
                    parent = "outcome"
                connect(parent, name, lag, f"w={edge['weight']:.2g}")
            if spec.get("kind") not in EXOGENOUS_KINDS:
                connect(f"{name} state", name, 1, "state")
                if spec["event_parent"]:
                    connect(
                        spec["event_parent"], name, spec.get("event_lag", 0), "event"
                    )
                if spec.get("operation"):
                    connect(
                        spec["operation"]["parent"],
                        name,
                        spec["operation"].get("lag", 0),
                        spec["operation"]["mode"],
                    )
                if spec.get("trend", {}).get("event_parent"):
                    connect(
                        spec["trend"]["event_parent"],
                        name,
                        spec["trend"].get("event_lag", 0),
                        "permanent slope break",
                    )
                if spec.get("ringing"):
                    connect(
                        spec["ringing"]["parent"],
                        name,
                        spec["ringing"].get("lag", 0),
                        "damped oscillation",
                    )
            elif spec.get("kind") in ("switch", "markov"):
                connect(f"{name} state", name, 1, "dwell state")
        for lag, weight in zip(model["lags"], model["outcome"]["ar_weights"]):
            if weight:
                connect("outcome residual", "outcome", lag, f"AR={weight:.2g}")
        for lag, weight in zip(
            model["lags"], model["outcome"].get("outcome_ar_weights", [])
        ):
            if weight:
                connect("outcome", "outcome", lag, f"AR={weight:.2g}")
        return {"vertices": vertices, "edges": edges}

    def _dag_view(self, model):
        """Unroll Y's inputs and one upstream generation at their actual times.

        Example: A[t-2] -> B[t-1] -> Y[t], not A[t] -> B[t-1].
        Stop after two edges to avoid infinitely expanding temporal feedback;
        graph_spec and scm.json retain every node's complete structural equation.
        """
        import networkx as nx

        if model.get("plant", {}).get("family") in (
            "coupled_storage",
            "rotating_machine",
        ):
            return physical_dag_view(model["plant"])
        full = self.graph_spec(model)
        graph, badges = nx.DiGraph(), set()
        incoming = {}
        for edge in full["edges"]:
            incoming.setdefault(edge["target"], []).append(edge)

        def vertex(variable, offset):
            name = variable if offset == 0 else f"{variable}[t{-abs(offset)}]"
            meta = full["vertices"].get(variable, {"visible": True, "kind": "history"})
            graph.add_node(
                name,
                **{
                    **meta,
                    "variable": variable,
                    "time_offset": offset,
                    "kind": meta["kind"] if offset == 0 else "history",
                },
            )
            return name

        def expand(variable, offset, depth):
            target = vertex(variable, offset)
            if depth == 2:
                return
            for edge in incoming.get(variable, []):
                if edge["label"] in ("state", "dwell state"):
                    badges.add(target)
                    continue
                meta = full["vertices"][edge["source"]]
                parent = meta.get("variable", edge["source"])
                source_offset = offset - edge["lag"]
                source = vertex(parent, source_offset)
                if graph.has_edge(source, target):
                    graph[source][target]["labels"].append(edge["label"])
                else:
                    graph.add_edge(
                        source, target, lag=edge["lag"], labels=[edge["label"]]
                    )
                if parent.startswith("covariate_"):
                    expand(parent, source_offset, depth + 1)

        expand("outcome", 0, 0)
        if not nx.is_directed_acyclic_graph(graph):
            raise ValueError("Causal graph contains a contemporaneous cycle")
        parents = set(graph.predecessors("outcome"))
        current = [
            n
            for n, meta in graph.nodes(data=True)
            if meta["variable"].startswith("covariate_")
        ]
        histories = [n for n in graph if n != "outcome" and n not in current]

        def history_order(name):
            return name.split("[")[0], int(name.rsplit("t-", 1)[1][:-1])

        groups = [
            (sorted(set(current) & parents), 1.8, 100, 260),
            (sorted(set(current) - parents), 3.2, 100, 260),
            (sorted(set(histories) & parents, key=history_order), 1.8, 55, -55),
            (sorted(set(histories) - parents, key=history_order), 3.2, 65, -65),
        ]
        positions = {"outcome": np.array([0.0, 0.0])}
        for names, radius, start, stop in groups:
            angles = (
                np.linspace(start, stop, len(names))
                if len(names) > 1
                else [(start + stop) / 2]
            )
            for name, angle in zip(names, angles):
                radians = np.deg2rad(angle)
                positions[name] = radius * np.array([np.cos(radians), np.sin(radians)])
        return graph, positions, badges

    def _plot_dag(self, model, path):
        import networkx as nx
        from matplotlib.backends.backend_agg import FigureCanvasAgg
        from matplotlib.figure import Figure
        from matplotlib.lines import Line2D

        graph, positions, badges = self._dag_view(model)
        figure = Figure(figsize=(16, 13))
        FigureCanvasAgg(figure)
        ax = figure.subplots()
        figure.subplots_adjust(top=0.91, bottom=0.15, left=0.05, right=0.95)
        colors = {
            "outcome": "#214e80",
            "visible": "#a8d9ca",
            "hidden": "#f1bd94",
            "history": "#d9d5eb",
        }
        sizes = {
            n: 3500
            if n == "outcome"
            else 1150
            if graph.nodes[n]["kind"] == "history"
            else 1700
            for n in graph
        }
        for n, meta in graph.nodes(data=True):
            # Historical covariates retain their source's visibility color.
            # Only outcome-derived history (including residuals) is purple.
            color = (
                colors["outcome"]
                if n == "outcome"
                else colors["visible" if meta["visible"] else "hidden"]
                if meta["variable"].startswith("covariate_")
                else colors["history"]
            )
            nx.draw_networkx_nodes(
                graph,
                positions,
                nodelist=[n],
                node_size=sizes[n],
                node_color=color,
                edgecolors="#596678",
                linewidths=1,
                ax=ax,
            )
        for source, target, edge in graph.edges(data=True):
            direct = target == "outcome"
            nx.draw_networkx_edges(
                graph,
                positions,
                edgelist=[(source, target)],
                node_size=[sizes[n] for n in graph],
                nodelist=list(graph),
                arrows=True,
                arrowstyle="-|>",
                arrowsize=16 if direct else 11,
                width=1.8 if direct else 0.85,
                alpha=0.85 if direct else 0.3,
                edge_color="#385d72",
                style="dashed" if edge["lag"] else "solid",
                connectionstyle="arc3,rad=0.04" if direct else "arc3,rad=0.18",
                ax=ax,
            )
        lag_roles = dict(zip(model["lags"], model.get("lag_roles", [])))
        for n, (x, y) in positions.items():
            label = (
                n.replace("covariate_", "C")
                .replace("outcome residual", "residual")
                .replace("outcome", "Y")
            )
            if n == "outcome":
                label = "OUTCOME\nY[t]"
            elif graph.nodes[n]["time_offset"] == 0:
                label += "[t]"
            if n.startswith("outcome[t-"):
                lag = int(n.split("t-")[1][:-1])
                if lag_roles.get(lag) == "seasonal":
                    label += "\nseasonal"
            ax.text(
                x,
                y,
                label,
                ha="center",
                va="center",
                fontsize=9 if n == "outcome" else 8,
                weight="bold" if n == "outcome" else "normal",
                color="white" if n == "outcome" else "#233244",
            )
        legend = [
            Line2D(
                [0],
                [0],
                marker="o",
                color="none",
                markerfacecolor=colors[k],
                markersize=11,
                label=label,
            )
            for k, label in [
                ("visible", "Visible covariate (any lag)"),
                ("hidden", "Hidden covariate (any lag)"),
                ("history", "Outcome-derived history"),
            ]
        ]
        legend += [
            Line2D([0], [0], color="#385d72", label="Current influence"),
            Line2D([0], [0], color="#385d72", linestyle="--", label="Lagged influence"),
        ]
        figure.legend(
            handles=legend,
            loc="lower center",
            ncol=3,
            frameon=False,
            bbox_to_anchor=(0.5, 0.055),
        )
        ax.set(
            xlim=(-4, 4),
            ylim=(-3.8, 3.8),
            aspect="equal",
            title=(
                "Source SCM: signs before observation inversion\n"
                if model.get("observation_transform")
                else "Source SCM: lags in original sample intervals\n"
                if model.get("observation_selection")
                else "Structural causal model: outcome at center\n"
            )
            + "Bold arrows feed Y directly; faint arrows show upstream relationships",
        )
        figure.text(
            0.5,
            0.025,
            "Y[t] inputs plus one upstream generation at actual times. Internal state, clock and innovations omitted. Full graph equations: scm.json.",
            ha="center",
            fontsize=9,
            color="#596678",
        )
        ax.axis("off")
        figure.savefig(path, dpi=150)
        figure.clear()

    def _plot(self, frame, path):
        from matplotlib.backends.backend_agg import FigureCanvasAgg
        from matplotlib.figure import Figure

        covariates = (
            [c for c in frame if c.startswith("covariate_")]
            if self.plot_covariates
            else []
        )
        # Separate axes keep binary events and small sensors visible next to
        # covariates that have accumulated large trends.
        figure = Figure(
            figsize=(12, 3 + 1.3 * len(covariates)), constrained_layout=True
        )
        FigureCanvasAgg(figure)
        axes = figure.subplots(1 + len(covariates), 1, squeeze=False, sharex=True)[:, 0]
        for ax, name in zip(axes, ["outcome"] + covariates):
            ax.plot(frame["time_index"], frame[name], linewidth=0.8)
            ax.set_ylabel(name, fontsize=8)
        axes[0].set_title("Synthetic operational time series")
        axes[-1].set_xlabel("Time step")
        figure.savefig(path, dpi=130)
        figure.clear()

    def _sample_accepted_model(self, trend_free, progress=None, completed=0):
        """Accept moderate trends too; linear and shape R-squared are distinct."""
        for attempt in range(100):
            if progress:
                progress.update(completed)
            model = self.sample_model(trend_free=trend_free)
            clean = self._simulate_process(model)
            # Screen the process itself on the rows that will be recorded;
            # sensor artifacts and missing values are applied after acceptance.
            window = recorded_window(clean, model.get("recorded_rows"))
            score = self.trend_score(window)
            strength = self.trend_strength(window, model)
            if self.reject_constant_outcome:
                # Nothing to learn from a flat line, and variation below
                # float32 resolution turns into rounding noise downstream.
                y = window["outcome"].to_numpy()
                if np.ptp(y) <= 1e-5 * max(1.0, abs(float(np.median(y)))):
                    continue
            # A linear score above the trend-free cutoff is sufficient evidence
            # of trend. Nonlinear trends can instead qualify by their shape fit.
            # Reclassify incidental trends in trend-free proposals without
            # correcting any values or resampling a valid moderate trend away.
            if score > 0.15 or (trend_free and score <= 0.15) or strength > 0.15:
                break
        else:
            raise RuntimeError(
                "Could not meet trend-profile screening criterion after 100 attempts"
            )
        model["trend_free"] = bool(trend_free and score <= 0.15)
        model["quality"] = {
            "linear_trend_r_squared": score,
            "screened_trend_free": bool(trend_free and score <= 0.15),
            "proposed_trend_free": trend_free,
            "trend_shape_r_squared": strength,
            "sampling_attempts": attempt + 1,
        }
        return model, self._observe(clean, model)

    def _write_table(self, model, frame, directory):
        directory.mkdir(exist_ok=False)
        frame.to_parquet(
            directory / "data.parquet",
            engine="pyarrow",
            index=False,
            compression="zstd",
        )
        if self.save_scm:
            (directory / "scm.json").write_text(
                json.dumps(model, indent=2, allow_nan=False), encoding="utf-8"
            )

    def _write_coarse_copy(self, model, frame, directory):
        """Cheap optional extra table, isolated from all simulation RNG draws."""
        if (
            len(frame) <= 512
            or model.get("observation_selection")
            or model.get("observation_transform")
            or not self.coarse_copy_probability
        ):
            return None
        # A dedicated seed namespace preserves original outputs, rejection counts,
        # and subsequent tables when augmentation is enabled or disabled.
        rng = np.random.default_rng(
            np.random.SeedSequence([model["observation_seed"], 0xC0A25E])
        )
        if rng.random() >= self.coarse_copy_probability:
            return None
        window_size = int(rng.integers(2, 6))
        offset = int(rng.integers(window_size))
        coarse = coarsen_frame(frame, window_size, offset)
        selection = {
            "kind": "fixed_stride",
            "source_rows": len(frame),
            "window_size": window_size,
            "offset": offset,
            "complete_windows_only": True,
        }
        if model["rows"] != len(frame):
            selection["simulated_rows"] = model["rows"]  # Includes prehistory.
        derived = copy.deepcopy(model)
        derived["rows"] = len(coarse)
        derived["observation_selection"] = selection
        # Source screening does not guarantee the thinned view has the same
        # score. Record that distinction rather than relabeling it as screened.
        derived["quality"] = {
            "source_quality": model.get("quality", {}),
            "screened": False,
            "linear_trend_r_squared": self.trend_score(coarse),
        }
        target = directory.with_name(f"{directory.name}_coarse_{window_size}_{offset}")
        self._write_table(derived, coarse, target)
        # Keep lineage even if saving full SCM JSON is disabled in the runner.
        (target / "provenance.json").write_text(
            json.dumps(
                {
                    "source_table": directory.name,
                    "observation_selection": selection,
                    "note": "Same SCM and realization; DAG lags use the source sampling interval.",
                },
                indent=2,
            ),
            encoding="utf-8",
        )
        _render_plot_job(self, derived, coarse, target)
        return target

    def _write_aggregate_copy(self, model, frame, directory):
        if (
            not self.aggregate_copy_probability
            or len(frame) <= 512
            or model.get("observation_selection")
            or model.get("observation_transform")
        ):
            return None
        # Independent from sampling, thinning and mirroring RNG streams.
        rng = np.random.default_rng(
            np.random.SeedSequence([model["observation_seed"], 0xA663])
        )
        if rng.random() >= self.aggregate_copy_probability:
            return None
        window = int(rng.integers(2, 6))
        statistic = str(rng.choice(["min", "max", "mean"]))
        recorded = aggregate_frame(frame, window, statistic)
        selection = {
            "kind": "trailing_aggregate",
            "window_size": window,
            "statistic": statistic,
            "source_rows": len(frame),
            "simulated_rows": model["rows"],
            "complete_windows_only": True,
            "availability": "exclusive_right_edge",
            "source_interval": "[t-window_size,t)",
        }
        derived = copy.deepcopy(model)
        derived.update(rows=len(recorded), observation_selection=selection)
        derived["quality"] = {
            "source_quality": model.get("quality", {}),
            "screened": False,
            "linear_trend_r_squared": self.trend_score(recorded),
        }
        target = directory.with_name(f"{directory.name}_aggregate_{statistic}_{window}")
        self._write_table(derived, recorded, target)
        (target / "provenance.json").write_text(
            json.dumps(
                {
                    "source_table": directory.name,
                    "observation_selection": selection,
                    "note": "All channels summarize the preceding completed window. Future aggregates are not future-known covariates. DAG describes the source clock.",
                },
                indent=2,
            ),
            encoding="utf-8",
        )
        _render_plot_job(self, derived, recorded, target)
        return target

    def _write_sign_inverted_copy(self, model, frame, directory):
        """Reflect an existing realization without sampling a second SCM."""
        if (
            not self.sign_inversion_probability
            or model.get("observation_selection")
            or model.get("observation_transform")
        ):
            return None
        # Separate from both the main sampler and coarse-copy selection. Turning
        # either augmentation on/off must not perturb the other's decisions.
        rng = np.random.default_rng(
            np.random.SeedSequence([model["observation_seed"], 0x519A])
        )
        if rng.random() >= self.sign_inversion_probability:
            return None
        mirrored = invert_frame(frame)
        derived = copy.deepcopy(model)
        derived["observation_transform"] = {
            "kind": "sign_inversion",
            "exclude": ["time_index"],
        }
        derived["quality"] = {
            "source_quality": model.get("quality", {}),
            "screened": False,
            "linear_trend_r_squared": self.trend_score(mirrored),
        }
        target = directory.with_name(f"{directory.name}_sign_inverted")
        self._write_table(derived, mirrored, target)
        (target / "provenance.json").write_text(
            json.dumps(
                {
                    "source_table": directory.name,
                    "observation_transform": derived["observation_transform"],
                    "note": "Same SCM and realization; all recorded values except time_index negated. DAG shows source equations.",
                },
                indent=2,
            ),
            encoding="utf-8",
        )
        _render_plot_job(self, derived, mirrored, target)
        return target

    def _write_extra_copies(self, model, frame, directory):
        # Independently select both views of the original; never copy copies.
        return [
            path
            for path in (
                self._write_coarse_copy(model, frame, directory),
                self._write_sign_inverted_copy(model, frame, directory),
                self._write_aggregate_copy(model, frame, directory),
            )
            if path is not None
        ]

    def _generate_seeded(self, profiles, dated_dir, run_id):
        """Parallelize complete tables, with bounded queues and independent RNGs."""
        count, batch_size = len(profiles), self.generation_batch_size
        workers = min(self.generation_workers, math.ceil(count / batch_size))
        main_file = getattr(sys.modules.get("__main__"), "__file__", None)
        if not main_file or not Path(main_file).is_file():
            workers = 1  # Spawn requires an importable Python entry point.
        progress = GenerationProgress(count, workers, self.progress_interval)
        progress.update(0, force=True)
        paths = [None] * count

        def batches():
            for start in range(0, count, batch_size):
                stop = min(count, start + batch_size)
                # Generate seeds lazily: never queue millions of tasks or frames.
                seeds = self.rng.integers(0, 2**63, size=stop - start)
                yield (
                    start,
                    [
                        (
                            int(seed),
                            bool(profiles[index]),
                            dated_dir / f"table_{run_id}_{index:06d}",
                        )
                        for index, seed in zip(range(start, stop), seeds)
                    ],
                )

        completed = 0
        if workers == 1:
            for start, jobs in batches():
                result = generate_batch(self, jobs)
                paths[start : start + len(result)] = result
                completed += len(result)
                progress.record_copies(result)
                progress.update(completed)
        else:
            executor = ProcessPoolExecutor(
                max_workers=workers,
                mp_context=get_context("spawn"),
                initializer=initialize_worker,
                initargs=(self,),
            )
            pending, source = {}, iter(batches())
            exhausted = False
            try:
                while pending or not exhausted:
                    while not exhausted and len(pending) < 2 * workers:
                        try:
                            start, jobs = next(source)
                        except StopIteration:
                            exhausted = True
                        else:
                            pending[executor.submit(generate_worker_batch, jobs)] = (
                                start
                            )
                    done, _ = wait(
                        pending,
                        timeout=min(1.0, self.progress_interval),
                        return_when=FIRST_COMPLETED,
                    )
                    for future in done:
                        start = pending.pop(future)
                        result = future.result()
                        paths[start : start + len(result)] = result
                        completed += len(result)
                        progress.record_copies(result)
                    # Keep console updates alive during worker startup and long
                    # batches. Only fully written/rendered tables count as done.
                    progress.update(completed)
            finally:
                executor.shutdown(wait=True, cancel_futures=True)
        progress.update(count, force=True)
        return [path for group in paths for path in group]

    def generate(self, num_tables=100):
        if (
            isinstance(num_tables, bool)
            or not isinstance(num_tables, int)
            or num_tables < 1
        ):
            raise ValueError("num_tables must be a positive integer")
        dated_dir = self.output_dir / date.today().isoformat()
        dated_dir.mkdir(parents=True, exist_ok=True)
        run_id, paths = uuid.uuid4().hex[:12], []
        # Enforce a batch quota, rather than relying on a Bernoulli draw to
        # happen to yield enough trend-free examples in small runs.
        profiles = [True] * math.ceil(num_tables * self.trend_free_fraction)
        profiles += [False] * (num_tables - len(profiles))
        self.rng.shuffle(profiles)
        if self._seeded_generation:
            return self._generate_seeded(profiles, dated_dir, run_id)
        # Sampling, RNG consumption, screening and table order stay serial and
        # unchanged. Only independent image rendering runs in other processes.
        workers = min(self.render_workers, num_tables)
        # Spawn needs an importable entry point. Interactive Python/notebooks
        # still work, and tiny automatic batches avoid process startup overhead.
        main_file = getattr(sys.modules.get("__main__"), "__file__", None)
        if (
            not main_file
            or not Path(main_file).is_file()
            or (self._auto_render_workers and num_tables < 4)
        ):
            workers = 1
        executor = (
            ProcessPoolExecutor(max_workers=workers, mp_context=get_context("spawn"))
            if workers > 1 and (self.save_plot or self.save_dag)
            else None
        )
        progress = GenerationProgress(
            num_tables, workers if executor else 1, self.progress_interval
        )
        progress.update(0, force=True)
        pending = deque()
        try:
            for index, trend_free in enumerate(profiles):
                model, frame = self._sample_accepted_model(
                    trend_free, progress, index - len(pending)
                )
                directory = dated_dir / f"table_{run_id}_{index:06d}"
                self._write_table(model, frame, directory)
                if executor:
                    pending.append(
                        executor.submit(_render_plot_job, self, model, frame, directory)
                    )
                    # Bound queued frames/SCMs; surface rendering failures early.
                    if len(pending) >= 2 * workers:
                        pending.popleft().result()
                else:
                    _render_plot_job(self, model, frame, directory)
                group = [directory] + self._write_extra_copies(model, frame, directory)
                paths.extend(group)
                progress.record_copies([group])
                while pending and pending[0].done():
                    pending.popleft().result()
                progress.update(index + 1 - len(pending))
            while pending:
                pending.popleft().result()
                progress.update(num_tables - len(pending))
        finally:
            if executor:
                executor.shutdown(wait=True, cancel_futures=True)
        progress.update(num_tables, force=True)
        return paths

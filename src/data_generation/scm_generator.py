"""Composable dynamic SCMs for diverse synthetic operational time series.

Sampling ranges are priors, not claims about a particular industry. All sampled
parameters are saved. Evaluation resamples innovations, never the equations.
"""

import json
import math
import uuid
from datetime import date
from pathlib import Path

import numpy as np
import pandas as pd


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
        hidden_probability=0.35,
        edge_probability=0.18,
        direct_probability=0.65,
        delayed_probability=0.15,
        drift_probability=0.5,
        walk_probability=0.15,
        trend_free_fraction=0.5,
        residual_history_probability=0.08,
        switching_probability=0.3,
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
        prior_ranges=None,
        noise_weights=None,
        burn_in=256,
        save_scm=True,
        save_plot=True,
        plot_covariates=True,
        save_dag=True,
    ):
        settings = locals().copy()
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
            "hidden_probability",
            "edge_probability",
            "direct_probability",
            "delayed_probability",
            "drift_probability",
            "walk_probability",
            "trend_free_fraction",
            "residual_history_probability",
            "switching_probability",
            "multiplicative_trend_probability",
            "variance_dynamics_probability",
            "multiplicative_constant_noise_probability",
            "multiplicative_constant_seasonality_probability",
            "seasonal_lag_probability",
            "long_lag_probability",
            "instantaneous_probability",
            "fourier_probability",
            "seasonal_shape_drift_probability",
        ):
            if not 0 <= settings[name] <= 1:
                raise ValueError(f"{name} must lie in [0, 1]")
            setattr(self, name, settings[name])
        if not np.isfinite(covariate_delay_scale) or covariate_delay_scale <= 0:
            raise ValueError("covariate_delay_scale must be finite and positive")
        self.covariate_delay_scale = float(covariate_delay_scale)
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
        signed = {"noise_location", "effect_offset", "persistence", "baseline"}
        for name, (low, high) in self.priors.items():
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
        ):
            if self.priors[name][1] >= 1:
                raise ValueError(f"{name} must be below 1")
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

    def _draw(self, name, *, log=False, signed=False):
        low, high = self.priors[name]
        value = (
            np.exp(self.rng.uniform(np.log(low), np.log(high)))
            if log
            else self.rng.uniform(low, high)
        )
        return float(value * (self.rng.choice([-1, 1]) if signed else 1))

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
        return spec["location"] + spec["scale"] * result

    @staticmethod
    def _quantize(value, step, offset):
        return offset + step * np.round((value - offset) / step) if step else value

    def _expression(self):
        """Build an expression DAG with reusable intermediate results.

        Evaluation in instruction order is dynamic programming: shared results
        are computed once. Each instruction consumes the previous result, plus
        an optional older result, so the program contains no dead instructions.
        """
        r, program = self.rng, []
        for i in range(int(r.integers(1, self.max_effect_components + 1))):
            op = str(r.choice(self.OPERATORS))
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
            program.append(instruction)
        return program

    @staticmethod
    def _evaluate(program, value):
        values = [value]
        for instruction in program:
            a, b = values[instruction["a"]], values[instruction["b"]]
            z = instruction["gain"] * a + instruction["offset"]
            op, limit = instruction["op"], instruction["limit"]
            if op == "affine":
                result = z
            elif op == "sine":
                result = np.sin(z)
            elif op == "cosine":
                result = np.cos(z)
            elif op in ("damped_sine", "damped_cosine"):
                # For nonnegative input this is exp(-decay*z)*sin/cos(w*z+phase).
                # abs(z) extends it to signed covariates without exponential
                # blow-up. Time-domain ringing is modeled separately below.
                angle = instruction["frequency"] * z + instruction["phase"]
                wave = np.sin(angle) if op == "damped_sine" else np.cos(angle)
                result = np.exp(-instruction["decay"] * abs(z)) * wave
            elif op == "tangent":
                cap = instruction["angle_limit"]
                result = np.tan(cap * np.tanh(z / cap))
            elif op == "sigmoid":
                result = np.tanh(z / 2)  # 2*sigmoid(z)-1, without exponential overflow.
            elif op == "exponential":
                result = np.expm1(limit * np.tanh(z / limit)) / limit
            elif op == "power":
                result = np.sign(z) * abs(z) ** instruction["power"]
            elif op == "inverse_square":
                # Exactly 1/z² outside a sampled near-zero floor, followed by
                # the same soft actuator saturation as the other operators.
                # The instruction's gain/offset shift and scale the singularity.
                result = (1 / max(abs(z), instruction["floor"])) ** 2
            elif op == "log":
                result = np.sign(z) * np.log1p(abs(z))
            elif op == "relu":
                result = max(0, z)
            elif op == "mlp":
                result = np.dot(
                    instruction["v"],
                    np.tanh(np.asarray(instruction["w"]) * z + instruction["bias"]),
                )
            elif op == "add":
                result = z + b
            elif op == "multiply":
                result = z * b
            elif op == "gate":
                result = z * (1 + np.tanh(b)) / 2
            else:
                raise ValueError(f"Unknown operator: {op}")
            # Soft saturation models finite capacity and controls nonlinear
            # feedback. Accumulated trend/walk states are NOT clipped this way.
            values.append(float(limit * np.tanh(result / limit)))
        return values[-1]

    def _edge(self, parent, lag=0):
        return {
            "parent": parent,
            "lag": lag,
            "weight": self._draw("edge_strength", log=True, signed=True),
            "input_scale": self._draw("input_scale", log=True),
            "input_offset": self._draw("effect_offset"),
            "output_offset": self._draw("effect_offset"),
            "program": self._expression(),
        }

    @classmethod
    def _effect(cls, edge, value):
        value = (value - edge["input_offset"]) / edge["input_scale"]
        return edge["weight"] * (
            cls._evaluate(edge["program"], value) + edge["output_offset"]
        )

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

    def _sample_lags(self, rows):
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
            candidates
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

    def sample_model(self, trend_free=None):
        r = self.rng
        rows = int(r.integers(self.min_rows, self.max_rows + 1))
        trend_free = (
            bool(r.random() < self.trend_free_fraction)
            if trend_free is None
            else bool(trend_free)
        )
        count = int(r.integers(self.min_covariates, self.max_covariates + 1))
        lags, lag_roles, seasonal_lags = self._sample_lags(rows)
        event = "covariate_00"
        nodes = [
            {
                "name": event,
                "kind": "event",
                "edges": [],
                "visible": bool(r.random() >= self.hidden_probability),
                "probability": self._draw("event_probability"),
            }
        ]
        # Branching/direct structures dominate. Short mediated paths remain
        # possible, but a full covariate chain is no longer a scaffold.
        topology = str(
            r.choice(["sparse", "fork", "diamond", "dense"], p=[0.35, 0.35, 0.25, 0.05])
        )
        switch = None
        if r.random() < self.switching_probability:
            switch = "covariate_01"
            nodes.append(
                {
                    "name": switch,
                    "kind": "switch",
                    "visible": bool(r.random() >= self.hidden_probability),
                    "edges": [],
                    "on_duration": [20, max(30, min(rows // 2, 200))],
                    "off_duration": [5, max(10, min(rows // 5, 80))],
                    "initial_on": True,
                }
            )
        for i in range(len(nodes), count):
            node = {
                "name": f"covariate_{i:02d}",
                "kind": "process",
                "visible": bool(r.random() >= self.hidden_probability),
                "edges": [],
                **self._dynamics(event),
            }
            probability = self.edge_probability * (1.8 if topology == "dense" else 1)
            for parent in nodes:
                if r.random() < probability:
                    lag = self._sample_covariate_lag()
                    node["edges"].append(self._edge(parent["name"], lag))
            # Scaffolds allow deep mediation, shared causes and colliders.
            # Random extra edges retain diversity; hiding is independently drawn.
            forced = []
            processes = [n for n in nodes if n["kind"] == "process"]
            if topology == "fork" and len(processes) in (1, 2):
                forced = [processes[0]["name"]]
            if topology == "diamond":
                if len(processes) in (1, 2):
                    forced = [processes[0]["name"]]
                elif len(processes) == 3:
                    forced = [n["name"] for n in processes[1:3]]
            for parent in forced:
                if not any(e["parent"] == parent for e in node["edges"]):
                    node["edges"].append(
                        self._edge(parent, self._sample_covariate_lag())
                    )
            if r.random() < self.delayed_probability:
                node["edges"].append(self._edge(f"outcome_lag_{int(r.choice(lags))}"))
            nodes.append(node)
        processes = [n for n in nodes if n["kind"] == "process"]
        selected = [n for n in processes if r.random() < self.direct_probability]
        # At least two independent inputs when available, selected without a
        # special preference for the last node in topological order.
        minimum = min(2, len(processes))
        remaining = [n for n in processes if n not in selected]
        while len(selected) < minimum:
            selected.append(remaining.pop(int(r.integers(len(remaining)))))
        outcome = {
            **self._dynamics(event),
            "edges": [
                self._edge(n["name"], self._sample_covariate_lag()) for n in selected
            ],
            "baseline": self._draw("baseline"),
            "scale": self._draw("outcome_scale", log=True),
            "ar_weights": [0.0] * len(lags),
            "outcome_ar_weights": self._lag_weights(lags, lag_roles),
        }
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
            node["interaction"] = (
                float(r.uniform(-0.8, 0.8))
                if len(node["edges"]) >= 2 and r.random() < 0.5
                else 0.0
            )
            self._revise_dynamics(node, rows, trend_free, node is outcome)
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
        if switch:
            outcome["operation"] = {
                "parent": switch,
                "mode": str(r.choice(["shutdown", "reset"], p=[0.7, 0.3])),
                "lag": self._sample_covariate_lag(),
                "flat_value": 0.0 if r.random() < 0.7 else self._draw("baseline"),
                "restart_rate": float(r.uniform(0.03, 0.3)),
                "ramp_rate": float(r.uniform(0.01, 0.08) * r.choice([-1, 1])),
            }
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
                    trend["initial_level"] * self._draw("relative_noise_scale"),
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
        return {
            "version": 3,
            "observation_seed": int(r.integers(0, 2**63)),
            "rows": rows,
            "trend_free": trend_free,
            "burn_in": self.burn_in,
            "lags": lags,
            "lag_roles": lag_roles,
            "seasonal_lags": seasonal_lags,
            "nodes": nodes,
            "outcome": outcome,
            "topology": topology,
            "prior_ranges": self.priors.copy(),
            "noise_weights": self.noise_weights.copy(),
            "equations": {
                "process": "x=quantize(intercept+level+seasonal+shock+residual); residual=persistence*previous_residual+forcing+heteroskedastic_noise",
                "level": "slope+=slope_walk*N(0,1); on event reverse/reset/add event_slope; level+=slope+walk_scale*N(0,1)+event*level_jump",
                "shock": "shock=shock_decay*previous_shock+event*shock_size",
                "seasonal": "sum(amplitude*(1+modulation*sin(2*pi*t/modulation_period))*sin(2*pi*t/period+phase))",
                "forcing": "sum(edge effects)+interaction*tanh(effect[0])*tanh(effect[1])",
                "noise": "saved noise draw*(1+heteroskedasticity*abs(tanh(previous_residual)))",
                "variance": "center innovations by their distribution mean; scale centered innovations and white noise by time-varying volatility and optional (latent_level/initial_level)**noise_elasticity; retain innovation mean",
                "multiplicative_seasonality": "seasonal * seasonal_base_scale * (latent_level/initial_level)**seasonal_elasticity",
                "outcome": "residual includes optional residual AR plus outcome_ar_weights times full normalized historical outcome; physical outcome=baseline+scale*x; operation overrides output during shutdown or adds a ramp reset on binary transitions",
                "trend": "optional uncapped outcome trend starts at t=0; linear/accelerating/decelerating rates, a permanent event-triggered slope multiplier, or multiplicative level L[t+1]=L[t]*exp(log_rate) contributing L[t]-initial_level; ordinary shock decay does not restore a changed slope",
                "ringing": "state[t]=exp(-decay+i*2*pi/period)*state[t-1]+amplitude*event[t]*exp(i*phase); add imag(state) for sine or real(state) for cosine",
                "seasonal_v3": "integrated phase with smoothly varying frequency; saved Fourier sine/cosine expansion (or legacy harmonic wave) times alternating-cycle envelope and optional exponential amplitude growth",
                "fourier": "sum_k(a_k(t)*cos(k*phase)+b_k(t)*sin(k*phase)); coefficients optionally interpolate smoothly between saved start/end spectra; no DC term",
                "quantization": "offset+step*round((x-offset)/step), bypassed when step=0",
                "initialization": "States zero except saved slopes; t starts at -burn_in; discard t<0. Outcome-lag edge parents use full normalized historical x; exported lags use physical outcome units.",
                "graph": "Topological contemporaneous edges plus positive temporal lags. Past values are separate vertices when time is unrolled; innovations are independent.",
            },
        }

    def _revise_dynamics(self, node, rows, trend_free, is_outcome):
        """Budget cumulative changes against the observation horizon.

        Disabling only slope is insufficient: hidden random walks, permanent
        jumps, and event-induced slopes would still transmit trends downstream.
        """
        r = self.rng
        if r.random() < self.variance_dynamics_probability:
            mode = str(r.choice(["growth", "cyclic", "regime"], p=[0.6, 0.2, 0.2]))
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
        if is_outcome:
            # A visible stochastic floor prevents seasonal/slow components from
            # producing nearly deterministic ramps. It is a causal innovation,
            # not noise added after seeing or normalizing the finished series.
            node["white_noise_scale"] = float(r.uniform(0.15, 0.45))
            if not node["seasons"] and r.random() < 0.85:
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
        values = frame["outcome"].to_numpy()
        if np.std(values) < 1e-12:
            return 0.0
        return float(np.corrcoef(np.arange(len(values)), values)[0, 1] ** 2)

    @staticmethod
    def trend_strength(frame, model):
        """Variance explained by the specified low-dimensional trend shape.

        Include a hinge at a saved shock time: a strong reversal may have almost
        zero global linear slope. This assesses the observed series, not merely
        the presence of nonzero parameters. No output values are modified.
        """
        y = frame["outcome"].to_numpy()
        t = np.arange(len(y), dtype=float) / max(1, len(y) - 1)
        columns = [np.ones(len(y)), t]
        trend = model["outcome"].get("trend", {})
        if trend.get("kind") == "multiplicative":
            # Fit in observation units without logging possibly signed/noisy
            # observations. The known exponential shape is a structural prior,
            # not a curve selected by searching over the generated outcome.
            columns = [
                np.ones(len(y)),
                np.expm1(trend["log_rate"] * np.arange(1, len(y) + 1)),
            ]
        if trend.get("kind") in ("accelerating", "decelerating"):
            columns.append(t**2)
        if trend.get("event_time") is not None:
            columns.append(np.maximum(0, t - trend["event_time"] / max(1, len(y) - 1)))
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
        if model["version"] == 1:
            try:
                from .legacy_scm_generator import OperationalSCMGenerator as Legacy
            except ImportError:
                from legacy_scm_generator import OperationalSCMGenerator as Legacy
            # Dispatch on a legacy instance so its noise/effect evaluators also
            # use the original schema, preserving existing dataset replay.
            return Legacy(self.output_dir).simulate(model)
        if model["version"] not in (2, 3):
            raise ValueError("Unsupported SCM schema version")
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
        randoms, states, season_values, white_errors = {}, {}, {}, {}
        noise_scales, noise_means = {}, {}
        time = np.arange(-warmup, rows)
        for name, spec in zip(names, specs):
            if spec.get("kind") == "event":
                history[name][:] = r.binomial(1, spec["probability"], length)
                for event_time in spec.get("forced_event_times", []):
                    history[name][warmup + event_time] = 1
            elif spec.get("kind") == "switch":
                # Semi-Markov dwell times create contiguous outages rather than
                # isolated Bernoulli blips. The node remains binary and can be hidden.
                position, on = 0, spec["initial_on"]
                while position < length:
                    low, high = spec["on_duration"] if on else spec["off_duration"]
                    duration = int(r.integers(low, high + 1))
                    history[name][position : position + duration] = int(on)
                    position += duration
                    on = not on
            else:
                randoms[name] = (
                    self._noise(r, spec["noise"], length),
                    r.normal(size=length),
                    r.normal(size=length),
                )
                states[name] = [0.0, spec["slope"], 0.0]  # level, slope, decaying shock
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
        recovery, ramp, previous_on = 1.0, 0.0, True

        def past(name, index):
            return history[name][index] if index >= 0 else 0.0

        for i in range(length):
            t = i - warmup
            for name, spec in zip(names, specs):
                if spec.get("kind") in ("event", "switch"):
                    continue
                noise, walk, slope_walk = randoms[name]
                event = (
                    past(spec["event_parent"], i - spec.get("event_lag", 0))
                    if spec["event_parent"]
                    else 0
                )
                level, slope, shock = states[name]
                slope += spec["slope_walk"] * slope_walk[i]
                if event:
                    mode = spec["event_mode"]
                    slope = (
                        -slope
                        if mode == "reverse"
                        else (
                            spec["event_slope"]
                            if mode == "reset"
                            else slope + spec["event_slope"]
                        )
                    )
                level += (
                    slope + spec["walk_scale"] * walk[i] + event * spec["level_jump"]
                )
                if "level_limit" in spec:
                    level = float(
                        np.clip(level, -spec["level_limit"], spec["level_limit"])
                    )
                shock = spec["shock_decay"] * shock + event * spec["shock_size"]
                states[name] = [level, slope, shock]
                seasonal = (
                    season_values[name][i]
                    if model["version"] == 3
                    else sum(
                        s["amplitude"]
                        * (
                            1
                            + s["modulation"]
                            * np.sin(2 * np.pi * t / s["modulation_period"])
                        )
                        * np.sin(2 * np.pi * t / s["period"] + s["phase"])
                        for s in spec["seasons"]
                    )
                )
                effects = []
                for edge in spec["edges"]:
                    parent = edge["parent"]
                    value = (
                        past("outcome", i - int(parent.rsplit("_", 1)[1]) - edge["lag"])
                        if parent.startswith("outcome_lag_")
                        else past(parent, i - edge["lag"])
                    )
                    effects.append(self._effect(edge, value))
                forcing = sum(effects)
                if spec["interaction"]:
                    forcing += (
                        spec["interaction"] * np.tanh(effects[0]) * np.tanh(effects[1])
                    )
                previous = residuals[name][i - 1] if i else 0.0
                ar = (
                    sum(
                        w * residuals[name][i - lag]
                        for w, lag in zip(spec["ar_weights"], lags)
                        if i >= lag
                    )
                    if name == "outcome"
                    else spec["persistence"] * previous
                )
                if name == "outcome":
                    ar += sum(
                        w * past("outcome", i - lag)
                        for w, lag in zip(spec.get("outcome_ar_weights", []), lags)
                    )
                innovation = noise[i]
                if name in noise_scales:
                    # Scale deviations, not the distribution's location. Thus
                    # changing variance need not also change the expected mean.
                    mean = noise_means[name]
                    innovation = mean + (innovation - mean) * noise_scales[name][i]
                residual = (
                    ar
                    + forcing
                    + innovation
                    * (1 + spec["heteroskedasticity"] * abs(np.tanh(previous)))
                )
                if model["version"] == 3:
                    residual += white_errors[name][i] * (
                        noise_scales[name][i] if name in noise_scales else 1
                    )
                residuals[name][i] = residual
                value = spec["intercept"] + level + shock + seasonal + residual
                if name in ringing_values:
                    value += ringing_values[name][i]
                if name == "outcome":
                    value += trend_values[i]
                value = self._quantize(
                    value, spec["quantization_step"], spec["quantization_offset"]
                )
                operation = spec.get("operation")
                if operation:
                    on = bool(past(operation["parent"], i - operation.get("lag", 0)))
                    if operation["mode"] == "shutdown":
                        recovery = (
                            min(1.0, recovery + operation["restart_rate"])
                            if on
                            else 0.0
                        )
                        flat = (operation["flat_value"] - spec["baseline"]) / spec[
                            "scale"
                        ]
                        value = flat + recovery * (value - flat)
                    else:
                        # Reset accumulated ramp on each binary transition:
                        # opposite slopes on the two states yield repeating
                        # sawtooth-like charging/discharging cycles.
                        if on != previous_on:
                            ramp = 0.0
                        ramp += operation["ramp_rate"] * (1 if on else -1)
                        value += ramp
                    previous_on = on
                history[name][i] = value
        outcome = model["outcome"]
        physical = outcome["baseline"] + outcome["scale"] * history["outcome"]
        operation = outcome.get("operation")
        if operation and operation["mode"] == "shutdown":
            physical[history[operation["parent"]] == 0] = operation["flat_value"]
        data = {"time_index": np.arange(rows), "outcome": physical[warmup:]}
        for lag in lags:
            data[f"outcome_lag_{lag}"] = physical[warmup - lag : length - lag]
        for node in model["nodes"]:
            if node["visible"]:
                data[node["name"]] = history[node["name"]][warmup:]
        frame = pd.DataFrame(data)
        if not np.isfinite(frame.to_numpy()).all():
            raise FloatingPointError(
                "Non-finite SCM observations; inspect sampled parameters"
            )
        return frame

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
            if spec.get("kind") not in ("event", "switch"):
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
            elif spec.get("kind") == "switch":
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
            title="Structural causal model: outcome at center\nBold arrows feed Y directly; faint arrows show upstream relationships",
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
        for index, trend_free in enumerate(profiles):
            for attempt in range(100):
                model = self.sample_model(trend_free=trend_free)
                frame = self.simulate(model)
                score = self.trend_score(frame)
                strength = self.trend_strength(frame, model)
                if (trend_free and score <= 0.15) or (
                    not trend_free and strength >= 0.4
                ):
                    break
            else:
                raise RuntimeError(
                    "Could not meet trend-profile screening criterion after 100 attempts"
                )
            model["quality"] = {
                "linear_trend_r_squared": score,
                "screened_trend_free": trend_free,
                "trend_shape_r_squared": strength,
                "sampling_attempts": attempt + 1,
            }
            directory = dated_dir / f"table_{run_id}_{index:06d}"
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
            if self.save_plot:
                self._plot(frame, directory / "series.png")
            if self.save_dag:
                self._plot_dag(model, directory / "dag.png")
            paths.append(directory)
            print(
                f"[{index + 1}/{num_tables}] {directory.name}: {len(frame)} rows, {len(frame.columns)} columns",
                flush=True,
            )
        return paths

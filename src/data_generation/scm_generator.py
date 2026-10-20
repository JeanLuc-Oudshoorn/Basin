"""Composable dynamic SCMs for diverse synthetic operational time series.

Sampling ranges are priors, not claims about a particular industry. All sampled
parameters are saved. Evaluation resamples innovations, never the equations.
"""

from datetime import date
import json
import math
from pathlib import Path
import uuid

import numpy as np
import pandas as pd


class OperationalSCMGenerator:
    # Override these ranges with prior_ranges. Distribution identities and
    # numerical safeguards, unlike model parameters, remain fixed in evaluation.
    DEFAULT_PRIORS = {
        "noise_scale": (0.02, 0.6), "noise_location": (-0.3, 0.3),
        "student_df": (2.1, 35.0), "beta_shape": (0.3, 8.0),
        "weibull_shape": (0.5, 5.0), "poisson_rate": (0.2, 40.0),
        "lognormal_sigma": (0.15, 1.3), "quantization_step": (0.01, 0.8),
        "mixture_separation": (0.5, 3.5), "component_scale": (0.3, 1.6),
        "contamination_probability": (0.01, 0.15), "outlier_scale": (3.0, 12.0),
        "edge_strength": (0.005, 3.0), "input_scale": (0.5, 20.0),
        "effect_gain": (0.2, 2.0), "effect_offset": (-2.0, 2.0),
        "effect_limit": (2.0, 12.0), "persistence": (-0.7, 0.97),
        "seasonal_period": (6.0, 512.0), "seasonal_amplitude": (0.05, 3.0),
        "drift_magnitude": (0.0005, 0.04), "walk_scale": (0.001, 0.12),
        "slope_walk_scale": (0.00001, 0.0005),
        "event_probability": (0.002, 0.05), "decay": (0.75, 0.998),
        "shock_magnitude": (0.1, 5.0), "ar_mass": (0.1, 0.94),
        "baseline": (10.0, 500.0), "outcome_scale": (0.5, 20.0),
        "perlin_frequency": (0.005, 0.2), "perlin_offset": (0.1, 250.0),
        "perlin_persistence": (0.25, 0.8), "perlin_lacunarity": (1.5, 3.0),
        "perlin_octaves": (1, 6),
        "pink_exponent": (0.8, 1.2), "pink_filter_length": (256, 2048),
        "inverse_square_floor": (0.01, 0.3),
    }
    BASE_NOISES = ("normal", "student_t", "laplace", "uniform", "skewed",
                   "poisson", "beta", "weibull", "quantization", "perlin", "pink")
    MIXTURES = ("student_t_mixture", "normal_mixture", "heterogeneous_mixture", "contamination")
    NOISES = BASE_NOISES + MIXTURES
    # Relative weights, not percentages: the three preferred families receive four
    # times the probability of each other family (1/6 each with these defaults).
    DEFAULT_NOISE_WEIGHTS = {name: (4.0 if name in ("perlin", "student_t", "pink") else 1.0)
                             for name in NOISES}
    OPERATORS = ("affine", "sine", "cosine", "tangent", "sigmoid", "exponential",
                 "power", "inverse_square", "log", "relu", "mlp", "add", "multiply", "gate")

    def __init__(self, output_dir, *, seed=42, min_rows=20, max_rows=2056,
                 min_covariates=8, max_covariates=18, max_lag=192,
                 max_lag_count=6, hidden_probability=0.35, edge_probability=0.18,
                 direct_probability=0.3, delayed_probability=0.3,
                 drift_probability=0.65, walk_probability=0.6,
                 max_effect_components=8, prior_ranges=None, noise_weights=None, burn_in=256,
                 save_scm=True, save_plot=True, plot_covariates=True, save_dag=True):
        settings = locals().copy()
        for name in ("min_rows", "max_rows", "min_covariates", "max_covariates",
                     "max_lag", "max_lag_count", "burn_in", "max_effect_components"):
            value = settings[name]
            if isinstance(value, bool) or not isinstance(value, int) or value < 1:
                raise ValueError(f"{name} must be a positive integer")
            setattr(self, name, value)
        if min_rows > max_rows or not 3 <= min_covariates <= max_covariates:
            raise ValueError("Invalid row/covariate bounds; at least 3 covariates required")
        for name in ("hidden_probability", "edge_probability", "direct_probability",
                     "delayed_probability", "drift_probability", "walk_probability"):
            if not 0 <= settings[name] <= 1:
                raise ValueError(f"{name} must lie in [0, 1]")
            setattr(self, name, settings[name])
        self.priors = dict(self.DEFAULT_PRIORS)
        for name, bounds in (prior_ranges or {}).items():
            if name not in self.priors:
                raise ValueError(f"Unknown prior: {name}")
            if len(bounds) != 2 or not np.isfinite(bounds).all() or bounds[0] > bounds[1]:
                raise ValueError(f"Invalid bounds for {name}")
            self.priors[name] = tuple(bounds)
        signed = {"noise_location", "effect_offset", "persistence", "baseline"}
        for name, (low, high) in self.priors.items():
            if name not in signed and low <= 0:
                raise ValueError(f"{name} bounds must be positive")
        if self.priors["student_df"][0] <= 2:
            raise ValueError("student_df must exceed 2 for finite variance normalization")
        for name in ("decay", "ar_mass", "event_probability", "contamination_probability"):
            if self.priors[name][1] >= 1:
                raise ValueError(f"{name} must be below 1")
        low, high = self.priors["perlin_octaves"]
        if int(low) != low or int(high) != high or low < 1 or high > 16:
            raise ValueError("perlin_octaves must have integer bounds within [1, 16]")
        if self.priors["perlin_persistence"][1] > 1:
            raise ValueError("perlin_persistence must be at most 1")
        low, high = self.priors["pink_filter_length"]
        if int(low) != low or int(high) != high or low < 2 or high > 65536:
            raise ValueError("pink_filter_length must have integer bounds within [2, 65536]")
        if self.priors["pink_exponent"][1] >= 2:
            raise ValueError("pink_exponent must be below 2")
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
        if self.noise_weights["heterogeneous_mixture"] and not any(self.noise_weights[k] for k in self.BASE_NOISES):
            raise ValueError("Heterogeneous mixtures require a positive base-family weight")
        self.burn_in = max(burn_in, max_lag)
        self.save_scm, self.save_plot = save_scm, save_plot
        self.plot_covariates, self.save_dag = plot_covariates, save_dag

    def _draw(self, name, *, log=False, signed=False):
        low, high = self.priors[name]
        value = np.exp(self.rng.uniform(np.log(low), np.log(high))) if log else self.rng.uniform(low, high)
        return float(value * (self.rng.choice([-1, 1]) if signed else 1))

    def _sample_noise(self, kind=None):
        """Separate shape, location, and scale so tail changes need not change SNR."""
        r = self.rng
        kind = self._choose_noise(self.NOISES) if kind is None else kind
        spec = {"kind": kind, "location": self._draw("noise_location"),
                "scale": self._draw("noise_scale", log=True)}
        if kind in self.MIXTURES:
            count = 2 if kind in ("student_t_mixture", "contamination") else int(r.integers(2, 5))
            families = {"student_t_mixture": ["student_t"] * count,
                        "normal_mixture": ["normal"] * count,
                        "contamination": ["normal", "student_t"]}
            chosen = families[kind] if kind in families else [self._choose_noise(self.BASE_NOISES) for _ in range(count)]
            components = [self._sample_noise(k) for k in chosen]
            separation = self._draw("mixture_separation")
            # Shifted components allow overlapping modes, rather than only a
            # zero-centered scale mixture. Avoid empirical sample recentering.
            for i, component in enumerate(components):
                component.update(location=float((i - (count - 1) / 2) * separation),
                                 scale=self._draw("component_scale", log=True))
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
            spec.update(seed=int(r.integers(0, 2**63)),
                        exponent=self._draw("pink_exponent"),
                        filter_length=int(r.integers(int(low), int(high) + 1)))
        elif kind == "perlin":
            low, high = self.priors["perlin_octaves"]
            spec.update(seed=int(r.integers(0, 2**63)),
                        frequency=self._draw("perlin_frequency", log=True),
                        offset=self._draw("perlin_offset"),
                        octaves=int(r.integers(int(low), int(high) + 1)),
                        persistence=self._draw("perlin_persistence"),
                        lacunarity=self._draw("perlin_lacunarity"))
        elif kind == "beta":
            spec.update(alpha=self._draw("beta_shape", log=True), beta=self._draw("beta_shape", log=True))
        elif kind == "poisson":
            spec["rate"] = self._draw("poisson_rate", log=True)
        elif kind == "weibull":
            spec["shape"] = self._draw("weibull_shape", log=True)
        elif kind == "skewed":
            spec["sigma"] = self._draw("lognormal_sigma")
        elif kind == "quantization":
            spec.update(step=self._draw("quantization_step", log=True),
                        offset=self._draw("effect_offset"), latent_scale=self._draw("component_scale"))
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
        return spec["kind"] in ("perlin", "pink") or any(
            cls._contains_correlated_noise(c) for c in spec.get("components", []))

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
            result = (rng.lognormal(0, spec["sigma"], size) - np.exp(variance / 2)) / np.sqrt(np.expm1(variance) * np.exp(variance))
        elif kind == "poisson":
            result = (rng.poisson(spec["rate"], size) - spec["rate"]) / np.sqrt(spec["rate"])
        elif kind == "beta":
            a, b = spec["alpha"], spec["beta"]
            result = (rng.beta(a, b, size) - a / (a + b)) / np.sqrt(a * b / ((a + b)**2 * (a + b + 1)))
        elif kind == "weibull":
            k = spec["shape"]
            mean = math.gamma(1 + 1 / k)
            result = (rng.weibull(k, size) - mean) / np.sqrt(math.gamma(1 + 2 / k) - mean**2)
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
            instruction = {"op": op, "a": i, "b": int(r.integers(0, i + 1)),
                           "gain": self._draw("effect_gain", signed=True),
                           "offset": self._draw("effect_offset"), "limit": self._draw("effect_limit")}
            if op == "power":
                instruction["power"] = float(r.uniform(0.4, 3.0))
            if op == "inverse_square":
                instruction["floor"] = self._draw("inverse_square_floor", log=True)
            if op == "tangent":
                instruction["angle_limit"] = float(r.uniform(0.7, 1.45))
            if op == "mlp":
                width = int(r.integers(3, 13))
                instruction.update(w=r.normal(size=width).tolist(),
                                   bias=r.normal(0, r.uniform(0.2, 2), size=width).tolist(),
                                   v=(r.normal(size=width) / np.sqrt(width)).tolist())
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
            elif op == "tangent":
                cap = instruction["angle_limit"]
                result = np.tan(cap * np.tanh(z / cap))
            elif op == "sigmoid":
                result = np.tanh(z / 2)  # 2*sigmoid(z)-1, without exponential overflow.
            elif op == "exponential":
                result = np.expm1(limit * np.tanh(z / limit)) / limit
            elif op == "power":
                result = np.sign(z) * abs(z)**instruction["power"]
            elif op == "inverse_square":
                # Exactly 1/z² outside a sampled near-zero floor, followed by
                # the same soft actuator saturation as the other operators.
                # The instruction's gain/offset shift and scale the singularity.
                result = (1 / max(abs(z), instruction["floor"]))**2
            elif op == "log":
                result = np.sign(z) * np.log1p(abs(z))
            elif op == "relu":
                result = max(0, z)
            elif op == "mlp":
                result = np.dot(instruction["v"], np.tanh(np.asarray(instruction["w"]) * z + instruction["bias"]))
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
        return {"parent": parent, "lag": lag,
                "weight": self._draw("edge_strength", log=True, signed=True),
                "input_scale": self._draw("input_scale", log=True),
                "input_offset": self._draw("effect_offset"),
                "output_offset": self._draw("effect_offset"), "program": self._expression()}

    @classmethod
    def _effect(cls, edge, value):
        value = (value - edge["input_offset"]) / edge["input_scale"]
        return edge["weight"] * (cls._evaluate(edge["program"], value) + edge["output_offset"])

    def _dynamics(self, event_name):
        r = self.rng
        drifting = r.random() < self.drift_probability
        return {
            "noise": self._sample_noise(), "persistence": self._draw("persistence"),
            "intercept": self._draw("effect_offset"),
            "seasons": [{"period": self._draw("seasonal_period", log=True),
                         "amplitude": self._draw("seasonal_amplitude", log=True),
                         "phase": float(r.uniform(-np.pi, np.pi)),
                         "modulation": float(r.uniform(0, 0.8)),
                         "modulation_period": self._draw("seasonal_period", log=True)}
                        for _ in range(int(r.integers(0, 4)))],
            "slope": self._draw("drift_magnitude", log=True, signed=True) if drifting else 0.0,
            "slope_walk": self._draw("slope_walk_scale", log=True) if drifting else 0.0,
            "walk_scale": self._draw("walk_scale", log=True) if r.random() < self.walk_probability else 0.0,
            "event_parent": event_name if r.random() < 0.7 else None,
            "event_mode": str(r.choice(["reverse", "reset", "accelerate"])),
            "event_slope": self._draw("drift_magnitude", log=True, signed=True),
            "level_jump": self._draw("shock_magnitude", log=True, signed=True) if r.random() < 0.5 else 0.0,
            "shock_size": self._draw("shock_magnitude", log=True, signed=True),
            "shock_decay": self._draw("decay"), "heteroskedasticity": float(r.uniform(0, 1.5)),
            "quantization_step": self._draw("quantization_step", log=True) if r.random() < 0.25 else 0.0,
            "quantization_offset": self._draw("effect_offset"),
        }

    def sample_model(self):
        r = self.rng
        count = int(r.integers(self.min_covariates, self.max_covariates + 1))
        lag_count = int(r.integers(1, min(self.max_lag_count, self.max_lag) + 1))
        lags = sorted([1] + r.choice(np.arange(2, self.max_lag + 1), lag_count - 1, replace=False).tolist())
        event = "covariate_00"
        nodes = [{"name": event, "kind": "event", "edges": [],
                  "visible": bool(r.random() >= self.hidden_probability), "probability": self._draw("event_probability")}]
        topology = str(r.choice(["sparse", "chain", "diamond", "dense"]))
        for i in range(1, count):
            node = {"name": f"covariate_{i:02d}", "kind": "process",
                    "visible": bool(r.random() >= self.hidden_probability), "edges": [], **self._dynamics(event)}
            probability = self.edge_probability * (1.8 if topology == "dense" else 1)
            for parent in nodes:
                if r.random() < probability:
                    lag = int(r.integers(1, self.max_lag + 1)) if r.random() < self.delayed_probability else 0
                    node["edges"].append(self._edge(parent["name"], lag))
            # Scaffolds allow deep mediation, shared causes and colliders.
            # Random extra edges retain diversity; hiding is independently drawn.
            forced = []
            if topology == "chain" and i > 1:
                forced = [nodes[-1]["name"]]
            if topology == "diamond" and count >= 5:
                forced = {2: ["covariate_01"], 3: ["covariate_01"],
                          4: ["covariate_02", "covariate_03"]}.get(i, [])
            for parent in forced:
                if not any(e["parent"] == parent and e["lag"] == 0 for e in node["edges"]):
                    node["edges"].append(self._edge(parent))
            if r.random() < self.delayed_probability:
                node["edges"].append(self._edge(f"outcome_lag_{int(r.choice(lags))}"))
            nodes.append(node)
        # The last process may mediate many upstream causes. Most covariates
        # need not directly cause the outcome; some may be irrelevant to it.
        selected = [n for n in nodes[1:-1] if r.random() < self.direct_probability] + [nodes[-1]]
        outcome = {**self._dynamics(event), "edges": [self._edge(n["name"],
                   int(r.integers(1, self.max_lag + 1)) if r.random() < self.delayed_probability else 0)
                   for n in selected], "baseline": self._draw("baseline"),
                   "scale": self._draw("outcome_scale", log=True),
                   "ar_weights": (r.dirichlet(np.ones(len(lags))) * self._draw("ar_mass")).tolist()}
        for node in nodes[1:] + [outcome]:
            node["interaction"] = (float(r.uniform(-0.8, 0.8)) if len(node["edges"]) >= 2 and r.random() < 0.5 else 0.0)
        return {"version": 2, "observation_seed": int(r.integers(0, 2**63)),
                "rows": int(r.integers(self.min_rows, self.max_rows + 1)),
                "burn_in": self.burn_in, "lags": lags, "nodes": nodes, "outcome": outcome,
                "topology": topology, "prior_ranges": self.priors.copy(),
                "noise_weights": self.noise_weights.copy(),
                "equations": {
                    "process": "x=quantize(intercept+level+seasonal+shock+residual); residual=persistence*previous_residual+forcing+heteroskedastic_noise",
                    "level": "slope+=slope_walk*N(0,1); on event reverse/reset/add event_slope; level+=slope+walk_scale*N(0,1)+event*level_jump",
                    "shock": "shock=shock_decay*previous_shock+event*shock_size",
                    "seasonal": "sum(amplitude*(1+modulation*sin(2*pi*t/modulation_period))*sin(2*pi*t/period+phase))",
                    "forcing": "sum(edge effects)+interaction*tanh(effect[0])*tanh(effect[1])",
                    "noise": "saved noise draw*(1+heteroskedasticity*abs(tanh(previous_residual)))",
                    "outcome": "same process except residual AR uses ar_weights and selected lags of residual; physical outcome=baseline+scale*x",
                    "quantization": "offset+step*round((x-offset)/step), bypassed when step=0",
                    "initialization": "States zero except saved slopes; t starts at -burn_in; discard t<0. Outcome-lag edge parents use full normalized historical x; exported lags use physical outcome units.",
                    "graph": "Topological contemporaneous edges plus positive temporal lags. Past values are separate vertices when time is unrolled; innovations are independent.",
                }}

    def simulate(self, model):
        if model["version"] == 1:
            try:
                from .legacy_scm_generator import OperationalSCMGenerator as Legacy
            except ImportError:
                from legacy_scm_generator import OperationalSCMGenerator as Legacy
            # Dispatch on a legacy instance so its noise/effect evaluators also
            # use the original schema, preserving existing dataset replay.
            return Legacy(self.output_dir).simulate(model)
        if model["version"] != 2:
            raise ValueError("Unsupported SCM schema version")
        r = np.random.default_rng(model["observation_seed"])
        rows, warmup, lags = model["rows"], model["burn_in"], model["lags"]
        length = rows + warmup
        names = [n["name"] for n in model["nodes"]] + ["outcome"]
        specs = model["nodes"] + [model["outcome"]]
        # Precompute innovations once. Visibility never affects random draws.
        history = {name: np.zeros(length) for name in names}
        residuals = {name: np.zeros(length) for name in names}
        randoms, states = {}, {}
        for name, spec in zip(names, specs):
            if spec.get("kind") == "event":
                history[name][:] = r.binomial(1, spec["probability"], length)
            else:
                randoms[name] = (self._noise(r, spec["noise"], length), r.normal(size=length), r.normal(size=length))
                states[name] = [0.0, spec["slope"], 0.0]  # level, slope, decaying shock

        def past(name, index):
            return history[name][index] if index >= 0 else 0.0

        for i in range(length):
            t = i - warmup
            for name, spec in zip(names, specs):
                if spec.get("kind") == "event":
                    continue
                noise, walk, slope_walk = randoms[name]
                event = history[spec["event_parent"]][i] if spec["event_parent"] else 0
                level, slope, shock = states[name]
                slope += spec["slope_walk"] * slope_walk[i]
                if event:
                    mode = spec["event_mode"]
                    slope = -slope if mode == "reverse" else (spec["event_slope"] if mode == "reset" else slope + spec["event_slope"])
                level += slope + spec["walk_scale"] * walk[i] + event * spec["level_jump"]
                shock = spec["shock_decay"] * shock + event * spec["shock_size"]
                states[name] = [level, slope, shock]
                seasonal = sum(s["amplitude"] * (1 + s["modulation"] * np.sin(2 * np.pi * t / s["modulation_period"])) *
                               np.sin(2 * np.pi * t / s["period"] + s["phase"]) for s in spec["seasons"])
                effects = []
                for edge in spec["edges"]:
                    parent = edge["parent"]
                    value = (past("outcome", i - int(parent.rsplit("_", 1)[1]) - edge["lag"])
                             if parent.startswith("outcome_lag_") else past(parent, i - edge["lag"]))
                    effects.append(self._effect(edge, value))
                forcing = sum(effects)
                if spec["interaction"]:
                    forcing += spec["interaction"] * np.tanh(effects[0]) * np.tanh(effects[1])
                previous = residuals[name][i - 1] if i else 0.0
                ar = (sum(w * residuals[name][i - lag] for w, lag in zip(spec["ar_weights"], lags) if i >= lag)
                      if name == "outcome" else spec["persistence"] * previous)
                residual = ar + forcing + noise[i] * (1 + spec["heteroskedasticity"] * abs(np.tanh(previous)))
                residuals[name][i] = residual
                value = spec["intercept"] + level + shock + seasonal + residual
                history[name][i] = self._quantize(value, spec["quantization_step"], spec["quantization_offset"])
        outcome = model["outcome"]
        physical = outcome["baseline"] + outcome["scale"] * history["outcome"]
        data = {"time_index": np.arange(rows), "outcome": physical[warmup:]}
        for lag in lags:
            data[f"outcome_lag_{lag}"] = physical[warmup - lag:length - lag]
        for node in model["nodes"]:
            if node["visible"]:
                data[node["name"]] = history[node["name"]][warmup:]
        frame = pd.DataFrame(data)
        if not np.isfinite(frame.to_numpy()).all():
            raise FloatingPointError("Non-finite SCM observations; inspect sampled parameters")
        return frame

    @staticmethod
    def graph_spec(model):
        """DAG with historical proxy vertices, avoiding misleading causal cycles."""
        vertices = {n["name"]: {"visible": n["visible"], "kind": n["kind"]} for n in model["nodes"]}
        vertices["outcome"] = {"visible": True, "kind": "outcome"}
        edges = []
        def connect(parent, target, lag, label):
            source = f"{parent}[t-{lag}]" if lag else parent
            vertices.setdefault(source, {"visible": vertices.get(parent, {}).get("visible", True), "kind": "history"})
            edges.append({"source": source, "target": target, "label": label, "lag": lag})
        for name, spec in [(n["name"], n) for n in model["nodes"]] + [("outcome", model["outcome"])]:
            for edge in spec["edges"]:
                parent, lag = edge["parent"], edge["lag"]
                if parent.startswith("outcome_lag_"):
                    lag += int(parent.rsplit("_", 1)[1])
                    parent = "outcome"
                connect(parent, name, lag, f"w={edge['weight']:.2g}")
            if spec.get("kind") != "event":
                connect(f"{name} state", name, 1, "state")
                if spec["event_parent"]:
                    connect(spec["event_parent"], name, 0, "event")
        for lag, weight in zip(model["lags"], model["outcome"]["ar_weights"]):
            connect("outcome residual", "outcome", lag, f"AR={weight:.2g}")
        return {"vertices": vertices, "edges": edges}

    def _plot_dag(self, model, path):
        from matplotlib.backends.backend_agg import FigureCanvasAgg
        from matplotlib.figure import Figure
        graph = self.graph_spec(model)
        vertices, edges = graph["vertices"], graph["edges"]
        layers, pending = {}, set(vertices)
        while pending:
            ready = sorted(n for n in pending if all(e["source"] in layers for e in edges if e["target"] == n))
            if not ready:
                raise ValueError("Contemporaneous graph contains a cycle")
            for name in ready:
                layers[name] = max([layers[e["source"]] + 1 for e in edges if e["target"] == name] or [0])
                pending.remove(name)
        grouped = {layer: [n for n in layers if layers[n] == layer] for layer in set(layers.values())}
        height = max(len(group) for group in grouped.values())
        figure = Figure(figsize=(max(14, len(grouped) * 3.2), max(7, height * 0.45)), constrained_layout=True)
        FigureCanvasAgg(figure)
        ax = figure.subplots()
        positions = {name: (layer, y) for layer, group in grouped.items()
                     for name, y in zip(group, np.linspace(0.04, 0.96, len(group)))}
        for edge in edges:
            source, target = positions[edge["source"]], positions[edge["target"]]
            ax.annotate("", xy=target, xytext=source, arrowprops={"arrowstyle": "->", "alpha": 0.35,
                        "color": "#8a539b" if edge["lag"] else "#526576",
                        "linestyle": "--" if edge["lag"] else "-", "shrinkA": 25, "shrinkB": 25})
        for name, (x, y) in positions.items():
            meta = vertices[name]
            color = "#d9e2eb" if meta["kind"] == "history" else ("#bde4d5" if meta["visible"] else "#f3c5a5")
            ax.text(x, y, name, ha="center", va="center", fontsize=7,
                    bbox={"boxstyle": "round,pad=0.4", "fc": color, "ec": "#637482"})
        ax.set(xlim=(-0.5, max(layers.values()) + 0.5), ylim=(-0.03, 1.05),
               title="SCM: green=visible, orange=hidden, gray=history; dashed=temporal edge\nIndependent innovations omitted; state includes level, slope, shock and residual. Weights are in scm.json.")
        ax.axis("off")
        figure.savefig(path, dpi=140)
        figure.clear()

    def _plot(self, frame, path):
        from matplotlib.backends.backend_agg import FigureCanvasAgg
        from matplotlib.figure import Figure
        covariates = [c for c in frame if c.startswith("covariate_")] if self.plot_covariates else []
        # Separate axes keep binary events and small sensors visible next to
        # covariates that have accumulated large trends.
        figure = Figure(figsize=(12, 3 + 1.3 * len(covariates)), constrained_layout=True)
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
        if isinstance(num_tables, bool) or not isinstance(num_tables, int) or num_tables < 1:
            raise ValueError("num_tables must be a positive integer")
        dated_dir = self.output_dir / date.today().isoformat()
        dated_dir.mkdir(parents=True, exist_ok=True)
        run_id, paths = uuid.uuid4().hex[:12], []
        for index in range(num_tables):
            model = self.sample_model()
            frame = self.simulate(model)
            directory = dated_dir / f"table_{run_id}_{index:06d}"
            directory.mkdir(exist_ok=False)
            frame.to_parquet(directory / "data.parquet", engine="pyarrow", index=False, compression="zstd")
            if self.save_scm:
                (directory / "scm.json").write_text(json.dumps(model, indent=2, allow_nan=False), encoding="utf-8")
            if self.save_plot:
                self._plot(frame, directory / "series.png")
            if self.save_dag:
                self._plot_dag(model, directory / "dag.png")
            paths.append(directory)
            print(f"[{index + 1}/{num_tables}] {directory.name}: {len(frame)} rows, {len(frame.columns)} columns", flush=True)
        return paths

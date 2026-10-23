"""Frozen version-1 implementation, retained only to replay existing SCM JSON."""

import json
import uuid
from datetime import date
from pathlib import Path

import numpy as np
import pandas as pd


class OperationalSCMGenerator:
    """Sample a temporal DAG, then simulate it in causal order.

    Covariates use normalized units. The outcome has a sampled baseline and
    scale; its autoregression operates in normalized units. Edges are bounded
    to prevent exponential/tangent/MLP relationships from causing explosions.
    This is a synthetic prior, not a calibrated model of a particular plant.
    """

    RELATIONSHIPS = (
        "linear",
        "exponential",
        "sine",
        "cosine",
        "tangent",
        "sigmoid",
        "quadratic",
        "mlp",
    )
    NOISES = ("normal", "student_t", "laplace", "uniform", "skewed", "mixture")

    def __init__(
        self,
        output_dir,
        *,
        seed=42,
        min_rows=20,
        max_rows=2056,
        min_covariates=3,
        max_covariates=10,
        max_lag=48,
        max_lag_count=5,
        hidden_probability=0.35,
        edge_probability=0.3,
        burn_in=256,
        save_scm=True,
        save_plot=True,
        plot_covariates=True,
    ):
        for name, value in {
            "min_rows": min_rows,
            "max_rows": max_rows,
            "min_covariates": min_covariates,
            "max_covariates": max_covariates,
            "max_lag": max_lag,
            "max_lag_count": max_lag_count,
            "burn_in": burn_in,
        }.items():
            if not isinstance(value, int) or isinstance(value, bool) or value < 1:
                raise ValueError(f"{name} must be a positive integer")
        if min_rows > max_rows or not 3 <= min_covariates <= max_covariates:
            raise ValueError("Invalid row bounds or covariate bounds (minimum 3)")
        if not 0 <= hidden_probability <= 1 or not 0 <= edge_probability <= 1:
            raise ValueError("Probabilities must lie in [0, 1]")
        self.output_dir = Path(output_dir)
        self.rng = np.random.default_rng(seed)
        self.min_rows, self.max_rows = min_rows, max_rows
        self.min_covariates, self.max_covariates = min_covariates, max_covariates
        self.max_lag, self.max_lag_count = max_lag, max_lag_count
        self.hidden_probability, self.edge_probability = (
            hidden_probability,
            edge_probability,
        )
        self.burn_in = max(burn_in, max_lag)
        self.save_scm, self.save_plot = save_scm, save_plot
        self.plot_covariates = plot_covariates

    def _edge(self, parent, weight=None):
        r = self.rng
        kind = str(r.choice(self.RELATIONSHIPS))
        edge = {
            "parent": parent,
            "kind": kind,
            "weight": float(r.uniform(-0.7, 0.7) if weight is None else weight),
            "gain": float(r.uniform(0.3, 1.5)),
            "bias": float(r.uniform(-0.5, 0.5)),
        }
        if kind == "mlp":
            # A randomly initialized one-hidden-layer neural network, frozen per SCM.
            edge["input_weights"] = r.normal(size=6).tolist()
            edge["hidden_bias"] = r.normal(size=6).tolist()
            edge["output_weights"] = (r.normal(size=6) / 3).tolist()
        return edge

    def sample_model(self):
        """Return a JSON-serializable specification, including all hidden nodes."""
        r = self.rng
        count = int(r.integers(self.min_covariates, self.max_covariates + 1))
        lag_count = int(r.integers(1, min(self.max_lag_count, self.max_lag) + 1))
        lags = sorted(
            [1]
            + r.choice(
                np.arange(2, self.max_lag + 1), size=lag_count - 1, replace=False
            ).tolist()
        )
        nodes = []
        for i in range(count):
            kind = (
                ("seasonal", "shock", "driver")[i]
                if i < 3
                else str(r.choice(["seasonal", "shock", "driver", "derived"]))
            )
            candidates = (
                []
                if kind != "derived"
                else (
                    [n["name"] for n in nodes] + [f"outcome_lag_{lag}" for lag in lags]
                )
            )
            parents = [p for p in candidates if r.random() < self.edge_probability]
            if kind == "derived" and not parents:
                parents = [str(r.choice(candidates))]
            nodes.append(
                {
                    "name": f"covariate_{i:02d}",
                    "kind": kind,
                    "visible": bool(r.random() >= self.hidden_probability),
                    "edges": [self._edge(p) for p in parents],
                    "noise": str(r.choice(self.NOISES)),
                    "noise_scale": float(r.uniform(0.03, 0.25)),
                    "persistence": float(r.uniform(0.5, 0.99)),
                    "period": float(
                        r.choice([12, 24, 48, 96, 168, 336]) * r.uniform(0.9, 1.1)
                    ),
                    "phase": float(r.uniform(0, 2 * np.pi)),
                    "amplitude": float(r.uniform(0.3, 1.5)),
                    "shock_probability": float(r.uniform(0.005, 0.06)),
                    "shock_decay": float(r.uniform(0.80, 0.995)),
                    "shock_scale": float(r.uniform(0.5, 3.0)),
                    "shock_sign": int(r.choice([-1, 1])),
                }
            )
        ar_weights = r.dirichlet(np.ones(len(lags))) * r.uniform(0.4, 0.94)
        outcome_edges = [self._edge(n["name"]) for n in nodes]
        # Ensure the operational seasonal and shock components survive the transform.
        for edge in outcome_edges[:2]:
            edge.update(
                kind="linear", weight=float(r.uniform(0.25, 0.8)), gain=1.0, bias=0.0
            )
        return {
            "version": 1,
            "observation_seed": int(r.integers(0, 2**63)),
            "rows": int(r.integers(self.min_rows, self.max_rows + 1)),
            "burn_in": self.burn_in,
            "lags": lags,
            "nodes": nodes,
            "outcome": {
                "ar_weights": ar_weights.tolist(),
                "edges": outcome_edges,
                "baseline": float(r.uniform(20, 500)),
                "scale": float(r.uniform(0.5, 15)),
                "noise": str(r.choice(self.NOISES)),
                "noise_scale": float(r.uniform(0.02, 0.3)),
                "heteroskedasticity": float(r.uniform(0, 0.5)),
            },
            "equations": {
                "edge": "weight * f(gain * clip(parent, -4, 4) + bias)",
                "functions": {
                    "linear": "z",
                    "exponential": "expm1(clip(z,-3,3))/6",
                    "sine": "sin(z)",
                    "cosine": "cos(z)",
                    "tangent": "tan(clip(z,-1.2,1.2))/3",
                    "sigmoid": "2/(1+exp(-z))-1",
                    "quadratic": "min(z*z,9)/3-1",
                    "mlp": "dot(output_weights,tanh(input_weights*z+hidden_bias))",
                },
                "driver": "x[t]=persistence*x[t-1]+sqrt(1-persistence**2)*innovation[t]",
                "seasonal": "x[t]=amplitude*(sin(2*pi*t/period+phase)+0.3*cos(4*pi*t/period+phase))+noise_scale*innovation[t]",
                "shock": "x[t]=shock_decay*x[t-1]+Bernoulli(shock_probability)*shock_sign*shock_scale*LogNormal(0,0.4)",
                "derived": "x[t]=0.35*x[t-1]+sum(edge(parent[t]))+noise_scale*innovation[t]",
                "outcome": "z[t]=sum(ar_weight[k]*z[t-lag[k]])+sum(edge(covariate[t]))/sqrt(number_of_covariates)+noise_scale*(1+heteroskedasticity*abs(tanh(covariate_00[t])))*innovation[t]; outcome[t]=baseline+scale*z[t]",
                "initialization": "All normalized states and pre-simulation outcome history are zero. Time starts at -burn_in; only t>=0 is recorded. Lag parents use normalized z; exported lag columns use outcome units.",
                "noise_distributions": {
                    "normal": "Normal(0,1)",
                    "student_t": "StudentT(df=4)/sqrt(2)",
                    "laplace": "Laplace(0,1/sqrt(2))",
                    "uniform": "Uniform(-sqrt(3),sqrt(3))",
                    "skewed": "(LogNormal(0,0.5)-exp(0.125))/sqrt((exp(0.25)-1)*exp(0.25))",
                    "mixture": "Normal(0,1)*(5 if Uniform(0,1)<0.03 else 1)/sqrt(1.72)",
                },
            },
        }

    @staticmethod
    def _noise(rng, kind, size):
        if kind == "normal":
            return rng.normal(size=size)
        if kind == "student_t":
            return rng.standard_t(4, size=size) / np.sqrt(2)
        if kind == "laplace":
            return rng.laplace(scale=1 / np.sqrt(2), size=size)
        if kind == "uniform":
            return rng.uniform(-np.sqrt(3), np.sqrt(3), size=size)
        if kind == "skewed":
            return (rng.lognormal(0, 0.5, size=size) - np.exp(0.125)) / np.sqrt(
                np.expm1(0.25) * np.exp(0.25)
            )
        if kind == "mixture":
            return (
                rng.normal(size=size)
                * np.where(rng.random(size) < 0.03, 5, 1)
                / np.sqrt(1.72)
            )
        raise ValueError(f"Unknown noise: {kind}")

    @staticmethod
    def _effect(edge, value):
        z = edge["gain"] * min(4.0, max(-4.0, value)) + edge["bias"]
        kind = edge["kind"]
        if kind == "linear":
            result = z
        elif kind == "exponential":
            result = np.expm1(min(3, max(-3, z))) / 6
        elif kind == "sine":
            result = np.sin(z)
        elif kind == "cosine":
            result = np.cos(z)
        elif kind == "tangent":
            result = np.tan(min(1.2, max(-1.2, z))) / 3
        elif kind == "sigmoid":
            result = 2 / (1 + np.exp(-z)) - 1
        elif kind == "quadratic":
            result = min(z * z, 9) / 3 - 1
        elif kind == "mlp":
            result = np.dot(
                edge["output_weights"],
                np.tanh(np.asarray(edge["input_weights"]) * z + edge["hidden_bias"]),
            )
        else:
            raise ValueError(f"Unknown relationship: {kind}")
        return edge["weight"] * result

    def simulate(self, model):
        """Replay a specification; hiding changes output columns, not causality."""
        rng = np.random.default_rng(model["observation_seed"])
        warmup, rows, lags = model["burn_in"], model["rows"], model["lags"]
        length, offset = warmup + rows, max(lags)
        time = np.arange(-warmup, rows)
        outcome = model["outcome"]
        z = np.zeros(length + offset)
        values = {n["name"]: np.zeros(length) for n in model["nodes"]}
        innovations = {}
        for node in model["nodes"]:
            innovations[node["name"]] = self._noise(rng, node["noise"], length)
            if node["kind"] == "shock":
                innovations[node["name"]] = (
                    rng.random(length) < node["shock_probability"]
                ) * (
                    node["shock_sign"]
                    * node["shock_scale"]
                    * rng.lognormal(0, 0.4, length)
                )
        errors = self._noise(rng, outcome["noise"], length)
        for i, t in enumerate(time):
            parents = {f"outcome_lag_{lag}": z[offset + i - lag] for lag in lags}
            for node in model["nodes"]:
                name, kind = node["name"], node["kind"]
                previous = values[name][i - 1] if i else 0.0
                error = innovations[name][i]
                if kind == "seasonal":
                    angle = 2 * np.pi * t / node["period"]
                    value = (
                        node["amplitude"]
                        * (
                            np.sin(angle + node["phase"])
                            + 0.3 * np.cos(2 * angle + node["phase"])
                        )
                        + node["noise_scale"] * error
                    )
                elif kind == "shock":
                    value = node["shock_decay"] * previous + error
                elif kind == "driver":
                    value = (
                        node["persistence"] * previous
                        + np.sqrt(1 - node["persistence"] ** 2) * error
                    )
                elif kind == "derived":
                    value = (
                        0.35 * previous
                        + sum(
                            self._effect(e, parents[e["parent"]]) for e in node["edges"]
                        )
                        + node["noise_scale"] * error
                    )
                else:
                    raise ValueError(f"Unknown node kind: {kind}")
                values[name][i] = parents[name] = value
            forcing = sum(
                self._effect(e, parents[e["parent"]]) for e in outcome["edges"]
            )
            noise_scale = outcome["noise_scale"] * (
                1
                + outcome["heteroskedasticity"]
                * abs(np.tanh(parents[model["nodes"][0]["name"]]))
            )
            z[offset + i] = (
                sum(
                    w * parents[f"outcome_lag_{lag}"]
                    for w, lag in zip(outcome["ar_weights"], lags)
                )
                + forcing / np.sqrt(len(values))
                + noise_scale * errors[i]
            )
        physical = outcome["baseline"] + outcome["scale"] * z
        start = offset + warmup
        data = {
            "time_index": np.arange(rows),
            "outcome": physical[start : start + rows],
        }
        for lag in lags:
            data[f"outcome_lag_{lag}"] = physical[start - lag : start + rows - lag]
        for node in model["nodes"]:
            if node["visible"]:
                data[node["name"]] = values[node["name"]][warmup:]
        frame = pd.DataFrame(data)
        if not np.isfinite(frame.to_numpy()).all():
            raise FloatingPointError("SCM generated non-finite observations")
        return frame

    def _plot(self, frame, path):
        # Object-oriented Agg canvas works in headless runs without changing a notebook's backend.
        from matplotlib.backends.backend_agg import FigureCanvasAgg
        from matplotlib.figure import Figure

        covariates = [c for c in frame if c.startswith("covariate_")]
        show_covariates = self.plot_covariates and bool(covariates)
        figure = Figure(
            figsize=(12, 7 if show_covariates else 4), constrained_layout=True
        )
        FigureCanvasAgg(figure)
        axes = figure.subplots(2 if show_covariates else 1, 1, squeeze=False)[:, 0]
        axes[0].plot(frame["time_index"], frame["outcome"], linewidth=1)
        axes[0].set(ylabel="Outcome", title="Synthetic operational time series")
        if show_covariates:
            for name in covariates:
                axes[1].plot(
                    frame["time_index"], frame[name], label=name, linewidth=0.8
                )
            axes[1].set_ylabel("Visible covariates (normalized units)")
            axes[1].legend(fontsize=7, ncol=3)
        axes[-1].set_xlabel("Time step")
        figure.savefig(path, dpi=130)
        figure.clear()

    def generate(self, num_tables=100):
        """Write one directory per SCM; repeated executions never overwrite tables."""
        if (
            not isinstance(num_tables, int)
            or isinstance(num_tables, bool)
            or num_tables < 1
        ):
            raise ValueError("num_tables must be a positive integer")
        dated_dir = self.output_dir / date.today().isoformat()
        dated_dir.mkdir(parents=True, exist_ok=True)
        run_id = uuid.uuid4().hex[:12]
        paths = []
        for index in range(num_tables):
            model = self.sample_model()
            frame = self.simulate(model)
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
                    json.dumps(model, indent=2), encoding="utf-8"
                )
            if self.save_plot:
                self._plot(frame, directory / "series.png")
            paths.append(directory)
            print(
                f"[{index + 1}/{num_tables}] {directory.name}: {len(frame)} rows, {len(frame.columns)} columns"
            )
        return paths

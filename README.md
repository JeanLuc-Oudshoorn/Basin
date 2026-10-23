# Basin

Basin is a time-series foundation model project targeting approximately 400
million parameters, built on RWKV-7. This initial repository sets up and validates
the fused CUDA training backbone; the full forecasting architecture, datasets,
and pretraining pipeline are future work.

## Development checks

Install the quality and security hooks in every new clone (using Python 3.12):

```bash
python -m pip install -r requirements-dev.txt
python -m pre_commit install
python -m pre_commit run --all-files
```

Commits run Ruff linting and formatting, Bandit security analysis, private-key
detection, syntax and configuration checks, and whitespace checks. Hooks that
fix files stop the commit; review and stage their changes, then commit again.
Vendored `third_party/` files are excluded to preserve their recorded hashes.
These static checks catch common issues but do not replace tests or dependency
vulnerability audits.

Run the data-generation and validation tests after installing their dependencies:

```bash
python -m pip install -e ".[data,validation]"
python -m unittest discover -s tests -v
```

## Runtime environment

Use Ubuntu 24.04 under WSL2, Python 3.12, and an NVIDIA GPU with bf16 support.
The exact setup and its local RWKV patches are documented in
[the original setup guide](docs/README_RWKV7_FAST_SETUP.md).
The initial installation and GPU checks passed; see
[the validation report](docs/RWKV7_ENVIRONMENT_VALIDATION.md).

From Bash inside Ubuntu-24.04, in this repository:

```bash
bash scripts/setup_rwkv7.sh
bash scripts/validate_rwkv7.sh
```

The first setup copies the verified patched source from the neighboring `Mamba`
project. Set `MAMBA_SOURCE` to change that source location. Once the vendored
files are present, setup no longer requires Mamba.

The virtual environment defaults to
`~/projects/basin-runtime/.venv-wsl` on the Linux filesystem. Set `BASIN_VENV`
consistently during setup and activation to use another location. In PyCharm,
select that environment's `bin/python` using the Ubuntu-24.04 WSL interpreter.
The native Windows `.venv` is not the CUDA training environment.

For subsequent training and smoke-test runs:

```bash
source scripts/activate_rwkv7.sh
python smoke_rwkv7_timeseries.py
```

Activation selects bf16 kernels for the RTX 5070 (`sm_120`), skips the unused LM
cross-entropy extension, and uses a persistent project-specific extension cache.
The validation script builds once, checks a fresh import with the builder
replaced by a function that raises, and runs continuous-valued forward,
backward, and optimizer steps. It records package versions, source hashes, and
dependency-check output.

Create a new extension-cache suffix in `scripts/activate_rwkv7.sh` after changing
the toolchain, GPU architecture, kernel sources, or compiler flags. Direct
library loading intentionally bypasses build freshness checks.

The repository can remain in this Windows checkout. For faster source imports,
clone it into WSL's Linux filesystem and run the same scripts there. The venv
and extension cache already live on the Linux filesystem.

## Convert public MATLAB datasets

Place `.mat` files in `data/public/`, install the data dependencies, then run
this file in IPython:

```bash
python -m pip install -e ".[data]"
%run scripts/convert_public_mat.py
```

The script writes tables under `data/public_cleaned/<dataset>/`, keeping
separate tables for nested groups with different measurements. Aligned signal
columns stay together, scalar fields are repeated as metadata, and struct
arrays get a `record_index` column. Differing sample lengths are kept separate
without truncating signals. The converter paths are based on the repository
location, so `%run` works regardless of the current working directory.

## Generate synthetic operational data

Install the data dependencies with `python -m pip install -e ".[data]"`, edit
the settings at the top of `src/data_generation/generate_operational_data.py`, then run:

```bash
python src/data_generation/generate_operational_data.py
```

The runner currently generates 30 independent structural causal models with
20–2056 observations each. Results go to
`data/generated/YYYY-MM-DD/table_<run-id>_<index>/`. Each directory contains
`data.parquet`, a replayable `scm.json`, `series.png`, and `dag.png`. Repeated executions
use new directory names. `SAVE_SCM`, `SAVE_PLOT`, `SAVE_DAG`, and `PLOT_COVARIATES` control
the optional artifacts; all default to true. Paths work from any working directory,
including IPython `%run`.

The reusable `OperationalSCMGenerator` lives in `src/data_generation/scm_generator.py`.
The runner samples 3–10 covariates. Sparse, fork, dense, and diamond
structures allow mediation, shared causes, colliders, and hidden confounding.
Multiple covariates directly affect the outcome: each process is selected with
probability 0.65, with at least two selected when available. Full-chain scaffolds
are no longer sampled. All sampled covariate influences, including mediated
edges, direct outcome inputs, event shocks, damped responses and shutdown
controls, use a shared delay prior. `INSTANTANEOUS_PROBABILITY=0.10` assigns
10% probability to lag zero. Otherwise the delay is one plus the floored draw
from an exponential distribution, truncated before discretization to respect
`MAX_COVARIATE_LAG` (24 by default). `COVARIATE_DELAY_SCALE=0.75` gives approximate
overall lag probabilities 10%/66%/17%/5% at zero/one/two/three steps, with a rapidly
decreasing tail. Larger scales produce longer delays. Scaffold edges follow the
same prior, rather than forcing same-time mediation. `OUTCOME_FEEDBACK_PROBABILITY`
separately controls how often historical outcomes feed covariates (the constructor
retains the older `delayed_probability` argument name for that probability).
`MAX_LAG` (192 by default) applies to outcome history; historical outcomes can
also influence covariates. Graph images show historical values as
separate vertices, so temporal feedback does not appear as a contemporaneous
cycle. Diagrams now use NetworkX with Y at the center, bold direct influences,
faint upstream edges and seasonal-lag labels. Green nodes represent visible
covariates at any timestep, orange nodes represent hidden covariates at any
timestep, and purple nodes represent outcome-derived history, including
residuals. The current outcome is blue. Solid/dashed arrows distinguish
same-time/delayed influences without changing the meaning of node colors.
The image focuses on Y[t]'s direct inputs plus one upstream generation.
Mediated paths are unrolled at consistent timestamps, such as
C0[t-3] -> C1[t-1] -> Y[t], with no unused C0[t] or C1[t] copies.
Deeper ancestry is omitted visually to bound recursive temporal feedback;
`graph_spec` and `scm.json` retain all structural equations. Internal state histories, clock and independent
innovations are omitted for readability. Before simulation or rendering, temporal
validation rejects negative delays, unknown parents and invalid same-time
evaluation order. Saved SCMs without explicit special-mechanism delays retain
their original zero-delay behavior when replayed. Install/update the data extra
to obtain NetworkX. Existing generated plots are not overwritten; generate a
fresh batch to inspect the new delay prior in the viewer.

Outcome history always includes t-1. Zero, one or two additional lags from t-2
through t-7 are selected with probabilities 15%, 75% and 10%, with a mild
preference for nearer lags. Coefficient priorities also decay with distance.
`SEASONAL_LAG_PROBABILITY=0.8` adds seasonal anchors with conditional counts
of one/two/three weighted 80%/16%/4%. Anchors come from 8, 12, 24, 48, 96,
and 168 samples, constrained by context length and `MAX_LAG`.
These periods also appear in the outcome's actual seasonal equations.
`LONG_LAG_PROBABILITY=0.20` adds one unstructured lag of at least eight samples,
excluding eligible seasonal anchors. `MAX_LAG_COUNT` caps the combined count;
small caps or short lag ranges can reduce these frequencies. Saved
`lag_roles` and `seasonal_lags` distinguish these mechanisms.

Each edge has a sampled signed strength spanning several orders of magnitude,
input/output offsets, and an expression DAG of up to `MAX_EFFECT_COMPONENTS`
instructions. Instructions compose affine, trigonometric, exponential, sigmoid,
power, logarithmic, rectifier, and random neural-network effects, with sums,
products, gates, and reused intermediate results. Some mechanisms additionally
interact across two parents. Evaluation computes each intermediate result once.

Covariates and the outcome combine independently sampled seasonal components,
amplitude modulation, persistent residuals, drift, random walks, decaying shocks,
and optional quantization. `covariate_00` is a binary event source, which can be
hidden. Events can reverse or reset slopes and cause level jumps and transient
shocks. Covariate cumulative levels have sampled limits; explicit outcome trends
start at the recorded window and are not clipped by those limits. An independent outcome
innovation floor keeps stochastic variation visible. Outcome autoregression
uses the full historical outcome. Each selected outcome lag independently has
an 8% chance of also contributing its residual, alongside its full value.
Exported lags always contain the complete observed outcome, including outages.

`TREND_FREE_FRACTION=0.5` reserves half of each generated batch (rounded up) for a
profile that disables cumulative drift, walks, level jumps, and event-induced
slopes throughout the graph. Those tables are accepted only when outcome linear
trend R² is at most 0.15; the score and sampling attempts are saved in `quality`.
This is an operational screening rule, not a test of formal stationarity or
absence of every possible nonlinear trend. The remaining tables always receive
an explicit rising/falling, accelerating/decelerating, or broken-slope outcome
trend, sized relative to seasonal/noise amplitude. They are accepted only when
the specified trend shape explains at least 40% of observed outcome variance.
For broken trends, this criterion includes a hinge at the saved event time so
strong reversals are not incorrectly rejected for having a small overall slope.
`DRIFT_PROBABILITY` controls additional covariate drift within trending profiles;
it is not a second probability that removes their outcome trend.

The runner's `MULTIPLICATIVE_TREND_PROBABILITY=0.3` selects multiplicative growth/decay for
30% of trending SCM proposals (roughly 15% of all proposals with the 50/50 split).
A positive latent level evolves through `L[t+1]=L[t]*exp(log_rate)` and contributes
`L[t]-initial_level` to the structural outcome equation. Its initial level and
rate are saved in the SCM. The sampled total absolute log change defaults to
0.7–2.5 over the recorded window; 80% grow and 20% decay. Warm-up leaves this
level unchanged. The isolated positive level is linear on a logarithmic axis;
covariates, seasonality, noise, baseline offsets and shutdowns can change the
appearance and positivity of the complete observed outcome. Trend screening
uses the saved exponential shape. Screening may change the accepted frequency
slightly; no generated values are reshaped or corrected.

Multiplicative outcomes now also have level-relative noise and seasonality.
The initial white-noise SD is at least a sampled 2.5–10% of initial latent
level, and seasonal amplitude is scaled relative to that level. Subsequent
noise and seasonal scales use separate powers of `L[t]/L[0]`. In 85% of
multiplicative proposals, noise elasticity is 1–1.4 (proportional or faster
growth); the remaining cases use 0.7–1. Seasonal elasticity is 0.95–1.3.
These mechanisms are saved under `multiplicative_scaling` and operate inside
the structural equations, including before shutdown overrides.

`VARIANCE_DYNAMICS_PROBABILITY=0.55` independently adds a saved `volatility`
mechanism to process/outcome noise: increasing/decreasing SD, cyclical SD, or a
regime change. This is allowed even in trend-free profiles. Distribution means
are retained while deviations and white innovations are scaled, so increasing
variance does not inherently introduce a mean trend. `variance_growth_factor`
controls SD, so variance changes by its square. General volatility and
multiplicative scaling can coexist; existing outcome-dependent heteroskedasticity
also remains available. Screening can alter the realized mixture of profiles.

Broken trends use a saved binary shock event to latch a new slope permanently,
including reversals and increased/decreased rates. Ordinary shock decay does
not undo this change. Warm-up does not consume the explicit trend's excursion.
Rejected models are resampled, never detrended or reshaped after simulation.

Seasonal waveforms can include several harmonics, alternating high/low peaks,
growing amplitude, and smoothly changing cycle lengths. Phase integrates the
varying frequency to avoid artificial discontinuities. Periods are limited
relative to context length so an incomplete cycle less often resembles a ramp.

`SWITCHING_PROBABILITY=0.3` adds a binary operating-state covariate, visible or
hidden, with sampled on/off dwell durations. Shutdown mode sets the complete
outcome exactly to zero or a sampled value during outages, then blends back to
the evolving underlying process over a sampled restart interval. Reset mode
resets an accumulated ramp at each state transition, creating repeating
charging/discharging or sawtooth-like shapes. These controls appear in the DAG.

Noise laws include Gaussian, Student-t, Laplace, uniform, lognormal, Poisson,
beta, Weibull, rounding errors, 1D Perlin noise, pink noise, and brown/red noise. `NOISE_WEIGHTS` controls
relative selection frequencies for both outcome noise and covariate innovations,
including covariates without incoming edges. Perlin, Student-t, and pink each default
to weight 4 versus weight 1 for each other family (16% each at the top level).
The same relative base-family weights apply inside heterogeneous mixtures.
Perlin is a temporally correlated process, with sampled frequency, phase, seed,
octave count, persistence, and lacunarity saved in the SCM. A NumPy gradient-noise
implementation uses quintic interpolation without requiring a compiled package.
Mixture components retain the original time axis when sampling Perlin values.
Pink noise uses a finite causal fractional-integration filter with saved seed,
spectral exponent, and memory length. The default exponent range is 0.8–1.2;
1 gives an approximately 1/f power spectrum above the filter's low-frequency
cutoff. Random prehistory avoids zero-state startup effects. Pink noise also
retains its time axis inside mixtures and can drive exogenous covariates.
Brown noise (`brown`, also called red noise) integrates seeded Gaussian steps,
giving a nonstationary random walk with an approximately 1/f² spectrum.
Its initial value and step scale are sampled and saved; no empirical centering
or normalization changes its trajectory. It can drive fully exogenous covariates,
other process innovations, outcome noise, and heterogeneous mixtures. Components
retain their original time axis inside mixtures. Use `brown` in `NOISE_WEIGHTS`;
its default weight is 1. Such sources can introduce apparent drift, so the
existing trend-profile screening still applies to the finished SCM simulation.

Effects include `damped_sine` and `damped_cosine`, evaluated as
`exp(-decay*abs(z))*sin(frequency*z+phase)` (or cosine), where
`z=gain*input+offset`. Decay, frequency, and phase are sampled and saved.
For nonnegative z this is the standard exponentially damped wave; absolute
value extends the envelope safely to signed covariates. The existing soft
saturation applies after each expression instruction.

Some mechanisms also include actual time-domain ringing: each binary event
adds a damped sine/cosine response with a sampled period, decay, amplitude, and
phase. Responses superpose through a stable causal oscillator state, allowing
a second event to add energy before the first response has decayed. These
event dependencies appear in the DAG and are saved under `ringing` in the SCM.
Effects additionally include `inverse_square`: `1/max(abs(gain*x+offset),floor)^2`,
followed by the usual soft saturation. The sampled floor avoids division by zero;
outside that region this is the reciprocal-square law, with a movable offset.
Shape parameters such as Student-t degrees
of freedom are sampled per mechanism and saved. Mixtures include two shifted
Student-t components, multiple Gaussians, heterogeneous families, and rare
contamination. Mixture weights, component locations and scales vary by model;
overlapping or separated modes are possible. Poisson, beta, and Weibull draws
are centered and variance-normalized innovations, not count/proportion output
constraints. Noise location and scale are then applied, with optional
heteroskedasticity. Quantization also occurs directly in sensor equations,
producing errors dependent on the underlying signal.

`PRIOR_RANGES` overrides sampling ranges in the class's `DEFAULT_PRIORS`.
Other top-level settings control graph density, direct/lagged dependencies,
drift and random-walk prevalence, and effect complexity. Structural parameters
are sampled once per table; only innovations change during simulation.

Parquet columns are `time_index`, `outcome`, `outcome_lag_<k>`, and visible
`covariate_<id>` columns. Time is a regular abstract sample index; seasonal
periods are measured in samples. Covariates use normalized units. Outcome
units use a sampled baseline and scale and can be signed. Lag 1 is always
included, and every selected outcome lag is visible and exactly matches its
historical outcome. A discarded warm-up provides history for the first rows.
Covariates are hidden only from the exported table and plot, not from the
simulation. The JSON includes their equations, parameters, and visibility.

For replay, load the JSON and call `generator.simulate(model)`. The fixed seed
reproduces models and observations in the same numerical environment; filenames
are intentionally unique. Version-1 JSON files remain replayable through the
frozen `legacy_scm_generator.py`; version-2 simulation remains supported and
new models use schema version 3. Softly bounded
edge functions and stable combined autoregression limit runaway feedback while
allowing nonstationary levels. These priors cover a much broader range of
behaviors, but no finite generator guarantees every real operational process.
Domain calibration and checks against real data remain necessary. Short samples
may not contain a shock or a complete seasonal cycle.

Run the generator checks with `python -m unittest discover -s tests`.

Multiplicative trends also have two independent rare exceptions, configured by
`MULTIPLICATIVE_CONSTANT_NOISE_PROBABILITY=0.05` and
`MULTIPLICATIVE_CONSTANT_SEASONALITY_PROBABILITY=0.05`. These probabilities apply
among multiplicative-trend proposals. The first sets noise elasticity to zero
and disables outcome volatility envelopes and residual-dependent noise scaling.
The second sets seasonal elasticity to zero and disables separate exponential
growth of outcome seasonal amplitudes. Bounded seasonal modulation, waveform
evolution and clock variation remain possible. These choices are independent:
noise can remain fixed while seasonal amplitude grows, or vice versa. Fixed
direct noise scaling does not guarantee constant total observed variance, since
stochastic covariates, colored noise, feedback and shutdowns can still affect it.
The selected equations are persisted in the SCM; existing saved models replay
unchanged. Batch quality screening can shift final frequencies from these priors.

### Browse generated plots

Double-click `View generated plots.bat` in the project root, or run
`python src/data_generation/view_generated_plots.py`. A local viewer opens in
your browser with the latest date selected. Choose another date, scroll through
series plots and DAGs side by side, filter by table name, or click a plot to
enlarge it (Esc closes it). Refresh discovers newly generated tables. Images load
as you scroll; missing plots are indicated. No additional dependencies are needed.
Keep the launcher window open while browsing and press Ctrl+C there to stop it.
`GENERATED_DIR`, `INITIAL_DATE` and the local port can be edited in the Python script.

### Fourier seasonal shapes

The seasonal equations now sample finite Fourier expansions for both covariates
and the outcome. In addition to the older sine-plus-harmonics mechanism, shape
priors include duty-cycle pulses, rising/falling sawtooth ramps, asymmetric
triangles, full/half rectified waves, rounded parabolic cycles, irregular
multi-peak cycles and random smooth spectra. These are inspired by the reference
in `data/external/common_fourier_series_examples.png`. Template parameters,
spectral smoothing and coefficient perturbations vary between components.

`FOURIER_PROBABILITY=0.85` selects the new expansion per seasonal component;
`FOURIER_WEIGHTS` controls the relative shape frequencies. Set a family weight
to zero to disable it. `MAX_FOURIER_HARMONICS=16` caps complexity, further limited
by the period and fastest warped clock so carrier harmonics remain below the
sampling limit. Short cycles therefore have fewer resolvable shape details.
The DC coefficient is omitted to separate cyclic variation from the SCM's level
and trend. Spectral normalization gives each static expansion the same RMS as
a unit sine, so adding harmonics does not automatically overwhelm other effects.

`SEASONAL_SHAPE_DRIFT_PROBABILITY=0.25` allows smooth interpolation between two
sampled spectra from the same family. Shape evolution combines with existing
amplitude growth/modulation, alternating peaks, multiple cycles and smoothly
varying frequency. Some shapes remain stable. All coefficients and transition
parameters are saved in `scm.json`; these are structural equations evaluated
at each timestamp, with no post-hoc correction or dependence on future samples.
Older SCMs without the new `fourier` field retain their original seasonal equations.

Run `python src/data_generation/preview_fourier_seasons.py` for a gallery of
isolated seasonal shapes and their saved specifications under
`results/data_generation/fourier_seasons.png` and `fourier_seasons.json`.
This broadens the seasonal prior; it does not establish coverage of every
real-world operational process, which still requires validation against data
from the domains of interest.

## Prophet validation

Install the validation dependencies with `python -m pip install -e ".[validation]"`.
Edit the settings at the top of `src/data_validation/validate_generated_data.py`,
particularly `INPUT_DIR` (a generated day folder), then run:

```powershell
python src/data_validation/validate_generated_data.py
```

Each table's rows are sorted by `time_index`; the first floor(80% of rows)
train the models and the remainder form one fixed forecast horizon. The
baseline predicts the last training outcome throughout that horizon. Two
independent Prophet fits forecast the outcome: one without regressors and one
with visible `covariate_*` columns. Constant training covariates are omitted.
Recorded `outcome_lag_*` columns are excluded because they expose held-out
targets during a multi-step forecast. No SCM metadata is used.

`COVARIATE_MODE="known_future"` supplies actual held-out covariates: this is a
conditional forecast benchmark, assuming those inputs are known in advance.
They may contain information about test outcomes through the generating SCM;
these scores should not be interpreted as performance with unknown future inputs.
Set the mode to `"forecast"` to extrapolate each covariate using a separate
Prophet fit to its training values only. Prophet requires future regressor
values, as described in its [regressor documentation](https://facebook.github.io/prophet/docs/seasonality%2C_holiday_effects%2C_and_regressors.html#additional-regressors).
Synthetic steps map to daily timestamps by default (`STEP_DURATION="1D"`).
Prophet uses automatic weekly/yearly seasonality; optional `SEASONAL_PERIODS`
are specified in timesteps and used only when training contains two cycles.
Both target fits use the same Prophet settings, without test-set tuning.

Every execution creates a unique folder under `results/data_validation` with
`metrics.csv`, `metrics.parquet`, and `settings.json`. Each table has one row:

| Column | Meaning |
| --- | --- |
| `baseline_rmse`, `univariate_rmse`, `multivariate_rmse` | Test RMSE in outcome units |
| `univariate_beats_baseline`, `multivariate_beats_baseline` | 1 if strictly lower RMSE, otherwise 0 |
| `univariate_improvement_pct`, `multivariate_improvement_pct` | 100 × (baseline RMSE − model RMSE) / baseline RMSE |
| `rmse_reduction_multivariate_vs_univariate` | Univariate RMSE − multivariate RMSE |
| `multivariate_vs_univariate_improvement_pct` | 100 × (univariate RMSE − multivariate RMSE) / univariate RMSE |
| `improvement_difference_pp` | Multivariate minus univariate baseline improvement, in percentage points |

Positive improvements mean better predictions; negative values mean worse.
A zero reference RMSE makes percentage improvement undefined (blank/NaN).
Errors are recorded per table/model with missing metrics and win flags, rather
than treating failures as losses or successes. Rows also record split sizes,
covariate names/count, source path, and covariate mode. With no usable covariates,
the second fit is another univariate model and is identified by a zero count.
The CSV is checkpointed after each table; Parquet is written on completion.

Run validation checks with
`python -m unittest discover -s tests -p test_prophet_validator.py`.

## Backend constraints

Use contiguous bf16 activations, sequence lengths divisible by 16, equal model
and attention widths divisible by 64, FFN widths divisible by 16, and at least
two layers. Compute regression losses in fp32 and preserve training
initialization. These sequence kernels do not expose streaming recurrent state.

The vendored RWKV source retains its upstream Apache-2.0
[license](third_party/rwkv-lm/LICENSE). See
[provenance](docs/RWKV7_PROVENANCE.md) for the upstream revision and local patches.

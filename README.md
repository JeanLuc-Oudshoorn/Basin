# Basin

Basin is a time-series foundation model project currently configured at approximately 500
million parameters, built on RWKV-7. The repository includes a fused CUDA training
backbone, synthetic data generation, and a patch-based multivariate forecasting
architecture with an end-to-end training smoke test. Training supports sharded
corpora and resumable timed checkpoints. See [the forecasting guide](docs/forecasting/FORECASTING.md).

The active configuration has 28 layers and 500,056,352 total parameters,
and predicts up to 64 timesteps per
backbone pass, with CUDA graph replay enabled for forecasting. The new direct
forecast head requires retraining. Four lead losses use weights 80%, 16%, 3%, 1%.
Longer requests roll forward in blocks of at most 64 points. See the
[short-horizon training and migration guide](docs/model_training/SHORT_HORIZON_TRAINING.md)
and [inference benchmark](docs/time_benchmark/fast_inference/REPORT.md).

The default trainer uses full training tables, with no reserved validation tail.
Every hour of active training it saves a resumable checkpoint; every four hours it validates
25,000 fixed reserved validation tables at five origins each (125,000 forecasts),
using Basin only. Training, validation and test tables are disjoint: no window
from a validation table enters gradient training. Originals with coarse or
sign-inverted copies and their copies are training-only. Every forecast spans
four tokens with 80/16/3/1 lead weights.
Checkpoints retain optimizer moments/settings, scheduler and RNG states, plus
the exact shuffled batch cursor for continuation within an epoch.
For large sharded corpora, `training.storage_locality=true` visits each shard
in one shuffled section per epoch and shuffles tables within compatible shape
buckets. Compact record handles and reusable shape plans reduce startup memory
and repeated data reads. See the
[million-table pipeline review](docs/audits/2026-11-08/pipeline_scaling.md).
`training.max_run_hours` sets a per-invocation soft duration limit (`null` means
unlimited); the trainer saves at a safe optimizer boundary before exiting.
Early-stopping and scheduler patience use validation checks rather than epochs.
See the [hourly training guide](docs/model_training/HOURLY_TRAINING.md) for settings,
timing semantics, migration, validation selection and restart behavior.
Compiled cross-variable attention is now enabled for training through
`training.compile_variate`, with about 17% higher measured update throughput.
See the [follow-up performance audit](docs/audits/2026-10-28/training_speed_round2/REPORT.md)
for benchmark limits, compilation warmup and verification.

Training uses physical microbatches of up to eight tables, bounded by 3,072
padded variable-tokens, with a 1,792-token activation budget. The
[November 8 throughput benchmark](docs/time_benchmark/TRAINING_THROUGHPUT_2026-11-08.md)
measured approximately 8% higher throughput on recorded training batches and
verified all 48 length/covariate-count buckets without OOM retries. Default logs
show epoch progress/ETA, memory and preparation/model-update timings.
Bounded training CUDA replay remains optional, but is
disabled by default after capture-inclusive mixed-data tests. See the
[training schedule benchmark](docs/time_benchmark/training_graph_schedule/REPORT.md)
for measured update times, memory use, numerical checks and replay settings.

## Repository and model outputs

`configs/config.json` selects the active model and its artifact paths.
`configs/legacy_447m.json` selects the archived 447M model. Pipeline stages write
to the following layout (data artifacts and model files appear when their stage runs):

```text
configs/
  config.json                     # Default model configuration
  legacy_447m.json                 # Archived 447M configuration
src/
  01_data_generation/
  02_data_validation/
  03_transforming_input_data/
  04_creating_embeddings/
  05_model_training/
  06_holdout_test/
  07_forecasting/
data/
  01_generated/YYYY-MM-DD/.../data.parquet
  02_processed/
    model/windows.pt              # Default model's processed windows
    447m_model/                   # Archived model's processed artifacts
  03_embedded/
    model/embeddings.pt            # Default model's lazy patch index
    447m_model/                   # Archived model's embedding index
results/
  model/
    training/                     # model.pt, resumable timed checkpoint and metrics
    holdout_test/                 # Out-of-sample metrics, predictions and plots
    forecasts/                    # Forecast-stage outputs
  447m_model/
    training/
    holdout_test/
    forecasts/
logs/                             # Timestamped logs and training JSONL journals
docs/                             # Architecture, benchmark reports and audit trail
```

The default exported model is `results/model/training/model.pt`; resumable training
state is `results/model/training/latest_checkpoint.pt`. Holdout evaluation reads
the exported model and writes to `results/model/holdout_test`. The general
forecasting stage writes to `results/model/forecasts`; when it delegates to the
holdout evaluator, it uses the holdout directory. Paths remain configurable.

The folder rename preserves existing binary data and checkpoint contents. It does
not convert the archived 447M weights into the default 395M architecture. The
[path migration audit](docs/audits/2026-10-31/model_folder_names/REPORT.md) records
the directory changes and text-reference updates.

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
[the original setup guide](docs/RWKV-7_setup/README_RWKV7_FAST_SETUP.md).
The initial installation and GPU checks passed; see
[the validation report](docs/RWKV-7_setup/RWKV7_ENVIRONMENT_VALIDATION.md).

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
python src/01_data_generation/generate_operational_data.py
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
instructions. `COMPOSITE_EFFECT_PROBABILITY=0.25` selects expressions of two or
more operations; otherwise the edge has one operation. Evaluation computes each
intermediate result once. Set `MAX_EFFECT_COMPONENTS=1` to disable composition.

`EFFECT_WEIGHTS` overrides relative operator priors: often=8, sometimes=2,
rare=0.15. Weights are normalized over enabled operators, not percentages per
tier; zero disables an operator. The defaults are in `DEFAULT_EFFECT_WEIGHTS`.
Common maps include affine, saturation, power, exponential/Arrhenius, Antoine,
relu/deadband/clip/knee/leaky-relu, log, quadratic pump/friction curves, positive
T⁴, products/ratios and bottleneck min/max/soft-min. Less common maps include
Hill/Michaelis–Menten/Langmuir, Gaussian/parabolic optima and queueing blow-up.
Inverse-square, static damped waves, tangent, SwiGLU and random neural maps are
rare. Sine/cosine/tangent are sampled only for angular parents: a 5% covariate
prior (`ANGLE_COVARIATE_PROBABILITY`) creates wrapped-radian covariates, recorded
with `domain="angle"`. Saved models without that field keep their original output.

Positive concentration/temperature inputs use a stable softplus mapping.
Antoine uses log-pressure form A−B/(C+T); queueing uses 1/(1−ρ) with bounded
utilization headroom. These names describe shape priors on standardized inputs,
not unit-calibrated physical models. Singularities and instruction outputs have
finite capacity bounds. New shape and temporal parameter ranges can be overridden
through `PRIOR_RANGES`, using the `effect_*` keys in `DEFAULT_PRIORS`.

Temporal operators have independent per-edge, per-instruction state: first-order
lag, transport delay, bounded leaky integration, hysteresis, asymmetric rate
limiting, cascaded Erlang/broader diffusion-like filters, damped resonance,
hysteretic regime switching, PID control and volatility coupling. They consume
only current/past input and retain state through warm-up. PID compares the input
against a sampled setpoint with an anti-windup integral; delayed feedback edges
can put it in a closed loop. Volatility coupling uses a causal variance estimate
without introducing a new random draw. Filter memory and internal transport
delay are additional to the explicit edge lag; internal states are omitted from
the compact DAG. Coefficients are sampled once and saved, while runtime state
is initialized afresh on each simulation.

`PARENT_COMBINATION_WEIGHTS` controls a separate node-level prior over weighted
sum, positive product/ratio, rare signed saddle, min/max/soft-min, gain gating,
exponent gating, thresholding, convex mixing, small multi-input MLPs, log-sums,
regime switching, volatility coupling and SwiGLU. These maps use **different
parents**. Participating indices and their roles are sampled independently of
edge ordering and saved. Ordinary interactions select a pair with probability
`PAIRWISE_INTERACTION_PROBABILITY=0.85` when more parents are available; otherwise
they select up to five. MLPs favor subsets of three to five when available, while
power laws involve all parents. The sum mode remains most frequent.

Products map inputs positive with softplus (usually) or exponentiation, then
multiply. They can dominate an additive contribution; `saddle` separately retains
the signed tanh-product as a rare prior. A joint term either replaces the selected
group's additive contribution or supplements the full sum, with
`ADDITIVE_INTERACTION_PROBABILITY=0.35` for product-like maps. Unselected parents
remain additive. Thresholds, gates, bottlenecks and convex mixtures replace the
selected group so a gated input cannot bypass the gate through a parallel sum.
Convex weights are sampled from a Dirichlet distribution and sum to one within
the selected group. MLP weights, biases, coefficients and all other joint shape
parameters are sampled once and stored in the SCM. Per-edge weights and nonlinear
effects remain active before the joint map.

`NODE_LINK_WEIGHTS` controls the final link over the parent aggregate: identity,
saturation, softplus, exponential, sigmoid or logarithm. Thus forcing can be
`g(sum(parent effects))`, while innovations, seasonality and intrinsic state
remain outside this link. A `log_sum` combination pairs with an exponential link
to implement `k * product(positive(parent effect_i)**power_i)`, including shapes
such as area times the square root of pressure. Disabling exponential links
also excludes log-sum combinations. Link choices and parameters are saved on
each node; identity is most common. Numerical overflow guards
(`JOINT_NUMERIC_LIMIT=1e6`) are separate from optional physical saturation links;
the newer joint maps have no mandatory tanh capacity bound.

The `joint_*` entries in `PRIOR_RANGES` control interaction strength, convex-mix
concentration, exponents, MLP width and weight scale. Existing `effect_*` and
`input_scale` priors also set map/link parameters. These are priors over normalized
relationships, not a claim of unit-consistent physical calibration.
Saved unversioned parent combinations retain the original signed-product and
tanh arithmetic. SCMs with no combination retain additive forcing and the old
optional two-parent interaction; models without `node_link` retain identity.
The new selected-parent schema is marked `parent_combination.version=2`.

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

`SWITCHING_PROBABILITY` adds a binary operating-state covariate, visible or
hidden, with sampled on/off dwell durations. Shutdown mode sets the complete
outcome exactly to zero or a sampled value during outages, then blends back to
the evolving underlying process over a sampled restart interval. Reset mode
resets an accumulated ramp at each state transition, creating repeating
charging/discharging or sawtooth-like shapes. These controls appear in the DAG.

`SWITCH_PROFILE_WEIGHTS` mixes ordinary alternating operation (weight 0.4),
long-dwell operation (0.5), and a single scheduled operating interval (0.1).
Long outages can span most or all of the recorded context; the initial state is
also sampled. These are duration ranges, rather than a fixed periodic schedule.

`PROCESS_EPISODE_PROBABILITY` independently selects a sustained process run.
New episodes use an explicit exogenous `formula` node in the SCM with a saved
process program, rather than a binary command followed by a fixed hump. Its
equation follows sampled setpoint commands through finite causal transitions.
The outcome's saved gating equation uses the delayed driver value; its measured
result enters outcome history before subsequent rows read their lags. Source
nodes can also influence other covariates through the usual DAG edges. No
finished observations are reshaped to create the process run.

Idle-before and idle-after durations have independent sampled weights, including
near-zero periods. Rise and release durations are drawn independently across a
wide range; either can be much faster than the other. Each transition separately
chooses linear, cosine, smoothstep, power, or exponential-release response.
The finite release can have a long tail and still reach idle exactly.
`PROCESS_MULTILEVEL_PROBABILITY=0.55` adds intermediate operating setpoints to
episodes, with independently sampled transitions and dwell periods; rapid new
commands can interrupt a ramp. `PROCESS_PRIMING_PROBABILITY=0.30` adds small,
saved priming pulses during pre-start idle when enough idle time is available.
All durations and levels are sampled before observations, not inferred from
the generated trajectory. Active variation still includes original parent
effects, oscillations, seasonality, and innovations; idle noise can be zero or
small. Screening uses the saved driver shape, without fitting a new schedule.
These options retain the runner's configured episode frequency and operate even
when `SWITCHING_PROBABILITY=0`. Existing saved binary-response SCMs replay using
their original equations.

`FORMULA_COVARIATE_PROBABILITY=0.08` selects formulaic exogenous sources for
eligible additional covariate slots after the first continuous process.
Families are ramps, powers, sigmoid transitions, Gaussian bumps, damped waves,
chirps (changing frequency), sawtooths, triangles, and state-machine programs.
Sampled amplitude, offset, timing, shape and optional noise law are saved in each
node. The usual `NOISE_FREE_COVARIATE_PROBABILITY` can disable private noise;
otherwise the existing noise families supply small innovations relative to the
formula amplitude. Formula sources can be hidden or visible under the Poisson
visibility sampler and can drive other nodes or the outcome. Their source
equations depend only on the clock and their own innovations, never future
observations. `PRIOR_RANGES` exposes the new `process_*` and `formula_*` ranges.

Run `python src/01_data_generation/preview_process_drivers.py` to create
`results/data_generation/process_drivers.png` (actual SCM outcomes),
`formula_sources.png` (formulaic source examples), and `process_drivers.json`
(the corresponding full SCMs). Tests in `test_process_drivers.py` cover
asymmetry, multiple settings, idle boundaries, optional noise, causal prefixes,
graph connections, visibility invariance and exact replay.

`DISCRETE_COVARIATE_PROBABILITY=0.25` replaces eligible additional source slots
after the first continuous covariate with discrete exogenous inputs.
`DISCRETE_WEIGHTS` selects independent Bernoulli indicators (0.35), persistent
binary Markov indicators (0.35), or multi-state Markov sources (0.30). Markov
sources have sampled levels, transition matrices, initial states, and dwell-time
scales. They emit exact discrete values without continuous noise, can influence
later covariates or the outcome, and follow the usual visibility sampling.

`SENSOR_BOUNDS_PROBABILITY=0.01` independently adds a hard floor, ceiling, or both
to each continuous covariate and the outcome. Bounds are sampled in the recorded
units from SCM scale parameters before simulation, never fitted to observed
quantiles. Capped measurements feed downstream equations and future lags. A
bound may remain inactive if the process never reaches it. Discrete sources
retain their exact categories.

`VISIBLE_COVARIATE_LAMBDA=1.3` draws the number of visible covariates from a
Poisson distribution, then chooses that many source nodes uniformly without
replacement. Counts above the number of available sources are capped, rather
than redrawn. All other covariates remain latent in the SCM. Zero visible
covariates produces a strictly univariate table containing only `time_index`
and `outcome`, while retaining the full range of latent causal mechanisms.
Before capping, probabilities for 0/1/2/3 visible covariates are approximately
27.3% / 35.4% / 23.0% / 10.0%; larger counts become progressively rarer.
There is no per-source hiding probability or enforced univariate batch quota.
Finite batches fluctuate naturally. Set lambda to zero for entirely univariate
generation. For controlled comparisons, `sample_model(visible_covariate_count=N)`
overrides the count for one model. The saved SCM records the count sampling
parameters and each node's visibility; replay uses the saved flags unchanged.

These probabilities and weights are editable at the top of
`src/01_data_generation/generate_operational_data.py`; duration, amplitude,
transition, and bound ranges can be overridden through `PRIOR_RANGES`. Screening may
change the proportions in accepted tables. The saved SCM contains every new
mechanism, and existing saved SCMs retain their previous replay behavior.
For a compact illustrated gallery, run
`python src/01_data_generation/preview_process_mechanisms.py`. It writes
`results/data_generation/process_mechanisms.png` and the underlying example SCMs.

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

Parquet columns are `time_index`, `outcome`, and visible current
`covariate_<id>` columns. Time is a regular abstract sample index; seasonal
periods are measured in samples. Covariates use normalized units. Outcome
units use a sampled baseline and scale and can be signed. Outcome and covariate
lag dependencies remain inside the SCM, with actual historical values read
from internal buffers. Lag 1 is always an internal outcome dependency; no lag
predictor columns are exported. A discarded warm-up provides history for the first rows.
Covariates are hidden only from the exported table and plot, not from the
simulation. The JSON includes their equations, parameters, and visibility.

For replay, load the JSON and call `generator.simulate(model)`. The fixed seed
reproduces models and observations in the same numerical environment; filenames
are intentionally unique. Version-1 JSON files remain replayable through the
frozen `legacy_scm_generator.py`; version-2 simulation remains supported and
new models use schema version 3. Replaying older models through the current
generator returns the current training schema without redundant lag columns.
Softly bounded edge functions and stable combined autoregression limit runaway feedback while
allowing nonstationary levels. These priors cover a much broader range of
behaviors, but no finite generator guarantees every real operational process.
Domain calibration and checks against real data remain necessary. Short samples
may not contain a shock or a complete seasonal cycle.

Run the generator checks with `python -m unittest discover -s tests -p test_scm_generator.py`
and `python -m unittest discover -s tests -p test_effect_mechanisms.py`.

Generation pre-resolves parent buffers and lag aliases and prepares fixed joint
map arrays once per simulation. Scalar clipping avoids repeated NumPy array
dispatch. These changes retain the arithmetic and RNG draw order.

The runner now sets `GENERATION_WORKERS=None`, automatically using up to eight
processes (half the available logical CPUs). Each process samples, screens,
simulates, and writes complete tables, including any enabled plots. This speeds
up runs even when plots and SCM JSON exports are disabled. `GENERATION_BATCH_SIZE=16`
amortizes task dispatch, with at most twice the worker count of queued batches;
frames and models stay in workers. Arrow threads are limited per worker to
avoid multiplying its thread pools. Tables are returned in their original order.

Parallel generation uses independent per-table RNG seeds and the same equation
sampling, simulation arithmetic, trend quotas and rejection criteria. A fixed
seed produces the same data and SCMs across parallel worker counts and batch
sizes, although its table sequence differs from the older sequential RNG stream.
Set `GENERATION_WORKERS=1` to preserve that older stream exactly. The generator
class defaults to this compatible sequential mode; the runner enables automatic
parallelism. Seeded mode falls back to in-process work when spawning is unavailable
or there is only one batch. Keep the main guard for custom Windows entry points.

Console progress reports completed tables, percentage, elapsed time, tables per
second, and ETA, initially shown as `estimating`. `PROGRESS_INTERVAL=1.0` limits
console updates to once per second, avoiding millions of print operations.
Throughput and ETA include startup, screening, writes and requested image
rendering; batches count only after their files are complete. Estimates become
more stable after several batches and are approximate for variable-length tables.

`COARSE_COPY_PROBABILITY=0.03` gives each accepted original with **more than 512
rows** a 3% chance of one extra, less granular table. A window size is sampled
uniformly from 2, 3, 4, 5, and one zero-based offset is sampled inside that
window. The same offset is reused across every non-overlapping window and every
column, including all visible covariates. This selects observations without
averaging or filtering. Incomplete trailing windows are dropped, and the copy's
`time_index` starts again at zero with unit steps.

The original remains unchanged. The copy is a sibling folder named
`<original>_coarse_<window>_<offset>`, containing `data.parquet` and an always-saved
`provenance.json` linking the source table and selection parameters. If enabled,
SCM JSON and plots are saved too. The saved SCM retains the original equations
and records the observation-selection step; replay runs at the original sample
rate before thinning, and the DAG labels its lags in original sample intervals.
The copy records its own linear trend score without claiming to have passed the
original trend-profile screen. Copies are not recursively augmented.

Selection uses a separate deterministic RNG, so enabling this feature changes
neither original tables nor subsequent SCM sampling. `NUM_TABLES` still counts
original SCMs; returned paths and the runner's final saved-table count include
extras. Console progress reports `+N coarse copies` separately. Set the probability
to zero to disable augmentation. Run
`python -m unittest discover -s tests -p test_coarse_copies.py` for alignment,
replay, provenance, RNG isolation and multiprocessing checks.

`SIGN_INVERSION_PROBABILITY=0.03` independently gives each original table a 3%
chance of a mirrored sibling named `<original>_sign_inverted`. Every recorded
value in the outcome and visible covariates is multiplied by -1, including
binary/discrete covariates; `time_index`, row order, and row count are unchanged.
The original realization and its latent equations are retained. An observation
transform in the optional saved SCM makes replay reproduce the mirror exactly;
the optional DAG labels itself as showing source equations before inversion.
An always-saved `provenance.json` links back to the original even without SCM
export. Optional series plots show the reflected data.

Mirroring uses a separate deterministic RNG from both SCM sampling and coarse
copy selection. It can coexist with a coarse copy of the same original, but
neither augmentation is recursively applied to copies. `NUM_TABLES` still means
original SCMs, and the console counts sign-inverted copies separately from
coarse copies. Set the probability to zero to disable mirroring. Centering and
scaling do not generally erase a sign inversion: for mean/std normalization,
the normalized mirrored series is the negative of the normalized original.
Run `python -m unittest discover -s tests -p test_sign_inversion.py` to verify
value/clock alignment, RNG isolation, replay, exports and parallel determinism.

In sequential generation, plot rendering is independent of simulation:
`RENDER_WORKERS=None` automatically uses up to four
processes for batches of at least four tables. Set it to `1` for serial rendering,
or an explicit positive count. Interactive sessions without an importable main
file fall back to serial rendering. Queued work is bounded, worker errors are
propagated, and `generate()` waits for all images before returning. Keep the main
guard when launching generation from a custom script on Windows. Full generation
workers render their own plots and do not create nested rendering pools.

Run `python src/01_data_generation/benchmark_generation.py` to benchmark serial
and parallel rendering with both plots enabled and verify exact data/model/PNG
equality. Its top-level settings control repetitions and batch size. Optionally
point `REFERENCE_SOURCE` at an archived generator source directory to compare
the prior implementation, allowing only removal of exported lag columns and
the corresponding initialization-description text. The report is saved to
`results/data_generation/generation_benchmark.json`; timings depend on hardware
and workload. SCM sampling, rejection criteria, and simulated values remain unchanged.

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

`NOISE_FREE_COVARIATE_PROBABILITY=0.10` in
`src/01_data_generation/generate_operational_data.py` independently selects
process covariates with no intrinsic noise. Selected nodes use zero innovations,
no random walks or slope walks, and no auxiliary random event/shock mechanism.
Their sampled deterministic seasonal/trend equations remain active. Explicit
causal parents remain connected and can transmit stochastic variation; a node
with no parents is a deterministic function of time. Binary event/switch sources
and outcome noise are outside this probability. The saved SCM marks selected
nodes with `noise_free: true` and stores their zero-noise equations for replay.
The 10% is a sampling prior; batch quality screening can shift final frequencies.

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

These finite expansions are Fourier **partial sums**. A square wave, for
example, uses `(4/pi) * sum(sin(k*phase)/k)` over odd orders `k=1,3,5,...`.
Six nonzero terms therefore use orders 1, 3, 5, 7, 9 and 11. The evaluator
already supports these exact coefficients. The sampled `pulse` family varies
the duty cycle, smoothing and coefficients, producing a wider range of
square-like operational cycles rather than only textbook 50% duty waves.
Increasing the harmonic count sharpens transitions subject to the sampling
limit; finite sums can overshoot near discontinuities (Gibbs ringing).

Run `python src/01_data_generation/preview_fourier_partial_sums.py` to compare
one, two, three and six terms, their component waves, and the amplitude spectrum.
It writes `results/01_data_generation/fourier_partial_sums.png` and the saved
seasonal equations in `fourier_partial_sums.json`. This demonstration uses
textbook amplitudes and does not change the generator's sampling distribution.

Run `python src/01_data_generation/preview_fourier_seasons.py` for a gallery of
isolated seasonal shapes and their saved specifications under
`results/01_data_generation/fourier_seasons.png` and `fourier_seasons.json`.
This broadens the seasonal prior; it does not establish coverage of every
real-world operational process, which still requires validation against data
from the domains of interest.

## Data admission screening

`configs/config.json` currently sets **`validation.enabled=false`** to skip
predictive screening for large cohorts. Transform admits all selected tables,
while retaining its normal data and sequence-length checks. Running the validator
in this mode exits successfully without forecasting or scanning tables.
Set `validation.enabled=true` to restore screening. `validation.auto_run=false`
only disables automatic screening runs; it does not bypass the admission gate.

The default Stage 02 filter is **Chronos-2 Small with covariates versus naive**.
A table passes when its median forecast MAE is no more than **15% worse** than
last-observation carry-forward (`validation.max_relative_degradation=0.15`).
All eligible historical covariates selected by the training configuration are
included, with future values supplied only for explicitly known covariates.
The transformer automatically runs screening when reports are missing or stale,
including previous Bolt/Prophet reports. The pinned 28M model is loaded once and tables are batched;
its cache stays under `.cache/huggingface` on the project drive.
See [Chronos-2 screening](docs/data_validation/CHRONOS2_SCREENING.md) for
the temporal split, configuration, admission rule and outputs.

### Legacy Prophet validation

The following documents the retained compatibility mode
`validation.engine="prophet"`; it is not the default admission filter.

Install the validation dependencies with `python -m pip install -e ".[validation]"`.
Configure `validation` and `data.generated_dates` in `configs/config.json`, then run:

```powershell
python src/02_data_validation/validate_generated_data.py --config configs/config.json
```

Each table's rows are sorted by `time_index`. The final `data.horizon` rows are
reserved for downstream testing and excluded from screening. Of the remaining
history, the first floor(`validation.train_fraction` times the row count) rows
train the models and the rest form one fixed forecast horizon. The baseline
predicts the last training outcome throughout that horizon. One multivariate
Prophet target model uses visible `covariate_*` columns; there is no separate
univariate target test. Constant training covariates and lag columns are omitted.
Lag columns can expose held-out targets during a multi-step forecast. No SCM
metadata is used. With no usable covariates, the single target fit uses only history.

`validation.covariate_mode="known_future"` supplies actual held-out covariates: this is a
conditional forecast benchmark, assuming those inputs are known in advance.
They may contain information about test outcomes through the generating SCM;
these scores should not be interpreted as performance with unknown future inputs.
Set the mode to `"forecast"` to extrapolate each covariate using a separate
Prophet fit to its training values only. Prophet requires future regressor
values, as described in its [regressor documentation](https://facebook.github.io/prophet/docs/seasonality%2C_holiday_effects%2C_and_regressors.html#additional-regressors).
The default mode is `"forecast"`; with K usable covariates it needs K covariate
fits plus one target fit. Synthetic steps map to daily timestamps by default
(`validation.step_duration="1D"`). The default config disables daily, weekly,
and yearly seasonality and uncertainty simulation. Holidays are inactive unless
explicitly supplied to Prophet. Optional `validation.seasonal_periods` are
specified in timesteps and used only when training contains two cycles.
Settings are shared across target and covariate models, without test-set tuning.

`validation.workers=null` automatically uses up to four processes, capped by
half the logical CPU count and the number of tables. Set a positive integer to
choose a worker count, or `1` for serial execution. Each worker processes one
table at a time, with native math/Stan thread counts set to one to avoid CPU
oversubscription. Workers use Windows-compatible process spawning. When calling
the batch API from your own script, protect the call with `if __name__ == "__main__":`.

With the default `validation.run_folders=true`, Stage 02 uses the parent of
`validation.metrics_path` as its results root and creates a unique folder per run:

```text
results/02_data_validation/
  2026-10-25_<source-selection-id>/
    latest.json
    <UTC-timestamp>_<run-id>/
      metrics.parquet
      metrics_progress.csv
      summary.json
      settings.json
      status.json
```

The cohort folder includes all selected dataset dates (or a compact range for
long lists) and an identifier for the source selection. `settings.json` preserves
the full configuration, exact `dataset_dates`, execution date, and start time;
`summary.json` also records completion time. Repeating the same date selection
creates another timestamped run, and changing dates creates a separate cohort.
Earlier metrics are not overwritten. Transformation automatically uses that
cohort's latest finished report, published through `latest.json` only after the
metrics and summary have been written. Failed/interrupted new runs leave the
previous pointer unchanged. The log prints the run folder before fitting.
`status.json` updates progress every 100 completed tables and at completion.

Only successful, finite multivariate **MAE** scores no greater than the
baseline pass the downstream transformation gate; ties, including perfect-zero
ties, pass. Failed fits are rejected. RMSE is retained for comparison but does not
control admission. RMSE-only reports must be regenerated; their scores cannot be
reused as MAE. Changing only workers, output paths, or folder layout does not
invalidate screening scores.
Each table has one row:

| Column | Meaning |
| --- | --- |
| `baseline_mae`, `multivariate_mae` | Screening MAE in outcome units; controls admission |
| `baseline_rmse`, `multivariate_rmse` | Secondary RMSE diagnostics |
| `admission_metric` | `mae` |
| `multivariate_beats_baseline` | 1 if strictly lower MAE, otherwise 0 |
| `multivariate_not_worse_than_baseline` | 1 if MAE is less than or equal to the baseline, otherwise 0 |
| `multivariate_improvement_pct` | 100 × (baseline MAE − model MAE) / baseline MAE |

Positive improvements mean better predictions; negative values mean worse.
A zero reference MAE makes percentage improvement undefined (blank/NaN).
Errors are recorded per table with missing metrics and win flags, rather
than treating failures as losses or successes. Rows also record split sizes,
covariate names/count, source path/hash, reserved tail, and covariate mode.
`metrics_progress.csv` checkpoints each completion by appending one row; final
Parquet rows retain source order regardless of completion order. This avoids
rewriting all previous results after every table in a large cohort. The standalone
`ProphetValidator.run()` API creates a unique run folder with `metrics.csv`,
`metrics.parquet`, and `settings.json` and uses the same concurrent evaluator.

Run validation checks with
`python -m unittest discover -s tests -p test_prophet_validator.py`.

Prophet screening places temporary Stan files under `.cache/prophet` on the
project drive, rather than the Windows system temp directory. Each fit's output
is deleted immediately after forecasting, so long-lived workers do not accumulate
CSV/stdout files across hundreds of thousands of models. Disk-full errors abort
the batch instead of silently turning all remaining tables into failed scores.

To recover a run with failed tables while keeping unchanged successful scores:

```powershell
uv run python src/02_data_validation/validate_generated_data.py --retry-errors
```

Reuse requires matching source content hashes and MAE screening settings. Failed,
new, or changed tables are evaluated again. Retry reads the cohort's latest report
and writes another independent run folder, retaining its predecessor. Explicit
legacy `run_folders=false` mode uses `metrics_path` directly and archives the old
metrics, summary, and progress CSV under `archive/<timestamp>/` before replacing
them. Summaries distinguish `successful` and `failed`
tables; `rejected` continues to include all tables that were not admitted.

On this Windows installation, `scripts/relocate_pycharm_storage.ps1` copies
PyCharm 2025.3 caches and Local History to `E:/JetBrains/PyCharm2025.3/system`,
sets logs and Java temporary files to E:, and preserves the original C: storage.
It must operate with PyCharm closed; `-WaitForExit` waits for that automatically.
Wait for `E:/JetBrains/PyCharm2025.3/migration-status.json` to show `complete`
before reopening PyCharm. Generated data, results, logs, and `.cache` are excluded
from this project's IDE indexing. This IDE relocation is machine-specific and
does not change global Windows TEMP/TMP settings.

## Backend constraints

Use contiguous bf16 activations, sequence lengths divisible by 16, equal model
and attention widths divisible by 64, FFN widths divisible by 16, and at least
two layers. Compute regression losses in fp32 and preserve training
initialization. These sequence kernels do not expose streaming recurrent state.

The vendored RWKV source retains its upstream Apache-2.0
[license](third_party/rwkv-lm/LICENSE). See
[provenance](docs/RWKV-7_setup/RWKV7_PROVENANCE.md) for the upstream revision and local patches.
# Trainable patch-based forecaster

The RWKV-7 pipeline lives in the numbered `src/` stage folders and uses
`configs/config.json`. Dependencies, including Torch and Loguru, are managed by `uv`.
See [the architecture, training commands and forecasting example](docs/forecasting/FORECASTING.md).
Choose one or more generated-data date folders with data.generated_dates in
configs/config.json; paths.raw points at data/01_generated. See the
[date-selection configuration](docs/audits/2026-10-25/generated_dates/REPORT.md).
Run the stages separately with `uv run python`:

```powershell
uv sync --extra data --extra validation
uv run python src/02_data_validation/validate_generated_data.py
uv run python src/03_transforming_input_data/transform.py
uv run python src/04_creating_embeddings/embed.py
uv run python src/05_model_training/train.py
uv run python src/06_holdout_test/holdout_test.py
```

See the [transformation guide](docs/transformation/README.md) for input selection,
causal training records, evaluation policies, outputs and checkpoint recovery.
The [embedding preparation guide](docs/embeddings/README.md) explains bounded
inspection batches, progress bars and resumable metadata indexing in `embed.py`.

Next-patch transformation resumes automatically when the same command is rerun.
The console shows a progress bar, completed/total tables, percentage, tables per
second, elapsed time and ETA. It updates every 100 tables, at least every five
seconds when a table finishes, and at completion. Source-inventory checking and
checkpoint restoration have separate bars; progress counts tables rather than rows.

The checkpoint cadence is configured independently of progress logging:

```json
"transform": {
  "batch_size_tables": 1000,
  "batch_memory_limit_mb": 64,
  "checkpoint_every_tables": 100000,
  "checkpoint_interval_seconds": null
}
```

By default, completed tables are checkpointed every 100,000 tables and after the
final table, with no time trigger. Set either option to `null` to disable that
trigger, or set a positive table count/time in seconds. When both are enabled,
whichever comes first triggers a save (checked between tables). Changing these
intervals does not change the dataset signature. A forced exit can lose
only work since the last saved checkpoint; an unfinished table is processed again.
Checkpoints live under
`data/02_processed/model/.transform_checkpoints/<identity>/` with an inspectable
`progress.json` and checksummed tensor batches. Restarting checks source names,
sizes and modification times and verifies saved shards without loading their tensors.
Changing the source inventory, data/model configuration or transform implementation
uses a separate checkpoint directory. Keep source tables unchanged during a run.

Transformation spills buffers after 1,000 tables or approximately 64 MiB of tensor
data, independently of checkpoint frequency. The final `windows.pt` is a small
shard index, published atomically without merging the corpus into RAM. Embed and
training read shards on demand through the artifact loader; older single-file
artifacts remain supported. The referenced `batch_*.pt` files in the checkpoint
directory are now the dataset itself and **must be kept** after completion.
The largest source table is still read in full, and source/record metadata still
uses memory. This resume mechanism covers next-patch transformation;
optional Chronos/Prophet screening remains a separate upstream phase.

Training ends after saving the model; holdout testing is a separate stage.
General forecasting lives under `src/07_forecasting/forecast.py`. GPU stages
launched on Windows automatically use the configured WSL CUDA environment.
The outputs include logs, training loss, forecast RMSE and Prophet/last-observation
carry-forward comparison plots. Holdout evaluation uses up to five origins per
table, configured by `holdout.percentiles`, with table-level progress and ETA.
Holdout comparisons are interactive Plotly HTML files with zoom, hover, a range
slider and model toggles. Regenerate them from existing saved predictions with
`uv run python src/06_holdout_test/plotting.py`; no model inference is repeated.
Results go to `results/model/holdout_test` by default
(`paths.holdout_test`). The holdout command automatically loads the latest
exported `paths.training/model.pt` and its saved test split.
Choose comparison models with `holdout.benchmark_models` (default
`["prophet", "naive", "chronos"]`; optional `"timestral"`) and evaluate a
reproducible portion of the test tables with `holdout.sample_percent` (default
100, e.g. 1 for a quick 1% run). The percentage applies to the admitted saved
holdout split. See the [foundation benchmark audit](docs/audits/2026-10-29/foundation_benchmarks/REPORT.md).
Fresh training now records full epoch-zero validation before the first update;
resumed runs retain their recorded baseline, if available. See the
[sampled evaluation and inference audit](docs/audits/2026-10-29/sampled_evaluation/REPORT.md).

### Pre-training token inventory

After embedding, inspect the amount of distinct input without starting training
or allocating the GPU model:

```powershell
uv run python src/05_model_training/token_inventory.py --config configs/config.json
```

Exact counting is an optional audit and can be expensive on large corpora.
Normal training uses `training.token_inventory.mode="cached"`: reuse a report
matching the processed artifact and dataset partition, otherwise proceed without
scanning. `token_inventory_status.json` records availability; missing unique
counts are not estimated or fabricated. The standalone command above generates
`token_inventory.txt`, `token_inventory.csv`, and `token_inventory.json` under
`paths.training` for reuse. Set mode to `exact` to explicitly require a fresh
audit before model initialization, or `off` to skip it entirely. See the
[startup performance audit](docs/audits/2026-10-27/training_startup/REPORT.md).
The text report shows training, validation, and out-of-sample test counts for:

- Input token occurrences in one pass, including repeated context use.
- Unique patch contents, deduplicated across windows and tables within each split.
- Unique source/variable/row positions, distinguishing repeated values from reused windows.
- Unique prediction-target contents, counted separately from model inputs.
- Unique-input and unique-target counts per model parameter (CSV/JSON include both ratios).

A token is one patch of one variable, not one embedding dimension. The report
reads the patch size and width from the configuration; it does not assume a
900-dimensional embedding. Content identity uses raw float32 patch values,
observation masks, and variable roles. It is independent of normalization,
position features, and learned embedding weights, which can change during
training. These counts describe distinct observed patches, rather than equality
of continuous embedding vectors. Kernel padding and entirely unobserved patches
are excluded; partly observed patches retain their masks.

Counts use the normal trainer's actual split: full training windows on training
tables, five-origin validation on reserved validation tables, and rolling forecasts
on separate test tables. Held-out-table training windows are excluded. Test counts
cover the selected holdout origins, capped contexts, and the final partial target patch,
rather than only the final window saved by embedding. Repeated rolling context
use is counted analytically, without running forecasts. JSON also records split
overlaps and validation/test content not seen in training. Source-table overlap
between splits must be zero; identical numeric content can occur in distinct tables.

The CPU-only parameter count follows the current vendored RWKV-7 architecture.
The trainer updates the ratios with its measured parameter count after model
initialization. A temporary SQLite index keeps deduplication memory bounded;
the patch index maps raw tensor storage and prepares features per accessed window.
The learned embedding projection runs inside training, without corpus-sized
embedding snapshots. See the [memory investigation](docs/audits/2026-10-27/embedding_memory/REPORT.md).
No optional packages
are required. This diagnostic currently supports the `next_patch` objective.

Exact uniqueness does not establish statistical independence or sufficient data:
near-identical or noisy patches can all have different contents. The report
provides descriptive ratios without imposing a sufficiency threshold.

The default configuration now runs a 394.9M-parameter causal next-patch model
with 16-point patches, width 1,024, and a 2,048-point history cap. See the earlier
[301.8M-parameter, 100-table experiment and baseline comparison](docs/audits/2026-10-24/next_patch_300m/REPORT.md).

The earlier admission policy used pre-test multivariate Prophet MAE
compared with last-observation MAE. The earlier strict univariate-or-multivariate
policy is documented in the historical
[learnability gate and embedding audit](docs/audits/2026-10-24/learnability_gate/REPORT.md).

Every regular and capacity training run automatically writes a timestamped JSONL
journal under `logs/`, with configuration, metrics, memory and completion/failure
events. See the [JSONL format and verification audit](docs/audits/2026-10-25/training_jsonl/REPORT.md).

Training now bounds GPU execution separately from the optimizer batch. Defaults
in `configs/config.json` retain `batch_size=8` and
`gradient_accumulation_steps=1`, but split each update into physical microbatches
of at most eight tables (`microbatch_size=8`) and 1,536 padded variable-patches
(`microbatch_token_limit=1536`). For example, eight tables with eight variables
and 128 patches each run as eight single-table forward/backward passes, followed
by one optimizer update. Losses are weighted by observed target points. Physical
microbatching preserves context, padding, validation coverage and the chosen
batch schedule.

`cuda_memory_limit_gib=8.0` caps PyTorch's CUDA allocator, including its cache,
at the smaller of 8 GiB and 80% of device capacity. CUDA libraries and desktop
applications use additional memory outside this budget. A shape that cannot fit
raises an allocation error rather than letting PyTorch's reservation grow without
bound. Each execution limit accepts `null` to disable that limit; a token budget
always permits at least one table and does not truncate its sequence. These
settings apply when the next training worker starts. See the
[VRAM investigation and benchmarks](docs/audits/2026-10-28/vram_budget/REPORT.md).

Training replay is available with `training_cuda_graphs=true`, but disabled by
default: the measured larger-microbatch execution was faster without replay.
Inference graph replay remains enabled. The training replay implementation uses
a schedule intended to amortize capture costs.
`training.shape_run_batches=128` shuffles runs of up to 128 batches from the
same variable-role/length bucket. Tables are shuffled within buckets each epoch;
observations inside each table remain chronological. Batch membership and epoch
coverage are preserved, but optimizer update order changes. The setting is part
of checkpoint identity: continue an older checkpoint with `shape_run_batches=1`,
or use `resume=false` to start training with the new order.

`training_graph_min_reuse=8` requires eight consecutive optimizer updates using
a shape before capturing it; accumulation microbatches count only once per
update. `training_graph_max_tokens=32` limits capture to short contexts (up to
512 input observations at patch size 16). Existing cached profiles replay
immediately; other shapes use compiled execution. Set the limit to `null` to
experiment with all lengths. The two-profile/1.5 GiB graph cache and 8 GiB CUDA
allocator limits still apply. See the
[capture-inclusive schedule benchmark](docs/time_benchmark/training_graph_schedule/REPORT.md)
for total-update timings, correctness checks and limits of the measurement.

## Reproducing the pipeline

The repository contains pipeline code, configurations, dependency locks, tests, experiment scripts and narrative audit reports. Generated datasets, model weights, numeric audit artifacts, caches and run outputs remain local and are not committed. Follow the stage commands above to regenerate these artifacts. Historical reports may link to local evidence files intentionally excluded from Git.

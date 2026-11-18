# Basin technical overview

Basin is a time-series foundation model for operational data, built on RWKV-7 and
trained on synthetic tables only. The repository includes a fused CUDA training
backbone, synthetic data generation, and a patch-based multivariate forecasting
architecture with an end-to-end training smoke test. Training supports sharded
corpora and resumable timed checkpoints. See [the forecasting guide](forecasting/FORECASTING.md).

The active configuration is a hybrid model with 221,507,808 parameters: 24 RWKV-7
blocks of width 768, with causal temporal attention and cross-variable attention
after every fourth block. One direct head forecasts 7 patches (112 timesteps) per
backbone pass, each timestep with equal weight in the loss; longer requests roll
forward in blocks of 112 points. Inputs and targets are causally standardised and
passed through `asinh`. Training uses optimizer parameter groups, a warm-up/cosine
schedule over a fixed epoch budget, and an exponential moving average of the
weights, which is what validation selects and the export contains. It has not yet
been trained at scale. See the
[220M architecture and recipe](model_training/HYBRID_220M_2026-11-15.md) for
the reasons, measurements and limits, and the
[inference benchmark](time_benchmark/fast_inference/REPORT.md).

The previous model (28 layers, width 1024, 500,056,352 parameters, four patches
with 80/16/3/1 lead weights) is archived under `results/500m_model/` and still
loads with `configs/legacy_500m.json`. Its
[short-horizon training guide](model_training/SHORT_HORIZON_TRAINING.md)
describes that protocol.

The default trainer uses full training tables, with no reserved validation tail.
Every hour of active training it saves a resumable checkpoint; every five hours it validates
25,000 fixed reserved validation tables at five origins each (125,000 forecasts),
using Basin only. Training, validation and test tables are disjoint: no window
from a validation table enters gradient training. Originals with coarse or
sign-inverted copies and their copies are training-only. Every validation and
holdout forecast spans the model's full 7-token block. Best weights are chosen
on the 10th, 50th and 90th percentile of table MASE, which a handful of
extreme forecasts cannot move. Missing values are forecast but never scored.
Checkpoints retain optimizer moments/settings, scheduler and RNG states, plus
the exact shuffled batch cursor for continuation within an epoch.
For large sharded corpora, `training.storage_locality=true` visits each shard
in one shuffled section per epoch and shuffles tables within compatible shape
buckets. Compact record handles and reusable shape plans reduce startup memory
and repeated data reads. See the
[million-table pipeline review](audits/2026-11-08/pipeline_scaling.md).
`training.max_run_hours` sets a per-invocation soft duration limit (`null` means
unlimited); the trainer saves at a safe optimizer boundary before exiting.
Early-stopping and scheduler patience use validation checks rather than epochs.
See the [hourly training guide](model_training/HOURLY_TRAINING.md) for settings,
timing semantics, migration, validation selection and restart behavior.
Compiled cross-variable attention is now enabled for training through
`training.compile_variate`, with about 17% higher measured update throughput.
See the [follow-up performance audit](audits/2026-10-28/training_speed_round2/REPORT.md)
for benchmark limits, compilation warmup and verification.

Training uses physical microbatches of up to eight tables, bounded by 3,072
padded variable-tokens, with a 3,072-token activation budget (1,792 for the
archived 500M model). The
[November 8 throughput benchmark](time_benchmark/TRAINING_THROUGHPUT_2026-11-08.md)
measured approximately 8% higher throughput on recorded training batches and
verified all 48 length/covariate-count buckets without OOM retries. Default logs
show epoch progress/ETA, memory and preparation/model-update timings, once per
1,000 tables (`training.log_every_tables`).

Since 2026-11-17 the FP32-master AdamW update runs as one CUDA kernel and
gradient clipping skips work that changes nothing. Both give the same weights
bit for bit. On the corpus's own mix of batch shapes an update takes 109.9 ms
instead of 143.7 ms (72.8 instead of 55.7 tables per second) with unchanged peak
VRAM. Checkpoints are written by a background thread while training continues,
and `training.checkpoint_upload` copies each one to an Azure Blob Storage
container while it is written; one checkpoint is kept in each place and replaced
only by a complete new one. The fixed validation panel is saved after the first
check (`training.validation_cache`), so later checks no longer open 25,000
source tables. See the [training speed report](model_training/TRAINING_SPEED_2026-11-17.md)
for measurements, tests and what was tried and rejected.

Bounded training CUDA replay remains optional, but is
disabled by default after capture-inclusive mixed-data tests. See the
[training schedule benchmark](time_benchmark/training_graph_schedule/REPORT.md)
for measured update times, memory use, numerical checks and replay settings.

## Repository and model outputs

`configs/config.json` selects the active model and its artifact paths.
`configs/legacy_500m.json` and `configs/legacy_447m.json` select the archived
500M and 447M models. Pipeline stages write
to the following layout (data artifacts and model files appear when their stage runs):

```text
configs/
  config.json                     # Default model configuration (220M hybrid)
  legacy_500m.json                 # Archived 500M configuration
  legacy_447m.json                 # Archived 447M configuration
  smoke_v2.json                    # Small end-to-end check (scripts/smoke_v2.sh)
src/
  01_data_generation/
  02_data_validation/
  03_transforming_input_data/
  04_patch_indexing/
  05_model_training/
  06_holdout_test/
  07_forecasting/
  08_model_export/               # Portable (pure PyTorch / FLA) model exports
data/
  01_generated/YYYY-MM-DD/.../data.parquet
  02_processed/
    model/windows.pt              # Default model's processed windows
    447m_model/                   # Archived model's processed artifacts
  03_patch_index/
    model/patch_index.pt          # Default model's lazy patch index
    447m_model/                   # Archived model's patch index
results/
  model/
    training/                     # model.pt, resumable timed checkpoint and metrics
    holdout_test/                 # Out-of-sample metrics, predictions and plots
    forecasts/                    # Forecast-stage outputs
  500m_model/                     # Archived 500M model, see ARCHIVE_MANIFEST.json
    training/
    holdout_test/
    portable/
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
not convert archived weights into the current default architecture. The
[path migration audit](audits/2026-10-31/model_folder_names/REPORT.md) records
the directory changes and text-reference updates.

## Portable model exports

The trained model runs on custom CUDA kernels. For sharing (for example on the
Hugging Face Hub) the evaluated checkpoint can be converted into two
self-contained folders under `results/model/portable/`. The ports implement the
archived 500M architecture (its exports are in `results/500m_model/portable/`);
they do not yet cover the hybrid 220M model and refuse its checkpoints:

| Folder | Needs | Runs on |
|---|---|---|
| `basin-rwkv7-pytorch` | `torch`, `safetensors` | CPU or GPU, any OS, no compiler |
| `basin-rwkv7-fla` | `flash-linear-attention` (Triton) | NVIDIA GPU on Linux/WSL, gcc or clang |

```bash
python src/08_model_export/export_models.py           # write both folders (CPU only)
python experiments/portable_models/run_pytorch.py     # demo forecast, plot and error metrics
python experiments/portable_models/run_fla.py         # same with FLA (relaunches in WSL)
python -m unittest discover -s tests -p test_model_export.py
```

The demo scripts forecast the last 20% of `data/00_public/airline-passengers.csv`
from the first 80%; `--data`, `--column`, `--device` and `--dtype` change that.
In Python:

```python
import sys

sys.path.insert(0, "results/model/portable/basin-rwkv7-pytorch")
from modeling_basin_pytorch import BasinPyTorch

model = BasinPyTorch.from_pretrained(
    "results/model/portable/basin-rwkv7-pytorch", device="cpu"
)
forecast = model.forecast(
    context, horizon=64, past_covariates=covariates
)  # [target, time] tensors
forecast.point, forecast.quantiles
```

Both variants keep the trained mixed precision (bfloat16 with float32 inside the
recurrence, gates and norms) and hold the exact weights the holdout evaluated.
Their forecasts differ from the original by about 0.4% of a series' standard
deviation, the same as the original's own two code paths differ from each other.
Median latency per forecast on the RTX 5070: original 7.9 ms, FLA 43.6 ms, pure
PyTorch 61.9 ms (213 ms on CPU). Method, evidence and limits are in the
[portable models audit](audits/2026-11-13/portable_models/REPORT.md).

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
[the original setup guide](RWKV-7_setup/README_RWKV7_FAST_SETUP.md).
The initial installation and GPU checks passed; see
[the validation report](RWKV-7_setup/RWKV7_ENVIRONMENT_VALIDATION.md).

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

WSL reads a Windows drive through a mount (`/mnt/e`) that is slow for large
files Windows no longer holds in its cache: 5 MiB/s was measured on 2026-11-17,
against 106 MiB/s for the same drive read natively and 110 MiB/s for WSL's own
disk. Writes are not affected in the same way (59 MiB/s). Resuming from a
4.5 GB checkpoint under `results/` therefore takes about 14 minutes, and the
saved validation panel is kept on WSL's disk (`~/.cache/basin/validation_panels`).

The setup script also installs `azure-storage-blob` (optional dependency group
`cloud`) for checkpoint copies. Their credentials, `AZURE_ENDPOINT` and
`AZURE_STORAGE_KEY`, are read from the environment or from a git-ignored `.env`
file in the repository root.

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

The generator creates sensor and measurement series for cyber-physical systems.
Each original table has its own sampled structural causal model (SCM), with
optional hidden covariates, process dynamics and recording artifacts.

### Quick start

From the repository root:

```bash
python -m pip install -e ".[data]"
python src/01_data_generation/generate_operational_data.py
```

Edit the variables at the top of
[generate_operational_data.py](../src/01_data_generation/generate_operational_data.py)
before running. The runner uses Python settings rather than command-line flags.
Its output path is resolved relative to the repository.

**Current runner settings** are summarized below. These are not necessarily the
reusable generator class's defaults; the runner is the source of truth.

| Setting | Current value | Meaning |
|---|---|---|
| `NUM_TABLES` | `6_500_000` | Number of original SCMs; optional copies are extra |
| `SEED` | `7` | Reproducible sampling; `None` requests fresh randomness |
| `MIN_ROWS`, `MAX_ROWS` | `112`, `2056` | Recorded length range for original tables |
| `FULL_LENGTH_PROBABILITY` | `0.5` | Point mass at the maximum length |
| `PREHISTORY_PROBABILITY` | `0.5` | Among full-length tables, simulate earlier history and retain the tail |
| `MAX_PREHISTORY_ROWS` | `2048` | Maximum extra history, in addition to burn-in |
| `MIN_COVARIATES`, `MAX_COVARIATES` | `3`, `6` | Sampled source-node count before extra plant/sensor channels |
| `VISIBLE_COVARIATE_LAMBDA` | `1.3` | Poisson count of visible SCM covariates, capped at those available |
| `BURN_IN` | `128` | Warm-up steps excluded from the recorded table |
| `SAVE_SCM`, `SAVE_PLOT`, `SAVE_DAG` | `False` | Replay specification and both plot types; off for bulk production |
| `PLOT_COVARIATES` | `False` | Include recorded covariates in the series plot |
| `GENERATION_WORKERS` | `10` | Worker processes; see "Parallel generation and progress" |

These are the settings of the latest bulk run. For a small sample to inspect, lower
`NUM_TABLES` and turn the three `SAVE_` settings on.

### Files, dates and repeated runs

Outputs are placed under the execution day's date:

```text
data/01_generated/YYYY-MM-DD/
  table_<run-id>_<index>/
    data.parquet
    scm.json       # when SAVE_SCM=True
    series.png     # when SAVE_PLOT=True
    dag.png        # when SAVE_DAG=True
```

`data.parquet` contains `time_index`, `outcome`, and the visible covariates.
Lagged predictors are internal simulation state and are not exported as feature
columns. Extra plant channels and redundant instruments can add covariates.

Each execution uses a new run identifier, so rerunning on the same date appends
tables without overwriting earlier ones. A different seed produces a new random
realization; a new folder name alone does not guarantee different data. Reusing
the same seed, settings and execution mode can reproduce the same values.

### How a table is simulated

1. Sample a causal graph, equations, parameters and observation seed.
2. Generate exogenous inputs, including noise, events, discrete states or formulas.
3. Advance the system timestep by timestep, preserving real historical values
   for lagged dependencies. Optional plant states model closed-loop behavior.
4. Screen the clean process for the requested trend profile and reject invalid
   or intentionally excluded constant outcomes.
5. Apply the recording layer, export the table and optionally create copies.

The specification in `scm.json` supports replay through
`OperationalSCMGenerator(...).simulate(model)`. Replaying a saved specification
is distinct from regenerating from an old seed after changing the sampling code.

### Causal graph and hidden covariates

Sparse graphs can include direct effects, mediation, common causes, colliders
and delayed feedback. Current-row dependencies must respect evaluation order;
negative delays and invalid parent references are rejected.

- **Outcome history:** `t-1` is always included. Additional recent, seasonal and
  unstructured lags are sampled up to `MAX_LAG=512` and `MAX_LAG_COUNT=4`.
- **Seasonal history:** `SEASONAL_LAG_PROBABILITY=0.6`; selected seasonal anchors
  can represent multiple cycles, subject to available lag capacity.
- **Other historical outcome effects:** `LONG_LAG_PROBABILITY=0.035` adds an
  unstructured lag when capacity permits. Residual history supplements selected
  outcome lags with `RESIDUAL_HISTORY_PROBABILITY=0.05`.
- **Covariate delays:** `INSTANTANEOUS_PROBABILITY=0.02`; otherwise a positive
  delay is drawn from a discretized exponential, with additional transport-delay
  cases. `MAX_COVARIATE_LAG=96` caps the delay.
- **Visibility:** a Poisson draw with lambda 1.3 chooses the number of visible
  SCM covariates. Hidden variables still affect the simulation. Optional plant
  exports or redundant sensors can make an otherwise univariate table multivariate.

DAG colors distinguish blue outcome, purple outcome-derived history, green
visible covariates and orange hidden covariates. Historical covariate nodes keep
their visibility color. The image shows a bounded upstream view; the saved SCM
retains the complete equations and dependencies.

### Relationships and stochastic inputs

Edges support affine, saturating, exponential, power, logarithmic, physical
response and neural-network-like maps. Temporal operators include lag, delay,
integration, hysteresis, rate limits, resonance and second-order responses.
Composite expressions are sampled with `COMPOSITE_EFFECT_PROBABILITY=0.25`, up
to `MAX_EFFECT_COMPONENTS=6` instructions.

Multiple parents can interact through products, ratios, bottlenecks, gates,
thresholds, convex mixtures and small neural networks. A node-level link can
then transform the combined parent effect. See
[effect_mechanisms.py](../src/01_data_generation/effect_mechanisms.py).

Noise includes Gaussian, Student-t, mixtures, Poisson, beta, Weibull,
quantization error, Perlin, pink and brown/red processes. Distribution parameters
are sampled per SCM. `NOISE_WEIGHTS`, `EFFECT_WEIGHTS`, `FOURIER_WEIGHTS` and
`PRIOR_RANGES` customize the available families and parameter ranges; weights
are relative weights, not percentages.

`NOISE_FREE_COVARIATE_PROBABILITY=0.35` allows deterministic process covariates
given their parents. Formulaic exogenous sources and Bernoulli/Markov inputs
provide alternatives to continuous noisy sources.

### Trends, seasonality and changing variance

`TREND_FREE_FRACTION=0.5` controls the share of **proposals**, not a guaranteed
final trend-free fraction. Screening records both proposed and accepted profiles:

- A trend-free proposal with linear R-squared at most 0.15 is accepted as trend-free.
- A linear R-squared above 0.15 is accepted as trending, including incidental
  trends in trend-free proposals.
- A trending proposal can also qualify through trend-shape R-squared above 0.15.

Linear and trend-shape R-squared are different metrics. Moderate trends are
accepted; no 0.4 minimum is imposed. Screening resamples unsuitable proposals
rather than correcting their generated values. `REJECT_CONSTANT_OUTCOME=True`
rejects constant or nearly constant clean targets; flat stretches remain possible.

Trend families include linear/broken trends, multiplicative growth, wear,
commissioning, onset, maintenance resets and level shifts. Event shocks can
permanently change the course of a trend.

| Control | Current value | Scope |
|---|---|---|
| `SEASONAL_TABLE_PROBABILITY` | `0.4` | Select clock-driven outcome seasonality |
| `FOURIER_PROBABILITY` | `0.7` | Per seasonal component, use a Fourier expansion |
| `MAX_FOURIER_HARMONICS` | `16` | Also limited by the sampling resolution |
| `SEASONAL_SHAPE_DRIFT_PROBABILITY` | `0.15` | Allow seasonal shape to evolve |
| `MULTIPLICATIVE_TREND_PROBABILITY` | `0.22` | Among trending proposals |
| `VARIANCE_DYNAMICS_PROBABILITY` | `0.2` | Allow changing noise envelopes |

Fourier partial sums produce pulse/square-like, sawtooth, triangular, rectified,
multi-peak and smooth random seasonal shapes. Multiple cycles, alternating peaks,
amplitude changes and smooth timing variation are supported. Multiplicative
trends normally scale noise and seasonal variation with level; independent 5%
exceptions allow constant noise or constant seasonal scaling.

`NOISE_PROFILE_WEIGHTS` selects clean, low, moderate or noisy tables. Wander and
variance dynamics can also occur without a trend or clock-driven seasonality.

### Process episodes and closed-loop equipment

`PROCESS_EPISODE_PROBABILITY=0.025` creates a sustained idle/startup/active/shutdown
run using an exogenous process driver. Ramp shapes, durations and idle periods
can be asymmetric; active periods may contain oscillations, multiple operating
levels and startup jitter. `SWITCHING_PROBABILITY=0.14` separately supports
shutdown/restart and ramp-reset behavior.

With `PLANT_PROBABILITY=0.3`, eligible proposals can use a plant whose demand is
driven by the SCM. Plant outputs and optional companion measurements come from
the same evolving state. Totalizers are selected within trending proposals;
other plant families are selected within trend-free proposals.

| Plant family | Example behavior |
|---|---|
| `relay` | Thermostat/compressor cycling with hysteresis, wear and inrush |
| `pid` | Setpoint tracking with delay, saturation, windup and disturbance rejection |
| `batch` | Fill, hold and discharge/reset cycles |
| `step_response` | First- or second-order settling, including overshoot |
| `totalizer` | Integrated flow with pauses |
| `stick_slip` | Force builds while position sticks, then releases during sliding |
| `interlock` | A protective trip remains latched until safe conditions and a new reset command |
| `coupled_storage` | Two inventories share an exactly balanced transfer flow |
| `rotating_machine` | Speed integrates into continuous phase; shared modes set speed, vibration amplitude and torque |

`PLANT_WEIGHTS` controls relative selection. Stick-slip and interlock each have
runner weight `0.08`. They use lightweight causal state updates rather than an
ODE solver. See [plant_mechanisms.py](../src/01_data_generation/plant_mechanisms.py)
and [process_mechanisms.py](../src/01_data_generation/process_mechanisms.py).

### Shared states and predictable drivers

The runner enables three sampling controls; none adds an acceptance/rejection test:

| Setting | Default | Effect |
|---|---|---|
| `MAX_DOMINANT_DRIVERS` | `2` | Retain at most two full-strength covariate sources; scale other sources' outgoing edge weights by a saved factor in `[0.03, 0.15]` |
| `PERSISTENT_HIDDEN_PROBABILITY` | `0.8` | Prefer persistent hidden inputs: causal low-pass innovations, longer Markov dwell, or persistent binary states instead of independent Bernoulli draws |
| `SHARED_STATE_PROBABILITY` | `0.2` | Sample one categorical Markov mode that influences multiple channels, when the node budget permits |

The driver budget is a prior on input strength, not a guarantee about observed
importance: nonlinear interactions can amplify weak inputs. Outcome innovations,
rare shock events and explicit shutdown mechanisms remain possible. Set the cap
to `None`, or either probability to zero, to disable that control. Constructor
defaults leave these controls off for compatibility; the runner enables them.

The two new plant families each have runner weight `0.10`. Both always use one
persistent categorical mode to coordinate their internal channels. Storage uses
`A[t] = A[t-1] + inflow[t] - transfer[t]` and
`B[t] = B[t-1] + transfer[t] - outflow[t]`. Noise perturbs inlet flow, not inventory
independently, so the joint material balance holds. The rotating machine uses
`phase[t] = phase[t-1] + 2*pi*speed[t-1]`; changing speed or mode never resets phase.
Harmonics share this phase, and sampled speeds keep them below Nyquist.

These are causal structural equations, saved with parameters in `scm.json`.
Their DAG images show the physical update and shared mode. Conservation applies
to latent physical units; the recorded target's affine unit conversion and later
sensor artifacts can obscure the raw numerical balance. Existing saved SCMs
remain replayable. Newly sampled tables intentionally use the expanded prior.

### Sensor artifacts and missing data

The [recording layer](../src/01_data_generation/measurement_layer.py) runs after the
clean simulation and does not feed its artifacts back into the physical process.
Separate structural sensor bounds and selected plant sensor effects can affect
causal state inside the simulator.

| Setting | Current value | Behavior |
|---|---|---|
| `MEASUREMENT_PROBABILITY` | `0.2` | Per continuous column, select 1-3 recording artifacts |
| `REDUNDANT_SENSOR_PROBABILITY` | `0.12` | Add extra instruments observing an existing signal |
| `MISSINGNESS_PROBABILITY` | `0.15` | Select a table-level missingness mechanism |
| `ROW_GAP_PROBABILITY` | `0.04` | Remove blocks or scattered rows |
| `SENSOR_BOUNDS_PROBABILITY` | `0.05` | Sample structural sensor bounds |

Artifacts include noise, clipping, coarse quantization, deadband, held/stuck
readings, spikes, dropout, drift and recalibration. Missingness can depend on
chance, another measurement, past values, a schedule or censoring thresholds.
Missing values are exported as NaN. The final 80 outcome rows are protected from
the missingness mask for evaluation labels.

Removed rows currently receive a consecutive recorded index. These tables
represent gaps in the observed sequence without preserving actual elapsed-time
gaps. The generator does not yet provide a full asynchronous timestamp model.

### Extra copies from existing tables

Copies reuse an existing realization and do not sample another SCM. Selection
uses independent random streams, so enabling copies does not perturb original
tables. Copies are made from originals only, without recursively copying copies.

| Copy | Setting | Eligibility | Transformation |
|---|---|---|---|
| Coarse | `COARSE_COPY_PROBABILITY=0.03` | More than 512 rows | Select one fixed offset from each complete window of 2-5 rows, shared by every channel |
| Historian aggregate | `AGGREGATE_COPY_PROBABILITY=0.03` | More than 512 rows | One min, max or mean copy using complete windows of 2-5 rows |
| Sign inversion | `SIGN_INVERSION_PROBABILITY=0.05` | Original tables | Negate all measured columns while preserving the clock |

**Historian copies are strictly trailing.** At source time `t=w`, all channels
summarize `[0,w)`; at `t=2w`, they summarize `[w,2w)`. No sample at or after `t`
contributes. The timestamp is the window's exclusive right edge, incomplete
final windows are discarded, and NaNs propagate. A binary covariate's mean is
the fraction of that window spent active.

These covariates are historical measurements. Future-window aggregates must not
be supplied as future-known inputs when forecasting. Original/copy families are
kept together by the pipeline's evaluation-family isolation: their members are
training-only, preventing the same realization from entering both training and
held-out evaluation.

Copies are written as sibling folders with `_coarse_...`, `_aggregate_...` or
`_sign_inverted` suffixes. Each has `provenance.json`, even if SCM export is off.
When saved, its SCM supports replay of the observation transformation. DAGs show
the source model and source-clock lags.

### Parallel generation and progress

`GENERATION_WORKERS=None` automatically selects half the logical processors, at
most eight. Each worker samples, simulates and writes complete tables. Use a
positive integer to set the worker count explicitly; the runner sets 10 of the
12 logical processors of the 6-core generation machine. The console shows
completed originals, throughput, elapsed time, ETA and additional copy counts.

Measured on 2026-11-15 (Ryzen 5 7600X3D, 3,000 tables, production settings, no
plots): 76 tables/s with six workers and 105 tables/s with twelve. The tables
are byte-identical for any worker count.

Simulation is a per-row Python loop. Each node's constants and each edge's
operator are therefore bound once per table rather than looked up on every row
(`compile_instruction` and `run_steps` in `effect_mechanisms.py`,
`_bind_effect` in `scm_generator.py`), short parent sums skip the NumPy array,
and Markov inputs search plain lists. None of this may change a value:
`tests/test_effect_mechanisms.py` and `tests/test_process_mechanisms.py` compare
the bound path with the reference path bit for bit, and after any such change
`benchmark_generation_exact.py` should be rerun against a copy of the previous
generator.

`GENERATION_BATCH_SIZE=16` batches work while keeping queues bounded.
`PROGRESS_INTERVAL=1.0` controls console update frequency. In sequential
simulation, `RENDER_WORKERS` can parallelize plotting separately; generation
workers render their own plots without nesting another process pool.

For bulk production, disabling `SAVE_PLOT` and `SAVE_DAG` avoids rendering and
image-writing costs. `SAVE_SCM=False` also reduces disk use but removes the full
replay specification. The viewer needs saved images to display plots.

Parallel generation uses independent per-table seeds and is reproducible across
worker counts/batch sizes within that mode. Explicit `GENERATION_WORKERS=1`
uses the legacy serial random stream, so switching between serial and parallel
modes can change the generated sequence. Keep the script's `__main__` guard for
Windows multiprocessing; interactive sessions may fall back to serial work.

### Browse generated plots

```bash
python src/01_data_generation/view_generated_plots.py
```

The local browser viewer defaults to the newest date folder and loads 50 tables
per page. Use Previous/Next, filter names on the current page, or click a plot to
enlarge it. Older date folders are not scanned for tables by default.

Edit [view_generated_plots.py](../src/01_data_generation/view_generated_plots.py):

- `LATEST_ONLY=False` restores the full date dropdown.
- `INITIAL_DATE="YYYY-MM-DD"` selects a particular day.
- `PAGE_SIZE` changes the number of tables loaded per page.
- `GENERATED_DIR` changes the source directory.

Keep the server running while browsing; Ctrl+C stops it. The viewer reads saved
`series.png` and `dag.png` files. It reports missing images explicitly rather
than regenerating them from Parquet.

### Preparing newly generated data for modeling

Set `data.generated_dates` in [configs/config.json](../configs/config.json) to the
new folders, then run stages 2-4 with that configuration. The generator's output
date does not automatically update the pipeline's date selection.

```bash
python src/02_data_validation/validate_generated_data.py
python src/03_transforming_input_data/transform.py
python src/04_patch_indexing/index_patches.py
```

Validation is controlled by `validation.enabled`; when false, stage 2 skips
screening. Transformation and patch indexing identify their input data
and support resumable work. Successful runs update the main artifacts; old
checkpoint directories can remain on disk. See the screening section below for
admission rules.

For a fresh training run on the new corpus, set `training.resume=false` before
launching training. Clearing generated/processed/patch-index data does not remove
training checkpoints or change `resume="auto"`.

### Inspection, tests and coverage review

Useful entry points:

| Tool | Purpose |
|---|---|
| `preview_process_drivers.py` | Process-run and formulaic-input examples |
| `preview_fourier_seasons.py` | Seasonal-shape gallery |
| `preview_fourier_partial_sums.py` | Fourier partial-sum demonstration |
| `benchmark_generation.py` | Simulation/rendering benchmark with optional archived reference |
| `benchmark_generation_exact.py` | Exact before/after audit using an archived generator directory |

These scripts live in `src/01_data_generation/` and have editable settings.
The exact benchmark reads runner settings without launching the production job;
it checks model parameters, values, copy outputs and sampled image bytes.
Use it for behavior-preserving optimizations. Intentional sampling/acceptance
changes require a separate performance comparison because outputs will differ.

Representative checks:

```bash
python -m unittest discover -s tests -p test_scm_generator.py
python -m unittest discover -s tests -p test_generator_realism.py
python -m unittest discover -s tests -p test_historian_mechanics.py
python -m unittest discover -s tests -p test_generation_runtime.py
python -m unittest discover -s tests -p test_plot_viewer.py
```

Additional tests cover effect/process mechanisms, copy replay and evaluation
isolation. `experiments/claude_review/generator_coverage.py` and its companion
scripts provide exploratory coverage statistics and contact sheets.

The [operational coverage review](audits/2026-11-14/generator_operational_coverage.md)
is a historical assessment with proposed extensions, some subsequently
implemented. The generator offers broad mechanism coverage, not a guarantee
that every real-world cyber-physical dataset is represented.

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
See [Chronos-2 screening](data_validation/CHRONOS2_SCREENING.md) for
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
[license](../third_party/rwkv-lm/LICENSE). See
[provenance](RWKV-7_setup/RWKV7_PROVENANCE.md) for the upstream revision and local patches.

# Trainable patch-based forecaster

The RWKV-7 pipeline lives in the numbered `src/` stage folders and uses
`configs/config.json`. Dependencies, including Torch and Loguru, are managed by `uv`.
See [the architecture, training commands and forecasting example](forecasting/FORECASTING.md).
Choose one or more generated-data date folders with data.generated_dates in
configs/config.json; paths.raw points at data/01_generated. See the
[date-selection configuration](audits/2026-10-25/generated_dates/REPORT.md).
Run the stages separately with `uv run python`:

```powershell
uv sync --extra data --extra validation
uv run python src/02_data_validation/validate_generated_data.py
uv run python src/03_transforming_input_data/transform.py
uv run python src/04_patch_indexing/index_patches.py
uv run python src/05_model_training/train.py
uv run python src/06_holdout_test/holdout_test.py
```

See the [transformation guide](transformation/README.md) for input selection,
causal training records, evaluation policies, outputs and checkpoint recovery.
The [patch indexing guide](patch_index/README.md) explains what
`index_patches.py` checks, its worker processes, progress bars and resumable
metadata indexing.

Next-patch transformation resumes automatically when the same command is rerun.
The console shows a progress bar, completed/total tables, percentage, tables per
second, elapsed time and ETA. It updates every 100 tables (less often for a corpus
of a million tables or more), at least every five seconds when a table finishes,
and at completion. Source-inventory checking and
checkpoint restoration have separate bars; progress counts tables rather than rows.

The checkpoint cadence is configured independently of progress logging:

```json
"transform": {
  "batch_size_tables": 100000,
  "batch_memory_limit_mb": 512,
  "checkpoint_every_tables": 100000,
  "checkpoint_interval_seconds": null
}
```

These are the values in `configs/config.json`. Without the first two settings,
buffers are spilled after 1,000 tables or 64 MiB.
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

Transformation spills buffers after `batch_size_tables` tables or approximately
`batch_memory_limit_mb` of tensor data, whichever comes first, independently of
checkpoint frequency. The final `windows.pt` is a small
shard index, published atomically without merging the corpus into RAM. Patch indexing and
training read shards on demand through the artifact loader; older single-file
artifacts remain supported. The referenced `batch_*.pt` files in the checkpoint
directory are now the dataset itself and **must be kept** after completion.
The largest source table is still read in full, and source/record metadata still
uses memory. This resume mechanism covers next-patch transformation;
optional Chronos/Prophet screening remains a separate upstream phase.

Millions of small tables on a hard disk are limited by head movement, so the
stage avoids visiting files one at a time in name order. Source discovery lists
each date folder once and makes one metadata lookup per table folder, in
file-table order on Windows; a folder that holds a `data.parquet` is a table and
is not searched further, any other folder is searched recursively. Next-patch
transformation then reads the tables in windows, requesting the pieces of all
files in a window in on-disk order (`source_io.py`), decodes them in worker
processes (`table_decoder.py`) and assembles records in table order in the main
process. None of this changes the records, shards or manifest. Four optional
`transform` settings tune it; they are not part of the dataset signature:

| Setting | Default | Meaning |
|---|---|---|
| `workers` | `null` | Decoding processes. `null` uses half the logical CPUs (at most 8) for 4,096 tables or more and none below; `0` decodes in the main process. |
| `read_threads` | `8` | Threads issuing read requests within a window. |
| `read_window_tables` | `16384` | Most tables read ahead as one window. |
| `read_window_mb` | `512` | Most source bytes held per window; up to three windows are in memory. |

Each checkpoint also logs the output written so far, the amount still expected
and the free space on the output disk, with a warning when it will not fit.

Patch indexing (`index_patches.py`, called `embed.py` until 2026-11-17) follows
the same pattern. One thread verifies each shard's checksum in file order, so
the disk reads the shards as one stream; worker processes open the verified
shards and inspect their records; the main process writes the results in record
order. The output is the same as with a single process. `patch_index.workers`
sets the number of processes (default `null`: half the logical CPUs, at most 8,
when eight or more shards remain; `0` inspects in the main process).

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
100, e.g. 1 for a quick 1% run). `configs/config.json` currently compares with
`["naive", "chronos", "timestral"]` on 15% of the test tables. The percentage
applies to the admitted saved holdout split. See the [foundation benchmark audit](audits/2026-10-29/foundation_benchmarks/REPORT.md).
Fresh training now records full epoch-zero validation before the first update;
resumed runs retain their recorded baseline, if available. See the
[sampled evaluation and inference audit](audits/2026-10-29/sampled_evaluation/REPORT.md).

### Pre-training token inventory

After patch indexing, inspect the amount of distinct input without starting training
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
[startup performance audit](audits/2026-10-27/training_startup/REPORT.md).
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
rather than only the final window saved by transformation. Repeated rolling context
use is counted analytically, without running forecasts. JSON also records split
overlaps and validation/test content not seen in training. Source-table overlap
between splits must be zero; identical numeric content can occur in distinct tables.

The CPU-only parameter count follows the current vendored RWKV-7 architecture.
The trainer updates the ratios with its measured parameter count after model
initialization. A temporary SQLite index keeps deduplication memory bounded;
the patch index maps raw tensor storage and prepares features per accessed window.
The learned embedding projection runs inside training, without corpus-sized
embedding snapshots. See the [memory investigation](audits/2026-10-27/embedding_memory/REPORT.md).
No optional packages
are required. This diagnostic currently supports the `next_patch` objective.

Exact uniqueness does not establish statistical independence or sufficient data:
near-identical or noisy patches can all have different contents. The report
provides descriptive ratios without imposing a sufficiency threshold.

Earlier defaults were a 394.9M-parameter causal next-patch model with 16-point
patches, width 1,024 and a 2,048-point history cap, and before it the
[301.8M-parameter, 100-table experiment and baseline comparison](audits/2026-10-24/next_patch_300m/REPORT.md).
The current default is the 221.5M hybrid described at the top of this document.

The earlier admission policy used pre-test multivariate Prophet MAE
compared with last-observation MAE. The earlier strict univariate-or-multivariate
policy is documented in the historical
[learnability gate and embedding audit](audits/2026-10-24/learnability_gate/REPORT.md).

Every regular and capacity training run automatically writes a timestamped JSONL
journal under `logs/`, with configuration, metrics, memory and completion/failure
events. See the [JSONL format and verification audit](audits/2026-10-25/training_jsonl/REPORT.md).
With `training.log_every_tables=1000` (the value in `configs/config.json`) the
journal holds one `metrics` record per 1,000 tables, summed over the updates in
between, plus one before every checkpoint and validation; `null` restores one
record per update with the ids of its tables.

Training now bounds GPU execution separately from the optimizer batch. Defaults
in `configs/config.json` retain `batch_size=8` and
`gradient_accumulation_steps=1`, but split each update into physical microbatches
of at most eight tables (`microbatch_size=8`) and 3,072 padded variable-patches
(`microbatch_token_limit=3072`; the code default is 1,536). For example, eight
tables with eight variables and 128 patches each (8,192 variable-patches) run as
three forward/backward passes of three, three and two tables, followed by one
optimizer update. Losses are weighted by observed target points. Physical
microbatching preserves context, padding, validation coverage and the chosen
batch schedule. Running batches of 3,072 to 4,096 variable-patches in one pass
was measured on 2026-11-17 and was 2.7% slower on the real mix of shapes.

`cuda_memory_limit_gib=11.0` caps PyTorch's CUDA allocator, including its cache,
at the smallest of 11 GiB, device capacity minus `cuda_memory_headroom_gib=0.5`,
and the memory free at start minus that headroom (10.26 GiB on the 12 GB
RTX 5070 with a desktop running). CUDA libraries and desktop applications use
additional memory outside this budget. A shape that cannot fit raises an
allocation error rather than letting PyTorch's reservation grow into shared GPU
memory; with `auto_microbatch=true` the update is then repeated with smaller
passes and the smaller size is remembered for that shape.
Each execution limit accepts `null` to disable that limit; a token budget
always permits at least one table and does not truncate its sequence. These
settings apply when the next training worker starts. See the
[VRAM investigation and benchmarks](audits/2026-10-28/vram_budget/REPORT.md).

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
experiment with all lengths. The two-profile/1.5 GiB graph cache and the CUDA
allocator limit still apply. See the
[capture-inclusive schedule benchmark](time_benchmark/training_graph_schedule/REPORT.md)
for total-update timings, correctness checks and limits of the measurement.

## Reproducing the pipeline

The repository contains pipeline code, configurations, dependency locks, tests, experiment scripts and narrative audit reports. Generated datasets, model weights, numeric audit artifacts, caches and run outputs remain local and are not committed. Follow the stage commands above to regenerate these artifacts. Historical reports may link to local evidence files intentionally excluded from Git.

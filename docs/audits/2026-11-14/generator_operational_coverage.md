# Operational generator coverage review

Date: 2026-11-14. Scope: sensor readings and other measurements of cyber-physical systems. This is a source-code review and a set of design suggestions, not an implementation or an empirical coverage benchmark. Sampling frequencies and realistic mixture proportions are deliberately out of scope.

## Main finding

The generator has broad waveform coverage. Its largest remaining gaps concern **what a measurement means**, **how measurements are acquired**, and **how physical states jointly evolve**. Adding more static nonlinear functions or noise distributions alone would not close these gaps.

A random SCM might accidentally produce a trace resembling one of the examples below. That is weaker coverage than explicitly generating the defining behavior and its relationship to covariates. Conversely, absence of a named industry-specific simulator does not mean every signal from that industry is missing. The suggestions target reusable mechanisms, not a digital twin of every asset.

## What is already covered

The reviewed implementation already includes:

- Delayed causal graphs, hidden covariates, outcome feedback, composite edge functions, multi-parent interactions, and node-level nonlinear links.
- Linear, multiplicative and lifecycle trends; permanent shock-induced slope changes; stochastic wander; level changes; maintenance-like resets.
- Multiple seasonal components, Fourier shapes, changing seasonal amplitude and shape, smooth periodic timing variation, alternating peaks, and rest-cycle gating.
- White/heavy-tailed/mixture noise, Poisson/beta/Weibull innovations, Perlin, pink and brown noise, heteroskedasticity, and volatility coupling.
- Formulaic exogenous inputs, Bernoulli and multistate Markov covariates, asymmetric process episodes, startup jitter, multiple operating levels, shutdown/restart and ramp resets.
- Stateful lag, delay, integration, hysteresis, resonance and second-order edge operators; relay, PID, batch, step-response and totalizer plants.
- Sensor clipping, quantization, deadband, hold, stuck readings, spikes, drift, recalibration, redundant instruments, removed rows and several missingness mechanisms.

These should not be proposed again as entirely absent features. The gaps below distinguish missing mechanisms from narrower existing approximations.

## Coverage gaps and suggested additions

### 1. Actual elapsed time and asynchronous sensor acquisition

**Status: acquisition mechanisms partly covered; timing semantics missing.** The simulator advances on one integer clock. `_simulate_process` exports consecutive `time_index` values, and `apply_observation` rewrites that index after removing rows. Multirate missingness and held values exist, but elapsed time across a removed block is not retained.

**Missed examples:** a pressure sample arriving 30 seconds late relative to flow; a logger restarting after a two-hour outage; sensors with independent sample clocks; jittered sampling; event-triggered observations.

**Suggestion:** distinguish physical time, source acquisition time, arrival time and observation age. Support independent channel schedules, variable time increments, timestamp jitter, delayed/batched arrivals and duplicate/out-of-order reports. A fixed-grid export could remain available, with age/availability indicators and explicit resampling rules. An asynchronous representation would require a separate schema decision.

**Coverage check:** removing an interval preserves its elapsed-time gap; delayed telemetry changes what was available at a forecast origin without changing the underlying process.

OPC UA explicitly separates source/server timestamps and measurement status, supporting this distinction. [OPC UA DataValue](https://reference.opcfoundation.org/specs/OPC-10000-6/5.2.2.17).

### 2. Historian aggregation and derived measurements

**Status: fixed-stride subsampling exists; aggregation semantics are missing.** A coarse copy selects one sample per window. It does not generate window means, extrema, integrals, RMS, rates or rolling statistics.

**Missed examples:** one-minute mean power versus accumulated energy; vibration RMS; peak pressure; an hourly maximum; rolling defect percentage; data reconstructed by historian interpolation or compression.

**Suggestion:** add causal observation operators for window mean/min/max/sum/RMS, integrals and differences, trailing-window statistics, filtered decimation and interpolated/held historian views. Save the interval, alignment and operator. Separate raw undersampling, which may alias, from filtered downsampling. Do not use centered windows that expose future values as current covariates.

**Coverage check:** one latent run can produce different but internally consistent sampled, averaged and accumulated tables. OPC UA defines distinct historical aggregates rather than treating these as interchangeable. [Aggregate details](https://reference.opcfoundation.org/specs/OPC-10000-13/5.4).

### 3. Discrete, bounded and intermittent **targets**

**Status: discrete covariates exist; there is no dedicated discrete target family.** The target is constructed through continuous equations; plant status channels are explicitly excluded from target selection. Poisson innovations and coarse quantization do not ensure a nonnegative integer-valued outcome.

**Missed examples:** cycle counts per interval, occupied slots, valve-open status, motor operating mode, alarm activity, reject counts, sparse detected particles, bounded occupancy and measured fractions.

**Suggestion:** allow target observation families: binary, categorical numeric states, nonnegative counts, bounded counts, fractions, and hurdle/zero-inflated event magnitudes. A latent physical state should control transition probability or event intensity. Decide how categorical labels are encoded for a numeric forecasting model; numeric labels should not accidentally imply a meaningful distance between states.

**Coverage check:** counts remain integral/nonnegative; binary targets remain binary; long zero stretches coexist with state-dependent nonzero bursts.

### 4. Real counter resets, rollover and rate/total consistency

**Status: totalization and batch resets partly cover the shapes.** The totalizer sums a rate; batch plants generate fill/reset traces. Neither explicitly provides hardware modulo rollover, reboot resets or a jointly consistent reset-aware counter and rate.

**Missed examples:** finite-width pulse counters, a rebooted energy meter, cycle totals reset per shift, cumulative flow with a changed meter origin.

**Suggestion:** a counter measurement driven by actual event increments or integrated flow, with finite register width, resets, replacement offsets and optional signed/bidirectional accumulation. Export corresponding rate/increment channels from the same state.

**Coverage check:** a rollover is distinguishable from negative physical flow when the rate is observed. Counter/gauge distinctions also matter for equipment telemetry. [Prometheus metric types](https://prometheus.io/docs/concepts/metric_types/).

### 5. Coupled storage and conservation laws

**Status: integrators and individual storage-like plants exist; general joint balances are missing.** Random graph equations do not enforce that multiple channels share a conserved inventory.

**Missed examples:** tank level with measured inlet/outlet flow; interconnected reservoirs; thermal mass with heat input and losses; battery charge, current, voltage and temperature; material concentrations whose fractions sum to one.

**Suggestion:** reusable coupled-state templates enforcing balance equations, such as `inventory_next = inventory + dt * (inflow - outflow)`, with capacity limits, overflow/leak channels and appropriate dynamics for derived measurements. Apply instrument errors after producing physically consistent latent channels. A few low-order templates would cover many assets.

**Coverage check:** level changes reconcile with the flow balance; a battery exhibits load-dependent voltage sag, recovery and charge-dependent behavior. The latter needs more than an arbitrary sigmoid plus a trend. [Battery equivalent-circuit model](https://www.mathworks.com/help/simscape-battery/ref/batteryequivalentcircuit.html).

### 6. Stateful queues and event-arrival processes

**Status: a static queue blow-up operator exists; queue occupancy dynamics are missing.** Poisson noise is not a general arrival/service model, and independent event draws do not explicitly model clustered or refractory event timing.

**Missed examples:** conveyor buffers, work-in-progress, automated inspection queues, intermittent actuator cycles, fault bursts, detector dead time, backlog that drains after demand subsides.

**Suggestion:** arrival/service state updates with queue capacity, variable service rate, blocking and loss counters; renewal processes with configurable dwell laws; clustered/self-exciting arrivals and refractory periods where appropriate. Export occupancy, arrivals, completions and throughput from one process.

**Coverage check:** overload accumulates backlog, recovery drains it, and event counts reconcile across channels. A formula resembling `1/(1-utilization)` alone cannot establish those relationships.

### 7. Operating modes that change the equations

**Status: on/off, scheduled levels, Markov covariates, gating and scalar regime operators exist; a general mode-dependent system is only partial.** Most saved edge equations and delays remain fixed throughout a table.

**Missed examples:** manual/automatic control transfer; heating versus cooling; different production recipes; normal/derated/fault operation; interlocked sequences; an emergency trip latched until an explicit reset.

**Suggestion:** guarded state machines whose mode selects coefficients, active edges, controller gains, delays, noise laws and constraints. Transitions should be able to depend on physical state, elapsed dwell, external commands and reset conditions, rather than only a preplanned clock or independent Markov draw.

**Coverage check:** the same input produces a different response after a mode change, and a latched trip persists even when the triggering threshold is no longer exceeded.

### 8. Endogenous degradation, failure and maintenance

**Status: wear curves, maintenance sawtooths, relay wear and shock effects already exist.** The broader gap is damage accumulated from actual exposure, with consequences for subsequent physics and fault risk.

**Missed examples:** fouling driven by cumulative throughput; motor heating shortening duty cycles; a valve becoming slower with wear; load-dependent failure followed by an imperfect repair.

**Suggestion:** a damage state updated by load, temperature, cycles or exposure. Let it change capacity, gain, friction, noise, lag and failure hazard. Maintenance resets some damage, changes parameters, or replaces an instrument; failure can trigger an absorbing or repairable operating mode.

**Coverage check:** identical clock time but different prior loading produces different remaining capacity. A failure changes the post-event dynamics, rather than just adding an outlier or bending a trend.

### 9. Nonlinear mechanics with memory beyond binary hysteresis

**Status: deadband, saturation, rate limits and binary hysteresis are present.** Missing families include backlash/play, breakaway friction, stick-slip, position-dependent contact and impact/rebound.

**Missed examples:** a reversing actuator moving without its load until gear slack is taken up; a valve sticking until pressure overcomes friction; intermittent contact in machining; position/force loops differing between loading and unloading.

**Suggestion:** small position/velocity/contact-state templates with backlash and friction memory, optionally coupled to a drive. Existing instantaneous deadband should remain a separate mechanism.

**Coverage check:** the output after direction reversal depends on recent travel, not just the current input or a binary threshold state.

### 10. Inverse response and richer coupled control

**Status: overshoot, ringing, negative covariate persistence and PID control exist.** Therefore, oscillation and negative dependence are not wholly absent. What is missing explicitly is a process that initially moves in the wrong direction before settling, plus interacting controllers that share a plant.

**Missed examples:** a nonminimum-phase level/thermal response, inverse initial response after a load change, cascade control, competing loops and load-sharing units.

**Suggestion:** opposing fast/slow response paths for inverse response; multiple controllers acting on coupled states; explicit actuator/sensor dynamics within those loops. Keep every dependency causal, potentially through internal solver substeps.

**Coverage check:** an input step causes a genuine wrong-way excursion followed by correct final settling; changing one loop perturbs another measured channel.

### 11. Speed-dependent vibration, impacts and electrical waveforms

**Status: Fourier waves, chirps, damped oscillations, event ringing and amplitude modulation already cover important components.** The gap is tying phase and faults to the physical operating state.

**Missed examples:** rotating-machine vibration whose harmonic frequencies follow RPM; bearing impacts followed by resonant decay; gear sidebands; run-up through a resonance; current/voltage phase relationships and switching ripple.

**Suggestion:** integrate shaft/electrical phase from speed/frequency, generate order-related harmonics and angle-triggered impacts, and modulate them with load and fault state. Produce vibration, speed, current and derived RMS channels jointly. Raw high-rate waveforms and low-rate health indicators should be distinct observation choices.

**Coverage check:** doubling speed changes harmonic/impact timing while a structural resonance can retain its natural frequency. Arbitrary clock chirps do not guarantee this relationship. [MathWorks vibration analysis example](https://www.mathworks.com/help/signal/ug/vibration-analysis-of-rotating-machinery.html).

### 12. Delays and response parameters controlled by the process

**Status: fixed sampled delays and time constants exist.** Smooth clock warping affects seasonal timing, but transport lag does not generally evolve with flow, speed or mode.

**Missed examples:** conveyor travel time changing with belt speed; downstream temperature delay changing with fluid flow; viscosity changing actuator response time.

**Suggestion:** causal transport buffers or accumulated-distance travel states, and state-dependent response coefficients. Distinguish a fractional physical delay with interpolation from acquisition latency in gap 1.

**Coverage check:** material arrives after the required integrated travel distance; an input change never appears downstream before it could physically arrive.

### 13. Persistent and shared stochastic state

**Status: colored noise, smooth variance envelopes, residual-dependent scaling and volatility coupling are present.** These are partial coverage of stochastic dependence, not a general multichannel stochastic state model.

**Missed examples:** persistent random high/low volatility episodes; intermittent turbulent bursts; common electrical interference across instruments; stochastic disturbances with a structured cross-channel covariance or phase relation.

**Suggestion:** optional stochastic variance states, innovation-driven variance recursion and shared disturbance factors with channel-specific filtering. Use hidden common-cause nodes when they already suffice; add correlated innovations only for dependence not expressed by those nodes. Fractional/long-memory dynamics could supplement the existing finite-filter pink and random-walk families if a concrete coverage test demonstrates a missing behavior.

**Coverage check:** bursty variance persists without a prescribed clock envelope; shared interference changes measured channels coherently without necessarily changing physical states.

### 14. Circular measurements and multi-axis observations

**Status: angle-domain covariates can wrap angles; a dedicated circular target/linked orientation family is missing.**

**Missed examples:** rotary encoder angle, wind direction at a turbine, robot orientation, multi-axis acceleration and rotating vector components.

**Suggestion:** phase/orientation states with wrapped observation maps, associated angular speed and optional vector components. Preserve valid relationships such as a unit direction vector. Account for encoder count rollover separately from physical direction.

**Coverage check:** crossing the angle boundary creates the correct wrapped discontinuity while physical angular velocity remains smooth; linked sine/cosine channels remain consistent.

### 15. Sensor errors that alter control behavior

**Status: some structural sensor clipping/quantization and PID sensor noise can affect feedback; the separate measurement-artifact layer does not.** That separation is useful, but it leaves a gap for other feedback-instrument faults.

**Missed examples:** a control sensor freezes while an independent reference sensor keeps moving; bias causes a controller to overheat a process; a calibration error shifts the actual controlled equilibrium.

**Suggestion:** distinguish controller measurements from archived measurements. Apply selected sensor faults before the controller's decision, while retaining optional independent reference channels. Keep historian-only faults outside the physical loop.

**Coverage check:** identical logger corruption leaves the process unchanged, whereas corruption of a feedback sensor changes actuator commands and subsequent true process state.

### 16. Physical forcing with boundaries and calendar semantics

**Status: cyclic/rest-day gates and formulaic inputs provide partial waveform coverage.** There is no explicit real calendar or constrained environmental forcing family.

**Missed examples:** solar generation exactly zero at night with cloud-driven fluctuations; weather-driven HVAC with occupancy; scheduled work interrupted by holidays; day-length changes across seasons.

**Suggestion:** optional timestamp-aware schedules and a few physical forcing templates with bounds, daylight/availability gates and coherent external drivers. Define which columns are future-known commands/calendar variables versus measurements unavailable at inference time. Fixed-grid models can still train on the resulting numeric traces, even if calendar metadata is kept separately.

**Coverage check:** night zeros, cloud transients and load/controller responses coexist; a known schedule and a noisy measured disturbance have distinct availability semantics.

## Coverage removed or narrowed by current rules

These are representation or acceptance restrictions, not requests to change mixture proportions:

| Current rule | Resulting coverage limit | Suggested design option |
|---|---|---|
| `REJECT_CONSTANT_OUTCOME=True` rejects constant/nearly constant clean targets. | Healthy equipment can be constant for an entire context; so can idle or fully saturated measurements. Some artifact-created flat outputs may still occur, so this is not a universal ban on flat tables. | An explicit constant/idle target family that bypasses this rejection when intentional. |
| `_sample_accepted_model` accepts clean trend-free proposals at linear R-squared <= 0.15, or trending proposals at saved trend-shape R-squared >= 0.4. | A valid process may fit neither criterion. Mixed transitions, weakly structured drift or irregular duty cycles can be rejected depending on their scores. | A general-process acceptance route based on validity rather than matching either trend profile. Do not infer missing coverage solely from a low sampling frequency. |
| `_revise_dynamics` caps seasonal periods relative to simulated span (`rows / 3`). | For a normal table without prehistory, clock seasonality cannot deliberately represent a cycle much longer than the context. Prehistory/cropping can already yield partial cycles, so this is a restriction rather than complete absence. | Draw a long physical trajectory and crop arbitrary contexts, independently of periodicity and event completion. |
| `_apply_artifacts` returns immediately for a zero-span clean signal. | A genuinely constant physical quantity cannot acquire recording noise, drift or failures through that helper. | Instruments with absolute engineering-unit noise/span specifications, independent of realized sample range. |
| Missingness is capped/thinned, the last 80 target rows are protected, and columns need at least two finite values. | No fully unavailable channel; limited severe target outages; no missing target tail from this mechanism. | Separate latent truth/evaluation labels from the observed measurement mask, so severe operational missingness need not destroy ground truth. |
| Row removal/subsampling resets `time_index`; no unit/sample-interval field is exported. | A short physical interval and a long one can become indistinguishable; timing and integral units are ambiguous. | Preserve sampling interval and original elapsed time in metadata, with an optional model-facing time representation. |
| The target and graph remain a fixed set of numeric columns within one table. | Device replacement, channels appearing/disappearing, schema changes and sensor identity changes are only approximated by missingness or recalibration. | Separate device/channel identity and validity intervals from the underlying physical variable; allow stitched operational records where appropriate. |

## Suggested order of work

This orders additions by how many qualitatively new dataset families they unlock, **not by their desired sampling probability**.

1. **Measurement semantics:** discrete/count targets, resettable counters, aggregated measurements, and explicit elapsed time. These are clear semantic holes that more random nonlinearities cannot repair.
2. **Coupled physical state:** a small inventory/thermal/battery/queue template library, producing consistent observable channels.
3. **Hybrid behavior:** guarded operating modes, exposure-driven degradation and repair, plus feedback-sensor faults.
4. **Dynamic signatures:** inverse response, backlash/stick-slip, state-dependent transport, and speed-coupled vibration.
5. **Specialized extensions:** circular/vector outputs, stochastic volatility/common interference and timestamp-aware environmental forcing.

Some clock/schema changes may need coordination with transformation and training code. That is a compatibility consideration, not a reason to count the corresponding coverage as present today.

## How to verify coverage before tuning any probabilities

Maintain a small deterministic coverage catalog. Force one mechanism at a time, plus combinations that matter: count target with a changing rate; process episode with irregular sampling; PID with a failed feedback sensor; totalizer with rollover; speed ramp with bearing impacts; conserved inventory with noisy instruments.

Each catalog entry should have a representative plot, its saved causal specification, a replay check, and a behavior-specific assertion. Test defining relationships (balance, integrality, travel time, phase, state guards), not merely a visual resemblance. This would distinguish an implemented capability from a mechanism that can be selected but is always rejected or never reaches export. No new probability targets are needed for this exercise.

## Code reviewed and evidence anchors

Links below are relative to this document. Function names are the durable anchors; line numbers may move.

- [Runner settings](../../../src/01_data_generation/generate_operational_data.py): active configuration and export choices.
- [SCM generator](../../../src/01_data_generation/scm_generator.py): `_sample_noise`, `_sample_lags`, `_lag_weights`, `_sample_process_driver`, `_revise_dynamics`, `_sample_recording`, `_simulate_process`, `_seasonal`, `_sample_accepted_model`, and both copy augmentations.
- [Effect mechanisms](../../../src/01_data_generation/effect_mechanisms.py): static/temporal operators, `evaluate_extra`, joint parent maps and links.
- [Process mechanisms](../../../src/01_data_generation/process_mechanisms.py): formula families, scheduled paths, discrete paths, episode envelope and sensor bounds.
- [Plant mechanisms](../../../src/01_data_generation/plant_mechanisms.py): the five plant families, eligible outcome/export channels, closed-loop equations and lifecycle shapes.
- [Measurement layer](../../../src/01_data_generation/measurement_layer.py): `_apply_artifacts`, `_missing_mask`, `_redundant`, `apply_observation` and `PROTECTED_TAIL`.
- [Mechanism tests](../../../tests/test_generator_realism.py), [effect tests](../../../tests/test_effect_mechanisms.py), and [process-driver tests](../../../tests/test_process_drivers.py): existing behavior and replay expectations.

External links support the measurement/physics examples. Assessments of current support and the proposed extensions are inferences from the repository source, not claims made by those external sources. No generator changes accompany this review.

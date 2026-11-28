# Annealing: a short final training phase on a chosen mix of tables

Written 2026-11-21. The code is in `src/10_annealing`; an example recipe is
`configs/annealing/flat25.json`.

**Status.** Building a set, creating an experiment and the holdout comparison
have been run on this machine. The training step itself has not: the GPU was
busy with the main run when this was written. It is covered by unit tests with
a stand-in for the trainer, and a prepared trial (`smoke`, see
[First run](#first-run-the-prepared-trial)) should be run once the GPU is free.

## The idea

After the main run has finished, training continues for a short time on a
small, fresh set of tables in which the kinds of series the model handles
least well are more common than usual. The learning rate starts where the main
run ended and falls to zero. The result is a second model next to the main
one. The main run's files are not changed, so an experiment that does not help
costs only its own time.

The first target is known from the holdout test after epoch 1
(`experiments/claude_review/flat_series_2026-11-21/RESULTS.md`):

| Kind of point | Share of holdout points | Basin, 10th percentile | Timestral, 10th percentile |
|---|---|---|---|
| Flat (equals the last observed value) | 4% | 0.021 | 0.000 |
| On a very smooth series | 11% | 0.39 | 0.24 |
| Other | 85% | 0.28 | 0.25 |
| First forecast patch (16 steps), all kinds | 14% | 0.13 | 0.07 |

Numbers are pointwise absolute errors in MASE units; lower is better. On flat
points Basin forecasts "stays flat" but lands slightly beside the level. That
is a precision problem, and Timestral shows that the error is not a property of
the data. This is the situation in which extra practice at a low learning rate
can be expected to help.

## What others do, and what follows for the set

- **Llama 3** changed the data mix towards its highest-quality sources for the
  last 40M tokens while the learning rate went to zero, and averaged the
  weights over that phase. It reports clear gains from annealing on small
  domain sets for its 8B model and none for its 405B model
  ([paper](https://arxiv.org/pdf/2407.21783), section 3.4.3).
- **MiniCPM** trains at a constant rate on the bulk of its data and then decays
  the rate quickly over high-quality data mixed with ordinary pretraining data
  ([paper](https://arxiv.org/abs/2404.06395)). Other models use the same
  two-phase pattern ([survey](https://arxiv.org/html/2510.23081v1)).
- **Mixing in ordinary data protects what the model already knows.** Studies of
  fine-tuning find that replaying generic pretraining data reduces forgetting
  and often improves the target task as well; in one of them a 1% share already
  recovered most of the loss ([replay study](https://arxiv.org/pdf/2603.04964),
  [mixing during pretraining](https://iclr.cc/virtual/2026/10012797)).
- **Selecting the examples with the highest loss is a trap.** High loss marks
  noise as readily as it marks something learnable, and the noise cannot be
  learned ([When does loss-based prioritization fail?](https://arxiv.org/abs/2107.07741),
  [Hardness-based resampling](https://arxiv.org/html/2504.07031v1)). In
  forecasting most of the high-loss series are simply unpredictable.
- **Oversampling changes less than its factor suggests.** In deep networks the
  effect of weighting examples fades as training goes on
  ([Byrd and Lipton](https://arxiv.org/abs/1812.03372)). A short phase at a low
  rate is the regime where the mix still matters, and a reason not to expect
  miracles from it.

The set is built accordingly:

1. **Categories are defined by a rule on the series, not by the model's loss.**
   A category is added only where there is evidence that the error can be
   reduced (another model does better on it).
2. **A table is chosen for what it is as a whole, never for what follows a
   forecast origin.** In the holdout, after a flat run of 5 to 16 points only
   35% of the following points are still flat. A set of flats that always
   continue would teach the model to over-predict flatness. Inside a chosen
   table, flat stretches begin and end as the generator makes them.
3. **Most of the set is ordinary.** The `standard` share is drawn exactly like
   the training corpus and protects the median and 90th percentile, where
   Basin already leads.
4. **All tables are new.** They come from the training generator with a new
   seed. None is taken from the corpus, its validation tables or the holdout,
   and nothing is selected by looking at holdout results.
5. **Oversampling means drawing more, not repeating.** The generator keeps
   running and tables of the wanted kind are kept until their share is full.
   Every kept table is an ordinary draw of the generator.

One limit of rule 2: tables with long flat stretches are more likely to pass
the `flat` rule than tables with short ones, so long stretches are somewhat
over-represented in that share. The holdout comparison is the check on this.

## Proposal

`configs/annealing/flat25.json`:

| Setting | Value | Reason |
|---|---|---|
| Tables | 200,000 | About 24,700 updates. At the main run's measured 30,000 updates per hour that is roughly 50 minutes of training (estimate). |
| Mix | 65% standard, 25% flat, 10% smooth | Flat is the largest gap; smooth is the second. |
| Start | best validated weights of the final checkpoint | The weights the holdout test of the main run evaluates. |
| Learning rate | 1e-5, 3% warm-up, cosine to 0 | The main run ends at 1e-5. The warm-up covers the fresh optimizer. |
| Passes over the set | 1 | No table is seen twice. |

Measured on 3,000 generated tables: 15.5% are `flat` and 12.9% are `smooth` by
the default rules. The standard share contains them at those rates, so the
finished set is about 35% flat (2.3 times the natural share) and about 18%
smooth (1.4 times). The exact shares of a built set are in its
`annealing_set.json`.

This is a starting point. Whether it helps, and by how much, is not known
until it has been run. Mixes worth trying after it, one change at a time:
flat at 40%; no smooth share; twice the tables; half the learning rate.

## How to run it

Run from the repository root. The first three steps use the CPU only.

```powershell
# 1. Draw the set and create the experiment folder.
uv run python src/10_annealing/build_annealing_set.py --recipe configs/annealing/flat25.json

# 2. Prepare the set for training (skip both if an earlier experiment already used this set).
uv run python src/03_transforming_input_data/transform.py --config results/model/annealing/flat25_lr1e-5/config.json
uv run python src/04_patch_indexing/index_patches.py --config results/model/annealing/flat25_lr1e-5/config.json

# 3. Train (GPU, routes itself to WSL like train.py).
uv run python src/10_annealing/anneal.py --experiment flat25_lr1e-5

# 4. Test on the main run's holdout tables (GPU), then compare (CPU).
uv run python src/06_holdout_test/holdout_test.py --config results/model/annealing/flat25_lr1e-5/config.json
uv run python src/10_annealing/compare_holdout.py --experiment flat25_lr1e-5
```

Step 1 prints the commands for the steps after it. Drawing the example set
needs about 450,000 generated tables. With three generator processes next to a
running training this machine drew 28 tables per second, which would be about
four and a half hours; on an idle machine with six processes expect about a
third of that (estimate from the generator's measured 76 tables per second).

If step 3 is interrupted, run the same command again: the experiment continues
from its own last checkpoint.

For the comparison, the main model must have a holdout run of the weights the
experiment started from. Run `holdout_test.py` with `configs/config.json` once
after the main run has finished; `compare_holdout.py` finds that run by the
checksum of the weights.

### Writing a recipe

```json
{
  "experiment": "flat40_lr1e-5",
  "description": "free text",
  "base_config": "configs/config.json",
  "set": {
    "name": "flat40_200k",
    "tables": 200000,
    "seed": 20261122,
    "mix": {"standard": 0.6, "flat": 0.4},
    "categories": {"flat": {"min_longest_run": 32}}
  },
  "start": {"checkpoint": null, "weights": "best"},
  "training": {"learning_rate": 1e-05}
}
```

- **`experiment`** names the results folder. A name can be used once.
- **`set.name`** names the data folder. Experiments that use the same set
  definition share the set, so trying another learning rate does not need new
  tables: change `experiment` and `training`, keep `set`.
- **`set.mix`** gives the share of each category and must add up to 1.
  `standard` is the natural distribution.
- **`set.categories`** changes the rules (optional). Defaults:

  | Category | Rule on the target series |
  |---|---|
  | `flat` | at least 5% of steps repeat the previous value (`min_flat_share`), and the longest such stretch has at least 16 steps (`min_longest_run`) |
  | `smooth` | mean absolute step at most 3% of the standard deviation (`max_roughness`), and fewer than 5% repeated steps (`max_flat_share`) |

- **`start.weights`**: `best` (the validated weights the holdout test uses) or
  `last` (the weights after the final update). `start.checkpoint` names another
  checkpoint than the base configuration's `latest_checkpoint.pt`.
  A checkpoint of a run that is still going is refused unless
  `start.allow_unfinished` is true.
- **`training`** overrides training settings. Anything not named is taken from
  the annealing defaults (below) and otherwise from the base configuration.

Annealing defaults that differ from the main run: learning rate 1e-5; warm-up
over 3% of the updates, then cosine to 0; one pass; validation and checkpoint
every 30 minutes of training; validation on 2,000 tables of the set and 500
tables set aside as the set's own test tables; no early stop.

## How to judge an experiment

`compare_holdout.py` prints the headline holdout numbers before and after, and
the pointwise error split into flat, smooth and other points and the first
forecast patch, next to Chronos-2 and Timestral. Both runs forecast the same
points of the same sealed holdout tables.

An experiment is a success if the flat and smooth rows and the pointwise 10th
percentile improve while the table MASE median and 90th percentile do not get
worse. If the median or 90th percentile gets worse, the mix was too far from
the natural one or the phase too long.

Two cautions:

- **The holdout is used for selection once you compare several mixes on it.**
  The best of many experiments looks better on the holdout than it is. Keep the
  number of experiments small, prefer differences that are large and consistent
  across the rows, and confirm the chosen recipe on an independent sample:
  run the holdout test of both models once more with another
  `holdout.sample_seed`, each from a copy of its configuration, and compare
  those two runs with `--baseline`.
- **Validation during annealing uses tables of the set**, so its numbers are
  for the mix, not for the natural distribution, and cannot be compared with
  the main run's validation history. It selects the exported weights: the check
  with the best value. If no check beats the start weights (`best_step` 0 in
  `training/early_stopping.json`), `model.pt` holds the last training weights,
  as in the main trainer, and the experiment should be treated as failed.

## Where things are

```text
configs/annealing/<recipe>.json          the recipe (kept in git)
data/04_annealing/<set name>/
  generated/<date>/table_*/data.parquet  the tables
  tables.parquet                         one row per table: category, statistics, seed, checksum
  annealing_set.json                     definition, counts, natural and effective shares
  processed/, patch_index/               written by transform.py and index_patches.py
results/model/annealing/<experiment>/
  recipe.json, config.json               what was asked for, and the derived pipeline configuration
  training/                              as in the main run, plus annealing_start.json
  holdout_test/<run>/                    one folder per holdout run
  logs/
```

**Nothing is overwritten.**

- A set or experiment folder is only ever created. A name that exists with
  other settings stops the command with an error; the same settings reuse it.
- An unfinished set (a build that was interrupted) is reported and left alone.
  Remove the folder yourself, or use a new name.
- Every output path of an experiment lies inside its own folder: checkpoints,
  holdout runs, forecasts, FakeBench results and logs. Checkpoint uploads, if
  enabled, go to a blob named after that folder.
- The main run's checkpoint is opened for reading and closed again before
  training starts.

## How it works

**Drawing the set** (`annealing_set.py`, `table_categories.py`). The generator
is built from `generate_operational_data.generator_settings()`, the same
settings as the training corpus, with the extra copies (coarse, aggregate,
sign-inverted) switched off so that every table is independent. Candidate
number *i* gets its own random stream from the set seed and *i*. The first
candidates fill the standard share unconditionally. After that each candidate
is classified from its target column, and kept if it belongs to a category
that still has room. Results are applied in candidate order, so a set depends
only on its definition, not on the number of processes or on timing. Tables
are written by the generator's own writer. Finally the checksums of the kept
tables are compared with those of the main run's holdout tables; one identical
table stops the build. `annealing_set.json` is written last and marks the set
as complete.

**The experiment's configuration** (`annealing_experiment.py`). `config.json`
is the base configuration with the data paths pointed at the set, the output
paths pointed at the experiment, and the training settings merged from the
base, the annealing defaults and the recipe. `training.resume` is always
`auto`, which here can only mean the experiment's own checkpoint. Because it is
an ordinary pipeline configuration, the existing transform, indexing, holdout
and FakeBench stages work on it unchanged.

**Starting from the main run** (`anneal.py`, `annealing_start.py`). The
weights are read from the main checkpoint (`best_state_dict` or `state_dict`),
after checking that it describes the same architecture and that its run has
finished. Their checksum, the same fingerprint the holdout stage records as
`evaluated_weights_sha256`, is saved in `training/annealing_start.json`. Then
the normal `train()` runs with the experiment's configuration. The weights are
put in place at the moment `train()` builds its optimizer: after the model has
been initialised, before the optimizer copies the weights into its 32-bit
masters and before the weight average takes its starting copy.

Only the weights are carried over. The optimizer starts empty, which is what
the warm-up is for, and the learning-rate schedule and update counter start at
zero. Taking the optimizer state along as well would be a possible refinement;
it was left out because the saved state belongs to the last training weights,
not to the best validated ones.

`anneal.py` reaches into `train()` by replacing `train.make_optimizer` for the
duration of the call, because the training code could not be edited while the
main run was using it. A test fails if `train()` stops building its optimizer
that way, and `anneal.py` raises an error if training ends without the weights
having been loaded. Once no run is active, a `train(config, initial_weights=...)`
parameter would be the cleaner form.

**Testing on the main holdout** (`holdout_test.py`). The experiment's
configuration names the main checkpoint as `holdout.split_checkpoint` and the
main configuration as `holdout.split_config`. The holdout stage loads the
annealed model from the experiment, takes the test tables from the main run's
saved split, checks that split against the main configuration, and refuses to
run if any test table is among the experiment's training tables. With the same
`sample_percent` and `sample_seed` it forecasts the same tables as the main
model's holdout run.

## First run: the prepared trial

A tiny experiment is ready to exercise the training step end to end once the
GPU is free: set `smoke_300` (300 tables, already transformed and indexed) and
experiment `smoke`. It starts from the main checkpoint even if that run is not
finished, so it also works between epochs while the main run is stopped.

```powershell
uv run python src/10_annealing/anneal.py --experiment smoke
```

It should run 30 updates in a few minutes. It inherits the main run's batch
settings and its 11 GiB GPU memory limit, so it should need no more memory
than the main run; that has not been measured. Check in the log that the line
`ANNEALING smoke starts from ...` appears and that the validation before the
first update gives plausible errors (a freshly initialised model would give
errors many times larger). Do not start it while the main run is training.

## Tests

`tests/test_annealing.py` (24 tests, CPU): category rules, recipe checks, the
derived configuration, that sets and experiments are never overwritten, the
mix and reproducibility of a drawn set, the holdout-overlap stop, reading the
start weights, and the order in which they are loaded.
`tests/test_holdout_stage.py` has one more test for `holdout.split_config`.

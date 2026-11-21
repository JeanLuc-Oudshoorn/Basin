# Annealing stage

A short final training phase after the main run, on a small fresh set of
tables in which chosen kinds of series (flat stretches, very smooth signals)
are more common than in the training corpus. Every experiment has its own
folders and never changes the main run's files.

The idea, a proposed first experiment, how to run and judge it, and how it is
implemented are in [docs/annealing/ANNEALING.md](../../docs/annealing/ANNEALING.md).

```powershell
uv run python src/10_annealing/build_annealing_set.py --recipe configs/annealing/flat25.json
uv run python src/03_transforming_input_data/transform.py --config results/model/annealing/flat25_lr1e-5/config.json
uv run python src/04_patch_indexing/index_patches.py --config results/model/annealing/flat25_lr1e-5/config.json
uv run python src/10_annealing/anneal.py --experiment flat25_lr1e-5
uv run python src/06_holdout_test/holdout_test.py --config results/model/annealing/flat25_lr1e-5/config.json
uv run python src/10_annealing/compare_holdout.py --experiment flat25_lr1e-5
```

| File | Purpose |
|---|---|
| `build_annealing_set.py` | draw the set of a recipe and create the experiment folder (CPU) |
| `anneal.py` | train the experiment, starting from the main run's weights (GPU, WSL) |
| `compare_holdout.py` | holdout results before and after, split by kind of point (CPU) |
| `annealing_experiment.py` | recipes, folders, derived configurations |
| `annealing_set.py` | drawing tables from the training generator in a chosen mix |
| `table_categories.py` | the rules that define `flat` and `smooth` |
| `annealing_start.py` | reading the start weights and handing them to the trainer |

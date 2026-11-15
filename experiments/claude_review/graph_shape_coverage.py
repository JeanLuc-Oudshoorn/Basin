"""How many CUDA-graph shapes does the new corpus need, and how many forecasts replay?

CPU only: samples table metadata (rows, variables) from the production generator
settings without simulating, then applies the pipeline's own shape rules. Writes
graph_shape_plan.json, the input of inference_graph_memory.py.

Usage: python graph_shape_coverage.py [tables]
"""
# ruff: noqa: E402

import json
import math
import sys
from collections import Counter
from pathlib import Path
from types import SimpleNamespace

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
for stage in (
    "01_data_generation",
    "03_transforming_input_data",
    "05_model_training",
    "06_holdout_test",
):
    sys.path.insert(0, str(ROOT / "src" / stage))
sys.path.insert(0, str(ROOT / "experiments" / "claude_review"))

from generator_coverage import production_generator
from graph_profiles import profile_counts

generator = production_generator()
config = json.loads((ROOT / "configs" / "config.json").read_text())
model = SimpleNamespace(
    context_length=config["model"]["context_length"],
    patch_size=config["model"]["patch_size"],
    width=config["model"]["width"],
    output_patches=config["model"]["output_patches"],
)
count = int(sys.argv[1]) if len(sys.argv) > 1 else 7500
tables = []
for seed in range(count):
    generator.rng = np.random.default_rng(seed)
    spec = generator.sample_model(trend_free=seed % 2 == 0)
    rows = spec.get("recorded_rows", spec["rows"])
    variables = 1 + sum(n["visible"] for n in spec["nodes"])
    variables += len(spec.get("plant", {}).get("exports", []))
    variables += len(spec.get("observation", {}).get("redundant", []))
    tables.append((rows, variables))
rows = np.array([t[0] for t in tables])
print(
    f"tables={count} rows mean={rows.mean():.0f} at_max={np.mean(rows == rows.max()):.3f}"
)
print("variables per table:", dict(sorted(Counter(t[1] for t in tables).items())))

STANDARD = (2048, 1536, 1024, 512)


def kernel_bucket(points):
    """What the pipeline does today: next multiple of 16 patches (256 points)."""
    return math.ceil(math.ceil(points / 16) / 16) * 256


def requested_rule(points):
    """Pad to a standard length when at most 10% shorter; None = not covered."""
    for length in STANDARD:
        if 0.9 * length <= points <= length:
            return length
    return None


# 1. Do the two rules ever disagree for a length the requested rule covers?
disagree = [
    p for p in range(32, 2049) if requested_rule(p) not in (None, kernel_bucket(p))
]
print("lengths where the 10% rule and the kernel bucket differ:", len(disagree))
print(
    "kernel bucket windows:",
    {
        edge: f"{edge - 255}-{edge} ({255 / edge:.0%} shorter at most)"
        for edge in STANDARD
    },
)

# 2. Training: one chunk per table; shape = (variables, kernel patches).
training = Counter()
real = padded = 0
for length, variables in tables:
    tokens = math.ceil(min(length, 2064) / 16) - 1
    training[(variables, math.ceil(tokens / 16) * 16)] += 1
    real += tokens * variables
    padded += math.ceil(tokens / 16) * 16 * variables
top = training.most_common()
print(f"training: distinct (variables, patches) shapes={len(training)}")
for share in (0.5, 0.9, 0.99):
    total, used = 0, 0
    for _, n in top:
        total += n
        used += 1
        if total >= share * count:
            break
    print(f"  {share:.0%} of tables fall in the {used} most common shapes")
print(f"  padding overhead: {padded / real - 1:.1%} extra variable-patches computed")
print(
    f"  share of tables already at 128 patches: {sum(n for (v, t), n in training.items() if t == 128) / count:.3f}"
)

# 3. Holdout: the pipeline's own plan (five origins per table, B=1).
records = [
    {
        "targets": ["outcome"],
        "roles": {"past": [f"covariate_{i:02d}" for i in range(v - 1)]},
        "source_rows": int(n),
    }
    for n, v in tables
    if n >= (model.output_patches + 6) * model.patch_size
]
counts = profile_counts(records, model, config, "cuda")
minimum = config["holdout"]["graph_min_profile_requests"]
admitted = {k: n for k, n in counts.items() if n >= minimum}
print(
    f"holdout: tables={len(records)} requests={sum(counts.values())} "
    f"unique_profiles={len(counts)} admitted(>={minimum})={len(admitted)} "
    f"planned_replay_share={sum(admitted.values()) / sum(counts.values()):.4f} "
    f"cache_limit={config['training']['inference_graph_cache_size']}"
)
by_length = Counter()
for (shape, _, _, _), n in counts.items():
    by_length[shape[2] * 16] += n
print("holdout requests by padded context length:", dict(sorted(by_length.items())))

plan = sorted(
    (
        {"variables": shape[1], "patches": shape[2], "requests": n}
        for (shape, _, _, _), n in counts.items()
    ),
    key=lambda row: -row["requests"],
)
(ROOT / "experiments" / "claude_review" / "graph_shape_plan.json").write_text(
    json.dumps(
        {"tables": len(records), "minimum_requests": minimum, "profiles": plan},
        indent=1,
    )
)

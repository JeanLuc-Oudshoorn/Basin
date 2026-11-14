"""Plot outcomes of tables sampled with the production settings (CPU, nothing saved to data/).

Usage: python generator_contact_sheet.py [first_seed] [count]
"""

import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from generator_coverage import production_generator


def label(model):
    parts = [model.get("noise_profile", {}).get("name", "")]
    if "plant" in model:
        parts.append(
            f"plant:{model['plant']['family']}/{model['plant']['outcome_channel']}"
        )
    trend = model["outcome"].get("trend")
    if trend:
        parts.append(f"trend:{trend['kind']}")
    if model["outcome"].get("operation"):
        parts.append(f"op:{model['outcome']['operation']['mode']}")
    if model["outcome"]["seasons"]:
        parts.append("seasonal")
    if "recorded_rows" in model:
        parts.append(f"pre:{model['rows'] - model['recorded_rows']}")
    observation = model.get("observation", {})
    parts += [a["kind"] for a in observation.get("artifacts", {}).get("outcome", [])]
    parts += [
        f"NaN:{m['paradigm']}/{m.get('pattern', m.get('driver', ''))}"
        for m in observation.get("missing", [])
    ]
    if "row_gaps" in model:
        parts.append("row_gaps")
    return " ".join(p for p in parts if p)


def main():
    first = int(sys.argv[1]) if len(sys.argv) > 1 else 0
    count = int(sys.argv[2]) if len(sys.argv) > 2 else 40
    generator = production_generator()
    figure, axes = plt.subplots(count // 4, 4, figsize=(26, 2.1 * (count // 4)))
    for axis, seed in zip(axes.ravel(), range(first, first + count)):
        generator.rng = np.random.default_rng(seed)
        model, frame = generator._sample_accepted_model(seed % 2 == 0)
        axis.plot(frame["outcome"].to_numpy(), linewidth=0.6)
        axis.set_title(f"{seed}: {label(model)}", fontsize=7)
        axis.tick_params(labelsize=6)
    figure.tight_layout()
    target = Path(__file__).with_name(f"generator_contact_sheet_{first}.png")
    figure.savefig(target, dpi=62)
    print(target)


if __name__ == "__main__":
    main()

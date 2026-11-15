"""Is the training loss dominated by a few flat-then-spike examples?

Runs a trained model over a sample of training tables exactly as the trainer
does (every causal anchor, every lead) and attributes the pinball loss to
individual labels. A label's "standardised size" is |label - mean so far| /
(standard deviation so far): a spike after a near-constant history has a huge
one. Reports, per size class, the share of labels, of the loss value and of the
gradient that reaches the forecast head.

The gradient share is the informative one. Pinball loss has a bounded slope:
however far a label lies from the forecast, each quantile output receives a
gradient of at most its quantile level. Loss value and gradient can therefore
be concentrated very differently.

    source scripts/activate_rwkv7.sh
    python experiments/claude_review/training_loss_concentration.py --config <config.json>
"""
# ruff: noqa: E402

import argparse
import json
import sys
from pathlib import Path

import torch
from torch.nn import functional as F

ROOT = Path(__file__).resolve().parents[2]
for directory in (
    ROOT,
    ROOT / "src/03_transforming_input_data",
    ROOT / "src/04_creating_embeddings",
    ROOT / "src/05_model_training",
):
    sys.path.insert(0, str(directory))
from multi_horizon import lead_quantiles, to_model_space
from train import RWKVForecaster
from transform import load_config, location, read_artifact

EDGES = [0.0, 3.0, 10.0, 100.0, 1e4, float("inf")]
NAMES = ["<= 3", "3 - 10", "10 - 100", "100 - 10,000", "> 10,000"]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True)
    parser.add_argument("--tables", type=int, default=3000)
    parser.add_argument("--output", default=None)
    args = parser.parse_args()
    config = load_config(args.config)
    model = RWKVForecaster.load(location(config, "training") / "model.pt").eval()
    settings = model.config
    leads, levels = settings.output_patches, torch.tensor(settings.quantiles).cuda()
    data = read_artifact(location(config, "embedded") / "embeddings.pt", config)
    records = [r for r in data["records"] if r["split"] == "train"]
    step = max(1, len(records) // args.tables)
    edges = torch.tensor(EDGES).cuda()
    count = torch.zeros(len(NAMES), dtype=torch.float64).cuda()
    loss = torch.zeros_like(count)
    gradient = torch.zeros_like(count)
    empty_history = torch.zeros(3, dtype=torch.float64).cuda()  # labels, loss, gradient
    table_loss, table_labels = [], []
    used = 0
    with torch.inference_mode():
        for record in records[::step]:
            prepared = {
                k: v.cuda() if isinstance(v, torch.Tensor) else v
                for k, v in record["prepared"].items()
            }
            targets, p = prepared["targets"], settings.patch_size
            labels = prepared["labels"]
            if not torch.isfinite(labels).any():
                continue
            x = model.encode_tokens(prepared)
            b, v, t, d = x.shape
            v_first = torch.empty_like(x.reshape(b * v, t, d))
            for layer, block in enumerate(model.blocks):
                flat, v_first = block(x.reshape(b * v, t, d).contiguous(), v_first)
                x = model.mix(layer, flat.reshape(b, v, t, d))
            n = prepared["tokens"]
            loc, scale = prepared["loc"][:, :targets], prepared["scale"][:, :targets]
            quantiles = lead_quantiles(model, x[:, :targets, :n])
            future = F.pad(labels, (0, 0, 0, leads - 1), value=float("nan"))
            future = future.unfold(2, leads, 1).permute(0, 1, 2, 4, 3)
            valid = torch.isfinite(future)
            standardized = (
                (future.float() - loc[..., None]) / scale[..., None]
            ).nan_to_num()
            error = to_model_space(model, standardized)[..., None] - quantiles
            pinball = torch.maximum(levels * error, (levels - 1) * error).mean(-1)
            # Size of the gradient each label sends into the head's nine outputs.
            slope = torch.where(error > 0, levels, 1 - levels).mean(-1)
            size = standardized.abs()
            bucket = torch.bucketize(size, edges[1:-1], right=False)
            for index in range(len(NAMES)):
                chosen = valid & (bucket == index)
                count[index] += chosen.sum()
                loss[index] += (pinball * chosen).sum()
                gradient[index] += (slope * chosen).sum()
            # Anchors whose target has not been observed at all so far.
            observed = (
                prepared["features"][:, :targets, :n, p : 2 * p].sum(-1).cumsum(-1)
            )
            unseen = valid & (observed == 0)[..., None, None]
            empty_history += torch.stack(
                [unseen.sum(), (pinball * unseen).sum(), (slope * unseen).sum()]
            )
            table_loss.append(float((pinball * valid).sum()))
            table_labels.append(int(valid.sum()))
            used += 1
    total_count, total_loss, total_gradient = count.sum(), loss.sum(), gradient.sum()
    rows = []
    for index, name in enumerate(NAMES):
        rows.append(
            {
                "standardised_label_size": name,
                "share_of_labels": float(count[index] / total_count),
                "share_of_loss_value": float(loss[index] / total_loss),
                "share_of_head_gradient": float(gradient[index] / total_gradient),
                "mean_loss_per_label": float(loss[index] / count[index].clamp_min(1)),
            }
        )
    ordered = sorted(zip(table_loss, table_labels), reverse=True)
    top = max(1, used // 100)
    report = {
        "config": args.config,
        "value_transform": settings.value_transform,
        "tables": used,
        "labels": int(total_count),
        "mean_loss": float(total_loss / total_count),
        "by_label_size": rows,
        "worst_1_percent_of_tables": {
            "tables": top,
            "share_of_loss_value": sum(x for x, _ in ordered[:top]) / float(total_loss),
            "share_of_labels": sum(y for _, y in ordered[:top]) / int(total_count),
        },
        "anchors_without_any_observed_history": {
            "share_of_labels": float(empty_history[0] / total_count),
            "share_of_loss_value": float(empty_history[1] / total_loss),
            "share_of_head_gradient": float(empty_history[2] / total_gradient),
        },
    }
    print(json.dumps(report, indent=2))
    if args.output:
        Path(args.output).write_text(json.dumps(report, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()

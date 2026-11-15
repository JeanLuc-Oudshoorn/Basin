"""Does CUDA-graph replay speed up validation batches, and is it still correct
after the weights change?

For every planned shape (graph_shape_plan.json) one validation batch of
`validation_batch_size` requests is run through the plain forward and through
replay on an UNTRAINED model built from configs/config.json. Then every weight
is changed in place, as an optimizer update or the weight average does, and
both paths are compared again WITHOUT recapturing.

Run inside the WSL CUDA environment with no training job on the GPU:

    source scripts/activate_rwkv7.sh
    python experiments/claude_review/validation_graph_replay.py
"""
# ruff: noqa: E402

import json
import statistics
import sys
from pathlib import Path
from time import perf_counter

import torch

ROOT = Path(__file__).resolve().parents[2]
for directory in (
    ROOT,
    ROOT / "src/03_transforming_input_data",
    ROOT / "src/04_creating_embeddings",
    ROOT / "src/05_model_training",
):
    sys.path.insert(0, str(directory))
from embed import ForecastConfig
from multi_horizon import weighted_forecast_loss
from rolling_validation import validation_engine
from train import RWKVForecaster
from variate_attention import configure_variate_execution


def batch(model, profile, size, generator):
    patch = model.config.patch_size
    length = profile["patches"] * patch
    # Real lengths differ inside a bucket; the shortest is one patch above the
    # previous bucket. Shorter rows are right-padded with missing values, which
    # tokenization masks, and each row keeps its own last real patch.
    lengths = torch.randint(
        length - 15 * patch, length + 1, (size,), generator=generator
    )
    lengths[0] = length
    lengths = lengths.clamp_min(2 * patch)
    series = torch.randn(
        size, profile["variables"], length, generator=generator
    ).cumsum(-1)
    prepared = model.tokenize_next(
        series[:, :1],
        past_covariates=series[:, 1:] if profile["variables"] > 1 else None,
    )
    prepared["last_token_indices"] = (lengths + patch - 1) // patch - 1
    prepared["prediction_patches"] = model.config.output_patches
    prepared.update(is_training=False, labels=None)
    labels = torch.randn(
        size, 1, model.config.output_patches * patch, generator=generator
    )
    move = lambda v: v.cuda() if isinstance(v, torch.Tensor) else v  # noqa: E731
    return {k: move(v) for k, v in prepared.items()}, labels.cuda()


def plain(model, prepared, labels):
    output = model.forward_encoded(model.encode_tokens(prepared), prepared, labels)
    return output.point.float(), output.per_sample_loss.float()


def replayed(model, engine, prepared, labels):
    block = engine.scored_block(prepared)
    median = model.config.quantiles.index(0.5)
    loss = weighted_forecast_loss(
        model,
        block[0],
        labels,
        block[-2],
        block[-1],
        normalized=block[1] if len(block) == 4 else None,
    )[1]
    return block[0][..., median].float(), loss.float()


def milliseconds(function, repeats):
    seconds = []
    for _ in range(repeats):
        torch.cuda.synchronize()
        start = perf_counter()
        function()
        torch.cuda.synchronize()
        seconds.append(perf_counter() - start)
    return statistics.median(seconds) * 1000


def gap(a, b):
    """Largest difference between two paths, as a share of the value spread."""
    return float((a - b).abs().max() / a.std().clamp_min(1e-9))


@torch.inference_mode()
def main():
    here = Path(__file__).resolve().parent
    config = json.loads((ROOT / "configs/config.json").read_text())
    plan = json.loads((here / "graph_shape_plan.json").read_text())
    training = dict(
        config["training"],
        validation_cuda_graphs=True,
        # Graphs for batches are large: an unlimited cache overflows the card.
        validation_graph_memory_gib=float(sys.argv[1]) if len(sys.argv) > 1 else 4.0,
    )
    size = training["validation_batch_size"]
    settings = dict(config["model"], quantiles=tuple(config["model"]["quantiles"]))
    torch.manual_seed(config["seed"])
    model = RWKVForecaster(ForecastConfig(**settings)).to(
        device="cuda", dtype=torch.bfloat16
    )
    configure_variate_execution(model, training, ROOT)
    model.eval()
    engine = validation_engine(model, {"training": training})
    graphs = engine.graphs
    generator = torch.Generator().manual_seed(5)
    rows, batches = [], []
    limit = int(sys.argv[2]) if len(sys.argv) > 2 else 12
    for profile in plan["profiles"][:limit]:
        # Most frequent shapes only, and stop well short of the card's memory:
        # spilling into shared GPU memory makes every timing meaningless.
        if torch.cuda.memory_reserved() > 6 * 2**30:
            break
        prepared, labels = batch(model, profile, size, generator)
        before, started, captures = (
            graphs.retained_bytes,
            perf_counter(),
            graphs.captures,
        )
        replayed(model, engine, prepared, labels)
        torch.cuda.synchronize()
        capture = perf_counter() - started
        point, loss = plain(model, prepared, labels)
        fast_point, fast_loss = replayed(model, engine, prepared, labels)
        rows.append(
            {
                **profile,
                "captured": graphs.captures > captures,
                "graph_mib": (graphs.retained_bytes - before) / 2**20,
                "capture_seconds": capture,
                "plain_ms": milliseconds(lambda: plain(model, prepared, labels), 5),
                "replay_ms": milliseconds(
                    lambda: replayed(model, engine, prepared, labels), 5
                ),
                "point_gap": gap(point, fast_point),
                "loss_gap": float((loss - fast_loss).abs().max() / loss.mean()),
            }
        )
        batches.append((prepared, labels))
        print(json.dumps(rows[-1]), flush=True)
    # An optimizer update and the weight average both write weights in place.
    captures = graphs.captures
    for parameter in model.parameters():
        parameter.mul_(1.02).add_(torch.randn_like(parameter) * 0.01)
    moved, stale = [], []
    for row, (prepared, labels) in zip(rows, batches):
        point, loss = plain(model, prepared, labels)
        fast_point, fast_loss = replayed(model, engine, prepared, labels)
        row["point_gap_after_update"] = gap(point, fast_point)
        stale.append(row["point_gap_after_update"])
        moved.append(float((loss - fast_loss).abs().max() / loss.mean()))
    weights = [r["requests"] for r in rows]
    total = sum(p["requests"] for p in plan["profiles"])

    def mean(name, captured=True):
        chosen = [(r, w) for r, w in zip(rows, weights) if r["captured"] == captured]
        return sum(r[name] * w for r, w in chosen) / max(1, sum(w for _, w in chosen))

    captured = sum(w for r, w in zip(rows, weights) if r["captured"])

    result = {
        "parameters": sum(p.numel() for p in model.parameters()),
        "batch_size": size,
        "gpu": torch.cuda.get_device_name(0),
        "shapes": len(rows),
        "request_share_in_measured_shapes": sum(weights) / total,
        "captured_shapes": sum(r["captured"] for r in rows),
        "request_share_replayed": captured / total,
        "graph_budget_gib": training["validation_graph_memory_gib"],
        "peak_reserved_gib": torch.cuda.max_memory_reserved() / 2**30,
        "plain_ms_per_batch_uncaptured_shapes": mean("plain_ms", False),
        "graph_gib": graphs.retained_bytes / 2**30,
        "reserved_gib": torch.cuda.memory_reserved() / 2**30,
        "capture_seconds_total": sum(r["capture_seconds"] for r in rows),
        "plain_ms_per_batch": mean("plain_ms"),
        "replay_ms_per_batch": mean("replay_ms"),
        "max_point_gap": max(r["point_gap"] for r in rows),
        "max_loss_gap": max(r["loss_gap"] for r in rows),
        "recaptures_after_weight_update": graphs.captures - captures,
        "max_point_gap_after_weight_update": max(stale),
        "max_loss_gap_after_weight_update": max(moved),
        "profiles": rows,
    }
    (here / "validation_graph_replay_output.json").write_text(
        json.dumps(result, indent=1)
    )
    print(json.dumps({k: v for k, v in result.items() if k != "profiles"}, indent=1))


if __name__ == "__main__":
    main()

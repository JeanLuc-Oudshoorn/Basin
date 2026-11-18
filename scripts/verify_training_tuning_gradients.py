"""Compare full-model BF16 gradients at identical weights across execution policies."""

# ruff: noqa: E402
import argparse
import json
import math
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
for stage in (
    "03_transforming_input_data",
    "04_patch_indexing",
    "05_model_training",
):
    sys.path.insert(0, str(ROOT / "src" / stage))
sys.path.insert(0, str(ROOT))
from runtime_storage import configure_runtime_storage

configure_runtime_storage(ROOT)
import torch
from epoch_training import _accumulated_gradients
from index_patches import LazyPatchRecord
from performance import configure_cuda_memory, make_optimizer, prepare_update
from train import ForecastConfig, RWKVForecaster
from transform import load_config, write_json
from variate_attention import configure_variate_execution


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--token-limit", type=int, required=True)
    parser.add_argument("--threshold", type=int, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    config = load_config()
    torch.set_num_threads(1)
    torch.manual_seed(config["seed"])
    configure_cuda_memory(config["training"])
    model = (
        RWKVForecaster(
            ForecastConfig(
                **dict(config["model"], quantiles=tuple(config["model"]["quantiles"]))
            )
        )
        .cuda()
        .bfloat16()
        .train()
    )
    configure_variate_execution(model, config["training"], ROOT)
    optimizer = make_optimizer(model, config["training"])
    sample = torch.load(
        ROOT / ".cache/training_tuning/live_sample.pt", weights_only=True
    )
    by_id = {record["id"]: LazyPatchRecord(record, model) for record in sample["train"]}
    groups = [[by_id[identity] for identity in ids] for ids in sample["batch_ids"]]
    actual = max(
        groups,
        key=lambda group: (
            len(group)
            * len(group[0]["prepared"]["roles"])
            * max(r["prepared"]["padded_tokens"] for r in group)
        ),
    )
    generator = torch.Generator().manual_seed(99)
    worst = [
        dict(
            split="train",
            prepared=model.tokenize_next(
                torch.randn(1, 1, 2048, generator=generator),
                past_covariates=torch.randn(1, 5, 2048, generator=generator),
                training=True,
            ),
        )
        for _ in range(8)
    ]
    report = {
        "parameters": sum(p.numel() for p in model.parameters()),
        "candidate": {
            "microbatch_token_limit": args.token_limit,
            "checkpoint_token_threshold": args.threshold,
        },
        "weights_changed": False,
        "optimizer_steps": 0,
        "cases": [],
    }
    for name, group in (
        ("recorded_batch", actual),
        ("maximum_length_and_covariates", worst),
    ):
        model.checkpoint_token_threshold = 1024
        baseline = _accumulated_gradients(
            model,
            [group],
            optimizer,
            config["training"]["gradient_clip"],
            prepare_update([group], microbatch_size=8, microbatch_token_limit=1536),
        )
        reference = {
            name: p.grad.detach().cpu().clone()
            for name, p in model.named_parameters()
            if p.grad is not None
        }
        model.checkpoint_token_threshold = args.threshold
        candidate = _accumulated_gradients(
            model,
            [group],
            optimizer,
            config["training"]["gradient_clip"],
            prepare_update(
                [group], microbatch_size=8, microbatch_token_limit=args.token_limit
            ),
        )
        error_squared = ref_squared = candidate_squared = dot = 0.0
        maximum_error = 0.0
        for parameter_name, parameter in model.named_parameters():
            if parameter.grad is None:
                if parameter_name in reference:
                    raise RuntimeError("Candidate omitted a reference gradient")
                continue
            left = reference[parameter_name].float()
            right = parameter.grad.detach().cpu().float()
            delta = right - left
            error_squared += float(delta.square().sum())
            ref_squared += float(left.square().sum())
            candidate_squared += float(right.square().sum())
            dot += float((left * right).sum())
            maximum_error = max(maximum_error, float(delta.abs().max()))
        relative = math.sqrt(error_squared / max(ref_squared, 1e-30))
        cosine = dot / max(math.sqrt(ref_squared * candidate_squared), 1e-30)
        loss_difference = abs(candidate["loss"] - baseline["loss"])
        row = {
            "case": name,
            "baseline_loss": baseline["loss"],
            "candidate_loss": candidate["loss"],
            "loss_absolute_difference": loss_difference,
            "relative_gradient_l2": relative,
            "gradient_cosine": cosine,
            "max_gradient_absolute_difference": maximum_error,
            "baseline_microbatches": baseline["microbatches"],
            "candidate_microbatches": candidate["microbatches"],
        }
        row["passed"] = loss_difference <= 1e-4 and relative <= 0.05 and cosine >= 0.995
        report["cases"].append(row)
        print(json.dumps(row), flush=True)
        del reference
    report["passed"] = all(row["passed"] for row in report["cases"])
    write_json(args.output, report)
    if not report["passed"]:
        raise RuntimeError(
            "Execution policy exceeds BF16 loss/gradient comparison tolerances"
        )


if __name__ == "__main__":
    main()

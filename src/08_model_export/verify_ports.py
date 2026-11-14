"""Compare an exported portable model with the original's reference forecasts and time it.

Loads the model from its export folder (the same files a Hub user would get),
replays the sampled holdout requests and reports, per reference execution mode,
how far the forecasts are from the original. Differences are expressed in units
of each request's context standard deviation (the model's own output scale).

    python src/08_model_export/verify_ports.py --variant pytorch --device cpu --limit 40
    python src/08_model_export/verify_ports.py --variant fla --device cuda --dtype bfloat16
"""

import argparse
import json
import platform
import sys
from pathlib import Path
from time import perf_counter

import torch

ROOT = Path(__file__).resolve().parents[2]
WORK = ROOT / ".cache/port_work"
CLASSES = {
    "pytorch": ("basin-rwkv7-pytorch", "modeling_basin_pytorch", "BasinPyTorch"),
    "fla": ("basin-rwkv7-fla", "modeling_basin_fla", "BasinFLA"),
}


def synchronize(device):
    if device.type == "cuda":
        torch.cuda.synchronize(device)


def timed_pass(model, requests, device):
    outputs, seconds = [], []
    for request in requests:
        synchronize(device)
        started = perf_counter()
        forecast = model.forecast(
            request["context"], past_covariates=request["past_covariates"]
        )
        synchronize(device)
        seconds.append(perf_counter() - started)
        outputs.append(forecast.quantiles[0])
    return outputs, seconds


def latency(seconds):
    values = torch.tensor(seconds, dtype=torch.float64) * 1000
    return {
        "requests": len(seconds),
        "median_ms": float(values.median()),
        "mean_ms": float(values.mean()),
        "p95_ms": float(values.quantile(0.95)),
        "min_ms": float(values.min()),
        "max_ms": float(values.max()),
    }


def context_scales(requests):
    scales = []
    for request in requests:
        context = request["context"][0].double()
        scale = torch.nan_to_num(context, nan=float("nan"))
        valid = torch.isfinite(scale)
        mean = torch.where(valid, scale, 0).sum(-1) / valid.sum(-1).clamp_min(1)
        variance = (torch.where(valid, scale - mean[:, None], 0) ** 2).sum(
            -1
        ) / valid.sum(-1).clamp_min(1)
        std = variance.sqrt()
        scales.append(torch.where(std > 0, std, torch.ones_like(std)))
    return scales


def difference(ours, theirs, scales, median_index):
    """Normalised absolute differences over all requests (quantiles and medians)."""
    q, m, mae_like = [], [], []
    for a, b, scale in zip(ours, theirs, scales):
        delta = (a.double() - b.double()).abs() / scale[:, None, None]
        q.append(delta.flatten())
        m.append(delta[..., median_index].flatten())
        mae_like.append(float(delta[..., median_index].mean()))
    q, m = torch.cat(q), torch.cat(m)
    return {
        "values": q.numel(),
        "identical_fraction": float((q == 0).double().mean()),
        "quantiles_mean_abs": float(q.mean()),
        "quantiles_p99_abs": float(q.quantile(0.99))
        if q.numel() < 16_000_000
        else None,
        "quantiles_max_abs": float(q.max()),
        "median_mean_abs": float(m.mean()),
        "median_max_abs": float(m.max()),
        "worst_request_median_mean_abs": max(mae_like),
    }


def holdout_mase(outputs, requests, median_index):
    """Per-request MASE of the median forecast, scored exactly like the holdout stage."""
    scores = []
    for output, request in zip(outputs, requests):
        actual, weight = request["actual"], request["metric_weight"]
        point = output[:, : actual.shape[-1], median_index].double()
        weight = weight / weight.sum(-1, keepdim=True)
        error = ((point - actual).abs() * weight).sum(-1) / request["mase_scale"]
        scores.append(float(error[torch.isfinite(error)].mean()))
    return torch.tensor(scores, dtype=torch.float64)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--variant", choices=CLASSES, required=True)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--dtype", default="float32", choices=("float32", "bfloat16"))
    parser.add_argument(
        "--limit", type=int, default=None, help="evenly spaced subset size"
    )
    parser.add_argument("--threads", type=int, default=None)
    parser.add_argument("--passes", type=int, default=2)
    parser.add_argument("--tag", default=None)
    parser.add_argument(
        "--fla-default-dispatch",
        action="store_true",
        help="let FLA use its chunked algorithm for 64+ tokens instead of the exact recurrence",
    )
    parser.add_argument("--models", default=str(ROOT / "results/model/portable"))
    args = parser.parse_args()
    if args.threads:
        torch.set_num_threads(args.threads)
    folder_name, module_name, class_name = CLASSES[args.variant]
    folder = Path(args.models) / folder_name
    sys.path.insert(0, str(folder))
    device, dtype = torch.device(args.device), getattr(torch, args.dtype)
    started = perf_counter()
    model_class = getattr(__import__(module_name), class_name)
    if args.variant == "fla":
        import fla.models.rwkv7.modeling_rwkv7  # noqa: F401  (time the import separately)
    import_seconds = perf_counter() - started
    started = perf_counter()
    model = model_class.from_pretrained(folder, device=device, dtype=dtype)
    synchronize(device)
    load_seconds = perf_counter() - started
    if args.fla_default_dispatch:
        model.exact_recurrence = False
    requests = torch.load(WORK / "requests.pt", weights_only=True)
    reference = torch.load(WORK / "reference.pt", weights_only=True)
    indices = list(range(len(requests)))
    if args.limit and args.limit < len(indices):
        step = len(indices) / args.limit
        indices = [int(i * step) for i in range(args.limit)]
    subset = [requests[i] for i in indices]
    scales = context_scales(subset)
    median_index = model.config.median_index

    started = perf_counter()
    model.forecast(subset[0]["context"], past_covariates=subset[0]["past_covariates"])
    synchronize(device)
    first_call_seconds = perf_counter() - started
    passes = []
    for _ in range(args.passes):
        outputs, seconds = timed_pass(model, subset, device)
        passes.append(seconds)

    name = args.variant + ("-chunk" if args.fla_default_dispatch else "")
    tag = args.tag or f"{name}_{device.type}_{args.dtype}_{platform.system().lower()}"
    result = {
        "tag": tag,
        "variant": args.variant,
        "device": str(device),
        "dtype": args.dtype,
        "platform": platform.platform(),
        "torch": torch.__version__,
        "cpu_threads": torch.get_num_threads(),
        "gpu": torch.cuda.get_device_name() if device.type == "cuda" else None,
        "requests": len(subset),
        "import_seconds": import_seconds,
        "load_seconds": load_seconds,
        "first_call_seconds": first_call_seconds,
        "latency_first_pass": latency(passes[0]),
        "latency": latency(passes[-1]),
        "versus_original": {
            mode: difference(
                outputs, [values[i] for i in indices], scales, median_index
            )
            for mode, values in reference["outputs"].items()
        },
    }
    # The stored holdout medians: independent evidence from the 2026-11-12 holdout run itself.
    saved = []
    for request, output, scale in zip(subset, outputs, scales):
        stored = request["holdout_point"]
        ours = output[:, : stored.shape[-1], median_index].double()
        saved.append(((ours - stored).abs() / scale[:, None]).flatten())
    saved = torch.cat(saved)
    result["versus_saved_holdout_medians"] = {
        "values": saved.numel(),
        "mean_abs": float(saved.mean()),
        "max_abs": float(saved.max()),
    }
    # Accuracy against the held-out truth, original vs this model on the same requests.
    ours = holdout_mase(outputs, subset, median_index)
    theirs = holdout_mase(
        [reference["outputs"]["graphs"][i] for i in indices], subset, median_index
    )
    keep = torch.isfinite(ours) & torch.isfinite(theirs)
    result["holdout_accuracy"] = {
        "requests": int(keep.sum()),
        "original_mean_mase": float(theirs[keep].mean()),
        "this_model_mean_mase": float(ours[keep].mean()),
        "original_median_mase": float(theirs[keep].median()),
        "this_model_median_mase": float(ours[keep].median()),
        "mean_abs_paired_difference": float((ours - theirs)[keep].abs().mean()),
        "max_abs_paired_difference": float((ours - theirs)[keep].abs().max()),
        "this_model_better_requests": int((ours < theirs)[keep].sum()),
        "original_better_requests": int((ours > theirs)[keep].sum()),
    }
    # Layer-by-layer activations for the requests the reference dumped.
    layers = {}
    for index, expected in reference.get("layers", {}).items():
        if index not in indices:
            continue
        request, captured = requests[index], []
        sys.path.insert(0, str(folder))
        from basin_common import tokenize

        prepared = tokenize(
            model.config, request["context"], request["past_covariates"]
        )
        with torch.inference_mode():
            model.quantiles(
                prepared,
                model.config.output_patches,
                observer=lambda _layer, x: captured.append(x.detach().float().cpu()),
            )
        rows = []
        for layer, (ours, theirs) in enumerate(zip(captured, expected)):
            theirs = theirs[
                :, :, : ours.shape[2]
            ]  # the original right-pads to 16 tokens
            delta = (ours - theirs).abs()
            rows.append(
                {
                    "layer": layer,
                    "identical_fraction": float((delta == 0).float().mean()),
                    "max_abs": float(delta.max()),
                    "mean_abs": float(delta.mean()),
                    "reference_rms": float(theirs.square().mean().sqrt()),
                }
            )
        layers[str(index)] = {
            "context_points": request["context_points"],
            "variables": request["variables"],
            "layers": rows,
        }
    result["layerwise_versus_original_direct"] = layers
    results = WORK / "results"
    results.mkdir(parents=True, exist_ok=True)
    (results / f"{tag}.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
    torch.save({"indices": indices, "outputs": outputs}, results / f"{tag}.pt")
    printable = {
        k: v for k, v in result.items() if k != "layerwise_versus_original_direct"
    }
    print(json.dumps(printable, indent=2))
    for index, entry in layers.items():
        first, last = entry["layers"][0], entry["layers"][-1]
        print(
            f"layers request {index} (points={entry['context_points']} variables={entry['variables']}): "
            f"layer0 identical={first['identical_fraction']:.4f} max={first['max_abs']:.3g} | "
            f"layer{last['layer']} identical={last['identical_fraction']:.4f} max={last['max_abs']:.3g} "
            f"rms={last['reference_rms']:.3g}"
        )


if __name__ == "__main__":
    main()

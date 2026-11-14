"""Run the ORIGINAL Basin model (custom CUDA kernels) on the sampled requests.

Needs the WSL CUDA runtime (``source scripts/activate_rwkv7.sh``). Stores the
forecasts, per-request latency and a few per-layer activations that the
portable models are then compared against. Execution modes:

* ``graphs``   - exactly the holdout path: compiled variable attention + CUDA-graph replay.
* ``direct``   - same kernels and arithmetic, no CUDA graphs.
* ``training`` - the training forward (clampw_v3 kernel) with eager attention,
  i.e. a second, independent execution path of the same original model.
"""

import argparse
import sys
from pathlib import Path
from time import perf_counter

import torch
from loguru import logger

ROOT = Path(__file__).resolve().parents[2]
for directory in (
    "",
    "src/03_transforming_input_data",
    "src/04_creating_embeddings",
    "src/05_model_training",
    "src/06_holdout_test",
):
    sys.path.insert(0, str(ROOT / directory))


def timed_pass(call, requests):
    outputs, seconds = [], []
    for request in requests:
        torch.cuda.synchronize()
        started = perf_counter()
        output = call(request)
        torch.cuda.synchronize()
        seconds.append(perf_counter() - started)
        outputs.append(output)
    return outputs, seconds


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--requests", default=str(ROOT / ".cache/port_work/requests.pt")
    )
    parser.add_argument("--output", default=str(ROOT / ".cache/port_work/reference.pt"))
    parser.add_argument("--layer-dumps", type=int, default=3)
    args = parser.parse_args()
    logger.remove()
    logger.add(sys.stderr, level="WARNING")

    from fast_inference import configure_fast_inference
    from inference import forecast_contexts
    from train import RWKVForecaster
    from transform import load_config
    from variate_attention import configure_variate_execution

    config = load_config(str(ROOT / "configs/config.json"))
    torch.set_num_threads(config["training"].get("cpu_threads", 4))
    payload = torch.load(
        ROOT / "results/model/training/latest_checkpoint.pt",
        map_location="cpu",
        weights_only=True,
        mmap=True,
    )
    started = perf_counter()
    model = RWKVForecaster.load(
        None,
        checkpoint={
            "config": payload["identity"]["model"],
            "state_dict": payload["best_state_dict"],
        },
    ).eval()
    selected_step = payload["best_step"]
    del payload
    configure_variate_execution(model, config["training"], ROOT)
    configure_fast_inference(model, config["training"])
    torch.cuda.synchronize()
    load_seconds = perf_counter() - started
    requests = torch.load(args.requests, weights_only=True)
    patches = model.config.output_patches

    def call(request):
        context = {k: request[k] for k in ("context", "past_covariates")}
        return list(forecast_contexts(model, [context], 1, prediction_patches=patches))[
            0
        ]

    result = {
        "meta": {
            "selected_step": selected_step,
            "load_seconds": load_seconds,
            "torch": str(torch.__version__),
            "gpu": torch.cuda.get_device_name(),
            "dtype": str(next(model.parameters()).dtype),
            "requests": len(requests),
        },
        "outputs": {},
        "seconds": {},
    }
    engine = model.fast_inference
    # 1) holdout path. First pass captures a CUDA graph per new shape; second is steady state.
    call(requests[0])
    _, result["seconds"]["graphs_first_pass"] = timed_pass(call, requests)
    result["outputs"]["graphs"], result["seconds"]["graphs"] = timed_pass(
        call, requests
    )
    result["meta"]["graph_cache"] = engine.graphs.summary()
    # 2) same arithmetic without graph replay.
    engine.graphs.enabled = False
    call(requests[0])
    timed_pass(call, requests)
    result["outputs"]["direct"], result["seconds"]["direct"] = timed_pass(
        call, requests
    )
    # Per-layer activations (after each block + variable mix) for a few varied requests.
    order = sorted(
        range(len(requests)),
        key=lambda i: (-requests[i]["variables"], -requests[i]["context_points"]),
    )
    picks = [order[0], order[len(order) // 2], order[-1]][: args.layer_dumps]
    result["layers"] = {}
    for index in picks:
        captured = []
        hooks = [
            mix.register_forward_hook(
                lambda _m, _i, out: captured.append(out.detach().float().cpu())
            )
            for mix in model.variate
        ]
        call(requests[index])
        for hook in hooks:
            hook.remove()
        result["layers"][index] = captured
    # 3) independent original path: training kernel + eager attention.
    model.fast_inference = None
    for mix in model.variate:
        mix.compiled_forward = None
    call(requests[0])
    timed_pass(call, requests)
    result["outputs"]["training"], result["seconds"]["training"] = timed_pass(
        call, requests
    )
    torch.save(result, args.output)
    for mode, seconds in result["seconds"].items():
        values = torch.tensor(seconds) * 1000
        print(
            f"{mode:18s} median={values.median():8.2f} ms  mean={values.mean():8.2f} ms  p95={values.quantile(0.95):8.2f} ms"
        )
    print("graph cache:", result["meta"]["graph_cache"])
    print("saved", args.output)


if __name__ == "__main__":
    main()

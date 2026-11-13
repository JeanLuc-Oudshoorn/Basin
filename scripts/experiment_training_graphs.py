"""Bounded-memory CUDA graph experiment on Basin's actual variate training op."""

# ruff: noqa: E402
import copy
import os
import sys
from pathlib import Path
from time import perf_counter

os.environ["TORCHINDUCTOR_COMPILE_THREADS"] = "1"
ROOT = Path(__file__).resolve().parents[1]
os.environ.setdefault("TORCHINDUCTOR_CACHE_DIR", str(ROOT / ".cache/torchinductor"))
os.environ.setdefault("TRITON_CACHE_DIR", str(ROOT / ".cache/torchinductor/triton"))
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "src/05_model_training"))
import torch
from train import VariateMix
from transform import load_config, run_context, write_json
from variate_attention import compiled_attention


def protect_gradients(module, inputs):
    # make_graphed_callables returns static gradient buffers. Do not let the
    # following replay overwrite storage retained by AccumulateGrad.
    for parameter in module.parameters():
        if parameter.grad is not None:
            parameter.grad = parameter.grad.clone()
    if inputs.grad is not None:
        inputs.grad = inputs.grad.clone()


def main():
    torch.set_num_threads(4)
    torch.cuda.set_per_process_memory_fraction(0.25)
    config = load_config()
    output = ROOT / "docs/time_benchmark/training_graphs"
    output.mkdir(parents=True, exist_ok=True)
    rows = []
    with run_context(config, "training_graphs_experiment"):
        for shape in [(1, 4, 32, 1024), (4, 4, 64, 1024)]:
            torch.manual_seed(7)
            base = VariateMix(1024, 16).cuda().bfloat16().train()
            base.compiled_forward = compiled_attention()
            reference = copy.deepcopy(base)
            graphed = copy.deepcopy(base)
            sample = torch.randn(
                shape, device="cuda", dtype=torch.bfloat16, requires_grad=True
            )
            # Warm compiled forward and backward outside CUDA capture.
            for _ in range(3):
                graphed(sample).float().square().mean().backward()
                graphed.zero_grad(set_to_none=True)
                sample.grad = None
            started = perf_counter()
            graph = torch.cuda.make_graphed_callables(graphed, (sample,))
            capture_seconds = perf_counter() - started
            parity = []
            for protected in (False, True):
                for accumulation in (1, 2, 4):
                    reference.zero_grad(set_to_none=True)
                    graphed.zero_grad(set_to_none=True)
                    for microstep in range(accumulation):
                        values = torch.randn_like(sample)
                        x = values.detach().requires_grad_()
                        y = values.detach().requires_grad_()
                        a, b = reference(x), graph(y)
                        a.float().square().mean().backward()
                        b.float().square().mean().backward()
                        if protected:
                            protect_gradients(graph, y)
                    errors = {
                        name: (
                            parameter.grad.float()
                            - dict(graphed.named_parameters())[name].grad.float()
                        )
                        .abs()
                        .max()
                        .item()
                        for name, parameter in reference.named_parameters()
                    }
                    parity.append(
                        dict(
                            protected=protected,
                            accumulation=accumulation,
                            max_gradient_error=max(errors.values()),
                            errors=errors,
                        )
                    )
            times = {}
            for name, module in (
                ("compiled", reference),
                ("compiled_graph", graph),
                ("compiled_graph_protected", graph),
            ):
                measured = []
                for repeat in range(25):
                    module.zero_grad(set_to_none=True)
                    sample.grad = None
                    torch.cuda.synchronize()
                    started = perf_counter()
                    for _ in range(4):
                        module(sample).float().square().mean().backward()
                        if name == "compiled_graph_protected":
                            protect_gradients(module, sample)
                    torch.cuda.synchronize()
                    if repeat >= 5:
                        measured.append((perf_counter() - started) * 1000)
                times[name] = measured
            rows.append(
                dict(
                    shape=shape,
                    capture_seconds=capture_seconds,
                    gradient_parity=parity,
                    times_ms=times,
                    peak_allocated_gib=torch.cuda.max_memory_allocated() / 2**30,
                )
            )
            write_json(output / "experiment.json", rows)
            del base, reference, graphed, graph, sample
            torch.cuda.empty_cache()
        write_json(
            output / "environment.json",
            dict(
                torch=torch.__version__,
                gpu=torch.cuda.get_device_name(),
                precision="bf16",
                cpu_threads=4,
            ),
        )


if __name__ == "__main__":
    main()

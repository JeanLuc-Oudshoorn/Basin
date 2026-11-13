"""Small, bounded-memory GPU experiment; writes results after every case."""

# ruff: noqa: E402
import os
import sys
from dataclasses import replace
from pathlib import Path
from time import perf_counter

os.environ["MAX_JOBS"] = "1"
os.environ["TORCHINDUCTOR_COMPILE_THREADS"] = "1"
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
sys.path.insert(0, str(ROOT / "src/05_model_training"))
from fast_inference import FastInference
from time_benchmark import (
    RWKVForecaster,
    configure_cuda_memory,
    configure_variate_execution,
    depth_variant,
    history,
    load_config,
    logger,
    run_context,
    rwkv_prediction,
    write_json,
)


def main():
    config = load_config()
    torch = __import__("torch")
    torch.set_num_threads(4)
    config["training"]["cuda_memory_limit_gib"] = 4
    configure_cuda_memory(config["training"])
    output = ROOT / "docs/time_benchmark/fast_inference"
    output.mkdir(parents=True, exist_ok=True)
    rows = []
    with run_context(config, "fast_inference_experiment"):
        model = RWKVForecaster.load(
            ROOT / "results/447m_model/training/model.pt"
        ).eval()
        configure_variate_execution(model, config["training"], ROOT)
        model = depth_variant(model, 17)
        engine = FastInference(model, max_graphs=2)
        direct = depth_variant(model, 17)
        direct.config = replace(model.config, output_patches=16)
        torch.manual_seed(7)
        direct.lead_embedding = torch.nn.Embedding(15, model.config.width).to(
            "cuda", dtype=torch.bfloat16
        )
        torch.nn.init.normal_(direct.lead_embedding.weight, std=0.02)
        direct_engine = FastInference(direct, max_graphs=2)
        for length, horizon in [
            (512, 16),
            (512, 64),
            (512, 256),
            (2048, 16),
            (2048, 256),
        ]:
            payload = history(length, 3, 7)
            reference = rwkv_prediction(model, payload, horizon)
            reference = torch.as_tensor(reference)
            for mode in ("reference", "graphs", "recurrent", "direct", "direct_graphs"):
                engine.clear()
                direct_engine.clear()

                def call():
                    if mode == "reference":
                        return torch.as_tensor(rwkv_prediction(model, payload, horizon))
                    if mode == "direct":
                        device_payload = {k: v.to("cuda") for k, v in payload.items()}
                        return (
                            direct.forecast(**device_payload, horizon=horizon)
                            .quantiles[0]
                            .float()
                            .cpu()
                        )
                    if mode == "direct_graphs":
                        return direct_engine.forecast(**payload, horizon=horizon)[0]
                    return engine.forecast(
                        **payload, horizon=horizon, recurrent=mode == "recurrent"
                    )[0]

                try:
                    start = perf_counter()
                    result = call()
                    torch.cuda.synchronize()
                    cold = perf_counter() - start
                    if mode.startswith("direct"):
                        expected = (
                            direct.forecast(
                                **{k: v.to("cuda") for k, v in payload.items()},
                                horizon=horizon,
                            )
                            .quantiles[0]
                            .float()
                            .cpu()
                        )
                    else:
                        expected = reference
                    difference = (result - expected).abs()
                    times = []
                    for _ in range(20):
                        torch.cuda.synchronize()
                        start = perf_counter()
                        result = call()
                        torch.cuda.synchronize()
                        times.append((perf_counter() - start) * 1000)
                    row = dict(
                        mode=mode,
                        context=length,
                        horizon=horizon,
                        cold_seconds=cold,
                        times_ms=times,
                        max_absolute_error=difference.max().item(),
                        mean_absolute_error=difference.mean().item(),
                        reference="direct eager (untrained leads)"
                        if mode.startswith("direct")
                        else "legacy eager rollout",
                        peak_allocated_gib=torch.cuda.max_memory_allocated() / 2**30,
                    )
                except Exception as exc:
                    logger.exception("Experiment failed: {}", mode)
                    row = dict(
                        mode=mode, context=length, horizon=horizon, error=str(exc)
                    )
                rows.append(row)
                write_json(output / "experiment.json", rows)
                logger.info("EXPERIMENT {}", row)
            engine.clear()
            direct_engine.clear()
        write_json(
            output / "environment.json",
            dict(
                torch=torch.__version__,
                gpu=torch.cuda.get_device_name(),
                parameters=sum(p.numel() for p in model.parameters()),
                direct_parameters=sum(p.numel() for p in direct.parameters()),
                precision="bf16",
                threads=4,
            ),
        )
        # One real CUDA training batch verifies that new lead parameters and the
        # original backbone train jointly. No trained checkpoint is overwritten.
        direct.train()
        optimizer = torch.optim.SGD(direct.parameters(), lr=0.001)
        payload = history(128, 3, 19)
        prepared = direct.tokenize_next(**payload, training=True)
        prepared = {
            k: v.to("cuda") if isinstance(v, torch.Tensor) else v
            for k, v in prepared.items()
        }
        losses = []
        before = direct.lead_embedding.weight.detach().float().clone()
        for step in range(3):
            torch.manual_seed(17)
            optimizer.zero_grad(set_to_none=True)
            result = direct.forward_encoded(direct.encode_tokens(prepared), prepared)
            result.loss.backward()
            torch.nn.utils.clip_grad_norm_(
                direct.parameters(), 1.0, error_if_nonfinite=True
            )
            optimizer.step()
            losses.append(result.loss.item())
            logger.info(
                "Direct-head CUDA training smoke step={} loss={}", step + 1, losses[-1]
            )
        write_json(
            output / "training_smoke.json",
            dict(
                losses=losses,
                lead_weight_change_l2=(direct.lead_embedding.weight.float() - before)
                .norm()
                .item(),
                peak_allocated_gib=torch.cuda.max_memory_allocated() / 2**30,
            ),
        )


if __name__ == "__main__":
    main()

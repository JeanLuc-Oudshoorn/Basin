"""End-to-end CUDA smoke: 01_generated data, training, reload, unseen forecast."""

import sys
import tempfile
from pathlib import Path

import torch
from train import ForecastConfig, RWKVForecaster, train_step
from transform import frame_window


def main():
    torch.manual_seed(7)
    model = RWKVForecaster(ForecastConfig(patch_size=8, max_patches=32)).to(
        device="cuda", dtype=torch.bfloat16
    )
    # Import after the backend: vendored RWKV owns the top-level src package.
    sys.path.insert(
        0, str(Path(__file__).resolve().parents[2] / "src/01_data_generation")
    )
    from scm_generator import OperationalSCMGenerator

    generator = OperationalSCMGenerator(
        "unused", seed=13, min_rows=96, max_rows=96, min_covariates=3, max_covariates=3
    )
    frame = generator.simulate(generator.sample_model())
    past = [c for c in frame if c not in ("time_index", "outcome") and "_lag_" not in c]
    # Deterministic calendar features are truly available at forecast time.
    frame["calendar"] = torch.sin(torch.arange(len(frame)) / 8).numpy()
    batch = frame_window(
        frame,
        cutoff=65,
        context_length=61,
        horizon=19,
        past_covariates=past,
        known_covariates=["calendar"],
        device="cuda",
    )
    batch["context"][..., 5] = float("nan")
    batch["future_target"][..., 2] = float("nan")
    before = model.blocks[0].att.output.weight.detach().clone()
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
    losses = [train_step(model, batch, optimizer) for _ in range(3)]
    assert not torch.equal(before, model.blocks[0].att.output.weight)
    assert all(
        torch.isfinite(p.grad).all() for p in model.parameters() if p.grad is not None
    )
    inputs = {k: v for k, v in batch.items() if k != "future_target"}
    expected = model.forecast(**inputs)
    altered = dict(batch, future_target=batch["future_target"] + 1000)
    with torch.no_grad():
        assert torch.equal(model(**batch).quantiles, model(**altered).quantiles), (
            "label leakage"
        )
    changed = dict(inputs, future_covariates=inputs["future_covariates"] + 5)
    assert not torch.equal(expected.quantiles, model.forecast(**changed).quantiles)
    with tempfile.TemporaryDirectory() as directory:
        path = Path(directory) / "model.pt"
        model.save(path)
        restored = RWKVForecaster.load(path)
        torch.testing.assert_close(
            restored.forecast(**inputs).quantiles, expected.quantiles, rtol=0, atol=0
        )
        fresh = generator.simulate(generator.sample_model())
        # New data, different variable count, multiple targets, non-divisible horizon.
        context = torch.tensor(
            fresh[["outcome", "outcome"]].iloc[:51].to_numpy().T.copy(),
            device="cuda",
            dtype=torch.float32,
        )[None]
        output = restored.forecast(context, 23)
        assert output.point.shape == (1, 2, 23)
        assert output.quantiles.shape == (1, 2, 23, 9)
        assert torch.isfinite(output.quantiles).all()
        assert (output.quantiles.diff(dim=-1) >= 0).all()
    torch.cuda.synchronize()
    print(
        f"PASS: losses={losses}; RWKV weights updated; label isolation; covariate influence; "
        f"checkpoint round-trip; unseen forecast={tuple(output.point.shape)}"
    )


if __name__ == "__main__":
    main()

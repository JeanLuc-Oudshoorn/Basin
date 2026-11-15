"""Convert the evaluated Basin checkpoint into two portable, self-contained model folders.

CPU only; needs torch + safetensors (no CUDA, no RWKV backend, no FLA).

    python src/08_model_export/export_models.py

Writes ``<output>/basin-rwkv7-pytorch`` and ``<output>/basin-rwkv7-fla``. Each
holds ``model.safetensors`` (bfloat16, exactly as trained), ``config.json``,
the model code and a short README, and can be uploaded to the Hub as is.
"""

import argparse
import hashlib
import json
import shutil
import subprocess  # nosec B404
import sys
from dataclasses import asdict
from pathlib import Path

import torch
from safetensors.torch import load_file, save_file

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
sys.path.insert(0, str(HERE))
from basin_common import CONFIG_NAME, WEIGHTS_NAME, BasinConfig  # noqa: E402
from modeling_basin_fla import convert_state_dict  # noqa: E402

VARIANTS = {
    "pytorch": ("basin-rwkv7-pytorch", "modeling_basin_pytorch.py", "BasinPyTorch"),
    "fla": ("basin-rwkv7-fla", "modeling_basin_fla.py", "BasinFLA"),
}
# Buffers that are constants derived from the config, not learned weights.
NON_WEIGHT_BUFFERS = ("levels",)


def weights_digest(weights):
    """Same fingerprint as src/06_holdout_test/startup_cache.py::weights_digest."""
    digest = hashlib.sha256()
    for name, tensor in sorted(weights.items()):
        header = json.dumps(
            [name, str(tensor.dtype), list(tensor.shape)], separators=(",", ":")
        ).encode()
        digest.update(len(header).to_bytes(8, "little"))
        digest.update(header)
        raw = tensor.detach().cpu().contiguous().reshape(-1).view(torch.uint8).numpy()
        digest.update(memoryview(raw))
    return digest.hexdigest()


# Model options the portable ports implement: the archived 500M architecture.
PORTED_ARCHITECTURE = {
    "head": "lead_offset",
    "variate_every": 1,
    "temporal_every": 0,
    "gated_mixing": False,
    "value_transform": "none",
    "fp32_io": False,
}


def portable_config(model, state):
    unsupported = {
        key: model[key]
        for key, value in PORTED_ARCHITECTURE.items()
        if model.get(key, value) != value
    }
    if unsupported:
        # Fail before writing anything: a port that ignores these options would
        # load without error only by accident and forecast something else.
        raise NotImplementedError(
            "The portable ports implement the archived 500M architecture only; "
            f"this checkpoint uses {unsupported}. Use configs/legacy_500m.json, or "
            "extend src/08_model_export for the hybrid direct-head model."
        )
    att = "blocks.1.att."
    return BasinConfig(
        patch_size=model["patch_size"],
        width=model["width"],
        layers=model["layers"],
        variate_heads=model["variate_heads"],
        quantiles=tuple(model["quantiles"]),
        context_length=model["context_length"],
        output_patches=model["output_patches"],
        head_size=state[att + "r_k"].shape[1],
        ffn_width=state["blocks.0.ffn.key.weight"].shape[0],
        decay_low_rank_dim=state[att + "w1"].shape[1],
        a_low_rank_dim=state[att + "a1"].shape[1],
        v_low_rank_dim=state[att + "v1"].shape[1],
        gate_low_rank_dim=state[att + "g1"].shape[1],
    )


def model_card(variant, class_name, module, settings):
    model, source = settings["model"], settings["source"]
    horizon = model["patch_size"] * model["output_patches"]
    requirement = {
        "pytorch": "`torch` and `safetensors`. Runs on CPU or GPU, any OS; no compiler, CUDA toolkit or Triton.",
        "fla": "`torch`, `safetensors` and `flash-linear-attention` (Triton kernels): in practice an NVIDIA GPU on Linux "
        "with gcc or clang available for Triton's JIT launcher. No CUDA toolkit and no hand-built kernels. "
        "For CPU-only or Windows-native use, take the pure PyTorch variant.",
    }[variant]
    device = {"pytorch": '"cpu"', "fla": '"cuda"'}[variant]
    return f"""# Basin RWKV-7 forecaster ({variant} variant)

Time-series foundation model for operational data: RWKV-7 time mixing interleaved
with cross-variable attention, {settings["weights"]["parameters"]:,} parameters,
{model["layers"]} layers, width {model["width"]}, patch size {model["patch_size"]}.
It reads up to {model["context_length"]} past points of one or more target series
plus optional past covariates, and returns {len(model["quantiles"])} quantiles for
the next {horizon} steps in one pass (longer horizons are rolled out).

This folder is a conversion of the trained Basin checkpoint (optimizer step
{source["selected_step"]}, weights SHA-256 `{source["evaluated_weights_sha256"]}`).
The weights are stored exactly as trained, in bfloat16.

**Requirements:** {requirement}

```python
import sys, torch
sys.path.insert(0, "path/to/this/folder")
from {module} import {class_name}

model = {class_name}.from_pretrained("path/to/this/folder", device={device})
context = torch.randn(1, 512).cumsum(-1)        # [target, time]; NaN marks missing values
covariates = torch.randn(3, 512)                # optional [covariate, time], same time axis
forecast = model.forecast(context, horizon={horizon}, past_covariates=covariates)
forecast.point       # [target, horizon] median
forecast.quantiles   # [target, horizon, {len(model["quantiles"])}] for levels {model["quantiles"]}
```

By default the model runs in the precision it was trained and evaluated in
(bfloat16 weights and activations, float32 inside the recurrence, gates and
norms). Pass `dtype=torch.float32` to widen the stored weights losslessly.
"""


def write_variant(variant, state, settings, output):
    folder_name, module, class_name = VARIANTS[variant]
    folder = output / folder_name
    folder.mkdir(parents=True, exist_ok=True)
    settings = dict(
        settings, variant=variant, model_class=f"{module[:-3]}.{class_name}"
    )
    settings["weights"] = {
        "file": WEIGHTS_NAME,
        "stored_dtype": "bfloat16",
        "tensors": len(state),
        "parameters": sum(t.numel() for t in state.values()),
    }
    save_file(
        {k: v.contiguous() for k, v in state.items()},
        str(folder / WEIGHTS_NAME),
        metadata={"format": "pt", "variant": variant},
    )
    (folder / CONFIG_NAME).write_text(json.dumps(settings, indent=2), encoding="utf-8")
    for name in ("basin_common.py", module):
        shutil.copyfile(HERE / name, folder / name)
    (folder / "README.md").write_text(
        model_card(variant, class_name, module[:-3], settings), encoding="utf-8"
    )
    # Read back what was written and require bit-identical tensors.
    saved = load_file(str(folder / WEIGHTS_NAME))
    if saved.keys() != state.keys() or any(
        saved[k].dtype != state[k].dtype or not torch.equal(saved[k], state[k])
        for k in state
    ):
        raise RuntimeError(f"{folder_name}: saved tensors differ from the checkpoint")
    print(
        f"{folder_name}: {len(state)} tensors, {settings['weights']['parameters']:,} "
        f"parameters, read back bit-identical -> {folder}"
    )
    return folder


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--checkpoint",
        default=str(ROOT / "results/model/training/latest_checkpoint.pt"),
    )
    parser.add_argument("--field", default="best_state_dict")
    parser.add_argument("--output", default=str(ROOT / "results/model/portable"))
    parser.add_argument(
        "--expected-sha256",
        default=None,
        help="defaults to evaluated_weights_sha256 in holdout_test/model_used.json",
    )
    args = parser.parse_args()
    checkpoint = Path(args.checkpoint)
    payload = torch.load(checkpoint, map_location="cpu", weights_only=True, mmap=True)
    state = dict(payload[args.field])
    model = payload["identity"]["model"] if "identity" in payload else payload["config"]
    step = (
        payload.get("best_step")
        if args.field == "best_state_dict"
        else payload.get("step")
    )
    digest = weights_digest(state)
    expected = args.expected_sha256
    used = ROOT / "results/model/holdout_test/model_used.json"
    if expected is None and used.is_file():
        expected = json.loads(used.read_text(encoding="utf-8")).get(
            "evaluated_weights_sha256"
        )
    print(f"checkpoint weights sha256={digest}\nholdout-evaluated  sha256={expected}")
    if expected is not None and digest != expected:
        raise RuntimeError(
            "Checkpoint weights are not the weights the holdout evaluated"
        )
    dtypes = {str(t.dtype) for t in state.values()}
    if dtypes != {"torch.bfloat16"}:
        raise RuntimeError(f"Expected bfloat16 training weights, found {dtypes}")
    constants = {k: state.pop(k) for k in NON_WEIGHT_BUFFERS if k in state}
    if "levels" in constants and constants["levels"].float().tolist() != [
        float(torch.tensor(q, dtype=torch.bfloat16)) for q in model["quantiles"]
    ]:
        raise RuntimeError("Checkpoint quantile levels disagree with its config")
    config = portable_config(model, state)
    try:
        commit = subprocess.run(  # nosec B603 B607
            ["git", "-C", str(ROOT), "rev-parse", "HEAD"],
            capture_output=True,
            text=True,
            check=True,
        ).stdout.strip()
    except (OSError, subprocess.CalledProcessError):
        commit = None
    settings = {
        "architecture": "basin-rwkv7",
        "model": asdict(config),
        "source": {
            "checkpoint": checkpoint.name,
            "field": args.field,
            "selected_step": step,
            "evaluated_weights_sha256": digest,
            "training_model_config": model,
            "basin_commit": commit,
            "omitted_buffers": sorted(constants),
        },
    }
    output = Path(args.output)
    write_variant("pytorch", state, settings, output)
    converted, dropped = convert_state_dict(state, config)
    settings["source"]["omitted_unused_tensors"] = dropped
    write_variant("fla", converted, settings, output)
    # Every FLA tensor must be a pure relabelling/transposition of a checkpoint tensor.
    used_values = sum(t.numel() for t in converted.values())
    dropped_values = sum(state[k].numel() for k in dropped)
    if used_values + dropped_values != sum(t.numel() for t in state.values()):
        raise RuntimeError("FLA conversion lost or duplicated parameters")
    print(
        f"fla: dropped {len(dropped)} unused layer-0 tensors ({dropped_values:,} values): {dropped}"
    )


if __name__ == "__main__":
    main()

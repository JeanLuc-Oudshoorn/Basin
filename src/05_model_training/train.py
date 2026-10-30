"""Train the RWKV forecaster from saved per-window patch features."""

import argparse
import os
import shlex
import subprocess  # nosec B404
import sys
from dataclasses import asdict, dataclass
from pathlib import Path
from types import SimpleNamespace

import pandas as pd
import torch
from loguru import logger
from torch import nn
from torch.utils.checkpoint import checkpoint

ROOT = Path(__file__).resolve().parents[2]
for directory in (
    ROOT,
    ROOT / "src/03_transforming_input_data",
    ROOT / "src/04_creating_embeddings",
):
    sys.path.insert(0, str(directory))
from embed import ForecastConfig, PatchEncoder, ResidualProjection
from epoch_training import fit_epochs, training_options
from performance import configure_cuda_memory, make_optimizer
from token_inventory import (
    prepare_token_inventory,
    save_inventory,
    update_parameter_ratios,
)
from training_log import (
    emit_training_event,
    logged_training,
    verify_worker_completed,
    worker_environment,
)
from training_protocol import split_datasets
from transform import (
    file_digest,
    load_config,
    location,
    read_artifact,
    run_context,
    save_artifact,
    signature,
    timed,
    write_json,
)
from variate_attention import configure_variate_execution

from rwkv7_backend import load_rwkv


@dataclass
class ForecastOutput:
    point: torch.Tensor  # [batch, target, horizon], median
    quantiles: torch.Tensor  # [batch, target, horizon, quantile]
    loss: torch.Tensor | None = None
    per_sample_loss: torch.Tensor | None = None


class VariateMix(nn.Module):
    def __init__(self, width, heads):
        super().__init__()
        self.norm = nn.LayerNorm(width)
        self.attention = nn.MultiheadAttention(width, heads, batch_first=True)
        self.compiled_forward = None
        self.compile_training = True
        self.compile_inference = False

    def forward(self, x):
        # Compile only the short variable axes measured in the throughput audit;
        # wider attention retains the default kernel's memory characteristics.
        if (
            self.compiled_forward is not None
            and (
                (self.training and torch.is_grad_enabled() and self.compile_training)
                or (
                    not self.training
                    and not torch.is_grad_enabled()
                    and self.compile_inference
                )
            )
            and x.shape[1] <= 8
            and self.attention.dropout == 0
        ):
            return self.compiled_forward(self, x)
        b, v, t, d = x.shape
        flat = x.permute(0, 2, 1, 3).reshape(b * t, v, d)
        normalized = self.norm(flat)
        mixed = self.attention(normalized, normalized, normalized, need_weights=False)[
            0
        ]
        return (flat + mixed).reshape(b, t, v, d).permute(0, 2, 1, 3)


class RWKVForecaster(PatchEncoder):
    def __init__(self, config: ForecastConfig = ForecastConfig()):
        super().__init__(config)
        self.config = config
        # Runtime-only execution choice; does not invalidate processed artifacts.
        self.checkpoint_token_threshold = 0
        # Training initialization is required even when first used for forecasting.
        rwkv = load_rwkv(ctx_len=config.max_patches, inference_only=False)
        args = SimpleNamespace(
            n_embd=config.width,
            dim_att=config.width,
            dim_ffn=4 * config.width,
            n_layer=config.layers,
            head_size=64,
            my_testing="x070",
            grad_cp=0,
        )
        self.blocks = nn.ModuleList([rwkv.Block(args, i) for i in range(config.layers)])
        self.variate = nn.ModuleList(
            [
                VariateMix(config.width, config.variate_heads)
                for _ in range(config.layers)
            ]
        )
        self.norm = nn.LayerNorm(config.width)
        self.head = ResidualProjection(
            config.width, config.patch_size * len(config.quantiles), 4 * config.width
        )
        self.register_buffer(
            "levels", torch.tensor(config.quantiles, dtype=torch.float32)
        )

    def forward(self, context, horizon, **kwargs):
        if self.config.objective == "next_patch":
            if (
                not isinstance(horizon, int)
                or horizon < 1
                or horizon > self.config.patch_size
            ):
                raise ValueError(
                    "forward predicts one patch; use forecast for longer horizons"
                )
            prepared = self.tokenize(context, horizon, **kwargs)
            output = self.forward_encoded(
                self.encode_tokens(prepared), prepared, kwargs.get("future_target")
            )
            return ForecastOutput(
                output.point[..., :horizon],
                output.quantiles[..., :horizon, :],
                output.loss,
            )
        prepared = self.tokenize(context, horizon, **kwargs)
        return self.forward_encoded(
            self.encode_tokens(prepared), prepared, kwargs.get("future_target")
        )

    def forward_encoded(self, x, prepared, future_target=None):
        if self.config.objective == "next_patch":
            return self.forward_next_encoded(x, prepared, future_target)
        b, targets, horizon, cp, hp, tokens, padded_tokens = [
            prepared[k]
            for k in ("b", "targets", "horizon", "cp", "hp", "tokens", "padded_tokens")
        ]
        loc, scale = prepared["loc"], prepared["scale"]
        roles = prepared["roles"]
        p = self.config.patch_size
        v_first = torch.empty_like(x.reshape(b * len(roles), padded_tokens, -1))
        for block, mix in zip(self.blocks, self.variate):
            flat = x.reshape(b * len(roles), padded_tokens, -1).contiguous()
            flat, v_first = block(flat, v_first)
            x = mix(flat.reshape(b, len(roles), padded_tokens, -1))
        raw = self.head(self.norm(x[:, :targets, cp:tokens])).float()
        raw = raw.reshape(b, targets, hp * p, len(self.config.quantiles))[
            :, :, :horizon
        ]
        # Sorting guarantees ordered quantiles and remains differentiable almost everywhere.
        normalized = raw.sort(dim=-1).values
        prediction = (
            normalized * scale[:, :targets, :, None] + loc[:, :targets, :, None]
        )
        loss = None
        if future_target is not None:
            self._validate(future_target, "future_target", b, horizon)
            if future_target.shape[1] != targets:
                raise ValueError("future_target target count mismatch")
            valid = torch.isfinite(future_target)
            if not prepared.get("observed_points") and not valid.any():
                raise ValueError("future_target contains no observed labels")
            labels = (future_target.float() - loc[:, :targets]) / scale[:, :targets]
            error = labels.nan_to_num()[..., None] - normalized
            levels = self.levels.float()
            pinball = torch.maximum(levels * error, (levels - 1) * error)
            loss = (pinball * valid[..., None]).sum() / (valid.sum() * len(levels))
        return ForecastOutput(
            prediction[..., self.config.quantiles.index(0.5)], prediction, loss
        )

    def forward_next_encoded(self, x, prepared, future_target=None):
        b, v, t, d = x.shape
        # Spend a fixed activation budget across layers, rather than abruptly
        # recomputing every layer when a batch is just above the threshold.
        uncheckpointed_layers = min(
            len(self.blocks),
            self.checkpoint_token_threshold * len(self.blocks) // (b * v * t),
        )
        v_first = torch.empty_like(x.reshape(b * v, t, d))
        for layer, (block, mix) in enumerate(zip(self.blocks, self.variate)):
            flat = x.reshape(b * v, t, d).contiguous()
            if (
                self.training
                and self.config.gradient_checkpointing
                and layer >= uncheckpointed_layers
            ):
                flat, v_first = checkpoint(block, flat, v_first, use_reentrant=False)
                x = checkpoint(mix, flat.reshape(b, v, t, d), use_reentrant=False)
            else:
                flat, v_first = block(flat, v_first)
                x = mix(flat.reshape(b, v, t, d))
        n, targets, p = prepared["tokens"], prepared["targets"], self.config.patch_size
        loc, scale = prepared["loc"][:, :targets], prepared["scale"][:, :targets]
        selected = x[:, :targets, :n]
        if "last_token_indices" in prepared:
            indices = prepared["last_token_indices"].view(b, 1, 1, 1)
            selected = selected.gather(2, indices.expand(b, targets, 1, d))
            loc = loc.gather(2, indices.expand(b, targets, 1, 1))
            scale = scale.gather(2, indices.expand(b, targets, 1, 1))
            n = 1
        elif not prepared["is_training"]:
            # A one-patch forecast needs only the final hidden position. Avoid
            # running the projection and sorting quantiles for the entire past.
            selected, loc, scale = selected[:, :, -1:], loc[:, :, -1:], scale[:, :, -1:]
            n = 1
        raw = self.head(self.norm(selected)).float()
        raw = (
            raw.reshape(b, targets, n, p, len(self.config.quantiles))
            .sort(dim=-1)
            .values
        )
        labels = prepared["labels"] if prepared["is_training"] else future_target
        if not prepared["is_training"]:
            raw, loc, scale = raw[:, :, -1], loc[:, :, -1], scale[:, :, -1]
        prediction = raw * scale[..., None] + loc[..., None]
        loss = None
        per_sample_loss = None
        if labels is not None:
            if labels.shape != raw.shape[:-1]:
                raise ValueError("Next-patch label dimensions mismatch")
            valid = torch.isfinite(labels)
            if not prepared.get("observed_points") and not valid.any():
                raise ValueError("No observed labels")
            normalized_labels = (labels.float() - loc) / scale
            error = normalized_labels.nan_to_num()[..., None] - raw
            # Model.to(bfloat16) casts registered buffers; retain exact float32 levels.
            levels = error.new_tensor(self.config.quantiles)
            pinball = (
                torch.maximum(levels * error, (levels - 1) * error) * valid[..., None]
            )
            count = (
                prepared["observed_points"]
                if "observed_points" in prepared
                else valid.sum()
            )
            loss = pinball.sum() / (count * len(levels))
            if "last_token_indices" in prepared:
                per_sample_loss = pinball.flatten(1).sum(1) / (
                    valid.flatten(1).sum(1) * len(levels)
                )
        return ForecastOutput(
            prediction[..., self.config.quantiles.index(0.5)],
            prediction,
            loss,
            per_sample_loss,
        )

    @torch.inference_mode()
    def forecast(self, context, horizon, **covariates):
        training = self.training
        self.eval()
        try:
            if self.config.objective == "next_patch":
                if not isinstance(horizon, int) or horizon < 1:
                    raise ValueError("horizon must be positive")
                outputs = []
                history = context
                past = covariates.get("past_covariates")
                if set(covariates) - {"past_covariates"}:
                    raise ValueError(
                        "Next-patch forecast accepts only observed historical covariates"
                    )
                for _ in range(
                    (horizon + self.config.patch_size - 1) // self.config.patch_size
                ):
                    part = self(history, self.config.patch_size, past_covariates=past)
                    outputs.append(part.quantiles)
                    history = torch.cat([history, part.point], dim=-1)[
                        ..., -self.config.context_length :
                    ]
                    if past is not None:
                        missing = past.new_full(
                            (*past.shape[:-1], self.config.patch_size), float("nan")
                        )
                        past = torch.cat([past, missing], dim=-1)[
                            ..., -self.config.context_length :
                        ]
                quantiles = torch.cat(outputs, dim=-2)[..., :horizon, :]
                return ForecastOutput(
                    quantiles[..., self.config.quantiles.index(0.5)], quantiles
                )
            return self(context, horizon, **covariates)
        finally:
            self.train(training)

    def save(self, path):
        torch.save(
            {"config": asdict(self.config), "state_dict": self.state_dict()}, Path(path)
        )

    @classmethod
    def load(cls, path, *, device="cuda", dtype=torch.bfloat16):
        checkpoint = torch.load(path, map_location="cpu", weights_only=True)
        model = cls(ForecastConfig(**checkpoint["config"])).to(
            device=device, dtype=dtype
        )
        model.load_state_dict(checkpoint["state_dict"])
        return model


def on_device(prepared):
    return {
        k: v.to("cuda") if isinstance(v, torch.Tensor) else v
        for k, v in prepared.items()
    }


def architecture_report(model):
    """Measured parameter counts and exact tensor shapes, independent of variable count."""
    parameters = {
        name: {
            "shape": list(p.shape),
            "parameters": p.numel(),
            "trainable": p.requires_grad,
        }
        for name, p in model.named_parameters()
    }
    groups = {}
    for name, info in parameters.items():
        key = (
            ".".join(name.split(".")[:2])
            if name.startswith(("blocks.", "variate."))
            else name.split(".")[0]
        )
        groups[key] = groups.get(key, 0) + info["parameters"]
    return {
        "config": asdict(model.config),
        "total_parameters": sum(p.numel() for p in model.parameters()),
        "trainable_parameters": sum(
            p.numel() for p in model.parameters() if p.requires_grad
        ),
        "groups": groups,
        "parameters": parameters,
        "rwkv_ffn_width": 4 * model.config.width,
        "rwkv_head_size": 64,
        "rwkv_heads": model.config.width // 64,
        "variate_head_size": model.config.width // model.config.variate_heads,
        "tied_embeddings": False,
        "input_feature_dimension": 3 * model.config.patch_size,
        "output_dimension": model.config.patch_size * len(model.config.quantiles),
    }


def train_step(model, batch, optimizer, gradient_clip=1.0):
    model.train()
    optimizer.zero_grad(set_to_none=True)
    output = model(**batch)
    if output.loss is None or not torch.isfinite(output.loss):
        raise FloatingPointError("Training requires finite loss")
    output.loss.backward()
    torch.nn.utils.clip_grad_norm_(
        model.parameters(), gradient_clip, error_if_nonfinite=True
    )
    optimizer.step()
    return float(output.loss.detach())


@logged_training("train")
def train(config):
    training_options(config["training"])
    configure_cuda_memory(config["training"])
    # Small patch preparation should not inherit all host CPU threads. Set this
    # before any optional audit or lazy tokenization, rather than after it.
    torch.set_num_threads(config["training"].get("cpu_threads", 4))
    data = read_artifact(location(config, "embedded") / "embeddings.pt", config)
    # The lazy-index loader already verified the full processed checksum.
    # Keep the legacy guard without rereading the same multi-GB file twice.
    if data.get("format") != "lazy_patch_index_v1" and data[
        "processed_sha256"
    ] != file_digest(location(config, "processed") / "windows.pt"):
        raise ValueError(
            "Processed data changed: rerun embedding stage before training"
        )
    records, validation_records, test_records, dataset_split = split_datasets(
        data["records"], config
    )
    inventory = None
    if config["model"].get("objective") == "next_patch":
        inventory = prepare_token_inventory(
            data,
            config,
            groups=(records, validation_records, test_records, dataset_split),
        )
    write_json(location(config, "training") / "dataset_split.json", dataset_split)
    emit_training_event("dataset_split", **dataset_split)
    if not records:
        raise ValueError("Embedding artifact contains no training windows")
    torch.manual_seed(config["seed"])
    torch.set_num_threads(config["training"].get("cpu_threads", 4))
    settings = ForecastConfig(
        **dict(config["model"], quantiles=tuple(config["model"]["quantiles"]))
    )
    with timed("initialize RWKV and restore embedding weights", cuda=True):
        model = RWKVForecaster(settings).to(device="cuda", dtype=torch.bfloat16)
        model.embedding.load_state_dict(
            {
                k.removeprefix("embedding."): v
                for k, v in data["encoder_state"].items()
                if k.startswith("embedding.")
            }
        )
        model.roles.load_state_dict(
            {
                k.removeprefix("roles."): v
                for k, v in data["encoder_state"].items()
                if k.startswith("roles.")
            }
        )
    t = config["training"]
    model.checkpoint_token_threshold = t.get("checkpoint_token_threshold", 0)
    configure_variate_execution(model, t, ROOT)
    for module in (model.embedding, model.roles):
        module.requires_grad_(t["train_embeddings"])
    report = architecture_report(model)
    if inventory is not None:
        update_parameter_ratios(inventory, report["total_parameters"])
        save_inventory(inventory, location(config, "training"))
    emit_training_event(
        "model_initialized",
        parameters=report["total_parameters"],
        architecture=report,
        dtype=str(next(model.parameters()).dtype),
        device=str(next(model.parameters()).device),
    )
    write_json(location(config, "training") / "architecture.json", report)
    logger.info("Measured model parameter groups={}", report["groups"])
    logger.info(
        "Device={} dtype={} parameters={} trainable={}",
        torch.cuda.get_device_name(),
        next(model.parameters()).dtype,
        sum(p.numel() for p in model.parameters()),
        sum(p.numel() for p in model.parameters() if p.requires_grad),
    )
    logger.info(
        "Embedding policy: {}",
        "prepare per-window features and learn embeddings jointly in the model"
        if t["train_embeddings"]
        else "frozen projection; compute only the current window on GPU",
    )
    optimizer = make_optimizer(model, t)
    initial_projection = model.embedding.skip.weight.detach().float().clone()
    history, training_summary, scheduler_state = fit_epochs(
        model, records, validation_records, config, optimizer
    )
    directory = location(config, "training")
    projection_delta = float(
        (model.embedding.skip.weight.detach().float() - initial_projection).norm()
    )
    write_json(
        directory / "embedding_optimization.json",
        {
            "jointly_trained": t["train_embeddings"],
            "output_dimension": settings.width,
            "projection_parameter_count": sum(
                p.numel() for p in model.embedding.parameters()
            ),
            "skip_weight_change_l2": projection_delta,
            "steps_with_nonzero_skip_gradient": sum(
                row["embedding_skip_gradient_norm"] > 0 for row in history
            ),
            "optimizer_updates": len(history),
            "selected_epoch": training_summary["best_epoch"],
            "selected_update": training_summary["best_step"],
        },
    )
    logger.info(
        "Embedding optimization: output_dimension={} skip_weight_change_l2={}",
        settings.width,
        projection_delta,
    )
    if t["train_embeddings"] and projection_delta == 0:
        logger.warning(
            "Embedding skip weights did not change; inspect gradients, precision and learning rate"
        )
    save_artifact(
        directory / "model.pt",
        {
            "signature": signature(config),
            "config": asdict(settings),
            "state_dict": model.state_dict(),
            "pipeline_config": config,
            "dataset_split": dataset_split,
            "lr_scheduler": scheduler_state,
            "training_summary": training_summary,
            "embedding_sha256": file_digest(
                location(config, "embedded") / "embeddings.pt"
            ),
        },
    )
    write_json(directory / "loss_history.json", history)
    write_json(
        directory / "learning_rate_scheduler.json",
        scheduler_state,
    )
    pd.DataFrame(history).to_csv(directory / "loss_history.csv", index=False)
    logger.success("Training complete: checkpoint={}", directory / "model.pt")
    emit_training_event(
        "training_complete",
        global_step=training_summary["completed_updates"],
        completed_global_step=training_summary["completed_updates"],
        completed_run_step=training_summary["completed_updates"],
        checkpoint_path=str(directory / "model.pt"),
        selected_step=training_summary["best_step"],
        selected_epoch=training_summary["best_epoch"],
        **{
            k: v
            for k, v in training_summary.items()
            if k not in ("best_step", "best_epoch")
        },
    )
    logger.info(
        "Training finished. Run src/06_holdout_test/holdout_test.py separately "
        "to evaluate the saved model on the sealed holdout tables."
    )


def launch_wsl(script, config_path):
    if os.name != "nt":
        return False
    relative_config = (ROOT / config_path).resolve().relative_to(ROOT).as_posix()
    relative_script = Path(script).resolve().relative_to(ROOT).as_posix()
    command = "source scripts/activate_rwkv7.sh && python -u " + shlex.quote(
        relative_script
    )
    command += " --config " + shlex.quote(relative_config)
    logger.info("Routing GPU stage to WSL: {}", command)
    # Fixed executable and shell-quoted repository paths; capture native backend output too.
    with subprocess.Popen(  # nosec B603
        ["C:/Windows/System32/wsl.exe", "--cd", str(ROOT), "bash", "-lc", command],
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        encoding="utf-8",
        errors="replace",
        env=worker_environment(),
    ) as process:
        for line in process.stdout:
            logger.info("WSL | {}", line.rstrip())
        code = process.wait()
    if code:
        raise RuntimeError(f"WSL stage failed with exit code {code}; see run log")
    verify_worker_completed()
    return True


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default=str(ROOT / "configs/config.json"))
    args = parser.parse_args()
    config = load_config(args.config)
    with run_context(config, "train"):
        if launch_wsl(__file__, args.config):
            return
        if not torch.cuda.is_available() or not torch.cuda.is_bf16_supported():
            raise RuntimeError("Use the configured WSL CUDA bf16 runtime")
        with timed("TRAIN stage", cuda=True):
            train(config)


if __name__ == "__main__":
    main()

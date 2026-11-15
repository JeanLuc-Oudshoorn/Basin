"""Train the RWKV forecaster from saved per-window patch features."""

import argparse
import os
import shlex
import subprocess  # nosec B404
import sys
from dataclasses import asdict, dataclass
from pathlib import Path
from types import SimpleNamespace

from runtime_storage import configure_runtime_storage

ROOT = Path(__file__).resolve().parents[2]
RUNTIME_STORAGE = configure_runtime_storage(ROOT)

import pandas as pd
import torch
from loguru import logger
from torch import nn
from torch.utils.checkpoint import checkpoint

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
    loss_mass: torch.Tensor | None = None


def apply_upstream_rwkv_init(blocks):
    """Initial weights of the upstream RWKV-7 training recipe.

    The vendored block constructors apply the initialisation upstream gives for
    using the blocks standalone. Upstream's own training entry point then
    overwrites part of it (`generate_init_weight` in the vendored model.py):
    orthogonal projections at a 3.5x larger scale than the constructor's uniform
    draw, and a per-layer gain on the time-mix GroupNorm that grows with depth.
    The decay, gate and mixing-ratio parameters keep their constructor values in
    both, so they are not touched here.
    """
    for index, block in enumerate(blocks):
        att, ffn = getattr(block, "att", None), getattr(block, "ffn", None)
        if att is None or ffn is None or att.receptance.weight.is_meta:
            continue  # test doubles, or weights about to be loaded from a file
        nn.init.orthogonal_(att.receptance.weight, gain=1.0)
        nn.init.orthogonal_(att.key.weight, gain=0.1)
        nn.init.orthogonal_(att.value.weight, gain=1.0)
        nn.init.zeros_(att.output.weight)
        nn.init.orthogonal_(ffn.key.weight, gain=1.0)
        nn.init.zeros_(ffn.value.weight)
        nn.init.constant_(att.ln_x.weight, ((1 + index) / len(blocks)) ** 0.7)


class VariateMix(nn.Module):
    def __init__(self, width, heads, *, gated=False):
        super().__init__()
        self.norm = nn.LayerNorm(width)
        self.attention = nn.MultiheadAttention(width, heads, batch_first=True)
        self.compiled_forward = None
        self.compile_training = True
        self.compile_inference = False
        # Gated: the branch starts closed and a table with one variable, which
        # has nothing to mix, skips it. Not a parameter, so nothing is saved.
        self.gated = gated
        if gated:
            nn.init.zeros_(self.attention.out_proj.weight)
            nn.init.zeros_(self.attention.out_proj.bias)

    def forward(self, x):
        if self.gated and x.shape[1] == 1:
            return x
        graphs = getattr(self, "training_graphs", None)
        if graphs is not None and self.training and torch.is_grad_enabled():
            return graphs.run(self.training_graph_index, x)
        return self.forward_ungraphed(x)

    def forward_ungraphed(self, x):
        if self.gated and x.shape[1] == 1:
            return x
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
        # Mixing follows every Nth RWKV block. The archived models mix variables
        # after every block (variate.<i> then belongs to block i); newer ones
        # mix less often and can add causal temporal attention at those points.
        variate_layers = [
            layer
            for layer in range(config.layers)
            if (layer + 1) % config.variate_every == 0
        ]
        temporal_layers = [
            layer
            for layer in range(config.layers)
            if config.temporal_every and (layer + 1) % config.temporal_every == 0
        ]
        self.variate = nn.ModuleList(
            [
                VariateMix(
                    config.width, config.variate_heads, gated=config.gated_mixing
                )
                for _ in variate_layers
            ]
        )
        if temporal_layers:
            from temporal_attention import TemporalAttention

            self.temporal = nn.ModuleList(
                [
                    TemporalAttention(config.width, config.max_patches)
                    for _ in temporal_layers
                ]
            )
        # Plain lookup lists (the modules are registered above): per block, the
        # temporal attention and the variable mixing that follow it, or None.
        self.temporal_after = [None] * config.layers
        self.variate_after = [None] * config.layers
        for module, layer in zip(self.variate, variate_layers):
            self.variate_after[layer] = module
        for module, layer in zip(getattr(self, "temporal", ()), temporal_layers):
            self.temporal_after[layer] = module
        self.norm = nn.LayerNorm(config.width)
        outputs = config.patch_size * len(config.quantiles)
        if config.head == "direct":
            # One projection emits every lead: each lead has its own output
            # rows, instead of an additive offset that LayerNorm can wash out.
            outputs *= config.output_patches
        self.head = ResidualProjection(config.width, outputs, 4 * config.width)
        if config.head == "lead_offset" and config.output_patches > 1:
            self.lead_embedding = nn.Embedding(config.output_patches - 1, config.width)
            nn.init.normal_(self.lead_embedding.weight, std=0.02)
        self.register_buffer(
            "levels", torch.tensor(config.quantiles, dtype=torch.float32)
        )
        if config.rwkv_init == "upstream":
            apply_upstream_rwkv_init(self.blocks)

    @property
    def hybrid(self):
        return any(module is not None for module in self.temporal_after)

    @property
    def compute_dtype(self):
        """The backbone's dtype; with fp32_io the first parameter is float32."""
        return self.roles.weight.dtype

    def mix(self, layer, x, *, ungraphed=False):
        """Mixing that follows RWKV block `layer`; x is [batch, variable, time, width]."""
        temporal, variate = self.temporal_after[layer], self.variate_after[layer]
        if temporal is not None:
            x = temporal(x)
        if variate is not None:
            x = variate.forward_ungraphed(x) if ungraphed else variate(x)
        return x

    def _apply(self, fn, recurse=True):
        # Every dtype cast funnels through here (.to, .bfloat16, .float, ...).
        # With fp32_io the first and last layers stay float32 whatever the
        # backbone is cast to: they are where rounding touches the data itself.
        super()._apply(fn, recurse)
        if self.config.fp32_io:
            for module in (self.embedding, self.norm, self.head):
                for parameter in module.parameters():
                    if parameter.dtype in (torch.bfloat16, torch.float16):
                        parameter.data = parameter.data.float()
        return self

    def forward(self, context, horizon, **kwargs):
        if self.config.objective == "next_patch":
            if (
                not isinstance(horizon, int)
                or horizon < 1
                or horizon > self.config.patch_size * self.config.output_patches
            ):
                raise ValueError(
                    "forward supports one direct block; use forecast for longer horizons"
                )
            prepared = self.tokenize(context, horizon, **kwargs)
            prepared["prediction_patches"] = (
                horizon + self.config.patch_size - 1
            ) // self.config.patch_size
            output = self.forward_encoded(
                self.encode_tokens(prepared), prepared, kwargs.get("future_target")
            )
            return ForecastOutput(
                output.point[..., :horizon],
                output.quantiles[..., :horizon, :],
                output.loss,
                output.per_sample_loss,
                output.loss_mass,
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
        for layer, block in enumerate(self.blocks):
            flat = x.reshape(b * len(roles), padded_tokens, -1).contiguous()
            flat, v_first = block(flat, v_first)
            x = self.mix(layer, flat.reshape(b, len(roles), padded_tokens, -1))
        raw = self.head(
            self.norm(x[:, :targets, cp:tokens].to(self.norm.weight.dtype))
        ).float()
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
        for layer, block in enumerate(self.blocks):
            flat = x.reshape(b * v, t, d).contiguous()
            variate = self.variate_after[layer]
            mixes = self.temporal_after[layer] is not None or (
                variate is not None and not (variate.gated and v == 1)
            )
            if (
                self.training
                and self.config.gradient_checkpointing
                and layer >= uncheckpointed_layers
            ):
                flat, v_first = checkpoint(block, flat, v_first, use_reentrant=False)
                x = flat.reshape(b, v, t, d)
                if mixes:
                    # Graphs keep their own static backward buffers; retain ordinary
                    # checkpoint recomputation for the layers that need memory savings.
                    x = checkpoint(
                        lambda value, layer=layer: self.mix(
                            layer, value, ungraphed=True
                        ),
                        x,
                        use_reentrant=False,
                    )
            else:
                flat, v_first = block(flat, v_first)
                x = flat.reshape(b, v, t, d)
                if mixes:
                    x = self.mix(layer, x)
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
        if self.config.head == "direct":
            return self.forward_direct(selected, prepared, loc, scale, future_target)
        prediction_patches = prepared.get("prediction_patches", 1)
        if prediction_patches > 1:
            from multi_horizon import direct_quantiles

            if prepared["is_training"]:
                raise ValueError("Direct block decoding requires an evaluation context")
            prediction = direct_quantiles(
                model=self,
                hidden=selected,
                loc=loc[:, :, -1],
                scale=scale[:, :, -1],
                patches=prediction_patches,
            )
            loss = per_sample_loss = mass = None
            if future_target is not None:
                if self.config.horizon_loss_weights is None:
                    raise ValueError(
                        "Multi-patch validation requires horizon_loss_weights"
                    )
                from multi_horizon import weighted_forecast_loss

                prediction = prediction[..., : future_target.shape[-1], :]
                loss, per_sample_loss, mass = weighted_forecast_loss(
                    self, prediction, future_target, loc[:, :, -1], scale[:, :, -1]
                )
            return ForecastOutput(
                prediction[..., self.config.quantiles.index(0.5)],
                prediction,
                loss,
                per_sample_loss,
                mass,
            )
        raw = self.read_out(selected).float()
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
        loss_mass = None
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
            if self.config.horizon_loss_weights is not None and prepared["is_training"]:
                from multi_horizon import weighted_training_loss

                loss, per_sample_loss, loss_mass = weighted_training_loss(
                    self, selected, labels, loc, scale, raw
                )
            elif (
                self.training
                and prepared["is_training"]
                and self.config.output_patches > 1
            ):
                from multi_horizon import auxiliary_loss

                extra, extra_count = auxiliary_loss(self, selected, labels, loc, scale)
                weight = self.config.multi_horizon_weight * (extra_count > 0)
                loss = (1 - weight) * loss + weight * extra
            if "last_token_indices" in prepared:
                per_sample_loss = pinball.flatten(1).sum(1) / (
                    valid.flatten(1).sum(1) * len(levels)
                )
        return ForecastOutput(
            prediction[..., self.config.quantiles.index(0.5)],
            prediction,
            loss,
            per_sample_loss,
            loss_mass,
        )

    def read_out(self, hidden):
        """Final norm and head; with fp32_io both compute in float32."""
        return self.head(self.norm(hidden.to(self.norm.weight.dtype)))

    def forward_direct(self, selected, prepared, loc, scale, future_target=None):
        """Direct head: every selected position forecasts all leads in one projection.

        Training supervises every causal anchor on every lead that still has
        labels inside the chunk. Evaluation reads one position per request.
        """
        from multi_horizon import (
            anchors_with_history,
            direct_training_loss,
            from_model_space,
            lead_quantiles,
            weighted_forecast_loss,
        )

        median = self.config.quantiles.index(0.5)
        if prepared["is_training"]:
            loss, per_sample_loss, mass, first = direct_training_loss(
                self,
                selected,
                prepared["labels"],
                loc,
                scale,
                supervised=anchors_with_history(
                    prepared["features"], prepared["targets"], self.config.patch_size
                ),
            )
            # Callers of a training pass use the loss. Keep the next-patch
            # forecast available at every anchor, as the legacy head did.
            prediction = (
                from_model_space(self, first) * scale[..., None] + loc[..., None]
            )
            return ForecastOutput(
                prediction[..., median], prediction, loss, per_sample_loss, mass
            )
        patches = prepared.get("prediction_patches", 1)
        if patches < 1 or patches > self.config.output_patches:
            raise ValueError("Requested block exceeds trained output_patches")
        b, targets = selected.shape[:2]
        normalized = lead_quantiles(self, selected, patches).reshape(
            b, targets, patches * self.config.patch_size, len(self.config.quantiles)
        )
        loc, scale = loc[:, :, -1], scale[:, :, -1]
        prediction = (
            from_model_space(self, normalized) * scale[..., None] + loc[..., None]
        )
        loss = per_sample_loss = mass = None
        if future_target is not None:
            steps = future_target.shape[-1]
            prediction, normalized = (
                prediction[..., :steps, :],
                normalized[..., :steps, :],
            )
            loss, per_sample_loss, mass = weighted_forecast_loss(
                self, prediction, future_target, loc, scale, normalized=normalized
            )
        return ForecastOutput(
            prediction[..., median], prediction, loss, per_sample_loss, mass
        )

    @torch.inference_mode()
    def forecast(self, context, horizon, **covariates):
        training = self.training
        self.eval()
        try:
            if self.config.objective == "next_patch":
                if (
                    isinstance(horizon, bool)
                    or not isinstance(horizon, int)
                    or horizon < 1
                ):
                    raise ValueError("horizon must be positive")
                if set(covariates) - {"past_covariates"}:
                    raise ValueError(
                        "Next-patch forecast accepts only observed historical covariates"
                    )
                engine = getattr(self, "fast_inference", None)
                if engine is not None:
                    quantiles = engine.forecast(context, horizon, **covariates).to(
                        context.device
                    )
                    return ForecastOutput(
                        quantiles[..., self.config.quantiles.index(0.5)], quantiles
                    )
                outputs = []
                history = context
                past = covariates.get("past_covariates")
                block_size = self.config.patch_size * self.config.output_patches
                for offset in range(0, horizon, block_size):
                    patches = min(
                        self.config.output_patches,
                        (horizon - offset + self.config.patch_size - 1)
                        // self.config.patch_size,
                    )
                    prepared = self.tokenize_next(history, past_covariates=past)
                    prepared["prediction_patches"] = patches
                    part = self.forward_encoded(self.encode_tokens(prepared), prepared)
                    outputs.append(part.quantiles)
                    history = torch.cat([history, part.point], dim=-1)[
                        ..., -self.config.context_length :
                    ]
                    if past is not None:
                        missing = past.new_full(
                            (*past.shape[:-1], part.point.shape[-1]), float("nan")
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
    def load(cls, path, *, device="cuda", dtype=torch.bfloat16, checkpoint=None):
        if checkpoint is None:
            checkpoint = torch.load(
                path, map_location="cpu", weights_only=True, mmap=True
            )
        # Restoring every saved tensor makes random initialization redundant.
        # Meta construction avoids allocating/orthogonalizing a full CPU model
        # before allocating the actual inference model on the destination.
        settings = ForecastConfig(**checkpoint["config"])
        # Import the backend outside the meta context so module-level CUDA
        # setup and any backend constants retain their intended device.
        load_rwkv(ctx_len=settings.max_patches, inference_only=False)
        with torch.device("meta"):
            model = cls(settings)
        model = model.to(dtype=dtype).to_empty(device=device)
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
            if name.startswith(("blocks.", "variate.", "temporal."))
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
        "variate_mixing_after_blocks": [
            layer
            for layer, module in enumerate(model.variate_after)
            if module is not None
        ],
        "temporal_attention_after_blocks": [
            layer
            for layer, module in enumerate(model.temporal_after)
            if module is not None
        ],
        "float32_modules": ["embedding", "norm", "head"]
        if model.config.fp32_io
        else [],
        "input_feature_dimension": 3 * model.config.patch_size,
        "output_dimension": model.head.skip.out_features,
        "direct_forecast_timesteps": model.config.patch_size
        * model.config.output_patches,
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
    torch.set_num_threads(config["training"].get("cpu_threads", 4))
    data = read_artifact(location(config, "embedded") / "embeddings.pt", config)
    # Detect an incompatible completed run before replacing its split, inventory
    # or architecture reports. The changed validation tail requires fresh data.
    from checkpoints import resume_path

    previous = resume_path(config)
    if previous is not None:
        metadata = torch.load(
            previous, map_location="cpu", weights_only=True, mmap=True
        )
        identity = metadata.get("identity", {})
        if identity.get("input_signature") != signature(config):
            raise ValueError(
                f"Saved training checkpoint uses a different model/data policy: {previous}. "
                f"Saved layers={identity.get('model', {}).get('layers', 'unknown')}; "
                f"requested layers={config['model']['layers']}. "
                "The prepared artifacts already match the current configuration; "
                "do not rerun transform.py or embed.py for this error. "
                "Archive the existing model results directory, then run train.py again "
                "to initialize fresh weights. Auto-resume cannot use this incompatible checkpoint."
            )
        if (
            config["data"].get("validation_policy") == "sampled_training_origins"
            and identity.get("optimization", {}).get("validation_split_policy")
            != "dataset_disjoint_v1"
        ):
            raise ValueError(
                "Saved checkpoint predates dataset-disjoint validation. Its weights may already "
                "have trained on the validation tables. Set training.resume=false to start fresh "
                "or choose a new training output directory; transform/embed do not need rerunning "
                "for this split change."
            )
        del metadata
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
    if config["training"].get("diagnostic_seconds") is not None:
        limit = config["training"].get("diagnostic_validation_tables", 64)
        if isinstance(limit, bool) or not isinstance(limit, int) or limit < 1:
            raise ValueError("training.diagnostic_validation_tables must be positive")
        validation_records = validation_records[:limit]
        logger.warning(
            "Diagnostic run: time budget={}s; final validation limited to {} tables; output={}",
            config["training"]["diagnostic_seconds"],
            len(validation_records),
            location(config, "training"),
        )
    inventory = None
    if config["model"].get("objective") == "next_patch":
        inventory = prepare_token_inventory(
            data,
            config,
            groups=(records, validation_records, test_records, dataset_split),
        )
    dataset_split["input_signature"] = signature(config)
    write_json(location(config, "training") / "dataset_split.json", dataset_split)
    emit_training_event(
        "dataset_split",
        **{
            k: v
            for k, v in dataset_split.items()
            if k
            not in (
                "train_sources",
                "validation_sources",
                "test_sources",
                "source_keys",
                "source_sha256",
                "excluded_short_sources",
                "evaluation_excluded_family_sources",
            )
        },
        split_path=str(location(config, "training") / "dataset_split.json"),
    )
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
    model.auto_microbatch = t.get("auto_microbatch", True)
    logger.info(
        "Training memory policy: optimizer tables={} x accumulation={} | physical "
        "microbatch<={} variable-patch tokens<={} | checkpoint threshold={} | OOM retry={}",
        t.get("batch_size", 1),
        t.get("gradient_accumulation_steps", 1),
        t.get("microbatch_size", 4),
        t.get("microbatch_token_limit", 1536),
        model.checkpoint_token_threshold,
        model.auto_microbatch,
    )
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
        dtype=str(model.compute_dtype),
        device=str(next(model.parameters()).device),
    )
    write_json(location(config, "training") / "architecture.json", report)
    logger.info("Measured model parameter groups={}", report["groups"])
    logger.info(
        "Forecast head: {} output patches ({} timesteps) per context pass",
        settings.output_patches,
        settings.patch_size * settings.output_patches,
    )
    if settings.horizon_loss_weights is not None:
        logger.info(
            "Exact per-lead loss weights={} (overrides legacy sampled auxiliary loss)",
            settings.horizon_loss_weights,
        )
    elif settings.output_patches > 1:
        logger.info(
            "Legacy sampled auxiliary loss weight={}", settings.multi_horizon_weight
        )
    logger.info(
        "Device={} dtype={} parameters={} trainable={}",
        torch.cuda.get_device_name(),
        model.compute_dtype,
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
            "steps_with_nonzero_skip_gradient": training_summary.get(
                "embedding_gradient_updates",
                sum(row["embedding_skip_gradient_norm"] > 0 for row in history),
            ),
            "optimizer_updates": training_summary["completed_updates"],
            "selected_epoch": training_summary.get(
                "exported_epoch", training_summary["best_epoch"]
            ),
            "selected_update": training_summary.get(
                "exported_step", training_summary["best_step"]
            ),
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
        selected_step=training_summary.get(
            "exported_step", training_summary["best_step"]
        ),
        selected_epoch=training_summary.get(
            "exported_epoch", training_summary["best_epoch"]
        ),
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


def launch_wsl(script, config_path, *, extra_args=()):
    if os.name != "nt":
        return False
    relative_config = (ROOT / config_path).resolve().relative_to(ROOT).as_posix()
    relative_script = Path(script).resolve().relative_to(ROOT).as_posix()
    command = "source scripts/activate_rwkv7.sh && python -u " + shlex.quote(
        relative_script
    )
    command += " --config " + shlex.quote(relative_config)
    command += "".join(" " + shlex.quote(argument) for argument in extra_args)
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
        logger.info("Training runtime storage: {}", RUNTIME_STORAGE)
        if launch_wsl(__file__, args.config):
            return
        if not torch.cuda.is_available() or not torch.cuda.is_bf16_supported():
            raise RuntimeError("Use the configured WSL CUDA bf16 runtime")
        with timed("TRAIN stage", cuda=True):
            train(config)


if __name__ == "__main__":
    main()

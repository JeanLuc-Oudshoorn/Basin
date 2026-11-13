"""Bounded CUDA graph execution and optional recurrent patch rollout.

Engines belong to one frozen evaluation model and are not thread safe. Clear them
after changing weights/device. Graph outputs are cloned so callers own results.
"""

from collections import OrderedDict
from time import perf_counter

import torch
from loguru import logger
from torch.nn import functional as F
from torch.utils._pytree import tree_map


class GraphCache:
    def __init__(self, max_entries=4, enabled=True):
        if (
            not isinstance(max_entries, int)
            or isinstance(max_entries, bool)
            or max_entries < 1
        ):
            raise ValueError("graph max_entries must be a positive integer")
        self.max_entries = max_entries
        self.enabled = enabled
        self.entries = OrderedDict()

    def clear(self):
        self.entries.clear()

    def run(self, key, function, tensors):
        if not self.enabled:
            return function(tensors)
        if key not in self.entries:
            # Retain warmed shapes: continual eviction on a varied corpus can
            # spend more time recapturing graphs than forecasting.
            if len(self.entries) >= self.max_entries:
                return function(tensors)
            start = perf_counter()
            static = {k: v.clone() for k, v in tensors.items()}
            stream = torch.cuda.Stream(device=next(iter(tensors.values())).device)
            stream.wait_stream(torch.cuda.current_stream())
            with torch.cuda.stream(stream):
                for _ in range(3):
                    function(static)
            torch.cuda.current_stream().wait_stream(stream)
            graph = torch.cuda.CUDAGraph()
            with torch.cuda.graph(graph, stream=stream):
                output = function(static)
            self.entries[key] = (graph, static, output)
            logger.info(
                "Inference CUDA graph captured shape={} seconds={:.3f} cache={}/{}",
                key,
                perf_counter() - start,
                len(self.entries),
                self.max_entries,
            )
        graph, static, output = self.entries[key]
        self.entries.move_to_end(key)
        for name, value in tensors.items():
            static[name].copy_(value)
        graph.replay()
        return tree_map(lambda value: value.clone(), output)


class FastInference:
    def __init__(self, model, *, cuda_graphs=True, max_graphs=4):
        self.model = model
        self.graphs = GraphCache(max_graphs, cuda_graphs)

    def clear(self):
        self.graphs.clear()

    def recurrent_quantiles(self, prepared, states, reuse):
        from recurrent import recurrent_quantiles

        inputs = {k: prepared[k] for k in ("features", "loc", "scale", "roles")}
        if reuse:
            for name in ("features", "loc", "scale"):
                inputs[name] = inputs[name][:, :, -1:].contiguous()
        for layer, state in enumerate(states):
            for component, tensor in enumerate(state):
                inputs[f"state_{layer}_{component}"] = tensor
        metadata = {
            k: v for k, v in prepared.items() if not isinstance(v, torch.Tensor)
        }

        def forward(tensors):
            state = [
                tuple(tensors[f"state_{layer}_{component}"] for component in range(3))
                for layer in range(len(states))
            ]
            quantiles = recurrent_quantiles(
                self.model, dict(metadata, **tensors), state
            )
            return quantiles, state

        key = ("recurrent", tuple(inputs["features"].shape), prepared["targets"], reuse)
        quantiles, next_states = self.graphs.run(key, forward, inputs)
        return quantiles, next_states

    @torch.inference_mode()
    def quantiles(self, prepared):
        model = self.model
        n = prepared["padded_tokens"]
        # Preserve the embedding GEMM's exact real-token shape. Padding raw
        # features before projection can change BF16 rounding for real tokens.
        inputs = {"encoded": model.encode_tokens(prepared)}
        for name, fill in (("loc", 0), ("scale", 1)):
            value = prepared[name]
            inputs[name] = F.pad(value, (0, 0, 0, n - value.shape[-2]), value=fill)
        inputs["roles"] = prepared["roles"]
        inputs["last_token_indices"] = prepared.get(
            "last_token_indices",
            torch.full(
                (prepared["b"],),
                prepared["tokens"] - 1,
                device=inputs["encoded"].device,
                dtype=torch.long,
            ),
        )
        # All real lengths in a kernel bucket share a graph. Last indices remain
        # dynamic device inputs; padding must never become a forecast origin.
        metadata = {
            k: v for k, v in prepared.items() if not isinstance(v, torch.Tensor)
        }
        metadata.update(tokens=n, labels=None, is_training=False)
        key = (
            tuple(inputs["encoded"].shape),
            prepared["targets"],
            prepared.get("prediction_patches", 1),
            str(inputs["encoded"].device),
        )
        if (
            key not in self.graphs.entries
            and len(self.graphs.entries) >= self.graphs.max_entries
        ):
            return model.forward_encoded(inputs["encoded"], prepared).quantiles

        def forward(tensors):
            from recurrent import stateless_quantiles

            batch = dict(metadata, **tensors)
            return stateless_quantiles(model, batch)

        return self.graphs.run(key, forward, inputs)

    @torch.inference_mode()
    def forecast(self, context, horizon, *, past_covariates=None, recurrent=False):
        if isinstance(horizon, bool) or not isinstance(horizon, int) or horizon < 1:
            raise ValueError("horizon must be a positive integer")
        model = self.model
        if model.config.objective != "next_patch":
            raise ValueError("Fast inference requires the next_patch objective")
        device = next(model.parameters()).device
        training = model.training
        model.eval()
        try:
            # CPU preprocessing avoids .any() synchronizations on the GPU.
            history = context.float().cpu()
            past = None if past_covariates is None else past_covariates.float().cpu()
            outputs, states = [], None
            patch = model.config.patch_size
            block_size = patch if recurrent else patch * model.config.output_patches
            for offset in range(0, horizon, block_size):
                cpu = model.tokenize_next(history, past_covariates=past)
                prepared = {
                    k: v.to(device) if isinstance(v, torch.Tensor) else v
                    for k, v in cpu.items()
                }
                if not recurrent:
                    prepared["prediction_patches"] = min(
                        model.config.output_patches,
                        (horizon - offset + patch - 1) // patch,
                    )
                if recurrent:
                    from recurrent import empty_state

                    # A rolling window alters prefix statistics and time features:
                    # discard state and prefill again instead of silently changing
                    # the trained model's 2048-point window semantics.
                    reuse = (
                        states is not None
                        and history.shape[-1] <= model.config.context_length
                    )
                    if not reuse:
                        states = empty_state(model, cpu["b"], len(cpu["roles"]))
                    quantiles, states = self.recurrent_quantiles(
                        prepared, states, reuse
                    )
                else:
                    quantiles = self.quantiles(prepared)
                quantiles = quantiles.float().cpu()
                if not torch.isfinite(quantiles).all():
                    raise FloatingPointError("Nonfinite forecast quantiles")
                outputs.append(quantiles)
                if offset + block_size >= horizon:
                    break
                point = quantiles[..., model.config.quantiles.index(0.5)]
                # Keep at most one extra patch to detect when to reset state.
                history = torch.cat(
                    (history[..., -model.config.context_length :], point), dim=-1
                )
                if past is not None:
                    missing = past.new_full(
                        (*past.shape[:-1], point.shape[-1]), float("nan")
                    )
                    past = torch.cat(
                        (past[..., -model.config.context_length :], missing), dim=-1
                    )
            return torch.cat(outputs, dim=-2)[..., :horizon, :]
        finally:
            model.train(training)


def configure_fast_inference(model, options):
    """Runtime option; no parameter names or saved checkpoint format change."""
    enabled = options.get("inference_cuda_graphs", False)
    model.fast_inference = (
        FastInference(model, max_graphs=options.get("inference_graph_cache_size", 4))
        if enabled
        and not model.training
        and next(model.parameters()).is_cuda
        and model.config.objective == "next_patch"
        else None
    )
    logger.info(
        "Inference CUDA graphs enabled={} cache_limit={} (first use captures each retained shape)",
        model.fast_inference is not None,
        options.get("inference_graph_cache_size", 4),
    )

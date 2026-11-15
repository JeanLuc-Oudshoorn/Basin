"""Bounded CUDA graph execution and optional recurrent patch rollout.

Engines belong to one frozen evaluation model and are not thread safe. Clear them
after changing weights/device. Graph outputs are cloned so callers own results.
"""

import math
from collections import OrderedDict
from time import perf_counter

import torch
from loguru import logger
from torch.nn import functional as F
from torch.utils._pytree import tree_map


class GraphCache:
    def __init__(self, max_entries=4, enabled=True, memory_gib=None):
        if (
            not isinstance(max_entries, int)
            or isinstance(max_entries, bool)
            or max_entries < 1
        ):
            raise ValueError("graph max_entries must be a positive integer")
        self.max_entries = max_entries
        self.enabled = enabled
        if memory_gib is not None and (
            isinstance(memory_gib, bool)
            or not isinstance(memory_gib, (float, int))
            or not math.isfinite(memory_gib)
            or memory_gib <= 0
        ):
            raise ValueError("graph memory_gib must be positive and finite")
        self.memory_limit = None if memory_gib is None else int(memory_gib * 2**30)
        self.entries = OrderedDict()
        self.retained_bytes = 0
        self.largest_capture_bytes = 0
        self.capture_bytes_per_input_byte = 0.0
        self.rejected = set()
        self.requests = self.replay_requests = self.captures = (
            self.fallback_requests
        ) = 0
        self.capture_seconds = 0.0
        self.fallback_reasons = {}
        self.allowed_keys = None

    def clear(self):
        self.entries.clear()
        self.retained_bytes = 0
        self.largest_capture_bytes = 0
        self.capture_bytes_per_input_byte = 0.0
        self.rejected.clear()
        self.allowed_keys = None

    def summary(self):
        return {
            "enabled": self.enabled,
            "requests": self.requests,
            "replay_requests": self.replay_requests,
            "replay_coverage": self.replay_requests / self.requests
            if self.requests
            else 0.0,
            "captures": self.captures,
            "capture_seconds": self.capture_seconds,
            "fallback_requests": self.fallback_requests,
            "fallback_reasons": dict(self.fallback_reasons),
            "cached_profiles": len(self.entries),
            "max_profiles": self.max_entries,
            "retained_cache_bytes": self.retained_bytes,
            "cache_memory_limit_bytes": self.memory_limit,
            "admitted_profiles": len(self.allowed_keys)
            if self.allowed_keys is not None
            else None,
        }

    def fallback(self, reason, function, tensors):
        if reason not in self.fallback_reasons and reason not in (
            "disabled",
            "low_reuse",
        ):
            logger.warning(
                "Inference replay admission reached {}: cached_profiles={} cache_MiB={:.1f}; uncached requests use forward-only execution; coverage is recorded in holdout metrics",
                reason,
                len(self.entries),
                self.retained_bytes / 2**20,
            )
        self.fallback_requests += 1
        self.fallback_reasons[reason] = self.fallback_reasons.get(reason, 0) + 1
        return function(tensors)

    def run(self, key, function, tensors):
        self.requests += 1
        already_replayed = False
        if not self.enabled:
            return self.fallback("disabled", function, tensors)
        if key not in self.entries:
            if self.allowed_keys is not None and key not in self.allowed_keys:
                return self.fallback("low_reuse", function, tensors)
            # Retain warmed shapes: continual eviction on a varied corpus can
            # spend more time recapturing graphs than forecasting.
            if len(self.entries) >= self.max_entries:
                return self.fallback("profile_limit", function, tensors)
            if key in self.rejected:
                return self.fallback("memory_limit", function, tensors)
            input_bytes = sum(t.numel() * t.element_size() for t in tensors.values())
            # Admission uses observed graph-pool growth, scaled for larger
            # inputs. Keep headroom before attempting a new capture.
            estimate = max(
                32 * 2**20,
                self.largest_capture_bytes,
                int(input_bytes * (1 + self.capture_bytes_per_input_byte)),
            )
            if (
                self.memory_limit is not None
                and self.retained_bytes + estimate > self.memory_limit
            ):
                self.rejected.add(key)
                return self.fallback("memory_limit", function, tensors)
            start = perf_counter()
            reserved_before_setup = torch.cuda.memory_reserved()
            free_before_setup = torch.cuda.mem_get_info()[0]
            static = {k: v.clone() for k, v in tensors.items()}
            stream = torch.cuda.Stream(device=next(iter(tensors.values())).device)
            stream.wait_stream(torch.cuda.current_stream())
            with torch.cuda.stream(stream):
                for _ in range(3):
                    function(static)
            torch.cuda.current_stream().wait_stream(stream)
            before = torch.cuda.memory_reserved()
            free_before = torch.cuda.mem_get_info()[0]
            graph = torch.cuda.CUDAGraph()
            with torch.cuda.graph(graph, stream=stream):
                output = function(static)
            # Some allocators do not include graph pools in reserved-byte
            # statistics. Measure committed device memory after first replay too.
            # Warmup/capture can mutate stateful inputs. Restore the caller's
            # values before the first replay just as for every later replay.
            for name, value in tensors.items():
                static[name].copy_(value)
            graph.replay()
            torch.cuda.synchronize()
            already_replayed = True
            profile_bytes = input_bytes + max(
                0,
                torch.cuda.memory_reserved() - before,
                free_before - torch.cuda.mem_get_info()[0],
            )
            # Include setup/warmup growth as well: compiler-owned buffers may
            # persist outside the outer graph's private allocation pool.
            retained = max(
                profile_bytes,
                input_bytes
                + max(
                    0,
                    torch.cuda.memory_reserved() - reserved_before_setup,
                    free_before_setup - torch.cuda.mem_get_info()[0],
                ),
            )
            if (
                self.memory_limit is not None
                and self.retained_bytes + retained > self.memory_limit
            ):
                # Retire only this new graph, after its work has completed.
                # Do not evict established profiles and trigger recapture churn.
                torch.cuda.synchronize()
                del graph, static, output
                torch.cuda.empty_cache()
                self.rejected.add(key)
                self.capture_seconds += perf_counter() - start
                return self.fallback("memory_limit", function, tensors)
            self.entries[key] = (graph, static, output)
            self.retained_bytes += retained
            self.largest_capture_bytes = max(self.largest_capture_bytes, profile_bytes)
            self.capture_bytes_per_input_byte = max(
                self.capture_bytes_per_input_byte, profile_bytes / max(input_bytes, 1)
            )
            self.captures += 1
            self.capture_seconds += perf_counter() - start
            logger.debug(
                "Inference CUDA graph captured shape={} seconds={:.3f} cache={}/{}",
                key,
                perf_counter() - start,
                len(self.entries),
                self.max_entries,
            )
        graph, static, output = self.entries[key]
        self.entries.move_to_end(key)
        if not already_replayed:
            for name, value in tensors.items():
                static[name].copy_(value)
            graph.replay()
        self.replay_requests += 1
        return tree_map(lambda value: value.clone(), output)


class FastInference:
    def __init__(
        self, model, *, cuda_graphs=True, max_graphs=256, graph_memory_gib=1.5
    ):
        self.model = model
        self.graphs = GraphCache(max_graphs, cuda_graphs, memory_gib=graph_memory_gib)

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
    def scored_block(self, prepared):
        """Batched forecast block for validation, replayed per batch shape.

        Returns (quantiles, model-space quantiles, loc, scale) for the direct
        head and (quantiles, loc, scale) otherwise: everything the validation
        loss needs. Weights are read at replay time, so graphs captured at one
        validation check stay valid after later optimizer updates.
        """
        inputs, metadata = self._padded_inputs(prepared)
        key = (
            "scored",
            tuple(inputs["encoded"].shape),
            prepared["targets"],
            prepared.get("prediction_patches", 1),
            str(inputs["encoded"].device),
        )

        def forward(tensors):
            from recurrent import stateless_block

            return stateless_block(self.model, dict(metadata, **tensors))

        return self.graphs.run(key, forward, inputs)

    @torch.inference_mode()
    def quantiles(self, prepared):
        inputs, metadata = self._padded_inputs(prepared)
        key = (
            tuple(inputs["encoded"].shape),
            prepared["targets"],
            prepared.get("prediction_patches", 1),
            str(inputs["encoded"].device),
        )

        def forward(tensors):
            from recurrent import stateless_quantiles

            batch = dict(metadata, **tensors)
            return stateless_quantiles(self.model, batch)

        return self.graphs.run(key, forward, inputs)

    def _padded_inputs(self, prepared):
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
        return inputs, metadata

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
        FastInference(
            model,
            max_graphs=options.get("inference_graph_cache_size", 256),
            graph_memory_gib=options.get("inference_graph_memory_gib", 1.5),
        )
        if enabled
        and not model.training
        and next(model.parameters()).is_cuda
        and model.config.objective == "next_patch"
        else None
    )
    logger.info(
        "Inference CUDA graphs enabled={} cache_limit={} memory_limit_GiB={} (first use captures each retained shape)",
        model.fast_inference is not None,
        options.get("inference_graph_cache_size", 256),
        options.get("inference_graph_memory_gib", 1.5),
    )


def inference_execution_summary(model):
    engine = getattr(model, "fast_inference", None)
    return (
        engine.graphs.summary()
        if isinstance(engine, FastInference)
        else {"enabled": False}
    )

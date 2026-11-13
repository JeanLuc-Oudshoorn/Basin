"""Bounded, accumulation-safe CUDA graphs for uncheckpointed variate layers.

Capture happens before the model forward, outside activation-checkpoint contexts.
The optimizer, clipping, scheduler, RNG-dependent loss and RWKV blocks stay eager.
Runtime objects never enter state_dict or training checkpoint identity.
"""

import gc
from collections import Counter, OrderedDict
from time import perf_counter

import torch
from loguru import logger
from torch import nn


class _AttentionProxy(nn.Module):
    def __init__(self, mix):
        super().__init__()
        self.norm = mix.norm
        self.attention = mix.attention
        self.execute = mix.compiled_forward
        # Compile against exactly the original module and its guards, not a new
        # proxy type: a separately compiled graph can round BF16 differently.
        object.__setattr__(self, "source", mix)

    def forward(self, x):
        return self.execute(self.source, x)


def _protect_accumulation(parameter):
    # A replay returns static gradient storage. Keep AccumulateGrad's retained
    # gradient separate before another microbatch can overwrite that storage.
    if parameter.grad is not None:
        parameter.grad = parameter.grad.clone()


class TrainingGraphs:
    def __init__(self, model, options):
        self.model = model
        self.max_shapes = options.get("training_graph_cache_shapes", 2)
        self.budget = int(options.get("training_graph_memory_gib", 0.75) * 2**30)
        self.allocator_limit = options.get("cuda_memory_limit_gib", 8.0)
        self.entries = {}
        self.blocked = set()
        self.shapes = OrderedDict()
        self.costs = {}
        self.hooks = {}
        self.bytes = 0
        self.counts = Counter()
        self.active_layers = set()
        self.disabled_reason = None
        self.min_reuse = options.get("training_graph_min_reuse", 8)
        self.max_tokens = options.get("training_graph_max_tokens", 32)
        self.update_index = 0
        self.last_seen = {}
        self.reuse_counts = Counter()

    def begin_update(self):
        self.update_index += 1

    def admit(self, shape):
        """Do not pay capture/eviction costs for isolated or expensive shapes."""
        previous = self.last_seen.get(shape)
        if previous != self.update_index:
            self.reuse_counts[shape] = (
                self.reuse_counts[shape] + 1 if previous == self.update_index - 1 else 1
            )
            self.last_seen[shape] = self.update_index
        if self.max_tokens is not None and shape[2] > self.max_tokens:
            self.counts["length_skips"] += 1
            return False
        if shape not in self.shapes and self.reuse_counts[shape] < self.min_reuse:
            self.counts["admission_skips"] += 1
            return False
        return True

    def prepare(self, prepared):
        """Capture eligible layers outside model forward; no corpus data retained."""
        if (
            self.disabled_reason is not None
            or not self.model.training
            or not prepared.get("is_training", False)
        ):
            return
        self.active_layers.clear()
        shape = (
            prepared["b"],
            len(prepared["roles"]),
            prepared["padded_tokens"],
            self.model.config.width,
        )
        if shape[1] > 8:
            return
        if not self.admit(shape):
            return
        layers = len(self.model.variate)
        if self.model.config.gradient_checkpointing:
            layers = min(
                layers,
                self.model.checkpoint_token_threshold
                * layers
                // (shape[0] * shape[1] * shape[2]),
            )
        if layers < 1:
            return
        if shape not in self.shapes:
            if len(self.shapes) >= self.max_shapes:
                self.retire(next(iter(self.shapes)))
            self.shapes[shape] = None
        self.shapes.move_to_end(shape)
        weight = next(self.model.parameters())
        # Evict complete old profiles BEFORE capturing any layer of this one.
        # Destroying private pools mid-profile can overlap autograd's retained
        # stream/gradient storage with newly captured graphs.
        parameter_bytes = sum(
            p.numel() * p.element_size() for p in self.model.variate[0].parameters()
        )
        per_layer = (
            3 * parameter_bytes
            + 24 * shape[0] * shape[1] * shape[2] * shape[3] * weight.element_size()
        )
        missing = sum((index, shape) not in self.entries for index in range(layers))
        desired = min(self.budget, missing * per_layer)
        while self.bytes + desired > self.budget and len(self.shapes) > 1:
            self.retire(next(s for s in self.shapes if s != shape))
        for index, mix in enumerate(self.model.variate[:layers]):
            key = (index, shape)
            if (
                key in self.entries
                or key in self.blocked
                or mix.compiled_forward is None
                or mix.attention.dropout != 0
            ):
                continue
            # Account conservatively for static parameter gradients and saved
            # attention activations before attempting a capture.
            parameter_bytes = sum(
                p.numel() * p.element_size() for p in mix.parameters()
            )
            estimate = (
                3 * parameter_bytes
                + 24 * shape[0] * shape[1] * shape[2] * shape[3] * weight.element_size()
            )
            allocated = torch.cuda.memory_allocated(weight.device)
            total = torch.cuda.get_device_properties(weight.device).total_memory
            limit = (
                min(total * 0.8, float(self.allocator_limit) * 2**30)
                if self.allocator_limit is not None
                else total * 0.8
            )
            if (
                self.bytes + estimate > self.budget
                or allocated + estimate > limit - 256 * 2**20
            ):
                self.counts["memory_fallbacks"] += 1
                self.blocked.add(key)
                continue
            start = perf_counter()
            before = torch.cuda.memory_allocated(weight.device)
            proxy = _AttentionProxy(mix).train()
            sample = torch.zeros(
                shape, device=weight.device, dtype=weight.dtype, requires_grad=True
            )
            # Compilation/warmup must not advance the training loss's random
            # lead sampling or change the exact RNG sequence after resume.
            stream = torch.cuda.current_stream(weight.device)
            try:
                with torch.random.fork_rng(devices=[weight.device.index]):
                    graph = torch.cuda.make_graphed_callables(proxy, (sample,))
            except RuntimeError as error:
                # A failed capture is an optional execution optimization, not a
                # failed training update. PyTorch's graph context can raise in
                # capture_end before restoring its stream, so restore it here
                # before checking that the CUDA context is still usable.
                if not self._recoverable_capture_error(error):
                    raise
                torch.cuda.set_stream(stream)
                torch.cuda.synchronize(weight.device)
                self.disabled_reason = f"{type(error).__name__}: {error}".splitlines()[
                    0
                ]
                self.counts["capture_failures"] += 1
                self.entries.clear()
                self.costs.clear()
                self.shapes.clear()
                self.blocked.clear()
                self.bytes = 0
                self.active_layers.clear()
                for handles in self.hooks.values():
                    for handle in handles:
                        handle.remove()
                self.hooks.clear()
                del proxy, sample
                gc.collect()
                logger.warning(
                    "Training CUDA graph capture failed at layer={} shape={}: {}. "
                    "CUDA health check passed; continuing this run with compiled "
                    "attention and graph capture disabled. No optimizer update was skipped.",
                    index,
                    shape,
                    error,
                )
                return
            finally:
                torch.cuda.set_stream(stream)
            cost = max(estimate, torch.cuda.memory_allocated(weight.device) - before)
            if self.bytes + cost > self.budget:
                self.blocked.add(key)
                self.counts["memory_fallbacks"] += 1
                del graph, proxy, sample
                gc.collect()
                continue
            self.entries[key] = graph
            self.costs[key] = cost
            self.bytes += cost
            if index not in self.hooks:

                def protect(parameter, layer=index):
                    if layer in self.active_layers:
                        _protect_accumulation(parameter)

                self.hooks[index] = [
                    p.register_post_accumulate_grad_hook(protect)
                    for p in mix.parameters()
                    if p.requires_grad
                ]
            self.counts["captures"] += 1
            logger.debug(
                "Training graph capture layer={} shape={} seconds={:.3f} accounted_MiB={:.1f}",
                index,
                shape,
                perf_counter() - start,
                self.bytes / 2**20,
            )

    def run(self, index, x):
        graph = self.entries.get((index, tuple(x.shape)))
        if graph is None:
            self.active_layers.discard(index)
            self.counts["eager_calls"] += 1
            return self.model.variate[index].forward_ungraphed(x)
        self.counts["replays"] += 1
        self.active_layers.add(index)
        return graph(x)

    def summary(self):
        return dict(
            self.counts,
            shapes=len(self.shapes),
            entries=len(self.entries),
            accounted_mib=self.bytes / 2**20,
            disabled_reason=self.disabled_reason,
            min_reuse=self.min_reuse,
            max_tokens=self.max_tokens,
        )

    @staticmethod
    def _recoverable_capture_error(error):
        # Never conceal illegal memory accesses, bad input, or unrelated model
        # errors. synchronize above must also succeed before fallback is safe.
        # CUDA exceptions append generic debugging advice mentioning device-side
        # assertions even for harmless capture errors. Classify the error itself.
        message = str(error).partition("\n")[0].lower()
        if any(
            marker in message
            for marker in ("illegal memory", "device-side assert", "misaligned address")
        ):
            return False
        return isinstance(error, torch.OutOfMemoryError) or any(
            marker in message
            for marker in (
                "during capture",
                "stream capture",
                "streamcapture",
                "when stream is capturing",
                "not permitted when stream",
            )
        )

    def retire(self, shape):
        # backward() returning means work was ENQUEUED, not completed on every
        # autograd stream. Private graph pools must outlive all of that work.
        torch.cuda.synchronize(next(self.model.parameters()).device)
        for key in [key for key in self.entries if key[1] == shape]:
            del self.entries[key]
            self.bytes -= self.costs.pop(key)
        self.blocked = {key for key in self.blocked if key[1] != shape}
        self.shapes.pop(shape, None)
        self.counts["profile_evictions"] += 1
        gc.collect()

    def close(self):
        if self.entries:
            torch.cuda.synchronize(next(self.model.parameters()).device)
        for handles in self.hooks.values():
            for handle in handles:
                handle.remove()
        self.hooks.clear()
        self.entries.clear()
        self.costs.clear()
        self.bytes = 0
        gc.collect()
        for mix in self.model.variate:
            mix.training_graphs = None


def configure_training_graphs(model, options):
    previous = getattr(model, "training_graphs", None)
    if previous is not None:
        previous.close()
    active = (
        options.get("training_cuda_graphs", False)
        and model.training
        and next(model.parameters()).is_cuda
    )
    model.training_graphs = TrainingGraphs(model, options) if active else None
    for index, mix in enumerate(model.variate):
        mix.training_graphs = model.training_graphs
        mix.training_graph_index = index
    logger.info(
        "Training CUDA graphs enabled={} max_shapes={} memory_budget_GiB={}",
        active,
        options.get("training_graph_cache_shapes", 2),
        options.get("training_graph_memory_gib", 0.75),
    )
    return model.training_graphs

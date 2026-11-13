"""FP32 AdamW updates with low-precision model parameters for computation."""

import torch
from loguru import logger


class FP32MasterAdamW(torch.optim.AdamW):
    """Accumulate sub-BF16 updates in FP32, including across checkpoint resume."""

    def __init__(self, parameters, *, gradient_bucket_bytes=64 * 2**20, **kwargs):
        if (
            isinstance(gradient_bucket_bytes, bool)
            or not isinstance(gradient_bucket_bytes, int)
            or gradient_bucket_bytes < 1
        ):
            raise ValueError("gradient_bucket_bytes must be a positive integer")
        self.gradient_bucket_bytes = gradient_bucket_bytes
        self.model_parameters = list(parameters)
        masters = [
            torch.nn.Parameter(p.detach().float().clone())
            for p in self.model_parameters
        ]
        super().__init__(masters, **kwargs)
        logger.info(
            "FP32 optimizer gradient staging budget={:.1f} MiB; masters and moments remain FP32",
            gradient_bucket_bytes / 2**20,
        )

    @property
    def master_parameters(self):
        return [p for group in self.param_groups for p in group["params"]]

    def zero_grad(self, set_to_none=True):
        super().zero_grad(set_to_none=set_to_none)
        for parameter in self.model_parameters:
            if parameter.grad is not None:
                if set_to_none:
                    parameter.grad = None
                else:
                    parameter.grad.detach_()
                    parameter.grad.zero_()

    @torch.no_grad()
    def step(self, closure=None):
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()
        bucket, size = [], 0
        for master, parameter in zip(self.master_parameters, self.model_parameters):
            if parameter.grad is None:
                continue
            required = master.numel() * master.element_size()
            if bucket and size + required > self.gradient_bucket_bytes:
                self._step_bucket(bucket)
                bucket, size = [], 0
            bucket.append((master, parameter))
            size += required
        if bucket:
            self._step_bucket(bucket)
        return loss

    def _step_bucket(self, bucket):
        # AdamW updates each parameter independently. Only this bucket has
        # master gradients, so parent AdamW skips all other parameters without
        # changing their step counters or applying weight decay. Keep the public
        # parameter groups intact for schedulers and checkpoint compatibility.
        # A tensor larger than the budget is processed alone.
        try:
            for master, parameter in bucket:
                master.grad = parameter.grad.detach().float()
            super().step()
            for master, parameter in bucket:
                parameter.copy_(master)
        finally:
            # Do not retain a model-sized FP32 gradient copy. No optimizer OOM
            # is retried: an earlier bucket may already have applied updates.
            for master, _ in bucket:
                master.grad = None

    def state_dict(self):
        result = super().state_dict()
        result["fp32_master_weights"] = [p.detach() for p in self.master_parameters]
        return result

    def load_state_dict(self, state_dict):
        weights = state_dict.get("fp32_master_weights")
        masters = self.master_parameters
        if weights is not None and (
            len(weights) != len(masters)
            or any(w.shape != p.shape for w, p in zip(weights, masters))
        ):
            raise ValueError("Saved FP32 master weights do not match model parameters")
        # AdamW loads moments onto the FP32 master parameters, promoting legacy
        # BF16 moments without throwing away steps, momentum or scheduler state.
        super().load_state_dict(state_dict)
        with torch.no_grad():
            for index, (master, parameter) in enumerate(
                zip(masters, self.model_parameters)
            ):
                master.copy_(parameter if weights is None else weights[index])
                parameter.copy_(master)
        if weights is None:
            logger.warning(
                "Upgraded legacy AdamW checkpoint to FP32 master weights and moments; "
                "training cursor and optimizer steps retained. Previously rounded "
                "updates cannot be recovered."
            )

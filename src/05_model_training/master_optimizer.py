"""FP32 AdamW updates with low-precision model parameters for computation.

Two execution paths give the same weights, moments and step counts bit for bit:

* the one-pass CUDA kernel (`master_adamw_kernel.cu`), used on CUDA when it can
  be built, and
* the portable three-pass path: cast gradients to float32 in bounded buckets,
  run PyTorch's AdamW on the masters, copy the masters back.
"""

import sys
from functools import lru_cache
from pathlib import Path

import torch
from loguru import logger

KERNEL_WEIGHT_TYPES = (torch.bfloat16, torch.float32)


@lru_cache(maxsize=1)
def load_fused_kernel():
    """Build on first use, then load, the one-pass CUDA update."""
    from torch.utils.cpp_extension import load

    try:
        from rwkv7_cuda_paths import configure_cuda_home
    except ImportError:
        sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
        from rwkv7_cuda_paths import configure_cuda_home

    configure_cuda_home()
    load(
        name="basin_master_adamw_v1",
        sources=[
            str(Path(__file__).with_name("master_adamw_kernel" + suffix))
            for suffix in (".cpp", ".cu")
        ],
        # No fast-math: the kernel must round exactly as PyTorch's AdamW does.
        # The project's toolkit pairs a newer nvcc with the CUDA 13.0 runtime
        # headers PyTorch was built against; PyTorch's CUDA headers pull in
        # CCCL, which refuses that pairing unless told otherwise. The parity
        # tests in tests/test_master_optimizer.py check the resulting kernel.
        extra_cuda_cflags=["-O3", "-DCCCL_DISABLE_CTK_COMPATIBILITY_CHECK"],
        is_python_module=False,
        verbose=False,
    )
    return torch.ops.basin_master_adamw_v1


class FP32MasterAdamW(torch.optim.AdamW):
    """Accumulate sub-BF16 updates in FP32, including across checkpoint resume."""

    def __init__(
        self,
        parameters,
        *,
        gradient_bucket_bytes=64 * 2**20,
        fused_kernel=True,
        **kwargs,
    ):
        if (
            isinstance(gradient_bucket_bytes, bool)
            or not isinstance(gradient_bucket_bytes, int)
            or gradient_bucket_bytes < 1
        ):
            raise ValueError("gradient_bucket_bytes must be a positive integer")
        self.gradient_bucket_bytes = gradient_bucket_bytes
        parameters = list(parameters)
        if parameters and isinstance(parameters[0], dict):
            # Parameter groups (per-group learning rate / weight decay). Masters
            # keep the same grouping and order as the model parameters.
            groups = [dict(group, params=list(group["params"])) for group in parameters]
            self.model_parameters = [p for group in groups for p in group["params"]]
            masters = [
                dict(
                    group,
                    params=[
                        torch.nn.Parameter(p.detach().float().clone())
                        for p in group["params"]
                    ],
                )
                for group in groups
            ]
        else:
            self.model_parameters = parameters
            masters = [
                torch.nn.Parameter(p.detach().float().clone())
                for p in self.model_parameters
            ]
        super().__init__(masters, **kwargs)
        self._kernel = None
        self._plan = None  # per group: handles to tensor lists held by the kernel
        self._waiting = []  # model parameters that have not had a gradient yet
        if (
            fused_kernel
            and self.model_parameters
            and all(
                p.is_cuda and p.dtype in KERNEL_WEIGHT_TYPES
                for p in self.model_parameters
            )
        ):
            try:
                self._kernel = load_fused_kernel()
            except Exception as error:  # a missing compiler must not stop training
                logger.warning(
                    "One-pass FP32-master AdamW kernel unavailable ({}: {}); using "
                    "the slower three-pass update. Results are identical.",
                    type(error).__name__,
                    str(error).splitlines()[0] if str(error) else "",
                )
        if self._kernel is not None:
            logger.info(
                "FP32 optimizer update: one-pass CUDA kernel, no gradient staging; "
                "masters and moments remain FP32"
            )
        else:
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
        if self._kernel is not None and self._kernel_applies():
            self._kernel_step()
            return loss
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

    def _kernel_applies(self):
        # The kernel implements plain AdamW with PyTorch's fused state layout.
        return all(
            group.get("fused")
            and not group.get("amsgrad")
            and not group.get("maximize")
            and not group.get("capturable")
            and not group.get("differentiable")
            and not isinstance(group["lr"], torch.Tensor)
            for group in self.param_groups
        )

    def _release_plan(self):
        plan, self._plan = self._plan, None
        for _, handles in plan or ():
            for handle in handles:
                self._kernel.release_plan(handle)

    def __del__(self):
        try:
            self._release_plan()
        except Exception:  # nosec B110 - interpreter shutdown; nothing to recover
            pass

    def _kernel_plan(self):
        """Register tensor lists per group and weight dtype with the kernel.

        The lists are handed over once and reused: passing thousands of
        tensors on every update costs milliseconds. As in torch.optim.AdamW,
        a parameter gets moments and a step counter at its first gradient.
        Until then it is left out and checked each step.
        """
        self._release_plan()
        plan, waiting, index = [], [], 0
        for group in self.param_groups:
            parts = {}
            for master in group["params"]:
                weight = self.model_parameters[index]
                index += 1
                state = self.state.get(master)
                if not state:
                    if weight.grad is None:
                        waiting.append(weight)
                        continue
                    state = self.state[master]
                    state["step"] = torch.zeros(
                        (), dtype=torch.float32, device=master.device
                    )
                    for key in ("exp_avg", "exp_avg_sq"):
                        state[key] = torch.zeros_like(
                            master, memory_format=torch.preserve_format
                        )
                part = parts.setdefault(weight.dtype, ([], [], [], [], []))
                for values, value in zip(
                    part,
                    (
                        master,
                        state["exp_avg"],
                        state["exp_avg_sq"],
                        state["step"],
                        weight,
                    ),
                ):
                    values.append(value)
            plan.append(
                (
                    group,
                    [self._kernel.create_plan(*part) for part in parts.values()],
                )
            )
        self._plan, self._waiting = plan, waiting
        return plan

    def _kernel_step(self):
        plan = self._plan
        if plan is None or any(w.grad is not None for w in self._waiting):
            plan = self._kernel_plan()
        for group, handles in plan:
            beta1, beta2 = group["betas"]
            for handle in handles:
                # The kernel reads each weight's current gradient itself.
                self._kernel.run_plan(
                    handle,
                    group["lr"],
                    beta1,
                    beta2,
                    group["weight_decay"],
                    group["eps"],
                )

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
        if self._kernel is not None:
            self._release_plan()  # the loaded moments are new tensors
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

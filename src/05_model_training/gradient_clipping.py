"""Gradient-norm clipping without the work that changes nothing.

`torch.nn.utils.clip_grad_norm_` does two things this module avoids, with
bit-identical results:

* It multiplies every gradient by the clipping coefficient even when that
  coefficient is exactly 1, which is the case whenever the norm is below the
  limit. Multiplying by one changes no value. Here the caller reads the
  coefficient together with its other metrics and scales only when it is
  below one.
* It stacks the per-tensor norms one by one. With float32 input/output layers
  next to a BF16 backbone the stack also converts hundreds of BF16 scalars one
  at a time. Here each dtype group is stacked first; the resulting vector, its
  order and its values are the same.

On the 221.5M model this takes clipping from about 15 ms to about 3 ms per
update (measured 2026-11-17, RTX 5070).
"""

import torch
from torch.utils._foreach_utils import _group_tensors_by_device_and_dtype


@torch.no_grad()
def measure_gradients(parameters, max_norm):
    """Total gradient norm and the coefficient `clip_grad_norm_` would apply.

    Returns `(total_norm, coefficient, groups)`. Nothing is modified. `groups`
    are the gradients by device and dtype, to pass to `scale_gradients` when
    the coefficient is below one.
    """
    gradients = [p.grad for p in parameters if p.grad is not None]
    if not gradients:
        return torch.tensor(0.0), torch.tensor(1.0), []
    first_device = gradients[0].device
    groups = [
        tensors
        for (_, _), ([tensors], _) in _group_tensors_by_device_and_dtype(
            [gradients]
        ).items()
    ]
    # Same vector as torch.stack over every norm in group order; torch.cat
    # promotes mixed dtypes exactly as torch.stack does.
    norms = torch.cat(
        [
            torch.stack(torch._foreach_norm(tensors, 2.0)).to(first_device)
            for tensors in groups
        ]
    )
    total_norm = torch.linalg.vector_norm(norms, 2.0)
    coefficient = torch.clamp(float(max_norm) / (total_norm + 1e-6), max=1.0)
    return total_norm, coefficient, groups


@torch.no_grad()
def scale_gradients(groups, coefficient):
    """Scale in place, as `clip_grad_norm_` does for every update."""
    for tensors in groups:
        torch._foreach_mul_(tensors, coefficient.to(tensors[0].device))


def clip_gradients(parameters, max_norm):
    """Drop-in for `clip_grad_norm_(..., foreach=True)`; one extra transfer.

    The trainer does not use this: it folds the coefficient into the metrics
    transfer it already makes. Kept for benchmarks and tests.
    """
    total_norm, coefficient, groups = measure_gradients(parameters, max_norm)
    if float(coefficient) != 1.0:
        scale_gradients(groups, coefficient)
    return total_norm

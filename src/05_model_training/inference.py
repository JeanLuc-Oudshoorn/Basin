"""Bounded batching of independent forecast contexts; no recurrent state reuse."""

import torch
from torch.nn import functional as F


@torch.inference_mode()
def forecast_contexts(model, contexts, batch_size=5, *, prediction_patches=1):
    """Yield CPU quantiles in input order for contexts with matching variable roles.

    Context tensors are prepared on CPU. Padding is on the right after each
    independently normalized sequence. Gather each real final token before the
    forecast head, so another context's padding never changes its forecast.
    """
    if (
        isinstance(batch_size, bool)
        or not isinstance(batch_size, int)
        or batch_size < 1
    ):
        raise ValueError("inference batch_size must be a positive integer")
    device = next(model.parameters()).device
    was_training = model.training
    model.eval()
    try:
        if (
            isinstance(prediction_patches, bool)
            or not isinstance(prediction_patches, int)
            or prediction_patches < 1
        ):
            raise ValueError("prediction_patches must be a positive integer")
        if prediction_patches > model.config.output_patches:
            # Archived one-patch models must roll out for the same requested
            # horizon; never silently change the benchmark request for them.
            for context in contexts:
                inputs = {k: v.to(device) for k, v in context.items() if v is not None}
                yield (
                    model.forecast(
                        horizon=prediction_patches * model.config.patch_size, **inputs
                    )
                    .quantiles[0]
                    .float()
                    .cpu()
                )
            return
        for start in range(0, len(contexts), batch_size):
            chunk = contexts[start : start + batch_size]
            examples = [
                model.tokenize_next(
                    c["context"], past_covariates=c.get("past_covariates")
                )
                for c in chunk
            ]
            roles = examples[0]["roles"]
            if any(not torch.equal(e["roles"], roles) for e in examples):
                raise ValueError("Forecast batch requires matching variable roles")
            n = max(e["tokens"] for e in examples)
            prepared = dict(
                examples[0], b=len(chunk), tokens=n, padded_tokens=((n + 15) // 16) * 16
            )
            prepared["prediction_patches"] = prediction_patches
            for name, fill in (("features", 0), ("loc", 0), ("scale", 1)):
                prepared[name] = torch.cat(
                    [
                        F.pad(e[name], (0, 0, 0, n - e["tokens"]), value=fill)
                        for e in examples
                    ]
                )
            prepared["last_token_indices"] = torch.tensor(
                [e["tokens"] - 1 for e in examples]
            )
            prepared = {
                k: v.to(device) if isinstance(v, torch.Tensor) else v
                for k, v in prepared.items()
            }
            engine = getattr(model, "fast_inference", None)
            if engine is not None:
                quantiles = engine.quantiles(prepared).float().cpu()
            else:
                output = model.forward_encoded(model.encode_tokens(prepared), prepared)
                quantiles = output.quantiles.float().cpu()
            if not torch.isfinite(quantiles).all():
                raise FloatingPointError("Nonfinite forecast quantiles")
            for value in quantiles:
                yield value
    finally:
        model.train(was_training)

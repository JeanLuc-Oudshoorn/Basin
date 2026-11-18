"""Operation-by-operation equivalence test: original CUDA kernels vs the PyTorch port.

Needs the WSL CUDA runtime. For real holdout requests it walks the ORIGINAL
model layer by layer and, at every step, feeds the original's own bfloat16
inputs to both the original kernel and the port's replacement function. Because
the inputs are identical, any difference is that one operation's arithmetic and
cannot hide inside accumulated rounding noise. Distances are counted in bfloat16
rounding steps (ULPs): 0 = bit-identical.
"""

import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path

import torch
from loguru import logger

ROOT = Path(__file__).resolve().parents[2]
WORK = ROOT / ".cache/port_work"
for directory in (
    "",
    "src/03_transforming_input_data",
    "src/04_patch_indexing",
    "src/05_model_training",
    "src/06_holdout_test",
):
    sys.path.insert(0, str(ROOT / directory))


def ordered_bits(x):
    bits = x.contiguous().view(torch.int16).to(torch.int32)
    magnitude = bits & 0x7FFF
    return torch.where(bits < 0, -magnitude, magnitude)


class Tally:
    """Per operation: share of bit-identical values and the size of the rest.

    ULP counts explode for values next to zero (a sign flip of 1e-30 is thousands
    of steps), so the worst difference is also given relative to the tensor's RMS.
    """

    def __init__(self):
        self.rows = defaultdict(lambda: [0, 0, 0, 0.0, 0.0, 0.0])

    def add(self, name, ours, theirs):
        row = self.rows[name]
        row[0] += ours.numel()
        if ours.dtype == torch.bfloat16:
            distance = (ordered_bits(ours) - ordered_bits(theirs)).abs()
            row[1] += int((distance == 0).sum())
            row[2] += int((distance <= 1).sum())
        else:  # float32 results: exact equality only
            same = int((ours == theirs).sum())
            row[1] += same
            row[2] += same
        delta = (ours.float() - theirs.float()).abs()
        rms = float(theirs.float().square().mean().sqrt())
        row[3] = max(row[3], float(delta.max()) / rms)
        row[4] += float(delta.sum())
        row[5] += rms * ours.numel()

    def report(self):
        return {
            name: {
                "values": values,
                "bit_identical_fraction": identical / values,
                "within_one_ulp_fraction": within / values,
                "max_abs_difference_over_rms": worst,
                "mean_abs_difference_over_rms": total / scale,
            }
            for name, (
                values,
                identical,
                within,
                worst,
                total,
                scale,
            ) in self.rows.items()
        }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--requests", type=int, default=12)
    parser.add_argument("--models", default=str(ROOT / "results/model/portable"))
    args = parser.parse_args()
    logger.remove()
    logger.add(sys.stderr, level="WARNING")
    from multi_horizon import direct_quantiles
    from recurrent import load_kernel
    from train import RWKVForecaster
    from transform import load_config
    from variate_attention import configure_variate_execution

    config = load_config(str(ROOT / "configs/config.json"))
    payload = torch.load(
        ROOT / "results/model/training/latest_checkpoint.pt",
        map_location="cpu",
        weights_only=True,
        mmap=True,
    )
    model = RWKVForecaster.load(
        None,
        checkpoint={
            "config": payload["identity"]["model"],
            "state_dict": payload["best_state_dict"],
        },
    ).eval()
    del payload
    configure_variate_execution(model, config["training"], ROOT)
    backend = sys.modules[type(model.blocks[0]).__module__]
    inference_kernel = load_kernel()

    folder = Path(args.models) / "basin-rwkv7-pytorch"
    sys.path.insert(0, str(folder))
    import modeling_basin_pytorch as port_module
    from basin_common import tokenize

    port = port_module.BasinPyTorch.from_pretrained(
        folder, device="cuda", dtype=torch.bfloat16
    )
    head = port.config.head_size
    requests = torch.load(WORK / "requests.pt", weights_only=True)
    step = len(requests) / args.requests
    tally = Tally()
    checks = {"tokenizer_identical": True, "requests": []}
    with torch.inference_mode():
        for index in (int(i * step) for i in range(args.requests)):
            request = requests[index]
            cpu = model.tokenize_next(
                request["context"], past_covariates=request["past_covariates"]
            )
            mine = tokenize(port.config, request["context"], request["past_covariates"])
            checks["tokenizer_identical"] &= all(
                torch.equal(cpu[k], mine[k])
                for k in ("features", "roles", "loc", "scale")
            )
            checks["requests"].append(
                {
                    "index": index,
                    "tokens": cpu["tokens"],
                    "variables": len(cpu["roles"]),
                }
            )
            prepared = {
                k: v.to("cuda") if isinstance(v, torch.Tensor) else v
                for k, v in cpu.items()
            }
            x = model.encode_tokens(prepared)
            tokens = cpu["tokens"]
            features = mine["features"].to("cuda", torch.bfloat16)
            embedded = (
                port.embedding(features)
                + port.roles(mine["roles"].to("cuda"))[None, :, None, :]
            )
            tally.add("patch embedding + role", embedded, x[:, :, :tokens])
            b, variables, padded, width = x.shape
            first = None
            for layer, (block, mix) in enumerate(zip(model.blocks, model.variate)):
                mine_block, att, mine_att = (
                    port.blocks[layer],
                    block.att,
                    port.blocks[layer].att,
                )
                flat = x.reshape(b * variables, padded, width).contiguous()
                whole, _ = mine_block(flat, first)
                if layer == 0:
                    flat = block.ln0(flat)
                normalized = block.ln1(flat)
                mixed = backend.tmix_mix6_bf16_v5(
                    normalized, *(getattr(att, "x_" + n).view(-1) for n in "rwkvag")
                )
                tally.add(
                    "token_mix (time-mix, 6 outputs)",
                    torch.stack(
                        port_module.token_mix(
                            normalized, *(getattr(mine_att, "x_" + n) for n in "rwkvag")
                        )
                    ),
                    torch.stack(mixed),
                )
                xr, xw, xk, xv, xa, xg = mixed
                r = att.receptance(xr)
                w = att.w0 + torch.tanh(xw @ att.w1) @ att.w2
                k, v = att.key(xk), att.value(xv)
                if layer == 0:
                    first = v
                else:
                    low_rank = (xv @ att.v1) @ att.v2
                    gated = backend.tmix_vres_gate_bf16_v3(v, first, att.v0, low_rank)
                    tally.add(
                        "value_residual",
                        port_module.value_residual(v, first, mine_att.v0, low_rank),
                        gated,
                    )
                    v = gated
                low_rank = (xa @ att.a1) @ att.a2
                a = backend.tmix_a_gate_bf16(att.a0, low_rank)
                tally.add(
                    "sigmoid_gate (learning rate)",
                    port_module.sigmoid_gate(mine_att.a0, low_rank),
                    a,
                )
                g = torch.sigmoid(xg @ att.g1) @ att.g2
                k_new, neg_kk, kka = backend.tmix_kk_pre_bf16_v5(
                    k, att.k_k.view(-1), a, att.k_a.view(-1)
                )
                tally.add(
                    "key_update (3 outputs)",
                    torch.stack(
                        port_module.key_update(k, mine_att.k_k, a, mine_att.k_a, head)
                    ),
                    torch.stack((k_new, neg_kk, kka)),
                )
                state = torch.zeros(
                    b * variables,
                    width // head,
                    head,
                    head,
                    dtype=torch.float32,
                    device="cuda",
                )
                y = inference_kernel(r, w, k_new, v, neg_kk, kka, state)
                y_training = backend.RWKV7_CLAMPW_CUDA(r, w, k_new, v, neg_kk, kka)
                y_mine = port_module.wkv7(r, w, k_new, v, neg_kk, kka, head)
                tally.add("wkv7 recurrence vs inference kernel", y_mine, y)
                tally.add("wkv7 recurrence vs training kernel", y_mine, y_training)
                tally.add(
                    "[original only] training kernel vs inference kernel", y_training, y
                )
                out = backend.tmix_lnx_rkvres_xg_bf16_v1(
                    y, r, k_new, v, att.r_k, att.ln_x.weight, att.ln_x.bias, g
                )
                tally.add(
                    "normalize_bonus_gate",
                    port_module.normalize_bonus_gate(
                        y,
                        r,
                        k_new,
                        v,
                        mine_att.r_k,
                        mine_att.ln_x.weight,
                        mine_att.ln_x.bias,
                        g,
                        head,
                    ),
                    out,
                )
                flat = flat + att.output(out)
                normalized = block.ln2(flat)
                mlp = block.ffn(normalized)
                tally.add("channel_mix (FFN)", mine_block.ffn(normalized), mlp)
                flat = flat + mlp
                tally.add("whole RWKV block, same input", whole, flat)
                before = flat.reshape(b, variables, padded, width)
                x = mix(before)
                compiled = mix.compiled_forward
                mix.compiled_forward = None
                eager = mix(before)
                mix.compiled_forward = compiled
                mine_mix = port.variate[layer](before)
                tally.add("variate attention vs original compiled", mine_mix, x)
                tally.add("variate attention vs original eager", mine_mix, eager)
                tally.add(
                    "[original only] compiled vs eager variate attention", x, eager
                )
            targets = cpu["targets"]
            selected = x[:, :targets, tokens - 1 : tokens]
            expected = direct_quantiles(
                model,
                selected,
                prepared["loc"][:, :targets, tokens - 1],
                prepared["scale"][:, :targets, tokens - 1],
                model.config.output_patches,
            )
            got = port.decode(x[:, :, :tokens], mine, port.config.output_patches)
            tally.add("forecast head (float32 quantiles)", got, expected)
    checks["operations"] = tally.report()
    (WORK / "results").mkdir(parents=True, exist_ok=True)
    (WORK / "results/block_equivalence.json").write_text(
        json.dumps(checks, indent=2), encoding="utf-8"
    )
    print("tokenizer identical:", checks["tokenizer_identical"])
    print(
        f"{'operation':56s} {'values':>11s} {'identical':>10s} {'<=1 ULP':>9s} {'max/rms':>9s} {'mean/rms':>9s}"
    )
    for name, row in checks["operations"].items():
        print(
            f"{name:56s} {row['values']:11d} {row['bit_identical_fraction']:10.6f} "
            f"{row['within_one_ulp_fraction']:9.6f} {row['max_abs_difference_over_rms']:9.2e} "
            f"{row['mean_abs_difference_over_rms']:9.2e}"
        )


if __name__ == "__main__":
    main()

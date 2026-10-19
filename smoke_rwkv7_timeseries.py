import torch
from torch import nn
from types import SimpleNamespace

from rwkv7_backend import load_rwkv

rwkv = load_rwkv(ctx_len=32, inference_only=False)
args = SimpleNamespace(
    n_embd=128, dim_att=128, dim_ffn=512, n_layer=2,
    head_size=64, my_testing='x070', grad_cp=0,
)


class ForecastSmoke(nn.Module):
    def __init__(self):
        super().__init__()
        self.input = nn.Linear(4, args.n_embd)
        self.blocks = nn.ModuleList([rwkv.Block(args, i)
                                     for i in range(args.n_layer)])
        self.norm = nn.LayerNorm(args.n_embd)
        self.output = nn.Linear(args.n_embd, 4)

    def forward(self, values):
        x = self.input(values)
        v_first = torch.empty_like(x)
        for block in self.blocks:
            x, v_first = block(x, v_first)
        return self.output(self.norm(x))


torch.manual_seed(7)
model = ForecastSmoke().to(device='cuda', dtype=torch.bfloat16).train()
optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)
series = torch.randn(2, 33, 4, device='cuda', dtype=torch.bfloat16)
optimizer.zero_grad(set_to_none=True)
prediction = model(series[:, :-1].contiguous())
loss = (prediction.float() - series[:, 1:].float()).square().mean()
loss.backward()
assert torch.isfinite(loss)
missing = {name for name, p in model.named_parameters()
           if p.requires_grad and p.grad is None}
# Layer zero stores v_first and does not use the value-residual gate.
assert missing <= {'blocks.0.att.v0', 'blocks.0.att.v1', 'blocks.0.att.v2'}
grads = [p.grad for p in model.parameters() if p.grad is not None]
assert all(torch.isfinite(g).all() for g in grads)
assert model.blocks[0].att.output.weight.grad.abs().max() > 0
optimizer.step()
torch.cuda.synchronize()
print(f'PASS: shape={tuple(prediction.shape)}, loss={loss.item():.4f}; forward/backward/update completed')

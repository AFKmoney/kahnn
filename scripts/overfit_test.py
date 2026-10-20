#!/usr/bin/env python
"""Overfit a single fixed batch. A healthy LM must drive loss -> ~0."""
import argparse, sys, time, math, copy
from pathlib import Path
import torch, torch.nn.functional as F
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from kuro_brain.model import KAHNN
from kuro_brain.config import CONFIGS

p = argparse.ArgumentParser()
p.add_argument("--config", default="nano"); p.add_argument("--steps", type=int, default=300)
p.add_argument("--lr", type=float, default=3e-3); p.add_argument("--tokens", type=int, default=64)
p.add_argument("--layers", type=int, default=None)
a = p.parse_args()
torch.manual_seed(0); torch.set_num_threads(8)
import tiktoken
enc = tiktoken.get_encoding("gpt2")
cfg = copy.deepcopy(CONFIGS[a.config]); cfg.vocab_size = enc.n_vocab
text = open(Path(__file__).resolve().parents[1] / "data/corpus.txt", encoding="utf-8").read(20000)
ids = torch.tensor(enc.encode_ordinary(text)[: a.tokens + 1]).unsqueeze(0)
x, y = ids[:, :-1], ids[:, 1:]
m = KAHNN(cfg)
if a.layers: m.set_active_layers(a.layers)
opt = torch.optim.AdamW(m.parameters(), lr=a.lr, weight_decay=0.0)
print(f"ln V = {math.log(cfg.vocab_size):.3f}")
t0 = time.time()
for s in range(a.steps + 1):
    logits = m(x)
    loss = F.cross_entropy(logits.reshape(-1, cfg.vocab_size), y.reshape(-1))
    opt.zero_grad(); loss.backward()
    if s % 25 == 0:
        gn = {n: (q.grad.norm().item() if q.grad is not None else None) for n, q in m.named_parameters()}
        nz = sum(1 for v in gn.values() if v); tot = len(gn)
        acc = (logits.argmax(-1) == y).float().mean().item()
        print(f"step {s:4d} loss {loss.item():.4f} acc {acc:.3f} logit_range [{logits.min().item():.3f},{logits.max().item():.3f}] params_with_grad {nz}/{tot} {time.time()-t0:.0f}s", flush=True)
    opt.step()

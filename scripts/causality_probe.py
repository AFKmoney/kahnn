#!/usr/bin/env python
"""Does the prediction at position t depend on tokens < t? (context mixing check)
Also checks causality: must NOT depend on tokens > t."""
import sys, copy, torch
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from kuro_brain.model import KAHNN
from kuro_brain.config import CONFIGS
cfg = copy.deepcopy(CONFIGS[sys.argv[1] if len(sys.argv) > 1 else "nano"]); cfg.vocab_size = 50257; cfg.noise = 0.0
torch.manual_seed(0); m = KAHNN(cfg).eval()
x = torch.randint(0, 50257, (1, 16)); x2 = x.clone(); x2[0, 3] = (x[0, 3] + 1) % 50257
with torch.no_grad():
    a, b = m(x), m(x2)
d = (a - b).abs().amax(-1)[0]
print("max |dlogit| per position after changing token 3:")
print(" ".join(f"{v:.2e}" for v in d.tolist()))
print("past->future influence (pos 4..15):", d[4:].max().item(), "| future->past leak (pos 0..2):", d[:3].max().item())

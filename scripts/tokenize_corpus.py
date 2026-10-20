#!/usr/bin/env python
"""Tokenize corpus.txt once (GPT-2 BPE) -> data/tok/{train,val}.bin (uint16).
Validation = every 100th ~1MB line-aligned chunk (held out, spread over all sources)."""
import argparse, numpy as np, tiktoken
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor
p = argparse.ArgumentParser()
p.add_argument("--corpus", default="data/corpus.txt"); p.add_argument("--out", default="data/tok")
p.add_argument("--val-frac", type=float, default=0.02)
a = p.parse_args()
def enc(chunk):
    return np.array(tiktoken.get_encoding("gpt2").encode_ordinary(chunk), dtype=np.uint16)
text = Path(a.corpus).read_text(encoding="utf-8", errors="replace")
def chunks(t, n=1 << 20):
    out, i = [], 0
    while i < len(t):
        j = t.rfind("\n", i, i + n) + 1 if i + n < len(t) else len(t)
        if j <= i: j = min(len(t), i + n)
        out.append(t[i:j]); i = j
    return out
Path(a.out).mkdir(parents=True, exist_ok=True)
cs = chunks(text, 1 << 20)
every = max(2, round(1 / a.val_frac))
with ProcessPoolExecutor(8) as ex:
    toks = list(ex.map(enc, cs))
for name, sel in (("train", [t for i, t in enumerate(toks) if i % every != every // 2]),
                  ("val", [t for i, t in enumerate(toks) if i % every == every // 2])):
    arr = np.concatenate(sel); arr.tofile(f"{a.out}/{name}.bin"); print(name, f"{arr.size:,} tokens")

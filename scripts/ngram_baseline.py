#!/usr/bin/env python
"""Unigram / interpolated-bigram cross-entropy (nats/token) on val.bin, fit on
the first N train tokens. Reference points for what 'learning' means."""
import sys, math, numpy as np
V = 50257
tr = np.fromfile("data/tok/train.bin", dtype=np.uint16).astype(np.int64)
va = np.fromfile("data/tok/val.bin", dtype=np.uint16).astype(np.int64)
for N in [int(x) for x in (sys.argv[1:] or ["5000000", str(len(tr))])]:
    t = tr[:N]
    uni = (np.bincount(t, minlength=V) + 0.5) / (N + 0.5 * V)
    lu = -np.log(uni[va]).mean()
    key = t[:-1] * V + t[1:]
    uk, uc = np.unique(key, return_counts=True)
    ctx = np.bincount(t[:-1], minlength=V)
    q = va[:-1] * V + va[1:]
    idx = np.searchsorted(uk, q); idx = np.clip(idx, 0, len(uk) - 1)
    c_big = np.where(uk[idx] == q, uc[idx], 0)
    best = None
    for lam in (0.3, 0.5, 0.6, 0.7, 0.8, 0.9):
        pb = np.where(ctx[va[:-1]] > 0, c_big / np.maximum(ctx[va[:-1]], 1), 0)
        p = lam * pb + (1 - lam) * uni[va[1:]]
        l = -np.log(p).mean(); best = min(best or 99, l)
    print(f"N_train={N:,}  unigram={lu:.3f}  bigram(interp,best lam)={best:.3f}  chance=ln V={math.log(V):.3f}")

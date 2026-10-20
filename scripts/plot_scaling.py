#!/usr/bin/env python
"""Collect runs/scaling/*/metrics.csv -> scaling.csv + scaling.png (+ power-law fit)."""
import csv, math, re
from pathlib import Path
import numpy as np, matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
R = Path(__file__).resolve().parents[1] / "runs/scaling"
rows, final = [], {}
fig, ax = plt.subplots(1, 3, figsize=(16, 4.5))
for c in ["nano", "commodity", "medium"]:
    f = R / c / "metrics.csv"
    if not f.exists(): continue
    log = (R / c / "train_universal.log").read_text()
    n = float(re.search(r"trainable params: ([\d.]+)M", log).group(1)) * 1e6
    d = list(csv.DictReader(open(f)))
    for r in d: r["config"] = c; r["params"] = int(n); rows.append(r)
    v = [(int(r["tokens"]), float(r["val_loss"]), float(r["elapsed_s"])) for r in d if r["val_loss"] != "nan"]
    t = [(int(r["tokens"]), float(r["train_loss"])) for r in d if r["train_loss"] != "nan"]
    if v:
        ax[0].plot(*zip(*[(a, b) for a, b, _ in v]), "o-", label=f"{c} val ({n/1e6:.1f}M)")
        ax[1].plot([s / 3600 for *_, s in v], [b for _, b, _ in v], "o-", label=c)
        if "[final-val]" in log: final[c] = (n, v[-1][1])
    if t: ax[0].plot(*zip(*t), alpha=.3, label=f"{c} train")
for a_, l in ((ax[0], "tokens"), (ax[1], "wall-clock hours (8-thread CPU)")):
    a_.set_xlabel(l); a_.set_ylabel("loss (nats)"); a_.legend(fontsize=7); a_.grid(alpha=.3)
for a_ in ax[:2]:
    a_.axhline(6.97, ls=":", c="gray"); a_.axhline(5.02, ls="--", c="gray")
fit = ""
if final:
    N = np.array([v[0] for v in final.values()]); L = np.array([v[1] for v in final.values()])
    ax[2].loglog(N, L, "o"); ax[2].set_xlabel("params"); ax[2].set_ylabel("final val loss")
    if len(N) >= 2:
        b, a = np.polyfit(np.log(N), np.log(L), 1)
        xs = np.geomspace(N.min(), N.max(), 20); ax[2].loglog(xs, np.exp(a) * xs ** b, "--")
        fit = f"L ~ {np.exp(a):.3g} * N^{b:.4f}"; ax[2].set_title(fit, fontsize=9)
ax[0].set_title("loss vs tokens (gray: unigram 6.97 / bigram 5.02 @2M)")
fig.tight_layout(); fig.savefig(R / "scaling.png", dpi=120)
keys = ["config", "params", "step", "tokens", "train_loss", "val_loss", "lr", "tps", "layers", "elapsed_s"]
with open(R / "scaling.csv", "w", newline="") as fh:
    w = csv.DictWriter(fh, keys, extrasaction="ignore"); w.writeheader(); w.writerows(rows)
print("final:", final, fit)

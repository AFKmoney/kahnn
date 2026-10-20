# Loss plateau debug (nano stuck at ~10.5)

Symptom: `nano` (D=2048, L=3, GPT-2 vocab 50257) trained 37M tokens with
`train_universal.py`; train loss stayed ~10.5 vs chance ln(50257)=10.825.
Old checkpoint `runs/nano_base/ckpt_final.pt` on our held-out split: **val 10.586**.

## Diagnostics (scripts in `scripts/`)

| test | old code | fixed code |
|---|---|---|
| `overfit_test.py` (nano, 64 tokens, AdamW 3e-3, 200 steps) | argmax acc 1.000 but **loss 9.839**, logits in [-0.12, 1.00] | **loss 0.027**, acc 1.000 |
| `causality_probe.py` (change token 3, max abs dlogit at later positions) | **0.0** (no context mixing) | 0.89 (future->past leak 2.5e-6, FFT roundoff) |
| gradients | 30/30 params get nonzero grad | 38/38 |

## Root causes

1. **No logit temperature.** Logits = cosine sim to a *fixed* random codebook, so
   in [-1, 1]. Even a perfect model gets loss >= ~ln(V)-1 = ~9.8. The overfit test
   showed exactly this: 100 % accuracy, loss 9.84.
   Fix: learnable `logit_log_scale` (init 10, max 100) and learnable per-token
   `out_bias` (optionally init to log-unigram with `--init-bias-unigram`).
2. **No sequence mixing at all.** Each position's hypervector went through the
   Kuramoto layers on its own. Positional binding does not mix tokens, so the
   prediction at position t did not depend on tokens < t: at best a bigram model.
   Fix: `HoloContext`, a causal holographic trace per layer
   `c_t = sum_{s<t} lambda_d^(t-1-s) x_s` (learnable per-dimension decay, causal FFT
   convolution over time, O(T log T); a recurrent O(D) update in generation).
   This is an HRR bundle over time. There is no attention.
3. **Progressive depth killed grown layers.** `activate_next_layer` zeroed A, B,
   phase_filter **and** gate of the HRR-MLP, so every gradient into it was exactly 0
   for good. Now only the output side (B, phase_filter) is zeroed. Default growth
   now reaches full depth by `--grow-frac 0.3` of training. Before, it was spread
   over the whole run (layer 3 only in the last third).
4. **LR too low** for this model: 3e-4 -> 3e-3 (sweep below).
5. Minor: Kuramoto noise is now applied only in `train()` mode.

## Real-data results (train/val = `scripts/tokenize_corpus.py`, 33.16M / 0.595M tokens; val = every 50th ~1MB chunk)

LR sweep, nano, 1M tokens, 2048 tok/step, val on 8,192 held-out tokens:

| lr | final val |
|---|---|
| 3e-4 | 9.428 |
| 1e-3 | 8.390 |
| 3e-3 | **6.380** (curve: 8.52 @205k, 7.65 @410k, 6.91 @614k, 6.53 @819k) |

Caveat: the 3e-3 run also used the new `--grow-frac 0.3` schedule. The other two
used the old schedule.

N-gram baselines on the same val (`scripts/ngram_baseline.py`; bigram interpolation
λ is picked on val, so it is slightly optimistic):

| train tokens | unigram | bigram |
|---|---|---|
| 1M | 7.611 | 6.426 |
| 2M | 6.971 | 5.021 |
| 5M | 6.802 | 4.375 |
| 33.2M | 6.498 | 3.871 |

So after 1M tokens, fixed nano (6.38) beats unigram and bigram fit on the same 1M
tokens. Before the fix: 10.59 after 37M tokens.

`teach.py smoke`, `python -m kuro_brain.smoke` and `teach.py probe` on the old
checkpoint (strict=False) all still pass.

Throughput note: the box was shared with a browser process, and tps fell
from ~1200 to ~350 during the sweep.

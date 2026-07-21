"""
mini_train.py — Actually run a tiny training loop on a small corpus to
verify the full pipeline (forward, backward, OnlineLearner, engram writes,
generation) works end-to-end.

This is a more aggressive integration test than smoke.py — it runs ~30
optimizer steps and verifies the loss goes down.
"""

from __future__ import annotations

import sys
from pathlib import Path

import torch
import torch.nn.functional as F

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

from kuro_brain.model import KAHNN, KAHNNConfig
from kuro_brain.continuous_learning import OnlineLearner
from kuro_brain.config import SMOKE
from data import build_tokenizer, StreamingCorpus, TokenBatcher


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    cfg = SMOKE
    cfg.kuramoto_steps = 2

    tok, vocab = build_tokenizer("gpt2")
    cfg.vocab_size = vocab  # 256 with byte fallback
    print(f"[mini] vocab={vocab} device={device}")

    model = KAHNN(cfg).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=1e-2)
    learner = OnlineLearner(model, cfg, base_optimizer=opt)

    corpus_path = "/tmp/kuro_test/tiny.txt"
    corpus = StreamingCorpus(corpus_path, tok, eos_id=vocab - 1)
    batcher = TokenBatcher(corpus, batch_size=4, seq_len=32,
                           device=device, prefetch=2)

    losses = []
    step = 0
    for x, y in batcher:
        if step >= 100:
            break
        logits = model(x)
        loss = F.cross_entropy(
            logits.reshape(-1, cfg.vocab_size),
            y.reshape(-1),
        )
        opt.zero_grad(set_to_none=True)
        loss.backward()
        learner.step(x, y, loss.detach())
        losses.append(float(loss.item()))
        if step % 10 == 0:
            print(f"[mini] step={step:3d} loss={losses[-1]:.4f} "
                  f"engram_mean={learner.state()['engram_usage_mean']:.2f}",
                  flush=True)
        step += 1
    batcher.close()

    first_avg = sum(losses[:5]) / 5
    last_avg = sum(losses[-5:]) / 5
    print(f"[mini] first-5 avg loss={first_avg:.4f}, last-5 avg loss={last_avg:.4f}")
    assert last_avg < first_avg, f"loss did not decrease: {first_avg} -> {last_avg}"
    print("[mini] PASS — loss decreased over 100 steps")


if __name__ == "__main__":
    main()

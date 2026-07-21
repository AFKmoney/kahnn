#!/usr/bin/env python
"""
evaluate.py — Compute perplexity on a held-out corpus.

Usage:
    python evaluate.py --checkpoint runs/run1/ckpt_final.pt \
                       --data /path/to/valid.txt --seq-len 1024
"""

from __future__ import annotations

import argparse
import math
import sys
from pathlib import Path

import torch
import torch.nn.functional as F

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

from kuro_brain.model import KAHNN, KAHNNConfig
from data import build_tokenizer, StreamingCorpus, TokenBatcher


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--checkpoint", required=True)
    p.add_argument("--data", required=True)
    p.add_argument("--seq-len", type=int, default=1024)
    p.add_argument("--batch-size", type=int, default=4)
    p.add_argument("--max-tokens", type=int, default=10_000_000)
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    return p.parse_args()


def main():
    args = parse_args()
    device = torch.device(args.device)
    ckpt = torch.load(args.checkpoint, map_location=device, weights_only=False)
    cfg = KAHNNConfig(**ckpt["config"])
    cfg.max_seq_len = args.seq_len
    model = KAHNN(cfg).to(device)
    model.load_state_dict(ckpt["model"])
    model.eval()

    tokenizer, vocab = build_tokenizer("gpt2")
    eos_id = vocab - 1
    corpus = StreamingCorpus(args.data, tokenizer, eos_id=eos_id)
    batcher = TokenBatcher(corpus, batch_size=args.batch_size,
                           seq_len=args.seq_len, device=device, prefetch=2)

    nll = 0.0
    n_tok = 0
    with torch.no_grad():
        for x, y in batcher:
            if n_tok >= args.max_tokens:
                break
            logits = model(x)
            loss = F.cross_entropy(
                logits.reshape(-1, cfg.vocab_size),
                y.reshape(-1),
                reduction="sum",
            )
            nll += float(loss.item())
            n_tok += x.numel()
    batcher.close()

    avg_nll = nll / max(n_tok, 1)
    ppl = math.exp(avg_nll)
    print(f"tokens={n_tok:,}  nll={avg_nll:.4f}  ppl={ppl:.3f}")


if __name__ == "__main__":
    main()

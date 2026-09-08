#!/usr/bin/env python
"""
generate.py — Sample text from a trained KAHNN checkpoint.

Usage:
    python generate.py --checkpoint runs/run1/ckpt_final.pt \
                       --prompt "The meaning of life is" \
                       --n-tokens 200 --temperature 0.8 --top-k 50
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import torch

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

from kuro_brain.model import KAHNN, KAHNNConfig
from data import build_tokenizer, encode_text


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--checkpoint", required=True)
    p.add_argument("--prompt", required=True)
    p.add_argument("--n-tokens", type=int, default=200)
    p.add_argument("--temperature", type=float, default=0.8)
    p.add_argument("--top-k", type=int, default=50)
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    p.add_argument("--seed", type=int, default=0)
    return p.parse_args()


def main():
    args = parse_args()
    torch.manual_seed(args.seed)
    device = torch.device(args.device)

    ckpt = torch.load(args.checkpoint, map_location=device, weights_only=False)
    cfg = KAHNNConfig(**ckpt["config"])
    cfg.max_seq_len = min(cfg.max_seq_len, 2048)  # keep generation cheap
    model = KAHNN(cfg).to(device)
    model.load_state_dict(ckpt["model"])
    model.eval()

    tokenizer, _ = build_tokenizer("gpt2")
    prompt_ids = encode_text(tokenizer, args.prompt)
    x = torch.tensor([prompt_ids], dtype=torch.long, device=device)

    out = model.generate(x, n_new=args.n_tokens,
                         temperature=args.temperature, top_k=args.top_k)
    text = tokenizer.decode(out[0].tolist())
    print(text)


if __name__ == "__main__":
    main()

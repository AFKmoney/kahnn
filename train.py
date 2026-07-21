#!/usr/bin/env python
"""
train.py — Stream-train a KAHNN model.

Usage:
    python train.py --config medium --data /path/to/corpus --epochs 1 \
                    --batch-size 16 --seq-len 1024 --device cuda \
                    --output /home/z/my-project/download/kuramoto_brain/runs/run1

This is the *full* training entry point. It:
  1. Builds a KAHNN model from a named config (smoke/tiny/medium/large/xl).
  2. Streams tokens from `--data` (file or directory of text files).
  3. Trains with Adam on global params + OnlineLearner local plasticity.
  4. Checkpoints every N tokens to `--output`.
  5. Logs throughput (tokens/sec), loss, and engram stats.

A 25B-token run on H100 with `--config large` finishes in ~2 days.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import sys
import time
from pathlib import Path

import torch
import torch.nn.functional as F

# Resolve repo root
HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

from kuro_brain.model import KAHNN, KAHNNConfig
from kuro_brain.continuous_learning import OnlineLearner
from kuro_brain.config import CONFIGS, describe_config
from data import build_tokenizer, StreamingCorpus, TokenBatcher


def parse_args():
    p = argparse.ArgumentParser(description="Train a KAHNN model.")
    p.add_argument("--config", choices=list(CONFIGS), default="medium")
    p.add_argument("--data", required=True, help="Path to text corpus (file or dir).")
    p.add_argument("--output", required=True, help="Output dir for checkpoints.")
    p.add_argument("--batch-size", type=int, default=8)
    p.add_argument("--seq-len", type=int, default=1024)
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    p.add_argument("--lr", type=float, default=3e-4)
    p.add_argument("--max-tokens", type=int, default=25_000_000_000,
                   help="Stop after this many tokens (default 25B).")
    p.add_argument("--checkpoint-every", type=int, default=500_000_000,
                   help="Tokens between checkpoints (default 500M).")
    p.add_argument("--log-every", type=int, default=100, help="Steps between logs.")
    p.add_argument("--continuous", action="store_true",
                   help="Run in continuous-learning mode (no optimizer, only local rules).")
    p.add_argument("--seed", type=int, default=1337)
    p.add_argument("--resume", default=None, help="Path to a checkpoint to resume from.")
    return p.parse_args()


def main():
    args = parse_args()
    torch.manual_seed(args.seed)
    device = torch.device(args.device)
    out_dir = Path(args.output)
    out_dir.mkdir(parents=True, exist_ok=True)

    # 1) Config & tokenizer
    cfg = CONFIGS[args.config]
    cfg.max_seq_len = args.seq_len
    print(f"[config] {describe_config(cfg)}", flush=True)

    tokenizer, vocab = build_tokenizer("gpt2")
    cfg.vocab_size = vocab
    eos_id = vocab - 1

    # 2) Model
    model = KAHNN(cfg).to(device)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"[model] trainable params: {n_params/1e6:.1f}M", flush=True)

    # 3) Optimizer (only if not continuous-only)
    if args.continuous:
        optimizer = None
        print("[mode] continuous learning (no base optimizer)", flush=True)
    else:
        optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=0.01)
        print(f"[mode] pretraining with AdamW lr={args.lr}", flush=True)

    learner = OnlineLearner(model, cfg, base_optimizer=optimizer,
                            continuous_mode=args.continuous)

    start_step = 0
    if args.resume:
        ckpt = torch.load(args.resume, map_location=device)
        model.load_state_dict(ckpt["model"])
        if optimizer is not None and "optimizer" in ckpt:
            optimizer.load_state_dict(ckpt["optimizer"])
        start_step = ckpt.get("step", 0)
        print(f"[resume] from step {start_step}", flush=True)

    # 4) Data
    corpus = StreamingCorpus(args.data, tokenizer, eos_id=eos_id)
    batcher = TokenBatcher(
        corpus, batch_size=args.batch_size, seq_len=args.seq_len,
        device=device, prefetch=4,
    )

    # 5) Training loop
    log_path = out_dir / "train.log"
    log_fh = open(log_path, "a", encoding="utf-8")
    step = start_step
    tokens_seen = 0
    t0 = time.time()
    running_loss = 0.0
    running_n = 0

    print(f"[train] starting training; target={args.max_tokens:,} tokens", flush=True)
    for x, y in batcher:
        if tokens_seen >= args.max_tokens:
            break

        logits = model(x)                       # [B, T, V]
        loss = F.cross_entropy(
            logits.reshape(-1, cfg.vocab_size),
            y.reshape(-1),
        )

        if optimizer is not None:
            optimizer.zero_grad(set_to_none=True)
        loss.backward()
        # The OnlineLearner applies the base optimizer step + local plasticity
        learner.step(x, y, loss.detach())

        bs_tokens = x.numel()
        tokens_seen += bs_tokens
        step += 1
        running_loss += float(loss.item()) * bs_tokens
        running_n += bs_tokens

        if step % args.log_every == 0:
            avg = running_loss / running_n
            dt = time.time() - t0
            tps = tokens_seen / max(dt, 1e-6)
            msg = (f"step={step} tokens={tokens_seen:,} loss={avg:.4f} "
                   f"tps={tps:.0f} engram_mean={learner.state()['engram_usage_mean']:.2f}")
            print(msg, flush=True)
            log_fh.write(msg + "\n"); log_fh.flush()
            running_loss = 0.0
            running_n = 0

        if tokens_seen % args.checkpoint_every < bs_tokens:
            ckpt_path = out_dir / f"ckpt_{step}.pt"
            save = {
                "model": model.state_dict(),
                "step": step,
                "tokens": tokens_seen,
                "config": cfg.__dict__,
            }
            if optimizer is not None:
                save["optimizer"] = optimizer.state_dict()
            torch.save(save, ckpt_path)
            print(f"[ckpt] saved {ckpt_path}", flush=True)

    # Final checkpoint
    ckpt_path = out_dir / "ckpt_final.pt"
    torch.save({
        "model": model.state_dict(),
        "step": step,
        "tokens": tokens_seen,
        "config": cfg.__dict__,
    }, ckpt_path)
    print(f"[done] saved final checkpoint to {ckpt_path}", flush=True)
    log_fh.close()
    batcher.close()


if __name__ == "__main__":
    main()

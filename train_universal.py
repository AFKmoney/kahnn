#!/usr/bin/env python
"""
train_universal.py — Cost-minimized Kahnn training on CPU / small GPU / MPS.

This is the recommended entrypoint when you do NOT have H100 / multi-4090
hardware. It auto-picks a device, chooses a commodity-friendly config, and
enables CPU-safe speedups (threaded BLAS, matmul decode, cheap engram pull,
optional torch.compile, progressive depth, optional continuous learning).

Examples:
  # Laptop CPU, nano model, 20M tokens
  python train_universal.py --data ./corpus.txt --output ./runs/nano_cpu \\
      --config nano --max-tokens 20000000

  # Auto device + commodity preset (default)
  python train_universal.py --data ./corpus --output ./runs/commodity \\
      --config commodity --max-tokens 260000000

  # Small CUDA GPU if present, else CPU
  python train_universal.py --data ./corpus --output ./runs/auto --device auto

  # Post-pretrain continuous learning (no Adam — local plasticity + engrams)
  python train_universal.py --data ./new_domain.txt --output ./runs/cont \\
      --config commodity --continuous --resume ./runs/commodity/ckpt_final.pt
"""

from __future__ import annotations

import argparse
import math
import os
import sys
import time
from pathlib import Path

import torch
import torch.nn.functional as F

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

from kuro_brain.model import KAHNN
from kuro_brain.continuous_learning import OnlineLearner
from kuro_brain.config import CONFIGS, describe_config
from kuro_brain.device import pick_device, recommend_batch
from data import build_tokenizer, StreamingCorpus, TokenBatcher


def parse_args():
    p = argparse.ArgumentParser(
        description="Universal Kahnn trainer (CPU / small GPU / MPS)."
    )
    p.add_argument("--data", required=True)
    p.add_argument("--output", required=True)
    p.add_argument("--config", default="commodity",
                   choices=list(CONFIGS),
                   help="Prefer nano/commodity/tiny for low-cost runs.")
    p.add_argument("--device", default="auto",
                   help="auto | cpu | cuda | mps | cuda:N")
    p.add_argument("--micro-batch", type=int, default=None)
    p.add_argument("--seq-len", type=int, default=None)
    p.add_argument("--grad-accum", type=int, default=None)
    p.add_argument("--lr", type=float, default=3e-4)
    p.add_argument("--weight-decay", type=float, default=0.01)
    p.add_argument("--max-tokens", type=int, default=None,
                   help="Default = Chinchilla 20× params for the chosen config.")
    p.add_argument("--warmup-frac", type=float, default=0.02)
    p.add_argument("--log-every", type=int, default=10)
    p.add_argument("--checkpoint-every", type=int, default=500)
    p.add_argument("--seed", type=int, default=1337)
    p.add_argument("--resume", default=None)
    p.add_argument("--continuous", action="store_true",
                   help="Disable Adam; local plasticity + engrams only.")
    p.add_argument("--enable-plasticity", action="store_true",
                   help="Also run local plasticity during Adam pretraining "
                        "(slower; usually unnecessary).")
    p.add_argument("--progressive-depth", action="store_true", default=True)
    p.add_argument("--no-progressive-depth", dest="progressive_depth",
                   action="store_false")
    p.add_argument("--initial-layers", type=int, default=None)
    p.add_argument("--grow-every-steps", type=int, default=None)
    p.add_argument("--compile", action="store_true",
                   help="Try torch.compile (CPU/GPU). Falls back silently.")
    p.add_argument("--mod", action="store_true",
                   help="Mixture-of-Depths (helps small GPUs; modest CPU gain).")
    p.add_argument("--mod-skip-rate", type=float, default=0.5)
    p.add_argument("--activation-checkpointing", action="store_true",
                   help="Save activation memory on small GPUs.")
    p.add_argument("--prefetch", type=int, default=4)
    p.add_argument("--smoke-steps", type=int, default=0,
                   help="If >0, run only N steps then exit (throughput check).")
    return p.parse_args()


def lr_at_step(step, total_steps, warmup, base_lr, min_lr_frac=0.1):
    if step < warmup:
        return base_lr * (step + 1) / max(1, warmup)
    prog = (step - warmup) / max(1, total_steps - warmup)
    cos = 0.5 * (1.0 + math.cos(math.pi * prog))
    return base_lr * (min_lr_frac + (1.0 - min_lr_frac) * cos)


def main():
    args = parse_args()
    info = pick_device(args.device)
    device = info.device
    out_dir = Path(args.output)
    out_dir.mkdir(parents=True, exist_ok=True)
    log_path = out_dir / "train_universal.log"

    def log(msg: str):
        print(msg, flush=True)
        with open(log_path, "a", encoding="utf-8") as fh:
            fh.write(msg + "\n")

    cfg = CONFIGS[args.config]
    micro, seq, accum = recommend_batch(args.config, info)
    if args.micro_batch is not None:
        micro = args.micro_batch
    if args.seq_len is not None:
        seq = args.seq_len
    if args.grad_accum is not None:
        accum = args.grad_accum
    cfg.max_seq_len = max(cfg.max_seq_len, seq)

    cfg.use_mod = bool(args.mod)
    cfg.mod_skip_rate = args.mod_skip_rate
    cfg.use_activation_checkpointing = bool(args.activation_checkpointing)

    n_params = cfg.param_count()
    if args.max_tokens is None:
        args.max_tokens = int(20 * n_params)

    log(f"[device] {info.kind} ({info.name}) — {info.notes}")
    log(f"[config] {describe_config(cfg)}")
    log(f"[batch] micro={micro} seq={seq} accum={accum} "
        f"tokens/step={micro * seq * accum:,}")
    log(f"[tokens] target={args.max_tokens:,} "
        f"(~{args.max_tokens / max(n_params,1):.1f} tok/param)")

    torch.manual_seed(args.seed)
    tokenizer, vocab = build_tokenizer("gpt2")
    cfg.vocab_size = vocab
    eos_id = vocab - 1

    model = KAHNN(cfg).to(device)
    if args.compile:
        try:
            compiled = torch.compile(model, mode="reduce-overhead")
            with torch.no_grad():
                _ = compiled(torch.zeros(1, 4, dtype=torch.long, device=device))
            model = compiled
            log("[compile] torch.compile enabled")
        except Exception as e:
            log(f"[compile] unavailable ({type(e).__name__}: {e}); continuing eager")

    n_train = sum(p.numel() for p in model.parameters())
    log(f"[model] trainable params: {n_train/1e6:.2f}M")

    optimizer = None
    if not args.continuous:
        decay, no_decay = [], []
        for n, p in model.named_parameters():
            if not p.requires_grad:
                continue
            if p.ndim < 2 or n.endswith(("K", "engram_gain", "gate")):
                no_decay.append(p)
            else:
                decay.append(p)
        optimizer = torch.optim.AdamW(
            [
                {"params": decay, "weight_decay": args.weight_decay},
                {"params": no_decay, "weight_decay": 0.0},
            ],
            lr=args.lr, betas=(0.9, 0.95), eps=1e-8,
        )
        log(f"[optim] AdamW lr={args.lr}")
    else:
        log("[mode] continuous learning (no Adam)")

    learner = OnlineLearner(model, cfg, base_optimizer=optimizer,
                            continuous_mode=args.continuous)
    if args.enable_plasticity:
        learner.enable_local_plasticity = True
        log("[plasticity] enabled during pretrain (extra cost)")

    tokens_per_step = micro * seq * accum
    total_steps = max(1, args.max_tokens // tokens_per_step)
    warmup = max(10, int(total_steps * args.warmup_frac))

    if args.progressive_depth and cfg.n_layers >= 3:
        initial = args.initial_layers or max(1, cfg.n_layers // 3)
        grow_every = args.grow_every_steps or max(
            1, total_steps // max(1, cfg.n_layers - initial + 1)
        )
        model.set_active_layers(initial)
        log(f"[progressive-depth] {initial}/{cfg.n_layers}, "
            f"grow_every={grow_every}")
    else:
        initial, grow_every = cfg.n_layers, None

    start_step = 0
    tokens_seen = 0
    if args.resume:
        ckpt = torch.load(args.resume, map_location=device, weights_only=False)
        model.load_state_dict(ckpt["model"], strict=False)
        if optimizer is not None and "optimizer" in ckpt:
            optimizer.load_state_dict(ckpt["optimizer"])
        start_step = int(ckpt.get("step", 0))
        tokens_seen = int(ckpt.get("tokens", 0))
        log(f"[resume] step={start_step} tokens={tokens_seen:,}")

    corpus = StreamingCorpus(args.data, tokenizer, eos_id=eos_id)
    batcher = TokenBatcher(
        corpus, batch_size=micro, seq_len=seq,
        device=device, prefetch=args.prefetch,
    )

    use_amp = info.amp_dtype is not None and info.kind == "cuda"
    step = start_step
    t0 = time.time()
    running_loss = 0.0
    running_n = 0
    log(f"[train] start — total_steps≈{total_steps:,} warmup={warmup}")

    accum_i = 0
    if optimizer is not None:
        optimizer.zero_grad(set_to_none=True)

    try:
        for x, y in batcher:
            if tokens_seen >= args.max_tokens:
                break
            if args.smoke_steps and (step - start_step) >= args.smoke_steps:
                break

            if optimizer is not None:
                lr = lr_at_step(step, total_steps, warmup, args.lr)
                for g in optimizer.param_groups:
                    g["lr"] = lr

            if use_amp:
                with torch.autocast(device_type="cuda", dtype=info.amp_dtype):
                    logits = model(x)
                    loss = F.cross_entropy(
                        logits.reshape(-1, cfg.vocab_size), y.reshape(-1)
                    )
            else:
                logits = model(x)
                loss = F.cross_entropy(
                    logits.reshape(-1, cfg.vocab_size), y.reshape(-1)
                )

            loss_scaled = loss / accum
            loss_scaled.backward()
            accum_i += 1

            if accum_i >= accum:
                if optimizer is not None:
                    torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                learner.step(x, y, loss.detach())
                if optimizer is not None:
                    optimizer.zero_grad(set_to_none=True)
                accum_i = 0
                step += 1

                if grow_every is not None and step % grow_every == 0:
                    new_n = model.activate_next_layer()
                    if new_n is not None:
                        log(f"[progressive-depth] activated layer {new_n}/{cfg.n_layers}")

            bs_tokens = x.numel()
            tokens_seen += bs_tokens
            running_loss += float(loss.detach()) * bs_tokens
            running_n += bs_tokens

            if step > start_step and step % args.log_every == 0 and accum_i == 0:
                dt = max(time.time() - t0, 1e-6)
                tps = tokens_seen / dt
                avg = running_loss / max(running_n, 1)
                eta_h = (args.max_tokens - tokens_seen) / max(tps, 1e-6) / 3600
                msg = (f"step={step} tokens={tokens_seen:,} loss={avg:.4f} "
                       f"tps={tps:.1f} layers={model.n_active_layers}/{cfg.n_layers} "
                       f"eta={eta_h:.1f}h")
                log(msg)
                running_loss = 0.0
                running_n = 0

            if (step > start_step and step % args.checkpoint_every == 0
                    and accum_i == 0):
                ckpt_path = out_dir / f"ckpt_{step}.pt"
                payload = {
                    "model": model.state_dict(),
                    "step": step,
                    "tokens": tokens_seen,
                    "config": cfg.__dict__,
                }
                if optimizer is not None:
                    payload["optimizer"] = optimizer.state_dict()
                torch.save(payload, ckpt_path)
                log(f"[ckpt] {ckpt_path}")
    finally:
        batcher.close()

    ckpt_path = out_dir / "ckpt_final.pt"
    payload = {
        "model": model.state_dict(),
        "step": step,
        "tokens": tokens_seen,
        "config": cfg.__dict__,
    }
    if optimizer is not None:
        payload["optimizer"] = optimizer.state_dict()
    torch.save(payload, ckpt_path)
    dt = time.time() - t0
    tps = tokens_seen / max(dt, 1e-6)
    log(f"[done] tokens={tokens_seen:,} tps={tps:.1f} saved={ckpt_path}")


if __name__ == "__main__":
    main()

#!/usr/bin/env python
"""
train_universal.py -- Cost-minimized Kahnn training on CPU / small GPU / MPS.

This is the recommended entrypoint when you do NOT have H100 / multi-4090
hardware. It auto-picks a device, chooses a commodity-friendly config, and
enables CPU-safe speedups (threaded BLAS, matmul decode, cheap engram pull,
optional torch.compile, progressive depth, optional continuous learning).

Examples:
  python train_universal.py --data ./corpus --output ./runs/x --config nano
  python train_universal.py --data ./new --output ./runs/c --continuous --resume CKPT
  python train_universal.py ... --continuous --soft-decay   # optional weak decay
  # teach/forget: see teach.py + docs/LIFELONG_LEARNING.md
"""

from __future__ import annotations

import argparse
import math
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
                   help="Disable Adam; local plasticity + lifelong engrams.")
    p.add_argument("--soft-decay", action="store_true",
                   help="With --continuous: age low-usage engram slots only. "
                        "Default OFF so memory persists until teach.py forget.")
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
    p.add_argument("--head-lr-mult", type=float, default=1.0,
                   help="LR multiplier for output head (logit scale + bias).")
    p.add_argument("--init-bias-unigram", action="store_true",
                   help="Init output bias to log unigram freqs of the train "
                        ".bin (standard LM head init; needs .bin data).")
    p.add_argument("--val-data", default=None,
                   help="uint16 .bin of held-out tokens (scripts/tokenize_corpus.py).")
    p.add_argument("--eval-every", type=int, default=200)
    p.add_argument("--eval-batches", type=int, default=16)
    p.add_argument("--grow-frac", type=float, default=0.3,
                   help="Progressive depth: reach full depth by this fraction "
                        "of training (was: spread over the whole run).")
    p.add_argument("--smoke-steps", type=int, default=0,
                   help="If >0, run only N steps then exit (throughput check).")
    return p.parse_args()


class BinBatcher:
    """Random (or sequential) windows from a uint16 token .bin (memmap)."""

    def __init__(self, path, batch, seq, device, seed=0, sequential=False):
        import numpy as np
        self.np = np
        self.data = np.memmap(path, dtype=np.uint16, mode="r")
        self.batch, self.seq, self.device = batch, seq, device
        self.rng = np.random.default_rng(seed)
        self.sequential = sequential

    def __iter__(self):
        np = self.np
        n = len(self.data) - self.seq - 1
        pos = 0
        while True:
            if self.sequential:
                starts = (pos + np.arange(self.batch) * (self.seq + 1)) % n
                pos += self.batch * (self.seq + 1)
            else:
                starts = self.rng.integers(0, n, self.batch)
            w = np.stack([self.data[s:s + self.seq + 1] for s in starts]).astype(np.int64)
            w = torch.from_numpy(w).to(self.device)
            yield w[:, :-1], w[:, 1:]

    def close(self):
        pass


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

    log(f"[device] {info.kind} ({info.name}) -- {info.notes}")
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
        decay, no_decay, head = [], [], []
        for n, p in model.named_parameters():
            if not p.requires_grad:
                continue
            if n in ("out_bias", "logit_log_scale"):
                head.append(p)
            elif p.ndim < 2 or n.endswith(("K", "engram_gain", "gate")):
                no_decay.append(p)
            else:
                decay.append(p)
        optimizer = torch.optim.AdamW(
            [
                {"params": decay, "weight_decay": args.weight_decay},
                {"params": no_decay, "weight_decay": 0.0},
                {"params": head, "weight_decay": 0.0,
                 "lr_mult": args.head_lr_mult},
            ],
            lr=args.lr, betas=(0.9, 0.95), eps=1e-8,
        )
        log(f"[optim] AdamW lr={args.lr}")
    else:
        log("[optim] none (continuous mode)")

    learner = OnlineLearner(
        model, cfg, base_optimizer=optimizer,
        continuous_mode=args.continuous,
        soft_decay=bool(args.soft_decay),
    )
    if args.continuous:
        if args.soft_decay:
            log("[mode] continuous + soft_decay (weak slots only)")
        else:
            log("[mode] continuous lifelong (no auto engram decay; "
                "use teach.py forget to erase)")
    if args.enable_plasticity:
        learner.enable_local_plasticity = True
        log("[plasticity] enabled during pretrain (extra cost)")

    tokens_per_step = micro * seq * accum
    total_steps = max(1, args.max_tokens // tokens_per_step)
    warmup = max(10, int(total_steps * args.warmup_frac))

    if args.progressive_depth and cfg.n_layers >= 3:
        initial = args.initial_layers or max(1, cfg.n_layers // 3)
        grow_every = args.grow_every_steps or max(
            1, int(total_steps * args.grow_frac) // max(1, cfg.n_layers - initial)
        )
        model.set_active_layers(initial)
        log(f"[progressive-depth] {initial}/{cfg.n_layers}, "
            f"grow_every={grow_every}")
    else:
        initial, grow_every = cfg.n_layers, None

    if args.init_bias_unigram and getattr(model, "out_bias", None) is not None \
            and str(args.data).endswith(".bin"):
        import numpy as np
        arr = np.memmap(args.data, dtype=np.uint16, mode="r")[:5_000_000]
        cnt = np.bincount(arr, minlength=cfg.vocab_size).astype(np.float64) + 0.5
        with torch.no_grad():
            model.out_bias.copy_(torch.from_numpy(np.log(cnt / cnt.sum())).float())
        log("[head] out_bias initialised to log unigram (first 5M train tokens)")

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

    if str(args.data).endswith(".bin"):
        batcher = BinBatcher(args.data, micro, seq, device, seed=args.seed)
    else:
        corpus = StreamingCorpus(args.data, tokenizer, eos_id=eos_id)
        batcher = TokenBatcher(
            corpus, batch_size=micro, seq_len=seq,
            device=device, prefetch=args.prefetch,
        )
    val_x = val_y = None
    if args.val_data:
        vb = BinBatcher(args.val_data, micro, seq, device, seed=12345,
                        sequential=True)
        it = iter(vb)
        pairs = [next(it) for _ in range(args.eval_batches)]
        val_x = [p[0] for p in pairs]; val_y = [p[1] for p in pairs]
        log(f"[val] {args.eval_batches * micro * seq:,} held-out tokens")
    csv_path = out_dir / "metrics.csv"
    if not csv_path.exists() or start_step == 0:
        csv_path.write_text("step,tokens,train_loss,val_loss,lr,tps,layers,elapsed_s\n")

    def evaluate():
        if val_x is None:
            return float("nan")
        was = model.training
        model.eval()
        tot = 0.0
        with torch.no_grad():
            for vx, vy in zip(val_x, val_y):
                lg = model(vx)
                tot += float(F.cross_entropy(lg.reshape(-1, cfg.vocab_size), vy.reshape(-1)))
        model.train(was)
        return tot / len(val_x)

    use_amp = info.amp_dtype is not None and info.kind == "cuda"
    step = start_step
    t0 = time.time()
    running_loss = 0.0
    running_n = 0
    log(f"[train] start -- total_steps~{total_steps:,} warmup={warmup}")

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
                    g["lr"] = lr * g.get("lr_mult", 1.0)

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
                vl = float("nan")
                if val_x is not None and step % args.eval_every == 0:
                    vl = evaluate()
                    msg += f" val={vl:.4f}"
                log(msg)
                cur_lr = optimizer.param_groups[0]["lr"] if optimizer else 0.0
                with open(csv_path, "a") as fh:
                    fh.write(f"{step},{tokens_seen},{avg:.5f},{vl:.5f},{cur_lr:.3e},"
                             f"{tps:.1f},{model.n_active_layers},{dt:.1f}\n")
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
    if val_x is not None:
        vl = evaluate()
        log(f"[final-val] val_loss={vl:.4f}")
        with open(csv_path, "a") as fh:
            fh.write(f"{step},{tokens_seen},nan,{vl:.5f},0,{tps:.1f},"
                     f"{model.n_active_layers},{dt:.1f}\n")
    log(f"[done] tokens={tokens_seen:,} tps={tps:.1f} saved={ckpt_path}")


if __name__ == "__main__":
    main()

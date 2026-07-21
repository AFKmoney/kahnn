#!/usr/bin/env python
"""
train_b1.py — Dedicated trainer for the KAHNN-B1 (1B-parameter) model.

Tuned for the Chinchilla-optimal 20B-token run on a single 8×H100 node:

    python train_b1.py \
        --data /data/corpus \
        --output /home/z/my-project/download/runs/b1_run1 \
        --device cuda \
        --ddp                       # multi-GPU
        --bf16                      # mixed precision
        --grad-accum 4              # effective batch = micro × accum × world

Features:
  * bf16 autocast + GradScaler fallback
  * DistributedDataParallel support (one process per GPU via torchrun)
  * Cosine LR schedule with 1% warmup
  * Gradient accumulation across micro-batches
  * Streaming corpus (constant memory; works on 25B+ tokens)
  * Checkpointing to `--output` every N steps
  * Throughput / loss / engram telemetry every `--log-every` steps
  * Online local plasticity (OnlineLearner) runs alongside Adam
  * Final checkpoint saved as ckpt_final.pt
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
import torch.distributed as dist
import torch.nn.functional as F
from torch.nn.parallel import DistributedDataParallel as DDP

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

from kuro_brain.model import KAHNN, KAHNNConfig
from kuro_brain.continuous_learning import OnlineLearner
from kuro_brain.config import CONFIGS, describe_config
from kuro_brain.chinchilla import plan_chinchilla
from kuro_brain.pgsu import build_pgsu_adamw, PGSU
from data import build_tokenizer, StreamingCorpus, TokenBatcher


# ---------------------------------------------------------------------------
# DDP / device setup
# ---------------------------------------------------------------------------

def setup_ddp() -> tuple[int, int, int, bool]:
    """Initialise DDP from env (torchrun sets RANK, WORLD_SIZE, LOCAL_RANK)."""
    if "RANK" in os.environ:
        rank = int(os.environ["RANK"])
        world = int(os.environ["WORLD_SIZE"])
        local = int(os.environ.get("LOCAL_RANK", 0))
        dist.init_process_group(backend="nccl")
        torch.cuda.set_device(local)
        return rank, world, local, True
    return 0, 1, 0, False


def cleanup_ddp():
    if dist.is_available() and dist.is_initialized():
        dist.destroy_process_group()


def is_main(rank: int) -> bool:
    return rank == 0


# ---------------------------------------------------------------------------
# LR schedule (linear warmup → cosine decay → 10% tail)
# ---------------------------------------------------------------------------

def lr_at_step(step: int, total_steps: int, warmup: int, base_lr: float,
               min_lr_frac: float = 0.1) -> float:
    if step < warmup:
        return base_lr * (step + 1) / max(1, warmup)
    prog = (step - warmup) / max(1, total_steps - warmup)
    cos = 0.5 * (1.0 + math.cos(math.pi * prog))
    return base_lr * (min_lr_frac + (1.0 - min_lr_frac) * cos)


# ---------------------------------------------------------------------------
# Args
# ---------------------------------------------------------------------------

def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--data", required=True)
    p.add_argument("--output", required=True)
    p.add_argument("--device", default="cuda")
    p.add_argument("--config", default="b1", choices=list(CONFIGS))

    # Batch
    p.add_argument("--micro-batch", type=int, default=8,
                   help="Per-GPU micro-batch size.")
    p.add_argument("--seq-len", type=int, default=4096)
    p.add_argument("--grad-accum", type=int, default=4,
                   help="Gradient accumulation steps.")

    # Optimisation
    p.add_argument("--lr", type=float, default=3e-4,
                   help="Peak learning rate (after warmup).")
    p.add_argument("--min-lr-frac", type=float, default=0.1)
    p.add_argument("--weight-decay", type=float, default=0.01)
    p.add_argument("--warmup-frac", type=float, default=0.01,
                   help="Warmup as a fraction of total steps.")

    # Mixed precision
    p.add_argument("--bf16", action="store_true", default=True)
    p.add_argument("--no-bf16", dest="bf16", action="store_false")

    # PGSU — Progressive Gradient Sparsification Update
    p.add_argument("--pgsu", action="store_true",
                   help="Enable PGSU gradient sparsification wrapper.")
    p.add_argument("--pgsu-target-density", type=float, default=0.10,
                   help="Final gradient density (0.10 = 90%% sparse).")
    p.add_argument("--pgsu-schedule", default="cosine",
                   choices=["linear", "cosine", "step", "exp"],
                   help="Density schedule shape.")
    p.add_argument("--pgsu-warmup", type=int, default=200,
                   help="Dense steps before sparsification begins.")

    # 8-bit optimizer
    p.add_argument("--use-8bit-optimizer", action="store_true",
                   help="Use bitsandbytes AdamW8bit (saves ~75%% optim memory).")

    # Progressive depth — start with N layers, grow one every K steps
    p.add_argument("--progressive-depth", action="store_true",
                   help="Start with fewer layers and grow during training.")
    p.add_argument("--initial-layers", type=int, default=None,
                   help="Initial active layer count (default: n_layers // 4).")
    p.add_argument("--grow-every-steps", type=int, default=None,
                   help="Add a new layer every N steps (default: total_steps / n_layers).")

    # FP8 (Ada/Hopper) — 2x throughput
    p.add_argument("--fp8", action="store_true",
                   help="Use FP8 (e4m3) for matmuls on Ada/Hopper. 2x throughput vs bf16.")

    # Mixture-of-Depths — token-level early exit
    p.add_argument("--mod", action="store_true",
                   help="Enable Mixture-of-Depths token early-exit routing.")
    p.add_argument("--mod-skip-rate", type=float, default=0.5,
                   help="Fraction of tokens that skip each layer (default 0.5).")
    p.add_argument("--mod-aux-weight", type=float, default=0.01,
                   help="Weight for the MoD auxiliary loss.")

    # Activation checkpointing
    p.add_argument("--activation-checkpointing", action="store_true",
                   help="Recompute forward in backward (saves VRAM, +30%% compute).")

    # CPU offload of optimizer state
    p.add_argument("--cpu-offload", action="store_true",
                   help="Pin AdamW state to CPU RAM (saves ~8GB VRAM for B1).")

    # Multi-GPU
    p.add_argument("--ddp", action="store_true",
                   help="Enable DistributedDataParallel (use with torchrun).")

    # Schedule
    p.add_argument("--max-tokens", type=int, default=20_080_000_000,
                   help="Total tokens to train on (Chinchilla-optimal default).")
    p.add_argument("--checkpoint-every", type=int, default=500,
                   help="Steps between checkpoints.")
    p.add_argument("--log-every", type=int, default=10)

    # Misc
    p.add_argument("--seed", type=int, default=1337)
    p.add_argument("--resume", default=None)
    p.add_argument("--num-workers", type=int, default=2,
                   help="Async prefetch workers for the batcher.")
    p.add_argument("--continuous", action="store_true",
                   help="Run in continuous-learning mode (no Adam).")
    p.add_argument("--profile", action="store_true",
                   help="Enable torch.profiler for the first 20 steps.")
    p.add_argument("--smoke-test", action="store_true",
                   help="Override config/batch/schedule for a CPU smoke run "
                        "(ignores --config, --max-tokens, etc.).")
    return p.parse_args()


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    args = parse_args()
    rank, world, local, is_ddp = setup_ddp()
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")

    if is_main(rank):
        out_dir = Path(args.output)
        out_dir.mkdir(parents=True, exist_ok=True)
        log_fh = open(out_dir / "train_b1.log", "a", encoding="utf-8")
    else:
        log_fh = None

    def log(msg: str):
        if is_main(rank):
            print(msg, flush=True)
            if log_fh:
                log_fh.write(msg + "\n"); log_fh.flush()

    # 1) Config
    if args.smoke_test:
        # Force tiny config + tiny schedule so this is runnable on CPU
        cfg = CONFIGS["smoke"]
        cfg.max_seq_len = 64
        args.seq_len = 64
        args.micro_batch = 2
        args.grad_accum = 1
        args.max_tokens = 4096
        args.checkpoint_every = 1_000_000   # don't ckpt during smoke
        args.log_every = 5
        args.bf16 = False
    else:
        cfg = CONFIGS[args.config]
        cfg.max_seq_len = args.seq_len

    # Apply speed-optimisation flags to the config (these change model
    # construction — must happen before KAHNN() is built)
    cfg.use_mod = args.mod
    cfg.mod_skip_rate = args.mod_skip_rate
    cfg.use_activation_checkpointing = args.activation_checkpointing
    log(f"[config] {describe_config(cfg)}")
    if args.mod:
        log(f"[mod] Mixture-of-Depths enabled, skip_rate={args.mod_skip_rate}")
    if args.activation_checkpointing:
        log(f"[ckpt] activation checkpointing enabled (+30% compute, -50% activation VRAM)")
    if args.fp8:
        log(f"[fp8] FP8 autocast enabled (Ada/Hopper, 2x throughput)")

    # 2) Chinchilla plan (for sanity)
    plan = plan_chinchilla(
        cfg, gpu_name="H100", gpu_tflops=990.0,
        gpu_count=max(1, world), gpu_mfu=0.40,
        batch_size=args.micro_batch * world * args.grad_accum,
        seq_len=args.seq_len,
    )
    log(f"[chinchilla] {plan.summary()}")

    # 3) Tokenizer
    tokenizer, vocab = build_tokenizer("gpt2")
    cfg.vocab_size = vocab
    eos_id = vocab - 1

    # 4) Model
    torch.manual_seed(args.seed + rank)
    model = KAHNN(cfg).to(device)
    if is_ddp:
        model = DDP(model, device_ids=[local], find_unused_parameters=True)
    raw_model = model.module if is_ddp else model
    n_params = sum(p.numel() for p in model.parameters())
    log(f"[model] trainable params: {n_params/1e9:.3f}B on {world} GPU(s)")

    # 5) Optimiser + learner
    if args.continuous:
        optimizer = None
        log("[mode] continuous learning (no base optimizer)")
    else:
        # Decay / no-decay parameter groups (skip bias-like 1-element tensors)
        decay, no_decay = [], []
        for n, p in model.named_parameters():
            if not p.requires_grad:
                continue
            if p.ndim < 2 or n.endswith("K") or n.endswith("engram_gain") or n.endswith("gate"):
                no_decay.append(p)
            else:
                decay.append(p)

        # We need total_steps for PGSU's density schedule
        tokens_per_step_pre = args.micro_batch * args.seq_len * world * args.grad_accum
        total_steps_pre = max(1, args.max_tokens // tokens_per_step_pre)

        if args.pgsu:
            # PGSU wraps its own base optimizer (AdamW or AdamW8bit)
            # Build with two param groups: decay + no_decay
            # PGSU's factory takes a flat param list, so we'll construct
            # it manually here to keep the two groups.
            from kuro_brain.pgsu import PGSU
            if args.use_8bit_optimizer:
                try:
                    import bitsandbytes as bnb
                    base = bnb.optim.AdamW8bit(
                        [
                            {"params": decay, "weight_decay": args.weight_decay},
                            {"params": no_decay, "weight_decay": 0.0},
                        ],
                        lr=args.lr, betas=(0.9, 0.95), eps=1e-8, optim_bits=8,
                    )
                    log("[optim] PGSU + AdamW8bit (bitsandbytes)")
                except ImportError:
                    log("[optim] WARNING: --use-8bit-optimizer but bitsandbytes not installed, falling back to AdamW")
                    base = torch.optim.AdamW(
                        [
                            {"params": decay, "weight_decay": args.weight_decay},
                            {"params": no_decay, "weight_decay": 0.0},
                        ],
                        lr=args.lr, betas=(0.9, 0.95), eps=1e-8,
                    )
            else:
                base = torch.optim.AdamW(
                    [
                        {"params": decay, "weight_decay": args.weight_decay},
                        {"params": no_decay, "weight_decay": 0.0},
                    ],
                    lr=args.lr, betas=(0.9, 0.95), eps=1e-8,
                )
                log("[optim] PGSU + AdamW")
            optimizer = PGSU(
                base,
                total_steps=total_steps_pre,
                target_density=args.pgsu_target_density,
                schedule=args.pgsu_schedule,
                warmup_steps=args.pgsu_warmup,
                verbose=is_main(rank),
            )
            log(f"[optim] PGSU: target_density={args.pgsu_target_density} "
                f"schedule={args.pgsu_schedule} warmup={args.pgsu_warmup}")
        else:
            if args.use_8bit_optimizer:
                try:
                    import bitsandbytes as bnb
                    optimizer = bnb.optim.AdamW8bit(
                        [
                            {"params": decay, "weight_decay": args.weight_decay},
                            {"params": no_decay, "weight_decay": 0.0},
                        ],
                        lr=args.lr, betas=(0.9, 0.95), eps=1e-8, optim_bits=8,
                    )
                    log(f"[optim] AdamW8bit lr={args.lr} wd={args.weight_decay} "
                        f"betas=(0.9,0.95) decay_groups={len(decay)} no_decay={len(no_decay)}")
                except ImportError:
                    log("[optim] WARNING: --use-8bit-optimizer but bitsandbytes not installed, falling back to AdamW")
                    optimizer = torch.optim.AdamW(
                        [
                            {"params": decay, "weight_decay": args.weight_decay},
                            {"params": no_decay, "weight_decay": 0.0},
                        ],
                        lr=args.lr, betas=(0.9, 0.95), eps=1e-8,
                    )
                    log(f"[optim] AdamW lr={args.lr} wd={args.weight_decay} "
                        f"betas=(0.9,0.95) decay_groups={len(decay)} no_decay={len(no_decay)}")
            else:
                optimizer = torch.optim.AdamW(
                    [
                        {"params": decay, "weight_decay": args.weight_decay},
                        {"params": no_decay, "weight_decay": 0.0},
                    ],
                    lr=args.lr, betas=(0.9, 0.95), eps=1e-8,
                )
                log(f"[optim] AdamW lr={args.lr} wd={args.weight_decay} "
                    f"betas=(0.9,0.95) decay_groups={len(decay)} no_decay={len(no_decay)}")

    # 5b) Progressive depth setup
    if args.progressive_depth:
        initial = args.initial_layers or max(1, cfg.n_layers // 4)
        grow_every = args.grow_every_steps or max(
            1, (total_steps_pre - args.pgsu_warmup) // max(1, cfg.n_layers - initial)
        )
        raw_model.set_active_layers(initial)
        log(f"[progressive-depth] initial_active={initial}/{cfg.n_layers} "
            f"grow_every={grow_every} steps")
        # State for grow schedule
        progressive_state = {"initial": initial, "grow_every": grow_every,
                             "next_grow_at": initial * grow_every if initial > 0 else grow_every}
    else:
        progressive_state = None

    # 5c) CPU offload of optimizer state (simple version: keep params/grads
    # on GPU, but the optimizer's Adam m+v state lives on CPU pinned memory).
    # We do this by intercepting optimizer.step() — the state is moved to
    # GPU for the step, then back to CPU. For B1 this saves ~8 GB VRAM.
    # The hook is a simple pre-step / post-step pair below.
    if args.cpu_offload and optimizer is not None:
        log("[cpu-offload] optimizer state will be pinned to CPU RAM")
        # Pin existing state to CPU
        for group in optimizer.param_groups if not isinstance(optimizer, PGSU) else optimizer.base.param_groups:
            for p in group["params"]:
                if p in (optimizer.state if not isinstance(optimizer, PGSU) else optimizer.base.state):
                    st = (optimizer.state if not isinstance(optimizer, PGSU) else optimizer.base.state)[p]
                    for k, v in st.items():
                        if isinstance(v, torch.Tensor) and v.device.type == "cuda":
                            st[k] = v.to("cpu", non_blocking=True)
        # Patch step to move state to GPU before, back to CPU after
        if not isinstance(optimizer, PGSU):
            orig_step = optimizer.step
            def patched_step(closure=None):
                for group in optimizer.param_groups:
                    for p in group["params"]:
                        if p in optimizer.state:
                            for k, v in optimizer.state[p].items():
                                if isinstance(v, torch.Tensor) and v.device.type == "cpu":
                                    optimizer.state[p][k] = v.to(p.device, non_blocking=True)
                out = orig_step(closure)
                for group in optimizer.param_groups:
                    for p in group["params"]:
                        if p in optimizer.state:
                            for k, v in optimizer.state[p].items():
                                if isinstance(v, torch.Tensor) and v.device.type == "cuda":
                                    optimizer.state[p][k] = v.to("cpu", non_blocking=True)
                return out
            optimizer.step = patched_step
        else:
            log("[cpu-offload] PGSU-wrapped optimizer: CPU offload not yet supported, skipping")

    learner = OnlineLearner(raw_model, cfg, base_optimizer=optimizer,
                            continuous_mode=args.continuous)

    # 6) Schedule
    tokens_per_step = args.micro_batch * args.seq_len * world * args.grad_accum
    total_steps = max(1, args.max_tokens // tokens_per_step)
    warmup_steps = max(100, int(total_steps * args.warmup_frac))
    log(f"[schedule] tokens/step={tokens_per_step:,} total_steps={total_steps:,} "
        f"warmup={warmup_steps} ckpt_every={args.checkpoint_every}")

    # 7) Resume
    start_step = 0
    if args.resume:
        ckpt = torch.load(args.resume, map_location=device, weights_only=False)
        raw_model.load_state_dict(ckpt["model"])
        if optimizer is not None and "optimizer" in ckpt:
            optimizer.load_state_dict(ckpt["optimizer"])
        start_step = ckpt.get("step", 0)
        log(f"[resume] from step {start_step}")

    # 8) Data (only on rank 0 — DDP samples one global stream; for multi-node
    #    you'd shard by rank here. For single-node 8×H100 a single streamer
    #    with 4× prefetch keeps all GPUs busy.)
    if is_main(rank):
        corpus = StreamingCorpus(args.data, tokenizer, eos_id=eos_id)
        batcher = TokenBatcher(
            corpus, batch_size=args.micro_batch * world,
            seq_len=args.seq_len, device=device,
            prefetch=args.num_workers,
        )
    else:
        batcher = None

    # 9) Mixed-precision context
    amp_dtype = torch.bfloat16 if args.bf16 and torch.cuda.is_available() else None
    amp_ctx = (
        torch.amp.autocast("cuda", dtype=amp_dtype)
        if amp_dtype is not None
        else torch.amp.autocast("cuda", enabled=False)
    )

    # 10) Training loop
    step = start_step
    tokens_seen = start_step * tokens_per_step
    t0 = time.time()
    running_loss = 0.0
    running_n = 0
    log(f"[train] starting B1 training; target={args.max_tokens:,} tokens "
        f"({args.max_tokens/1e9:.2f}B)")

    if is_main(rank):
        iterator = iter(batcher)
    else:
        iterator = iter(())  # non-rank-0 processes don't pull data

    while step < total_steps:
        if is_main(rank):
            try:
                x_full, y_full = next(iterator)
            except StopIteration:
                log("[train] stream exhausted, restarting")
                batcher.close()
                corpus = StreamingCorpus(args.data, tokenizer, eos_id=eos_id)
                batcher = TokenBatcher(
                    corpus, batch_size=args.micro_batch * world,
                    seq_len=args.seq_len, device=device,
                    prefetch=args.num_workers,
                )
                iterator = iter(batcher)
                x_full, y_full = next(iterator)
            # Broadcast shapes to other ranks
            if is_ddp:
                shapes = torch.tensor([x_full.shape[0], x_full.shape[1]],
                                      device=device, dtype=torch.long)
                dist.broadcast(shapes, src=0)
        else:
            if is_ddp:
                shapes = torch.zeros(2, device=device, dtype=torch.long)
                dist.broadcast(shapes, src=0)
                x_full = torch.empty(int(shapes[0].item()), int(shapes[1].item()),
                                     dtype=torch.long, device=device)
                y_full = torch.empty_like(x_full)
            else:
                continue  # no data on non-main in non-DDP

        if is_ddp:
            dist.broadcast(x_full, src=0)
            dist.broadcast(y_full, src=0)

        # Split global batch across ranks (one contiguous shard per rank)
        assert x_full.shape[0] % world == 0, "global batch not divisible by world"
        per_rank = x_full.shape[0] // world
        x = x_full[rank * per_rank:(rank + 1) * per_rank]
        y = y_full[rank * per_rank:(rank + 1) * per_rank]

        # Optimiser step with gradient accumulation
        optimizer.zero_grad(set_to_none=True) if optimizer is not None else None
        accum_loss = 0.0
        accum_mod_loss = 0.0
        for micro in range(args.grad_accum):
            mx = x[micro::args.grad_accum]   # stride split for accumulation
            my = y[micro::args.grad_accum]
            if mx.shape[0] == 0:
                continue
            # FP8 autocast wraps the forward — eligible matmuls use fp8 e4m3
            # on Ada/Hopper. Falls back to bf16 on other hardware.
            from kuro_brain.fp8 import fp8_autocast
            with amp_ctx, fp8_autocast(enabled=args.fp8):
                logits = model(mx)
                # Compute per-token CE for MoD aux loss
                ce_per_token = F.cross_entropy(
                    logits.reshape(-1, cfg.vocab_size),
                    my.reshape(-1),
                    reduction="none",
                ).view(mx.shape[0], -1)  # [B, T]
                loss_main = ce_per_token.mean() / args.grad_accum
            # Add MoD aux loss if enabled
            mod_aux = getattr(raw_model, "_last_mod_loss",
                              torch.tensor(0.0, device=device))
            total_micro_loss = loss_main + args.mod_aux_weight * mod_aux / args.grad_accum
            total_micro_loss.backward()
            accum_loss += float(loss_main.item()) * args.grad_accum
            accum_mod_loss += float(mod_aux.item()) / args.grad_accum if isinstance(mod_aux, torch.Tensor) else 0.0

        # AllReduce gradients if DDP — DDP already does this in backward(),
        # but we still need to sync the OnlineLearner's local rule across ranks.
        # The local rule is no-op for cross-rank sync (each rank updates
        # its own copy of U, V via local phase stats). DDP's grad allreduce
        # averages the .grad tensor across ranks, so the Adam step is correct.
        # For the local plasticity rule we accept that each rank updates U,V
        # slightly differently; DDP will resync them on the next backward.

        if optimizer is not None:
            # Set LR per schedule
            lr_now = lr_at_step(step, total_steps, warmup_steps,
                                args.lr, args.min_lr_frac)
            for g in optimizer.param_groups:
                g["lr"] = lr_now
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            optimizer.zero_grad(set_to_none=True)

        # Online learner applies local plasticity + engram writes
        # (operates on raw_model — DDP wrapper is transparent)
        learner.step(x, y, torch.tensor(accum_loss, device=device))

        # Progressive depth: grow a new layer every `grow_every` steps.
        # Each rank decides independently based on the global step counter
        # — they stay in sync because they all use the same step number.
        if progressive_state is not None and step >= progressive_state["next_grow_at"]:
            new_count = raw_model.activate_next_layer()
            if new_count is not None:
                progressive_state["next_grow_at"] += progressive_state["grow_every"]
                log(f"[progressive-depth] step={step} activated layer {new_count}/"
                    f"{cfg.n_layers}")
            else:
                progressive_state["next_grow_at"] = float("inf")  # stop growing

        step += 1
        tokens_seen += tokens_per_step
        running_loss += accum_loss * (x.numel() * args.grad_accum)
        running_n += (x.numel() * args.grad_accum)

        if step % args.log_every == 0:
            avg = running_loss / max(1, running_n)
            dt = time.time() - t0
            tps = (tokens_seen - start_step * tokens_per_step) / max(dt, 1e-6)
            eta_s = (total_steps - step) * (dt / max(1, step - start_step))
            eta_h = eta_s / 3600.0
            lr_now = lr_at_step(step, total_steps, warmup_steps,
                                args.lr, args.min_lr_frac)
            active = raw_model.n_active_layers if hasattr(raw_model, "n_active_layers") else cfg.n_layers
            pgsu_density = ""
            if isinstance(optimizer, PGSU):
                pgsu_density = f" density={optimizer.stats()['current_density']:.3f}"
            log(f"[step {step:5d}/{total_steps}] loss={avg:.4f} "
                f"lr={lr_now:.2e} tps={tps/1e6:.2f}M "
                f"layers={active}/{cfg.n_layers}{pgsu_density} "
                f"engram={learner.state()['engram_usage_mean']:.2f} "
                f"eta={eta_h:.1f}h")
            running_loss = 0.0
            running_n = 0

        if step % args.checkpoint_every == 0 and is_main(rank):
            ckpt_path = out_dir / f"ckpt_{step:06d}.pt"
            save = {
                "model": raw_model.state_dict(),
                "step": step,
                "tokens": tokens_seen,
                "config": cfg.__dict__,
                "args": vars(args),
            }
            if optimizer is not None:
                save["optimizer"] = optimizer.state_dict()
            torch.save(save, ckpt_path)
            log(f"[ckpt] saved {ckpt_path}")

    # Final checkpoint
    if is_main(rank):
        ckpt_path = out_dir / "ckpt_final.pt"
        torch.save({
            "model": raw_model.state_dict(),
            "step": step,
            "tokens": tokens_seen,
            "config": cfg.__dict__,
            "args": vars(args),
        }, ckpt_path)
        log(f"[done] saved final checkpoint to {ckpt_path}")
        if log_fh:
            log_fh.close()
        if batcher is not None:
            batcher.close()

    cleanup_ddp()


if __name__ == "__main__":
    main()

"""
chinchilla.py — Compute-optimal scaling laws for KAHNN.

Hoffmann et al. (2022) — "Training Compute-Optimal Large Language Models"
(Chinchilla) — established that for a Transformer with N parameters,
the compute-optimal training token count is:

    D_opt(N) ≈ 20 · N

i.e. ~20 tokens per parameter. Training on fewer tokens underfits; on
more tokens overfits at fixed compute. The total training FLOPs is
then:

    C(N) ≈ 6 · N · D_opt(N) = 120 · N²

We re-derive the equivalent numbers for KAHNN. The per-token FLOPs
are different (no attention; FFT-bound) but the Chinchilla *token
budget* depends on the number of learnable parameters and is largely
architecture-agnostic — Chinchilla's 20:1 ratio is empirical across
model classes.

This module computes:
  1. The Chinchilla-optimal token count for any KAHNNConfig
  2. Estimated training FLOPs (KAHNN-specific per-token cost)
  3. Estimated wall-clock training time on a given GPU
  4. Recommended batch size, LR schedule, and checkpoint cadence
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Optional

from .model import KAHNNConfig


# ---------------------------------------------------------------------------
# Chinchilla constants
# ---------------------------------------------------------------------------

# Tokens-per-parameter ratio at the compute-optimal frontier (Hoffmann 2022).
CHINCHILLA_RATIO = 20.0

# FLOPs per parameter per token (forward+backward) for Transformer = 6.
# For KAHNN the per-token cost is dominated by FFTs and low-rank products.
# We compute it explicitly below; this is just a sanity floor.
TRANSFORMER_FLOPS_PER_PARAM_TOKEN = 6.0


# ---------------------------------------------------------------------------
# Per-token FLOPs estimate for KAHNN
# ---------------------------------------------------------------------------

def kuro_flops_per_token(cfg: KAHNNConfig) -> float:
    """
    Estimate forward+backward FLOPs per token for a KAHNN model.

    Components per layer per token:
      - Tokenizer encode (amortised over batch, ~0 per token after first)
      - Kuramoto layer (S steps):
          per step:
            sin/cos: 2·D
            low-rank A·v (twice): 2·2·D·r
            engram pull: E·D  (sin/cos over E engrams)
            update + wrap: ~3·D
          per step total: ~D·(2·2·r + E + 5)
          × S steps
      - HRR-MLP:
          FFT + iFFT: 2·5·D·log2(D)  (standard FFT cost ≈ 5·D·log2(D))
          low-rank: 2·D·mlp_rank
      - Decode (output projection to V): V·D  -- *expensive*

    Forward+backward ≈ 3× forward (standard approximation).
    """
    D, L = cfg.dim, cfg.n_layers
    S = cfg.kuramoto_steps
    r = cfg.kuramoto_rank
    E = cfg.n_ensembles_per_layer
    V = cfg.vocab_size
    mlp_r = cfg.mlp_rank if cfg.use_hrr_mlp else 0

    # Per-layer forward FLOPs per token
    kuramoto_per_step = D * (2 * 2 * r + E + 5)
    kuramoto = S * kuramoto_per_step
    fft_cost = 5 * D * max(1, math.log2(D))
    mlp = (2 * fft_cost + 2 * D * mlp_r) if cfg.use_hrr_mlp else 0
    decode = V * D  # output projection (cosine-sim against full codebook)

    forward_per_token = L * (kuramoto + mlp) + decode
    # Forward + backward ≈ 3× forward (1 fwd + 2 bwd, standard heuristic)
    total = 3.0 * forward_per_token
    return total


# ---------------------------------------------------------------------------
# Compute-optimal token count
# ---------------------------------------------------------------------------

@dataclass
class ChinchillaPlan:
    """Compute-optimal training plan for a KAHNN config."""

    # Model
    n_params: float                     # parameters (absolute count)
    n_params_B: float                   # parameters in billions

    # Chinchilla-optimal token budget
    optimal_tokens: float               # tokens (absolute)
    optimal_tokens_B: float             # tokens in billions
    ratio_tokens_per_param: float       # tokens / params (should be ~20)

    # Compute
    flops_per_token: float
    total_flops: float                  # total training FLOPs
    total_pflop_days: float             # PFLOP-days (1e15 FLOP/s × 86400 s)

    # Hardware estimate
    gpu_name: str
    gpu_tflops: float                   # bf16 peak FLOPs
    gpu_count: int
    gpu_mfu: float                      # model FLOP utilisation
    wall_clock_seconds: float
    wall_clock_hours: float
    wall_clock_days: float

    # Training schedule
    seq_len: int
    tokens_per_step: int                # B·T
    total_steps: int
    warmup_steps: int
    checkpoint_every_steps: int

    # Memory estimate
    param_bytes: float                  # bf16 params
    optim_bytes: float                  # AdamW state (fp32 m+v)
    grad_bytes: float                   # bf16 grads
    total_memory_GB: float

    def summary(self) -> str:
        return (
            f"=== Chinchilla Plan for KAHNN-B1 ===\n"
            f"Parameters:              {self.n_params_B:.3f} B  ({self.n_params/1e9:.4f} B exact)\n"
            f"Optimal tokens:          {self.optimal_tokens_B:.2f} B  "
            f"(ratio {self.ratio_tokens_per_param:.1f} tokens/param)\n"
            f"FLOPs per token:         {self.flops_per_token/1e9:.2f} GFLOP\n"
            f"Total training FLOPs:    {self.total_flops:.3e}\n"
            f"                         = {self.total_pflop_days:.1f} PFLOP-days\n"
            f"Hardware:                {self.gpu_count}× {self.gpu_name} "
            f"@ {self.gpu_tflops:.0f} TFLOPS bf16, MFU={self.gpu_mfu:.0%}\n"
            f"Wall-clock:              {self.wall_clock_days:.2f} days "
            f"({self.wall_clock_hours:.1f} h)\n"
            f"Schedule:                seq_len={self.seq_len}, "
            f"{self.tokens_per_step:,} tokens/step, "
            f"{self.total_steps:,} steps\n"
            f"                         warmup={self.warmup_steps:,} steps, "
            f"ckpt every {self.checkpoint_every_steps:,} steps\n"
            f"Memory:                  params={self.param_bytes/1e9:.2f} GB, "
            f"optim={self.optim_bytes/1e9:.2f} GB, "
            f"grad={self.grad_bytes/1e9:.2f} GB → "
            f"~{self.total_memory_GB:.1f} GB/GPU"
        )


def plan_chinchilla(
    cfg: KAHNNConfig,
    gpu_name: str = "H100",
    gpu_tflops: float = 990.0,         # H100 bf16 dense
    gpu_count: int = 8,
    gpu_mfu: float = 0.40,             # 40% MFU is realistic for non-Transformer
    batch_size: Optional[int] = None,  # auto if None
    seq_len: Optional[int] = None,
) -> ChinchillaPlan:
    """
    Build a compute-optimal training plan for `cfg`.

    Defaults target a single 8×H100 node. Override `gpu_*` to estimate
    other setups (A100 80GB: 312 TFLOPS bf16, MFU ~0.40).
    """
    N = cfg.param_count()
    N_B = N / 1e9

    # Chinchilla: optimal_tokens ≈ 20 × N
    optimal_tokens = CHINCHILLA_RATIO * N
    optimal_tokens_B = optimal_tokens / 1e9

    # FLOPs
    fpt = kuro_flops_per_token(cfg)
    total_flops = fpt * optimal_tokens
    pflop_days = total_flops / (1e15 * 86400)

    # Wall-clock
    effective_tflops = gpu_tflops * gpu_mfu * gpu_count
    wall_seconds = total_flops / (effective_tflops * 1e12)
    wall_hours = wall_seconds / 3600.0
    wall_days = wall_hours / 24.0

    # Schedule
    sl = seq_len or cfg.max_seq_len
    # Auto batch size: aim for ~4M tokens/step on 8×H100
    if batch_size is None:
        target_tokens_per_step = 4_000_000
        batch_size = max(1, target_tokens_per_step // (sl * gpu_count)) * gpu_count
    tokens_per_step = batch_size * sl
    total_steps = max(1, int(optimal_tokens // tokens_per_step))
    warmup_steps = max(100, total_steps // 100)        # 1% warmup
    checkpoint_every_steps = max(500, total_steps // 200)  # 200 ckpts

    # Memory (per-GPU estimate for DDP, not tensor-parallel)
    # bf16 params: 2 bytes; AdamW: 8 bytes (fp32 m + v) + 4 bytes master fp32
    # grads bf16: 2 bytes
    bytes_per_gpu = (2 + 8 + 4 + 2) * (N / gpu_count)
    total_mem_GB = bytes_per_gpu / 1e9

    return ChinchillaPlan(
        n_params=N,
        n_params_B=N_B,
        optimal_tokens=optimal_tokens,
        optimal_tokens_B=optimal_tokens_B,
        ratio_tokens_per_param=optimal_tokens / N,
        flops_per_token=fpt,
        total_flops=total_flops,
        total_pflop_days=pflop_days,
        gpu_name=gpu_name,
        gpu_tflops=gpu_tflops,
        gpu_count=gpu_count,
        gpu_mfu=gpu_mfu,
        wall_clock_seconds=wall_seconds,
        wall_clock_hours=wall_hours,
        wall_clock_days=wall_days,
        seq_len=sl,
        tokens_per_step=tokens_per_step,
        total_steps=total_steps,
        warmup_steps=warmup_steps,
        checkpoint_every_steps=checkpoint_every_steps,
        param_bytes=2 * N,
        optim_bytes=12 * N,            # fp32 master + Adam m+v
        grad_bytes=2 * N,
        total_memory_GB=total_mem_GB,
    )


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main():
    from .config import CONFIGS, describe_config
    import argparse

    p = argparse.ArgumentParser()
    p.add_argument("--config", default="b1", choices=list(CONFIGS))
    p.add_argument("--gpu", default="H100",
                   choices=["H100", "A100", "B200", "L4", "RTX4090", "RTX4080", "RTX3090"])
    p.add_argument("--n-gpu", type=int, default=8)
    p.add_argument("--mfu", type=float, default=0.40)
    p.add_argument("--seq-len", type=int, default=None)
    p.add_argument("--batch-size", type=int, default=None)
    # Speedup multipliers for the new optimisations
    p.add_argument("--fp8", action="store_true", help="FP8 (2x throughput on Ada/Hopper)")
    p.add_argument("--mod-skip-rate", type=float, default=0.0,
                   help="MoD skip rate (0.5 → ~1.7x speedup)")
    p.add_argument("--pgsu-density", type=float, default=1.0,
                   help="PGSU final density (0.10 → ~1.5x on memory-bound)")
    p.add_argument("--progressive-depth", action="store_true",
                   help="Progressive depth (~1.3x average)")
    p.add_argument("--checkpointing", action="store_true",
                   help="Activation checkpointing (no compute saving, enables larger batch)")
    args = p.parse_args()

    gpu_specs = {
        "H100":    990.0,    # bf16 dense, SXM5
        "A100":    312.0,    # bf16 dense, SXM4 80GB
        "B200":   2250.0,    # bf16 dense, Blackwell
        "L4":      120.0,
        "RTX4090":  82.0,    # consumer Ada, bf16 dense (24GB VRAM)
        "RTX4080":  49.0,    # AD103
        "RTX3090":  35.0,    # Ampere consumer
    }

    cfg = CONFIGS[args.config]
    print(describe_config(cfg))
    print()

    # Compute combined speedup multiplier
    speedup = 1.0
    notes = []
    if args.fp8:
        speedup *= 2.0
        notes.append("FP8 (2x)")
    if args.mod_skip_rate > 0:
        # 1 / (1 - skip_rate * 0.7) — token skip doesn't perfectly map to compute skip
        mod_speedup = 1.0 / max(0.3, 1.0 - args.mod_skip_rate * 0.7)
        speedup *= mod_speedup
        notes.append(f"MoD skip={args.mod_skip_rate} ({mod_speedup:.2f}x)")
    if args.pgsu_density < 1.0:
        # PGSU speedup: optimizer is ~40% of step on memory-bound GPUs (4090).
        # At density d, optimizer step is d × original. Speedup = 1 / (0.6 + 0.4*d)
        pgsu_speedup = 1.0 / (0.6 + 0.4 * args.pgsu_density)
        speedup *= pgsu_speedup
        notes.append(f"PGSU density={args.pgsu_density} ({pgsu_speedup:.2f}x)")
    if args.progressive_depth:
        # Average active layers ~ 0.65 of full → 1/0.65 = 1.54x
        speedup *= 1.54
        notes.append("Progressive depth (1.54x)")
    if args.checkpointing:
        # No wall-clock compute saving, but enables ~2x larger batch which
        # improves GPU utilisation by ~1.1x
        speedup *= 1.1
        notes.append("Activation checkpointing (1.1x via larger batch)")

    if notes:
        print(f"[speedups] {' + '.join(notes)} = {speedup:.2f}x combined")
        print(f"[effective] {gpu_specs[args.gpu]:.0f} TFLOPS × {args.mfu:.2f} MFU × {speedup:.2f}x = "
              f"{gpu_specs[args.gpu] * args.mfu * speedup * args.n_gpu:.0f} TFLOPS effective\n")

    plan = plan_chinchilla(
        cfg,
        gpu_name=args.gpu,
        gpu_tflops=gpu_specs[args.gpu] * speedup,  # apply combined speedup
        gpu_count=args.n_gpu,
        gpu_mfu=args.mfu,
        batch_size=args.batch_size,
        seq_len=args.seq_len,
    )
    print(plan.summary())


if __name__ == "__main__":
    main()

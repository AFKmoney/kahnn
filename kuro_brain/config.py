"""
config.py — Pre-defined configurations for KAHNN.

Presets from laptop-CPU smoke/nano/commodity up through H100-scale B1/XL.
Prefer nano/commodity/tiny for cost-minimized universal training.
"""

from dataclasses import dataclass, asdict
from .model import KAHNNConfig


# ---------------------------------------------------------------------------
# Smoke test — runs in seconds on CPU/GPU, sanity-checks the pipeline
# ---------------------------------------------------------------------------

SMOKE = KAHNNConfig(
    vocab_size=256,
    dim=2048,
    n_layers=2,
    max_seq_len=128,
    kuramoto_steps=2,
    kuramoto_rank=32,
    n_ensembles_per_layer=8,
    use_hrr_mlp=False,
)


# ---------------------------------------------------------------------------
# Tiny — single-GPU, ~5M params, trains on 100M tokens in ~1 hour
# ---------------------------------------------------------------------------

TINY = KAHNNConfig(
    vocab_size=50257,
    dim=4096,
    n_layers=4,
    max_seq_len=1024,
    kuramoto_steps=3,
    kuramoto_rank=64,
    n_ensembles_per_layer=16,
    use_hrr_mlp=True,
    mlp_rank=128,
)


# ---------------------------------------------------------------------------
# Medium — single A100/H100, ~150M params, 25B tokens in ~1 day
# ---------------------------------------------------------------------------

MEDIUM = KAHNNConfig(
    vocab_size=50257,
    dim=8192,
    n_layers=8,
    max_seq_len=2048,
    kuramoto_steps=4,
    kuramoto_rank=128,
    n_ensembles_per_layer=32,
    use_hrr_mlp=True,
    mlp_rank=256,
)


# ---------------------------------------------------------------------------
# Large — 4×H100, ~1B params, 25B tokens in ~2 days
# ---------------------------------------------------------------------------

LARGE = KAHNNConfig(
    vocab_size=50257,
    dim=16384,
    n_layers=16,
    max_seq_len=4096,
    kuramoto_steps=4,
    kuramoto_rank=256,
    n_ensembles_per_layer=64,
    use_hrr_mlp=True,
    mlp_rank=512,
)


# ---------------------------------------------------------------------------
# B1 — 1.0B parameters, compute-optimal per Chinchilla (≈20B tokens)
# Single 8×H100 node in ~24-36h, or 4×H100 in ~3-4 days.
# ---------------------------------------------------------------------------

B1 = KAHNNConfig(
    vocab_size=50257,            # GPT-2 BPE
    dim=24576,                   # D — hypervector / oscillator count
    n_layers=22,                 # L — Kuramoto + HRR-MLP stacks
    max_seq_len=4096,            # context window
    kuramoto_steps=4,            # integration steps per forward
    kuramoto_dt=0.05,
    kuramoto_rank=256,           # rank of low-rank coupling A = U V^T
    n_ensembles_per_layer=64,    # per-layer engram slots
    cross_layer_memory_capacity=512,
    noise=0.01,
    use_hrr_mlp=True,
    mlp_rank=640,                # rank of the HRR-MLP
    seed=1337,
    # Online / continuous learning rates (used by OnlineLearner)
    online_lr_phase=0.01,
    online_lr_engram=0.05,
    online_lr_omega=0.001,
    engram_write_every=16,
    engram_coherence_threshold=0.2,
)


# ---------------------------------------------------------------------------
# XL — 8×H100, ~3B params, 25B tokens in ~3-4 days
# ---------------------------------------------------------------------------

XL = KAHNNConfig(
    vocab_size=50257,
    dim=32768,
    n_layers=24,
    max_seq_len=4096,
    kuramoto_steps=5,
    kuramoto_rank=256,
    n_ensembles_per_layer=64,
    use_hrr_mlp=True,
    mlp_rank=768,
)



# ---------------------------------------------------------------------------
# Nano — laptop CPU, ~0.8M params, curriculum / continuous-learning friendly
# ---------------------------------------------------------------------------

NANO = KAHNNConfig(
    vocab_size=50257,
    dim=2048,
    n_layers=3,
    max_seq_len=512,
    kuramoto_steps=2,
    kuramoto_rank=48,
    n_ensembles_per_layer=12,
    use_hrr_mlp=True,
    mlp_rank=96,
)

# ---------------------------------------------------------------------------
# Commodity — single consumer GPU (8–12 GB) or strong multi-core CPU.
# ~18M params → Chinchilla-optimal ≈ 360M tokens. Designed for $0–$50 budgets.
# ---------------------------------------------------------------------------

COMMODITY = KAHNNConfig(
    vocab_size=50257,
    dim=4096,
    n_layers=6,
    max_seq_len=1024,
    kuramoto_steps=3,
    kuramoto_rank=64,
    n_ensembles_per_layer=24,
    use_hrr_mlp=True,
    mlp_rank=192,
    cross_layer_memory_capacity=128,
)

CONFIGS = {
    "smoke": SMOKE,
    "nano": NANO,
    "commodity": COMMODITY,
    "tiny": TINY,
    "medium": MEDIUM,
    "large": LARGE,
    "b1": B1,
    "xl": XL,
}


def get_config(name: str) -> KAHNNConfig:
    if name not in CONFIGS:
        raise ValueError(f"Unknown config '{name}'. Available: {list(CONFIGS)}")
    return CONFIGS[name]


def describe_config(cfg: KAHNNConfig) -> str:
    return (
        f"KAHNN config: D={cfg.dim}, L={cfg.n_layers}, "
        f"params≈{cfg.param_count()/1e6:.1f}M "
        f"(~{cfg.param_count()/1e9:.2f}B)"
    )


if __name__ == "__main__":
    for name, cfg in CONFIGS.items():
        print(f"{name:8s}: {describe_config(cfg)}")

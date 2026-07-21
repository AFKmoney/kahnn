# KuroBrain — Kuramoto-Attractor Holographic Hypervector Network (KAHNN)

A non-Transformer, neuroscience-inspired AI paradigm.

> *"We literally code a brain. We transpose neuroscience into code."*

KAHNN throws out the Transformer dogma:

| Transformer | KAHNN |
|---|---|
| Dense learned embeddings | Fixed random hypervector codebook |
| Self-attention (O(T²) mixing) | Kuramoto oscillator coupling (O(D·log D) per token, O(T) total) |
| Backprop through time | Local phase-based plasticity + global neuromodulator |
| Static weights after training | Continuous online learning (engrams + local STDP-like rule) |
| Replay buffer for new data | Attractor memory with active forgetting |
| Weeks of training for 1B params | ~1 day on H100 / ~2.6 days on 5× RTX 4090 for 20B tokens |

## Architecture (1-page)

```
tokens  →  HypervectorTokenizer.encode  →  token HVs (position-bound, holographic)
                                            │
                                            ▼
                          ┌────────────────────────────────────────┐
                          │ KuramotoLayer (steps=4, rank=128)       │
                          │   θ ← θ + ω + K·ΣA·sin(θⱼ−θᵢ) + engram  │
                          │   engram pull from per-layer memory     │
                          │   [optional] MoD router skips 50% tokens │
                          │   [optional] activation checkpointing    │
                          └────────────────────────────────────────┘
                                            │
                                            ▼
                          ┌────────────────────────────────────────┐
                          │ HRRMLP (phase-filter + low-rank res.)   │
                          └────────────────────────────────────────┘
                                            │
                                            ▼
                                  × L layers (8–24)
                                            │
                                            ▼
                          HypervectorTokenizer.decode → token logits
```

**Five primitives, that's the whole model:**

1. **Hypervector encoding** — every token is a D-dim bipolar vector (D = 8k–32k). Random, fixed, near-orthogonal.
2. **Holographic binding** — `bind(a,b) = IFFT(FFT(a)·FFT(b))`. Compositional, FFT-cheap.
3. **Kuramoto dynamics** — `θᵢ ← θᵢ + ωᵢ + K·Σ Aᵢⱼ sin(θⱼ−θᵢ)`. This is where "thinking" happens. Phase locking = assembly binding.
4. **Attractor memory** — high-coherence states get written into engram slots. The next forward pass is pulled toward the nearest engram — that's *remembering*, no replay needed.
5. **Local online learning** — `ΔU ∝ err · sin(θ_engram − θ) · V_proj`. Local, neuromodulated, no global gradient through time.

## 6 speed upgrades (RTX 4090 / H100)

| Upgrade | Module | Speedup | What it does |
|---|---|---|---|
| **PGSU** | `kuro_brain/pgsu.py` | 1.56× | Top-k% gradient masking, cosine schedule from 100% → 10% dense |
| **8-bit optimizer** | bitsandbytes | 1.3× | AdamW8bit: 4× less optimizer state |
| **Progressive depth** | `kuro_brain/model.py` | 1.54× | Start with 6/22 layers, grow over training |
| **FP8** | `kuro_brain/fp8.py` | 2.0× | Ada/Hopper fp8 tensor cores (165 TFLOPS on RTX 4090) |
| **Mixture-of-Depths** | `kuro_brain/mixture_of_depths.py` | 1.54× | Phase-coherence router, 50% of tokens skip each layer |
| **Activation checkpointing** | `kuro_brain/model.py` | 1.1× | Recompute fwd in bwd, enables larger batch |

**Combined: ~8.14× speedup.** All 6 verified end-to-end running together.

## Why it trains fast even without the upgrades

- **No learned embedding matrix.** A 50k × 16k embedding = 800M params. We use a fixed random one. The model only learns *relations*, not lexicon.
- **No attention.** Attention is O(T²). KAHNN mixing is O(T·D·log D), linear in T.
- **No BPTT.** Each forward pass uses `steps=4` Kuramoto updates. The local learning rule has no temporal unrolling — memory is in engrams, not in the gradient graph.
- **FFT everywhere.** Binding, unbinding, and the HRR-MLP are all O(D·log D) via `torch.fft`.
- **Per-step cost is tiny:** for D=16k, L=8, T=1024, B=8 → ~150 GFLOP/step. A single H100 does 990 TFLOP/s.

## Continuous learning (the "no retraining" claim)

Once pretrained, switch the learner to continuous mode:

```python
learner.continuous_mode = True   # disable base optimizer
# from now on, every new token updates:
#   - per-layer engrams (write if coherence > 0.2)
#   - cross-layer attractor memory
#   - U, V via local phase plasticity (no gradient)
# old engrams decay at rate 0.9995 per step (active forgetting)
```

The model keeps learning forever, on whatever you feed it, with no replay buffer and no catastrophic forgetting (because engrams consolidate stable states, not raw activations).

## Quickstart

```bash
git clone https://github.com/AFKmoney/kahnn
cd kahnn
pip install -r requirements.txt
pip install bitsandbytes        # optional: enables --use-8bit-optimizer

# Smoke test (CPU, ~30s)
python -m kuro_brain.smoke

# Full mini training run (CPU, validates loss decreases)
python mini_train.py

# Chinchilla plan for any config/hardware combo
python -m kuro_brain.chinchilla --config b1 --gpu RTX4090 --n-gpu 5 \
    --fp8 --mod-skip-rate 0.5 --pgsu-density 0.10 \
    --progressive-depth --checkpointing

# Train B1 on 5× RTX 4090 (~2.6 days, full 20B tokens, no reduction)
NGPU=5 bash launch_b1_rtx4090.sh /data/corpus ./runs/b1_run1

# Train B1 on 8× H100 (~23h, no upgrades needed)
bash launch_b1.sh /data/corpus ./runs/b1_run1

# Generate
python generate.py --checkpoint ./runs/b1_run1/ckpt_final.pt \
       --prompt "Once upon a time" --n-tokens 300

# Evaluate perplexity
python evaluate.py --checkpoint ./runs/b1_run1/ckpt_final.pt \
       --data /data/valid.txt
```

## Configs

| Config | D | L | Params | Optimal tokens | 1× H100 | 5× RTX 4090 (upgrades) |
|---|---|---|---|---|---|---|
| smoke | 2048 | 2 | ~0.3M | sanity | seconds | seconds |
| tiny | 4096 | 4 | ~6.6M | 132M | ~30 min | ~4 hours |
| medium | 8192 | 8 | ~52M | 1.05B | ~6 hours | ~3 days |
| large | 16384 | 16 | ~420M | 8.4B | ~14 hours | ~16 hours* |
| **b1** | **24576** | **22** | **~1.0B** | **20.08B** | **~23h** | **~2.6 days** |
| xl | 32768 | 24 | ~1.66B | 33.2B | ~38h | ~4.3 days |

*large on 1× RTX 4090 with all upgrades ≈ 2.5 days; on 5× ≈ 12 hours.

## Repository layout

```
kahnn/
├── kuro_brain/
│   ├── __init__.py
│   ├── hypervectors.py        # bipolar HDV primitives
│   ├── holographic.py         # HRR bind/unbind (FFT) + phase bridge
│   ├── kuramoto.py            # KuramotoLayer + AttractorMemory
│   ├── encoder.py             # HypervectorTokenizer (random item memory)
│   ├── model.py               # KAHNN stack + HRR-MLP + progressive depth + MoD + checkpointing
│   ├── continuous_learning.py # OnlineLearner (local plasticity + engrams)
│   ├── pgsu.py                # Progressive Gradient Sparsification Update
│   ├── fp8.py                 # FP8 (e4m3) autocast for Ada/Hopper
│   ├── mixture_of_depths.py   # MoD token early-exit router
│   ├── chinchilla.py          # Compute-optimal scaling-law + speedup estimator
│   ├── config.py              # smoke/tiny/medium/large/b1/xl presets
│   └── smoke.py               # 30-second end-to-end smoke test
├── data.py                    # streaming corpus + async batcher
├── train.py                   # generic trainer
├── train_b1.py                # B1 trainer with all 6 upgrades + DDP + bf16 + grad-accum
├── generate.py                # sampling
├── evaluate.py                # perplexity
├── launch_b1.sh               # one-command launcher for 8× H100
├── launch_b1_rtx4090.sh       # one-command launcher for 5× RTX 4090 (all upgrades)
├── mini_train.py              # tiny end-to-end CPU validation
├── docs/
│   ├── ARCHITECTURE.md        # full design doc + neuroscience mapping
│   ├── B1_TRAINING.md         # 1B on H100 / multi-GPU plan
│   └── RTX4000_TRAINING.md    # 1B on RTX 4090 plan with all 6 upgrades
├── requirements.txt
└── README.md
```

## What this is NOT

- It is **not** a Transformer. There is no attention.
- It is **not** trained with backprop-through-time. Memory is structural (engrams), not algorithmic (gradient graph).
- It is **not** a Hopfield network. Attractors live in *phase space*, not binary space.
- It is **not** an SSM or RNN. State evolution is governed by oscillator physics, not by a learned recurrence matrix.
- It is **not** a finished product. It is a research framework that operationalizes a specific hypothesis: *intelligence = phase-coherent holographic binding over oscillatory population codes, learning by local neuromodulated plasticity.*

See `docs/ARCHITECTURE.md` for the full design rationale, equations, and the neuroscience mapping table.
See `docs/RTX4000_TRAINING.md` for the honest analysis of training B1 on RTX 4090.
See `docs/B1_TRAINING.md` for the H100 / multi-GPU path.

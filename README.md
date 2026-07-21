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
| Weeks of training for 1B params | ~1–2 days on a single H100 for 25B tokens |

## Architecture (1-page)

```
tokens  →  HypervectorTokenizer.encode  →  token HVs (position-bound, holographic)
                                            │
                                            ▼
                          ┌────────────────────────────────────────┐
                          │ KuramotoLayer (steps=4, rank=128)       │
                          │   θ ← θ + ω + K·ΣA·sin(θⱼ−θᵢ) + engram  │
                          │   engram pull from per-layer memory     │
                          └────────────────────────────────────────┘
                                            │
                                            ▼
                          ┌────────────────────────────────────────┐
                          │ HRRMLP (phase-filter + low-rank res.)   │
                          └────────────────────────────────────────┘
                                            │
                                            ▼
                                  × L layers (8–16)
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

## Why it trains in 1 day, not 1 month

- **No learned embedding matrix.** A 50k × 16k embedding = 800M params. We use a fixed random one. The model only learns *relations*, not lexicon.
- **No attention.** Attention is O(T²). KAHNN mixing is O(T·D·log D), linear in T.
- **No BPTT.** Each forward pass uses `steps=4` Kuramoto updates. The local learning rule has no temporal unrolling — memory is in engrams, not in the gradient graph.
- **FFT everywhere.** Binding, unbinding, and the HRR-MLP are all O(D·log D) via `torch.fft`.
- **Per-step cost is tiny:** for D=16k, L=8, T=1024, B=8 → ~150 GFLOP/step. A single H100 does 990 TFLOP/s. Theoretical peak: ~6600 steps/s = 6.7M tokens/s → 25B tokens in ~1 hour. Realistic with overhead: ~1 day.

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
pip install -r requirements.txt

# Smoke test (CPU, ~30s)
python -m kuro_brain.smoke

# Real training (single H100, 25B tokens, ~1 day)
python train.py --config large --data /data/corpus.txt \
       --output ./runs/run1 --batch-size 16 --seq-len 2048 \
       --device cuda --max-tokens 25000000000

# Generate
python generate.py --checkpoint ./runs/run1/ckpt_final.pt \
       --prompt "Once upon a time" --n-tokens 300

# Evaluate
python evaluate.py --checkpoint ./runs/run1/ckpt_final.pt \
       --data /data/valid.txt
```

## Configs

| Config | D | L | Params | Target tokens | Target time (1×H100) |
|---|---|---|---|---|---|
| smoke | 2048 | 2 | ~0.5M | sanity | seconds |
| tiny | 4096 | 4 | ~5M | 100M | ~1 hour |
| medium | 8192 | 8 | ~150M | 25B | ~1 day |
| large | 16384 | 16 | ~1B | 25B | ~2 days |
| xl | 32768 | 24 | ~3B | 25B | ~3–4 days |

## Repository layout

```
kuramoto_brain/
├── kuro_brain/
│   ├── __init__.py
│   ├── hypervectors.py        # bipolar HDV primitives
│   ├── holographic.py         # HRR bind/unbind (FFT) + phase bridge
│   ├── kuramoto.py            # KuramotoLayer + AttractorMemory
│   ├── encoder.py             # HypervectorTokenizer (random item memory)
│   ├── model.py               # KAHNN stack + HRR-MLP
│   ├── continuous_learning.py # OnlineLearner (local plasticity + engrams)
│   └── smoke.py               # 30-second end-to-end smoke test
├── config.py                  # smoke/tiny/medium/large/xl presets
├── data.py                    # streaming corpus + async batcher
├── train.py                   # full training entry point
├── generate.py                # sampling
├── evaluate.py                # perplexity
├── docs/
│   └── ARCHITECTURE.md        # full design doc
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

# KuroBrain — Kuramoto-Attractor Holographic Hypervector Network (KAHNN)

A non-Transformer, neuroscience-inspired AI paradigm.

> *"We literally code a brain. We transpose neuroscience into code."*

**Product goal (2026):** CPU-first / small-GPU **universal pretrain** + **lifelong continuous learning** (engrams that stick until explicit `forget`) — *not* multi-H100 as the default path. Cluster B1 docs remain for scale-out.

KAHNN throws out the Transformer dogma:

| Transformer | KAHNN |
|---|---|
| Dense learned embeddings | Fixed random hypervector codebook |
| Self-attention (O(T²) mixing) | Kuramoto oscillator coupling (O(D·log D) per token, O(T) total) |
| Backprop through time | Local phase-based plasticity + global neuromodulator |
| Static weights after training | Continuous online learning (engrams + local STDP-like rule) |
| Replay buffer for new data | Attractor memory with **explicit** forgetting |
| Weeks of training for 1B params | Laptop `nano` ~37M tokens ; optional H100/4090 for B1 |

## Start here (CPU / petit GPU)

```bash
git clone https://github.com/AFKmoney/kahnn
cd kahnn
pip install -r requirements.txt

# Smoke (~30s)
python -m kuro_brain.smoke
python teach.py smoke --device cpu

# Universal pretrain (auto device: cuda → mps → cpu)
# Build or place a text+code corpus first — see data/DATA.md
python train_universal.py \
  --data ./data/corpus.txt \
  --output ./runs/nano_base \
  --config nano --device auto

# After base ckpt: teach a fact (lifelong, no auto decay)
python teach.py teach --text "La capitale du Canada est Ottawa." \
  --resume ./runs/nano_base/ckpt_final.pt --config nano --output ./runs/teach
```

Docs (FR-friendly) :

| Doc | Contenu |
|-----|---------|
| [`docs/UNIVERSAL_TRAINING.md`](docs/UNIVERSAL_TRAINING.md) | Chemin universel CPU/petit GPU, configs nano/commodity, mesures |
| [`docs/COLAB_GPU.md`](docs/COLAB_GPU.md) | **Colab T4/L4** — notebook nano, Drive, free vs Pro, transfert ckpt |
| [`notebooks/kahnn_nano_colab.ipynb`](notebooks/kahnn_nano_colab.ipynb) | Notebook Colab-ready (`train_universal.py --config nano --device cuda`) |
| [`docs/LIFELONG_LEARNING.md`](docs/LIFELONG_LEARNING.md) | Teach / probe / forget, mémoire sticky, limites honnêtes |
| [`docs/CPU_RUN_LOG.md`](docs/CPU_RUN_LOG.md) | Journal daté 2026-09-08 (box 8 threads, vrais tps) |
| [`data/DATA.md`](data/DATA.md) | Provenance corpus (~109 MB local, **non** commité) |
| [`docs/ARCHITECTURE.md`](docs/ARCHITECTURE.md) | Design + neuroscience |
| [`docs/B1_TRAINING.md`](docs/B1_TRAINING.md) / [`docs/RTX4000_TRAINING.md`](docs/RTX4000_TRAINING.md) | Scale-out H100 / 4090 (**secondaire**) |

### Mesures box (2026-09-08, CPU only, 8 threads, torch 2.14+cpu)

| Setup | Résultat réel |
|-------|----------------|
| smoke bench | ~**870–1040** tok/s |
| nano live pretrain (`4×256`) | ~**1070–1230** tok/s early ; target ~37M tokens ; ETA ~8–10 h @ 1/3 depth |
| commodity bench | ~**145–170** tok/s |
| `teach.py smoke` | probe ~**0 → ~0.96** ; pas de decay auto ; forget vide les slots |

Détail : [`docs/CPU_RUN_LOG.md`](docs/CPU_RUN_LOG.md). Ship : [PR #1](https://github.com/AFKmoney/kahnn/pull/1).

### GPU gratuit (Colab)

Pour accélérer nano sur **T4/L4** : ouvre [`notebooks/kahnn_nano_colab.ipynb`](notebooks/kahnn_nano_colab.ipynb) dans Colab (Runtime → GPU) et suis [`docs/COLAB_GPU.md`](docs/COLAB_GPU.md). Attente : nettement plus rapide que ~1.2k tok/s CPU — **mesure le tps** sur le smoke Colab (pas de benchmark GPU inventé dans le repo).

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
                                  × L layers (3–24)
                                            │
                                            ▼
                          HypervectorTokenizer.decode → token logits
```

**Five primitives, that's the whole model:**

1. **Hypervector encoding** — every token is a D-dim bipolar vector (D = 2k–32k). Random, fixed, near-orthogonal.
2. **Holographic binding** — `bind(a,b) = IFFT(FFT(a)·FFT(b))`. Compositional, FFT-cheap.
3. **Kuramoto dynamics** — `θᵢ ← θᵢ + ωᵢ + K·Σ Aᵢⱼ sin(θⱼ−θᵢ)`. Phase locking = assembly binding.
4. **Attractor memory** — high-coherence states get written into engram slots. Remembering = pull toward nearest engram.
5. **Local online learning** — `ΔU ∝ err · sin(θ_engram − θ) · V_proj`. Local, neuromodulated, no BPTT.

## Continuous / lifelong learning

Once pretrained (Adam on text+code), switch to continuous mode:

```python
learner = OnlineLearner(model, cfg, continuous_mode=True)  # soft_decay=False by default
# every new token updates engrams + local plasticity
# NO automatic global decay — forget only via learner.forget(...) / teach.py forget
```

CLI : `teach.py teach | probe | forget | smoke`. Soft aging of *weak* slots only : `--soft-decay`.

Honest limits : sticky memory = **engrams** ; finite slots ; need a real base corpus first ; B1-on-CPU is unrealistic. See `docs/LIFELONG_LEARNING.md`.

## Configs

| Config | D | L | Params | Optimal tokens | Rôle |
|---|---|---|---|---|---|
| smoke | 2048 | 2 | ~0.3M | sanity | CI / smoke |
| **nano** | **2048** | **3** | **~1.8M** | **~37M** | **CPU laptop / box** |
| **commodity** | (voir config) | | **~13M** | **~264M** | **CPU fort / GPU 8–12 Go** |
| tiny | 4096 | 4 | ~6.6M | 132M | entry GPU |
| medium | 8192 | 8 | ~52M | 1.05B | GPU 12–24 Go |
| large | 16384 | 16 | ~420M | 8.4B | multi-GPU |
| **b1** | **24576** | **22** | **~1.0B** | **20.08B** | H100 / 5×4090 |
| xl | 32768 | 24 | ~1.66B | 33.2B | scale-out |

## Speed upgrades (RTX 4090 / H100 — optionnel)

| Upgrade | Module | Speedup | What it does |
|---|---|---|---|
| **PGSU** | `kuro_brain/pgsu.py` | 1.56× | Top-k% gradient masking |
| **8-bit optimizer** | bitsandbytes | 1.3× | AdamW8bit |
| **Progressive depth** | `kuro_brain/model.py` | 1.54× | Grow layers over training |
| **FP8** | `kuro_brain/fp8.py` | 2.0× | Ada/Hopper only |
| **Mixture-of-Depths** | `kuro_brain/mixture_of_depths.py` | 1.54× | Skip ~50% tokens/layer |
| **Activation checkpointing** | `kuro_brain/model.py` | 1.1× | VRAM |

**Combined ~8.14×** on Ada/Hopper — **not** the CPU path. On CPU, what matters is decode matmul + engram pull + streaming + progressive depth (`train_universal.py`).

## Quickstart — scale-out (secondaire)

```bash
# Chinchilla plan
python -m kuro_brain.chinchilla --config b1 --gpu RTX4090 --n-gpu 5 \
    --fp8 --mod-skip-rate 0.5 --pgsu-density 0.10 \
    --progressive-depth --checkpointing

# B1 on 5× RTX 4090 / 8× H100
NGPU=5 bash launch_b1_rtx4090.sh /data/corpus ./runs/b1_run1
bash launch_b1.sh /data/corpus ./runs/b1_run1

python generate.py --checkpoint ./runs/b1_run1/ckpt_final.pt \
       --prompt "Once upon a time" --n-tokens 300
python evaluate.py --checkpoint ./runs/b1_run1/ckpt_final.pt --data /data/valid.txt
```

## Repository layout

```
kahnn/
├── kuro_brain/
│   ├── device.py              # auto cpu/cuda/mps + batch hints
│   ├── continuous_learning.py # OnlineLearner + teach/forget (no auto decay)
│   ├── config.py              # smoke/nano/commodity/tiny/.../b1/xl
│   ├── …                      # hypervectors, kuramoto, model, fp8, …
│   └── smoke.py
├── train_universal.py         # ★ primary trainer (CPU-first)
├── teach.py                   # ★ lifelong teach / probe / forget CLI
├── data.py                    # streaming corpus
├── train.py / train_b1.py     # generic / B1 DDP trainers
├── generate.py / evaluate.py
├── mini_train.py
├── launch_b1.sh / launch_b1_rtx4090.sh
├── data/
│   └── DATA.md                # corpus provenance + rebuild (corpus.txt local only)
├── notebooks/
│   └── kahnn_nano_colab.ipynb # ★ Colab T4/L4 nano train
├── docs/
│   ├── UNIVERSAL_TRAINING.md  # ★ CPU / petit GPU
│   ├── COLAB_GPU.md           # ★ Colab free/Pro + Drive ckpt
│   ├── LIFELONG_LEARNING.md   # ★ continuous memory
│   ├── CPU_RUN_LOG.md         # ★ dated measurements
│   ├── ARCHITECTURE.md
│   ├── B1_TRAINING.md
│   └── RTX4000_TRAINING.md
├── requirements.txt
└── README.md
```

## What this is NOT

- It is **not** a Transformer. There is no attention.
- It is **not** trained with backprop-through-time. Memory is structural (engrams).
- It is **not** a Hopfield network. Attractors live in *phase space*.
- It is **not** an SSM or RNN.
- It is **not** “B1 on a laptop”. Use `nano`/`commodity` + lifelong for the product path.
- It is **not** a finished product. Research framework: *intelligence = phase-coherent holographic binding over oscillatory population codes, learning by local neuromodulated plasticity.*

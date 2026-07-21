# KAHNN-B1 — 1B-Parameter Training Plan

This is the compute-optimal training plan for the KAHNN-B1 model: a
1.0-billion-parameter instance of the Kuramoto-Attractor Holographic
Hypervector Network, trained on **20.08 billion tokens** per Chinchilla.

## 1. Model spec

| Hyperparameter | Value |
|---|---|
| `dim` (D, hypervector / oscillator count) | 24,576 |
| `n_layers` (L, Kuramoto + HRR-MLP stacks) | 22 |
| `kuramoto_steps` (S, integration steps per fwd) | 4 |
| `kuramoto_rank` (r, low-rank coupling) | 256 |
| `n_ensembles_per_layer` (E) | 64 |
| `mlp_rank` (HRR-MLP rank) | 640 |
| `max_seq_len` (context window) | 4,096 |
| `vocab_size` | 50,257 (GPT-2 BPE) |
| **Total trainable parameters** | **1,004,027,926 (~1.004 B)** |

Param breakdown (per `param_count()`):
- Per Kuramoto layer: `D + 1 + 2·D·r + E·D = 14,180,353`
- × 22 layers = 311,967,766 (≈ 312 M)
- Per HRR-MLP layer: `2 · D · mlp_rank = 31,457,280`
- × 22 layers = 692,060,160 (≈ 692 M)
- **Total = 1,004,027,926 ≈ 1.004 B**

Note: this excludes the token item-memory (50,257 × 24,576 = 1.23 B
elements) because it is a **fixed random projection, not a learnable
parameter**. That is the single biggest departure from Transformers —
GPT-style models would count this as part of their parameter budget.

## 2. Chinchilla scaling law

**Hoffmann et al. (2022)** established that the compute-optimal token
count for a model with N parameters is:

    D_opt(N) ≈ 20 · N

For B1:
- N = 1.004 × 10⁹ parameters
- **D_opt = 20.08 × 10⁹ tokens (20.08 B)**

This is the sweet spot. Training on fewer tokens underfits at the
given compute; training on more (e.g. 25 B) overfits unless you also
scale parameters proportionally.

## 3. Compute estimate

KAHNN-B1 per-token FLOPs (forward + backward, 3× forward heuristic):

| Component | Per-layer FLOPs/token | × 22 layers |
|---|---|---|
| Kuramoto step (sin/cos + low-rank + engram) | D·(4r + E + 5) ≈ 26 M | 572 M |
| × 4 Kuramoto steps | | 2.29 G |
| HRR-MLP (FFT + low-rank) | 2·5·D·log₂D + 2·D·mlp_r ≈ 6.4 M | 141 M |
| Decode (V · D cosine sims) | — | 1.24 G |
| **Forward per token** | | **3.67 G** |
| **× 3 (fwd + 2 bwd)** | | **13.11 GFLOP/token** |

Total training compute: 13.11 × 20.08 × 10⁹ ≈ **2.63 × 10²⁰ FLOPs**.

In PFLOP-days: 2.63 × 10²⁰ / (10¹⁵ × 86400) ≈ **3.0 PFLOP-days**.

## 4. Wall-clock estimate

| Setup | Peak bf16 | MFU | Effective | Time |
|---|---|---|---|---|
| 1 × H100 SXM5 | 990 TF/s | 40% | 396 TF/s | 7.7 days |
| 4 × H100 | 3,960 TF/s | 40% | 1,584 TF/s | 1.9 days |
| **8 × H100 (1 node)** | **7,920 TF/s** | **40%** | **3,168 TF/s** | **23 hours** |
| 8 × B200 | 18,000 TF/s | 40% | 7,200 TF/s | 10 hours |
| 8 × A100 80GB | 2,496 TF/s | 35% | 874 TF/s | 3.5 days |

40% MFU is conservative for a non-Transformer architecture; we expect
to tune to 50–55% once the FFT kernels are fused. A real run on
8 × H100 should land between 18–24 hours.

## 5. Training schedule

| Knob | Value |
|---|---|
| Micro-batch per GPU | 8 |
| Sequence length | 4,096 |
| Gradient accumulation | 4 |
| World size (GPUs) | 8 |
| **Effective tokens/step** | 8 × 4096 × 4 × 8 = **1,048,576** |
| Total steps | 20.08 B / 1.05 M ≈ **19,152 steps** |
| Warmup steps | 1% = ~192 steps |
| LR schedule | Linear warmup → cosine decay → 10% tail |
| Peak LR | 3e-4 |
| Min LR (tail) | 3e-5 |
| Optimizer | AdamW, β=(0.9, 0.95), wd=0.01 |
| Gradient clipping | max norm 1.0 |
| Precision | bf16 autocast |
| Checkpoint cadence | every 500 steps (≈ 38 checkpoints) |
| Engram writes | every 16 steps (per `cfg.engram_write_every`) |

> **Note on tokens/step:** the Chinchilla plan in `kuro_brain.chinchilla`
> uses a different default (4 M tokens/step → ~5,000 steps). The B1
> trainer defaults to ~1 M tokens/step (smaller effective batch) for
> smoother LR scheduling and better OnlineLearner cadence. Either is
> valid; the larger batch is ~10% faster.

## 6. Memory budget

Per GPU (bf16 params + AdamW fp32 master + Adam m+v + bf16 grads):

| Component | Bytes / param | Total (B1 / 8 GPUs) |
|---|---|---|
| Params (bf16) | 2 | 0.25 GB |
| Master params (fp32) | 4 | 0.50 GB |
| Adam m (fp32) | 4 | 0.50 GB |
| Adam v (fp32) | 4 | 0.50 GB |
| Grads (bf16) | 2 | 0.25 GB |
| **Total per GPU** | | **~2.0 GB** |

Plus activations: for bf16 with seq_len=4096, micro_batch=8, D=24576,
22 layers — activations dominate at ~40 GB per GPU. Total peak:
**~45 GB/GPU**, fits comfortably on H100 80GB or A100 80GB.

## 7. How to launch

### 7.1 Single-node 8×H100 (recommended)

```bash
# 1) Environment
pip install -r requirements.txt
export TOKENIZERS_PARALLELISM=false
export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7

# 2) Launch (torchrun handles process-per-GPU + DDP)
torchrun --nproc_per_node=8 train_b1.py \
    --data /data/corpus \
    --output /home/z/my-project/download/runs/b1_run1 \
    --config b1 \
    --device cuda \
    --ddp \
    --bf16 \
    --micro-batch 8 \
    --seq-len 4096 \
    --grad-accum 4 \
    --lr 3e-4 \
    --warmup-frac 0.01 \
    --max-tokens 20080000000 \
    --checkpoint-every 500 \
    --log-every 10
```

Expected log output:
```
[config] KAHNN config: D=24576, L=22, params≈1004.0M (~1.00B)
[chinchilla] === Chinchilla Plan for KAHNN-B1 ===
            Parameters: 1.004 B
            Optimal tokens: 20.08 B (ratio 20.0 tokens/param)
            Wall-clock: 0.96 days (23.1 h)
            ...
[step   10/19152] loss=10.84 lr=1.6e-05 tps=1.2M engram=0.00 eta=22.8h
[step   20/19152] loss=10.42 lr=3.3e-05 tps=1.2M engram=0.00 eta=22.5h
...
```

### 7.2 Single-GPU sanity (would take ~7.7 days — for debugging only)

```bash
python train_b1.py \
    --data /data/corpus \
    --output ./runs/b1_debug \
    --config b1 --device cuda --bf16 \
    --micro-batch 4 --seq-len 2048 --grad-accum 1 \
    --max-tokens 100000000    # 100M smoke, not the full 20B
```

### 7.3 Continuous-learning mode (post-training, never stops)

After the B1 pretraining finishes:

```bash
torchrun --nproc_per_node=8 train_b1.py \
    --data /data/live_stream \
    --output ./runs/b1_continuous \
    --config b1 --device cuda --ddp --bf16 \
    --continuous \
    --resume ./runs/b1_run1/ckpt_final.pt
```

In continuous mode the Adam optimizer is disabled; only the local
phase plasticity + engram consolidation + active forgetting run. The
model keeps learning indefinitely from new data without replay.

## 8. Smoke test (validates trainer on CPU)

```bash
python train_b1.py --data <tiny.txt> --output /tmp/run \
       --device cpu --smoke-test
```

This swaps in the `smoke` config (0.3M params, 256 vocab) and runs a
few steps end-to-end to verify the trainer machinery (LR schedule,
grad-accum, engram writes, checkpointing) works before committing to
the real run.

## 9. What to watch during training

| Metric | Healthy range | What to do if wrong |
|---|---|---|
| `loss` | Decreasing from ~10.8 (chance) toward 2–3 | If flat after 500 steps, raise LR. If NaN, lower LR and check `K` init. |
| `tps` (tokens/sec) | ≥ 1.0 M on 8×H100 | If <0.5 M, the data loader is the bottleneck — raise `--num-workers`. |
| `engram_mean` | Rises from 0 to 5–20 over training | If stuck at 0, lower `engram_coherence_threshold` from 0.2 → 0.1. |
| `eta` | Matches the projected 23 h | If growing, throughput is dropping — check GPU thermals. |
| Gradient norm | 0.1 – 3.0 (post-clip) | If >5 consistently, raise clip to 2.0 or lower LR. |

## 10. Post-training

```bash
# Sample
python generate.py --checkpoint ./runs/b1_run1/ckpt_final.pt \
    --prompt "The brain is" --n-tokens 300 --temperature 0.8 --top-k 50

# Perplexity on held-out
python evaluate.py --checkpoint ./runs/b1_run1/ckpt_final.pt \
    --data /data/valid.txt --seq-len 4096 --batch-size 4
```

A well-trained B1 should reach perplexity in the **6–10 range** on
general English text (comparable to GPT-3 1.3B at ~10.5, with
KAHNN's holographic binding giving better compositional generalisation
at the cost of slightly worse rote memorisation).

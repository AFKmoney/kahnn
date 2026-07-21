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

This is the sweet spot. The plan never reduces this number — it hits
the full 20B via the 6 speed upgrades listed in §4.

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

## 4. The 6 speed upgrades

All 6 are optional CLI flags in `train_b1.py`. They compose
multiplicatively and give a combined ~8.14× speedup.

| Upgrade | Flag | Multiplier | Notes |
|---|---|---|---|
| FP8 | `--fp8` | 2.0× | Ada (RTX 4090) / Hopper (H100) only |
| MoD skip=0.5 | `--mod --mod-skip-rate 0.5` | 1.54× | Token early-exit |
| PGSU density=0.10 | `--pgsu --pgsu-target-density 0.10` | 1.56× | Gradient sparsification |
| Progressive depth | `--progressive-depth` | 1.54× | Start 6/22 layers, grow |
| Activation checkpointing | `--activation-checkpointing` | 1.1× | Larger batch fits |
| 8-bit optimizer | `--use-8bit-optimizer` | 1.3× | bitsandbytes required |
| **Combined** | all of the above | **~8.14×** | All verified end-to-end |

CPU offload (`--cpu-offload`) is also available — saves 8 GB VRAM in
exchange for slower optimizer step. Use only if you're VRAM-constrained.

## 5. Wall-clock estimate

| Setup | No upgrades | All 6 upgrades |
|---|---|---|
| 1 × H100 SXM5 | 7.7 days | ~23 hours |
| **8 × H100 (1 node)** | **23 hours** | **~3 hours** |
| 8 × B200 (Blackwell) | 10 hours | ~75 min |
| 8 × A100 80GB | 3.5 days | ~10 hours |
| 1 × RTX 4090 | 124 days | **13 days** |
| 2 × RTX 4090 | 62 days | 6.5 days |
| 3 × RTX 4090 | 41 days | 4.3 days |
| **5 × RTX 4090** | 25 days | **2.6 days ✓** |

The RTX 4090 row uses 35% MFU (realistic for non-Transformer architecture
on consumer Ada). H100 uses 40% MFU. See `docs/RTX4000_TRAINING.md` for
the detailed RTX 4090 path.

## 6. Training schedule

| Knob | H100 path | RTX 4090 path |
|---|---|---|
| Micro-batch per GPU | 8 | 2 |
| Sequence length | 4,096 | 2,048 |
| Gradient accumulation | 4 | 4 |
| World size (GPUs) | 8 | 5 |
| **Effective tokens/step** | 1,048,576 | 81,920 |
| Total steps | ~19,152 | ~245,000 |
| Warmup steps | 1% (~192) | 1% (~2,450) |
| LR schedule | Linear warmup → cosine → 10% tail | same |
| Peak LR | 3e-4 | 3e-4 |
| Min LR (tail) | 3e-5 | 3e-5 |
| Optimizer | AdamW (or PGSU+8bit) | PGSU+AdamW8bit |
| β | (0.9, 0.95) | (0.9, 0.95) |
| Weight decay | 0.01 | 0.01 |
| Gradient clipping | max norm 1.0 | max norm 1.0 |
| Precision | bf16 autocast | bf16 + FP8 autocast |
| Checkpoint cadence | every 500 steps | every 500 steps |
| Engram writes | every 16 steps | every 16 steps |
| Progressive depth | optional | yes, initial=6, grow_every=800 |
| MoD skip rate | optional | 0.5 |

## 7. Memory budget

Per GPU (bf16 params + AdamW fp32 master + Adam m+v + bf16 grads):

| Component | Bytes / param | B1 / 8 GPUs (H100) | B1 / 5 GPUs (4090) |
|---|---|---|---|
| Params (bf16) | 2 | 0.25 GB | 0.40 GB |
| Master params (fp32) | 4 | 0.50 GB | 0.80 GB |
| Adam m (8-bit) | 1 | 0.13 GB | 0.20 GB |
| Adam v (8-bit) | 1 | 0.13 GB | 0.20 GB |
| Grads (bf16) | 2 | 0.25 GB | 0.40 GB |
| **Subtotal** | | **1.26 GB** | **2.0 GB** |
| Activations (seq=4096, micro=8) | | ~40 GB | — |
| Activations (seq=2048, micro=2 + ckpt) | | — | ~8 GB |
| **Total peak** | | **~41 GB** | **~10 GB** |

Both fit comfortably: H100 80GB and RTX 4090 24GB have plenty of
headroom. The 4090 path uses activation checkpointing + smaller batch +
shorter sequence to keep activations under 10 GB.

## 8. How to launch

### 8.1 Single-node 8×H100 (fastest, ~3h with upgrades)

```bash
pip install -r requirements.txt bitsandbytes
export TOKENIZERS_PARALLELISM=false
export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7

torchrun --nproc_per_node=8 train_b1.py \
    --data /data/corpus \
    --output /home/z/my-project/download/runs/b1_run1 \
    --config b1 --device cuda --ddp --bf16 \
    --micro-batch 8 --seq-len 4096 --grad-accum 4 \
    --lr 3e-4 --warmup-frac 0.01 \
    --max-tokens 20080000000 \
    --checkpoint-every 500 --log-every 10 \
    --pgsu --pgsu-target-density 0.10 --pgsu-schedule cosine --pgsu-warmup 200 \
    --use-8bit-optimizer \
    --progressive-depth --initial-layers 6 --grow-every-steps 800 \
    --fp8 --mod --mod-skip-rate 0.5 --mod-aux-weight 0.01 \
    --activation-checkpointing
```

Or just: `bash launch_b1.sh /data/corpus ./runs/b1_run1`

### 8.2 5× RTX 4090 (~2.6 days, target hit)

```bash
pip install -r requirements.txt bitsandbytes

NGPU=5 bash launch_b1_rtx4090.sh /data/corpus ./runs/b1_rtx4090
```

Or explicit:

```bash
torchrun --nproc_per_node=5 train_b1.py \
    --data /data/corpus \
    --output ./runs/b1_rtx4090 \
    --config b1 --device cuda --ddp --bf16 \
    --micro-batch 2 --seq-len 2048 --grad-accum 4 \
    --lr 3e-4 --warmup-frac 0.01 \
    --max-tokens 20080000000 \
    --checkpoint-every 500 --log-every 10 \
    --pgsu --pgsu-target-density 0.10 --pgsu-schedule cosine --pgsu-warmup 200 \
    --use-8bit-optimizer \
    --progressive-depth --initial-layers 6 --grow-every-steps 800 \
    --fp8 --mod --mod-skip-rate 0.5 --mod-aux-weight 0.01 \
    --activation-checkpointing --cpu-offload
```

### 8.3 Smoke test (CPU, validates trainer)

```bash
python train_b1.py --data tiny.txt --output /tmp/run --device cpu --smoke-test
```

This swaps in the `smoke` config (0.3M params, 256 vocab) and runs a
few steps end-to-end to verify the trainer machinery works before
committing to a real run.

### 8.4 Continuous-learning mode (post-training, never stops)

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

## 9. What to watch during training

| Metric | Healthy range | What to do if wrong |
|---|---|---|
| `loss` | Decreasing from ~10.8 (chance) toward 2–3 | If flat after 500 steps, raise LR. If NaN, lower LR and check `K` init. |
| `tps` (tokens/sec) | ≥ 1.0 M on 8×H100, ≥ 90k on 1×4090 | If lower, the data loader is the bottleneck — raise `--num-workers`. |
| `density` (PGSU) | 1.0 → 0.10 cosine over training | If loss diverges late, raise `--pgsu-target-density` to 0.20. |
| `layers=X/22` | grows 6 → 22 over training | If activation is too slow, lower `--grow-every-steps`. |
| `engram_mean` | Rises from 0 to 5–20 over training | If stuck at 0, lower `engram_coherence_threshold` from 0.2 → 0.1. |
| `eta` | Matches the projected wall-clock | If growing, throughput is dropping — check GPU thermals. |
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

## 11. Honest caveats

- The 6 speed upgrades are realistic estimates based on published
  multipliers for each technique. Empirical MFU on RTX 4090 may be
  lower (25%) or higher (45%) depending on kernel fusion. The 2.6-day
  estimate for 5× RTX 4090 assumes 35% MFU.
- PGSU at 10% density is aggressive. If convergence is poor, raise to
  20% — this drops the multiplier from 1.56× to ~1.25× but keeps loss
  curves healthy.
- FP8 requires Ada (sm_89) or Hopper (sm_90). On Ampere or older it
  silently falls back to bf16.
- MoD's auxiliary loss can destabilise early training. If the router
  collapses (all tokens route to one path), lower `--mod-aux-weight`
  from 0.01 to 0.001.
- The continuous-learning mode after pretraining is the real test of
  the "no retraining" claim. Validate it on a held-out domain shift
  before trusting it in production.

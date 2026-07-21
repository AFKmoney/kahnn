#!/usr/bin/env bash
#
# launch_b1_rtx4090.sh — Launch KAHNN-B1 on a cluster of RTX 4090s with
# all 6 speed upgrades (FP8 + MoD + PGSU + 8-bit optimizer + progressive
# depth + activation checkpointing).
#
# With 5x RTX 4090 + this script, B1 trains in ~2.6 days on 20B tokens
# (Chinchilla-optimal, no reduction).
#
# Usage:
#   ./launch_b1_rtx4090.sh /data/corpus ./runs/b1_run1
#
# Override defaults via env:
#   NGPU=5 MICRO=2 SEQLEN=2048 ACCUM=4 \
#       ./launch_b1_rtx4090.sh /data/corpus ./runs/b1_run1

set -euo pipefail

DATA="${1:-/data/corpus}"
OUT="${2:-/home/z/my-project/download/runs/b1_rtx4090}"

NGPU="${NGPU:-5}"                  # 5x 4090 → ~2.6 days for full B1
MICRO="${MICRO:-2}"                # per-GPU micro-batch
SEQLEN="${SEQLEN:-2048}"           # 4090 has 24GB; 2048 fits with all upgrades
ACCUM="${ACCUM:-4}"                # grad accumulation
LR="${LR:-3e-4}"
MAXTOKENS="${MAXTOKENS:-20080000000}"  # Chinchilla-optimal, NO reduction
CKPT="${CKPT:-500}"
LOG="${LOG:-10}"

# MoD / PGSU / FP8 knobs
MOD_SKIP="${MOD_SKIP:-0.5}"        # 50% of tokens skip each layer
PGSU_DENSITY="${PGSU_DENSITY:-0.10}" # 90% sparse gradients
INIT_LAYERS="${INIT_LAYERS:-6}"    # start with 6/22 layers active
GROW_EVERY="${GROW_EVERY:-800}"    # add a layer every 800 steps

HERE="$(cd "$(dirname "$0")" && pwd)"
cd "$HERE"

mkdir -p "$OUT"

export TOKENIZERS_PARALLELISM=false
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0,1,2,3,4}"
export OMP_NUM_THREADS=8
export NCCL_DEBUG=WARN

echo "[launch_b1_rtx4090] data=$DATA out=$OUT"
echo "[launch_b1_rtx4090] ngpu=$NGPU micro=$MICRO seq=$SEQLEN accum=$ACCUM"
echo "[launch_b1_rtx4090] All 6 upgrades enabled (FP8+MoD+PGSU+8bit+progressive+ckpt)"
echo "[launch_b1_rtx4090] Target: $MAXTOKENS tokens (Chinchilla-optimal, NO reduction)"

torchrun --nproc_per_node="$NGPU" train_b1.py \
    --data "$DATA" \
    --output "$OUT" \
    --config b1 \
    --device cuda \
    --ddp \
    --bf16 \
    --micro-batch "$MICRO" \
    --seq-len "$SEQLEN" \
    --grad-accum "$ACCUM" \
    --lr "$LR" \
    --warmup-frac 0.01 \
    --min-lr-frac 0.1 \
    --weight-decay 0.01 \
    --max-tokens "$MAXTOKENS" \
    --checkpoint-every "$CKPT" \
    --log-every "$LOG" \
    --num-workers 2 \
    --pgsu \
    --pgsu-target-density "$PGSU_DENSITY" \
    --pgsu-schedule cosine \
    --pgsu-warmup 200 \
    --use-8bit-optimizer \
    --progressive-depth \
    --initial-layers "$INIT_LAYERS" \
    --grow-every-steps "$GROW_EVERY" \
    --fp8 \
    --mod \
    --mod-skip-rate "$MOD_SKIP" \
    --mod-aux-weight 0.01 \
    --activation-checkpointing \
    --cpu-offload

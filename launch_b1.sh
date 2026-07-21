#!/usr/bin/env bash
#
# launch_b1.sh — Launch the KAHNN-B1 training run on a single 8xH100 node.
#
# Usage:
#   ./launch_b1.sh /data/corpus  /home/z/my-project/download/runs/b1_run1
#
# Override defaults via env:
#   NGPU=8 MICRO=8 SEQLEN=4096 ACCUM=4 LR=3e-4 MAXTOKENS=20080000000 \
#       ./launch_b1.sh /data/corpus ./out

set -euo pipefail

DATA="${1:-/data/corpus}"
OUT="${2:-/home/z/my-project/download/runs/b1_run1}"

NGPU="${NGPU:-8}"
MICRO="${MICRO:-8}"
SEQLEN="${SEQLEN:-4096}"
ACCUM="${ACCUM:-4}"
LR="${LR:-3e-4}"
MAXTOKENS="${MAXTOKENS:-20080000000}"
CKPT="${CKPT:-500}"
LOG="${LOG:-10}"

HERE="$(cd "$(dirname "$0")" && pwd)"
cd "$HERE"

mkdir -p "$OUT"

export TOKENIZERS_PARALLELISM=false
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0,1,2,3,4,5,6,7}"
export OMP_NUM_THREADS=8
export NCCL_DEBUG=WARN

echo "[launch_b1] data=$DATA out=$OUT"
echo "[launch_b1] ngpu=$NGPU micro=$MICRO seq=$SEQLEN accum=$ACCUM lr=$LR maxtok=$MAXTOKENS"

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
    --num-workers 2

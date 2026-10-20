#!/usr/bin/env bash
# Phase-2 scaling test: same data, tokens, batch, LR schedule for 3 sizes.
set -u
cd "$(dirname "$0")/.."
BUDGET=${BUDGET:-1000000}
for c in nano commodity medium; do
  python3 train_universal.py --data data/tok/train.bin --val-data data/tok/val.bin \
    --output runs/scaling/$c --config $c --device cpu \
    --micro-batch 1 --seq-len 256 --grad-accum 8 --lr 3e-3 \
    --max-tokens $BUDGET --log-every 10 --eval-every 50 --eval-batches 32 \
    --activation-checkpointing --checkpoint-every 100000 --seed 1337 \
    > runs/scaling/$c.stdout 2>&1
done
python3 scripts/plot_scaling.py

#!/bin/bash
# SYMSOL I baseline retrain WITH held-out val-selection (best.pth by val LL). Same config as the lost prior run
# (L5, bs64, lr 1e-4 head / 1e-5 backbone, OneCycle 5% warmup, 60k steps, fp32, clip 1.0) -- the only change is that
# overfitting is now observable/selectable, to isolate how much early-stopping alone recovers tet/cube/icosa.
# Shares the local GPU with resident jobs: grad-checkpointed, device 0, host RAM capped. Ends with test eval.
set -euo pipefail
GPY=/home/kavinder/miniconda3/envs/jepa/bin/python
DATA=${DATA:-/home/kavinder/SPACE-HOP/data/symsol}
OUT=${OUT:-/home/kavinder/SPACE-HOP/checkpoints/symsol_L5_base}
BS=${BS:-64}; STEPS=${STEPS:-60000}; LEVEL=${LEVEL:-5}; WORKERS=${WORKERS:-3}
cd /home/kavinder/SPACE-HOP
export CUDA_VISIBLE_DEVICES=${GPU:-0} PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
       OMP_NUM_THREADS=4 PYTHONUNBUFFERED=1
mkdir -p "$OUT"
RESUME=""; [ -f "$OUT/last.pth" ] && RESUME="--resume $OUT/last.pth"   # crash-safe: pick up where we left off
echo "=== train $(date) | bs $BS steps $STEPS L$LEVEL gpu $CUDA_VISIBLE_DEVICES ${RESUME} ==="
$GPY src/symsol_train.py train --data "$DATA" --out "$OUT" --level "$LEVEL" \
     --bs "$BS" --steps "$STEPS" --lr 1e-4 --lr_bb 1e-5 --workers "$WORKERS" \
     --eval_every 2500 --save_every 2500 --val_frac 0.04 --seed 0 --beam 8 --grad_ckpt $RESUME
echo "=== TRAIN DONE $(date) ==="
echo "=== decompose on best.pth (per-level coverage diagnostic) $(date) ==="
$GPY src/symsol_train.py decompose --data "$DATA" --ckpt "$OUT/best.pth" --level "$LEVEL" --n 500

#!/bin/bash
# Serial SYMSOL-II per-shape training -- the remaining main SO(3) comparison (SOTA target: SpyroPose avg 9.27).
# Protocol matches the SYMSOL-I headline: ViT-B/16, L6, 60k steps, bs 64, exact in-plane roll augmentation,
# single GPU (device 0). The three marked shapes run one after another: there is one physical GPU, so running
# them concurrently only time-slices the same compute (no wall-clock gain) while raising memory/thrash risk.
set -u
cd /home/kavinder/SPACE-HOP
PY=/home/kavinder/miniconda3/envs/jepa/bin/python
export CUDA_VISIBLE_DEVICES=0
for s in tetX cylO sphereX; do
  echo "=== $(date '+%F %T') START symsol_II_$s ==="
  "$PY" src/symsol_train.py train \
    --data "data/symsol_II_$s" --out "checkpoints/symsol_II_$s" \
    --level 6 --steps 60000 --bs 64 --workers 6 --save_every 5000 --augment \
    >> "data/symsol_II_${s}_train.log" 2>&1
  rc=$?
  echo "=== $(date '+%F %T') DONE symsol_II_$s rc=$rc ==="
done
echo "=== $(date '+%F %T') SYMSOL_II_ALL_DONE ==="

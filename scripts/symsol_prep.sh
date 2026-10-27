#!/bin/bash
# Regenerate SYMSOL I locally (CPU only -- no GPU contention with the resident cloud jobs):
#   dump    (tf env)   tfds symmetric_solids -> PNG blobs + rotations, 5 unmarked shapes.
#   prepare (jepa env) per-path tree geometry at L=5 (healpy+scipy).
#   ceiling (jepa env) analytic discretisation ceiling L3-6 -> fills paper tab:ceiling.
set -euo pipefail
DATA=${DATA:-/home/kavinder/SPACE-HOP/data/symsol}
TFPY=/home/kavinder/miniconda3/bin/python                       # conda base: tensorflow 2.21 + tfds
GPY=/home/kavinder/miniconda3/envs/jepa/bin/python              # healpy+scipy+torch+timm
cd /home/kavinder/SPACE-HOP
mkdir -p "$DATA"
export TFDS_DATA_DIR=/home/kavinder/tensorflow_datasets CUDA_VISIBLE_DEVICES="" OMP_NUM_THREADS=4
echo "=== dump $(date) ==="
$TFPY src/symsol_data.py dump --out "$DATA"
echo "=== prepare L5 $(date) ==="
$GPY src/symsol_data.py prepare --out "$DATA" --level 5
echo "=== ceiling $(date) ==="
$GPY src/symsol_train.py ceiling --data "$DATA" --levels 3,4,5,6
echo "=== PREP DONE $(date) ==="

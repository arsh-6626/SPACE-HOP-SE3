#!/bin/bash
# SYMSOL-II re-dump (CPU only, no GPU contention): the raw tfds already has the 3 marked shapes.
# One dir per marked shape (prior-work protocol = one model per marked shape). dump(base tf env) + prepare(jepa, L6).
set -euo pipefail
TFPY=/home/kavinder/miniconda3/bin/python          # conda base: tensorflow 2.21 + tfds
GPY=/home/kavinder/miniconda3/envs/jepa/bin/python # healpy+scipy+torch
cd /home/kavinder/SPACE-HOP
export TFDS_DATA_DIR=/home/kavinder/tensorflow_datasets CUDA_VISIBLE_DEVICES="" OMP_NUM_THREADS=4 PYTHONUNBUFFERED=1
for SH in tetX cylO sphereX; do
  OUT=/home/kavinder/SPACE-HOP/data/symsol_II_$SH
  mkdir -p "$OUT"
  echo "=== dump $SH $(date) ==="
  $TFPY src/symsol_data.py dump --out "$OUT" --shapes "$SH"
  echo "=== prepare $SH L6 $(date) ==="
  $GPY src/symsol_data.py prepare --out "$OUT" --level 6
done
echo "=== SYMSOL-II REDUMP DONE $(date) ==="
du -sh /home/kavinder/SPACE-HOP/data/symsol_II_* 2>/dev/null

#!/bin/bash
# Local YCB-V SITE fine-tune fan-out on kavi device 0. Serial over objects that have an _aug base ckpt.
# Warm-start _aug obj -> 8k steps on train_pbr + train_real(REAL) with bbox t_hat anchor. Idempotent (skips COMPLETE).
# Skips obj2,obj8 (no _aug base) and obj17 (already done as canary). Indexes train_real per object if missing.
set -u
cd /home/kavinder/SPACE-HOP
PY=~/miniconda3/envs/jepa/bin/python
YC=/media/kavinder/hdd2/liepose_data/bop_datasets/ycbv
CACHE=cache/ycbv_se3
OBJS="${OBJS:-1 3 4 5 6 7 9 10 11 12 13 14 15 16 18 19 20 21}"
STEPS="${STEPS:-8000}"
export PYTHONPATH=src CUDA_VISIBLE_DEVICES=0 OMP_NUM_THREADS=6 MKL_NUM_THREADS=6 PYTHONUNBUFFERED=1
MLOG=logs/ycbv_site_runner.log; mkdir -p logs
log(){ echo "$(date '+%F %T') $*" | tee -a "$MLOG"; }
log "=== YCB-V SITE fan-out start on device 0; objs: $OBJS ==="
for O in $OBJS; do
  RUN=checkpoints/se3_ycbv_site_obj$O; mkdir -p "$RUN"; LOG=$RUN/train_local.log
  INIT=checkpoints/se3_ycbv_aug_obj$O/last.pth
  [ -f "$RUN/COMPLETE" ] && { log "obj $O COMPLETE -> skip"; continue; }
  [ -f "$INIT" ] || { log "obj $O no _aug base ($INIT) -> skip"; continue; }
  if [ ! -f "$CACHE/train_real_obj$O.npz" ]; then
    log "obj $O indexing train_real"
    $PY src/se3_tless_data.py index --root $YC --out $CACHE --objs $O --split train_real >> "$LOG" 2>&1
  fi
  log "=== obj $O SITE fine-tune start -> $RUN ==="
  $PY src/se3_tless_train.py train --root $YC --cache $CACHE --obj $O \
    --out "$RUN" --test_split test \
    --init "$INIT" --t_hat_train bbox --real_split train_real \
    --test_that bbox --bbox_sj 0.08 --bbox_cj 0.06 \
    --L_rot 5 --L_trans 5 --steps $STEPS --bs 48 --workers 12 \
    --save_every 2000 --eval_every 1000 --lr 1e-4 --lr_bb 1e-5 >> "$LOG" 2>&1
  rc=$?
  if [ $rc -eq 0 ]; then touch "$RUN/COMPLETE"; log "=== obj $O DONE rc=0 ==="
  else log "=== obj $O EXIT rc=$rc ==="; fi
done
log "=== YCB-V SITE fan-out: all objects processed ==="
touch checkpoints/.ycbv_site_fanout_DONE

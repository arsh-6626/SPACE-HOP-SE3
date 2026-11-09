#!/usr/bin/env bash
# Auto-queue T-LESS SE(3) per-object training on device 0 after SYMSOL-T finishes.
# Resume-aware + reboot-robust: re-running this script continues each object from
# its last.pth and skips objects already marked COMPLETE. Does NOT evict anything;
# it only starts once the SYMSOL-T trainer PID has exited AND completed cleanly.
#
# Usage: scripts/tless_autoqueue.sh [SYMSOL_PID] [SYMSOL_LOG]
set -u
cd "$(dirname "$0")/.."                      # -> SPACE-HOP root
ROOT=/media/kavinder/hdd2/DP/tless
CACHE=/media/kavinder/hdd3/spacehop_tless_cache
JEPA=/home/kavinder/miniconda3/envs/jepa/bin/python
OBJS="1 14 25 27"
SYMSOL_PID="${1:-102295}"
SYMSOL_LOG="${2:-data/se3_train_L5.log}"
mkdir -p logs
echo "[autoqueue $(date '+%F %T')] waiting for SYMSOL-T PID $SYMSOL_PID to exit"

# 1) Wait for SYMSOL-T to release device 0.
while kill -0 "$SYMSOL_PID" 2>/dev/null; do sleep 60; done
echo "[autoqueue $(date '+%F %T')] SYMSOL-T PID gone"

# 2) Only proceed if SYMSOL-T actually finished (reached final eval / step 60000).
#    If it died early, leave device 0 and the decision alone.
if ! grep -qE "final eval \(test, best.pth\)|step 60000/60000" "$SYMSOL_LOG" 2>/dev/null; then
  echo "[autoqueue $(date '+%F %T')] SYMSOL-T did NOT complete cleanly (no final-eval marker in $SYMSOL_LOG)."
  echo "[autoqueue] NOT auto-starting T-LESS. Re-run this script once SYMSOL-T is sorted."
  exit 2
fi
echo "[autoqueue $(date '+%F %T')] SYMSOL-T completed cleanly -> starting T-LESS queue"

# 3) Train each object sequentially; resume if interrupted, skip if COMPLETE.
for OBJ in $OBJS; do
  OUT="checkpoints/se3_tless_obj${OBJ}"
  LOG="logs/tless_obj${OBJ}.log"
  if [ -f "$OUT/COMPLETE" ]; then
    echo "[autoqueue $(date '+%F %T')] obj $OBJ already COMPLETE -> skip"
    continue
  fi
  RESUME=""
  if [ -f "$OUT/last.pth" ]; then
    RESUME="--resume $OUT/last.pth"
    echo "[autoqueue $(date '+%F %T')] obj $OBJ resuming from $OUT/last.pth"
  fi
  nvidia-smi --query-gpu=memory.free,memory.used --format=csv,noheader -i 0 2>/dev/null \
    | sed "s/^/[autoqueue obj $OBJ VRAM dev0] /"
  echo "[autoqueue $(date '+%F %T')] obj $OBJ training -> $LOG"
  env CUDA_VISIBLE_DEVICES=0 OMP_NUM_THREADS=4 "$JEPA" src/se3_tless_train.py train \
      --root "$ROOT" --cache "$CACHE" --obj "$OBJ" \
      --out "$OUT" --steps 40000 --bs 48 --workers 4 \
      --save_every 2500 --eval_every 2000 --test_that gt \
      $RESUME > "$LOG" 2>&1
  rc=$?
  if [ $rc -eq 0 ]; then
    touch "$OUT/COMPLETE"
    echo "[autoqueue $(date '+%F %T')] obj $OBJ DONE (rc=0) -> marked COMPLETE"
  else
    echo "[autoqueue $(date '+%F %T')] obj $OBJ FAILED rc=$rc (see $LOG) -> continuing to next object"
  fi
done
echo "[autoqueue $(date '+%F %T')] === T-LESS QUEUE FINISHED ==="
for OBJ in $OBJS; do
  if [ -f "checkpoints/se3_tless_obj${OBJ}/COMPLETE" ]; then s="COMPLETE"; else s="INCOMPLETE"; fi
  echo "[autoqueue] obj $OBJ: $s"
done

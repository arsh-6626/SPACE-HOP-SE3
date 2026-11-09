#!/bin/bash
# Local serial YCB-V NeST SE(3) trainer (device 0 = RTX 6000 Ada, 48GB) -- HELPER in a 2-pool YCB fan-out.
# dtu004 HPC trains YCB ASCENDING (1->21) via the topper; this local runner trains DESCENDING (21->1),
# skipping any object dtu004 has marked COMPLETE. The two streams converge in the middle: self-balancing,
# no double-training. Reads BOP YCB-V read-only from hdd2; writes caches+checkpoints under ~/SPACE-HOP.
# Index caches prebuilt. Waits for L7 (device-0 occupant) to exit + GPU to free before starting.
set -u
cd ~/SPACE-HOP
PY=~/miniconda3/envs/jepa/bin/python
YC=/media/kavinder/hdd2/liepose_data/bop_datasets/ycbv
CACHE=cache/ycbv_se3
TAG=_aug
L7_PID="${L7_PID:-356218}"
OBJS="${OBJS:-21 20 19 18 17 16 15 14 13 12 11 10 9 8 7 6 5 4 3 2 1}"   # DESCENDING
STEPS="${STEPS:-40000}"
DTU004_W=/scratch/hpc/visitor/dtu004.visitor/spacehop/checkpoints
export PYTHONPATH=src CUDA_VISIBLE_DEVICES=0
export OMP_NUM_THREADS=6 MKL_NUM_THREADS=6 PYTHONUNBUFFERED=1
mkdir -p logs
MLOG=logs/local_ycbv_run.log
log(){ echo "$(date '+%F %T') $*" | tee -a "$MLOG"; }

# best-effort: should kavi SKIP obj O because dtu004 already has it covered?
# yes only if dtu004 marked it COMPLETE, or has a RUNNING (state R) ycba_o$O job. A merely QUEUED dtu004
# job is NOT a skip (dtu004's queue may be stuck at 0 running -> kavi must still plow through). Silent on error.
dtu004_complete(){
  local O=$1
  timeout 50 ssh -o ConnectTimeout=15 yashdeep \
    "ssh -o ConnectTimeout=20 hpc-login 'f=$DTU004_W/se3_ycbv_aug_obj$O/COMPLETE; test -f \$f && { echo yes; exit; }; qstat -u dtu004.visitor 2>/dev/null | awk \"\\\$4==\\\"ycba_o$O\\\" && \\\$10==\\\"R\\\"{print \\\"yes\\\"}\"'" 2>/dev/null | grep -m1 yes
}

log "=== kavi-local YCB helper start (descending, dtu004-skip); waiting for L7 (PID $L7_PID) ==="
while kill -0 "$L7_PID" 2>/dev/null; do sleep 60; done
log "L7 exited; waiting for device-0 free mem >40GB"
while :; do
  FREE=$(nvidia-smi --query-gpu=memory.free --format=csv,noheader,nounits -i 0 | head -1)
  [ "${FREE:-0}" -gt 40000 ] && break; sleep 30
done
log "device 0 free=${FREE}MiB -> start YCB (steps=$STEPS, order: $OBJS)"

for O in $OBJS; do
  RUN=checkpoints/se3_ycbv${TAG}_obj${O}; mkdir -p "$RUN"; LOG=$RUN/train_local.log
  [ -f "$RUN/COMPLETE" ] && { log "obj $O local COMPLETE -> skip"; continue; }
  if [ "$(dtu004_complete $O)" = "yes" ]; then log "obj $O COMPLETE on dtu004 -> skip (HPC did it)"; continue; fi
  touch "$RUN/CLAIMED"   # mutual-exclusion: dtu004 topper reads local CLAIMED/COMPLETE and will not submit (qdels) this obj
  RESUME=""; [ -f "$RUN/last.pth" ] && RESUME="--resume $RUN/last.pth" && log "obj $O resume from last.pth"
  log "=== obj $O training start (CLAIMED) -> $RUN ==="
  $PY src/se3_tless_train.py train \
      --root "$YC" --cache "$CACHE" --obj "$O" --out "$RUN" --test_split test --augment \
      --L_rot 5 --L_trans 5 --steps "$STEPS" --bs 48 --workers 12 \
      --save_every 2500 --eval_every 2000 --test_that gt $RESUME >> "$LOG" 2>&1
  rc=$?
  if [ $rc -eq 0 ]; then touch "$RUN/COMPLETE"; log "=== obj $O DONE rc=0 ==="
  else log "=== obj $O EXIT rc=$rc (resume next run) ==="; fi
done
log "=== kavi-local YCB helper: all objects processed ==="

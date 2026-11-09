#!/usr/bin/env bash
# Two-phase top-up loop for NeST SE(3) jobs on IIT-D HPC (scai_q, per-user limit 2).
# Phase 1: T-LESS objects 1/14/25/27. Phase 2 (only after ALL T-LESS are COMPLETE): YCB-V objects 1..21.
# Keeps up to LIMIT jobs queued/running, resumes any that exit before writing COMPLETE (the PBS resumes
# from last.pth), stops when everything is done. Reconciles from live qstat + COMPLETE markers, so it is
# safe to restart and never double-submits. Touches no non-SE(3) job.
set -u
# T-LESS all 30 objects (BOP AR is dataset-level over all 30). 1/14/25/27 (SpyroPose subset) first — already in flight.
TLESS="1 14 25 27 2 3 4 5 6 7 8 9 10 11 12 13 15 16 17 18 19 20 21 22 23 24 26 28 29 30"
YCBV="1 2 3 4 5 6 7 8 9 10 11 12 13 14 15 16 17 18 19 20 21"
LIMIT=2
SLEEP=${SLEEP:-180}
R=/scratch/hpc/visitor/dtu001.visitor/SPACE-HOP
HOST=iitd-hpc08
LOG=/home/kavinder/SPACE-HOP/logs/se3_hpc_queue_loop.log
mkdir -p "$(dirname "$LOG")"

rsh() { local b; b=$(printf '%s' "$1" | base64 -w0)
  timeout 120 ssh -o BatchMode=yes -o ConnectTimeout=30 "$HOST" "echo $b | base64 -d | bash" 2>/dev/null; }

in_list() { local x="$1"; shift; for e in "$@"; do [ "$e" = "$x" ] && return 0; done; return 1; }

echo "[$(date '+%F %T')] se3 queue-loop start (tless then ycbv, limit $LIMIT)" | tee -a "$LOG"
while true; do
  Q=$(rsh '
R='"$R"'
names=$(qstat -u dtu001.visitor 2>/dev/null | awk "{print \$4}")
echo "NJOBS=$(echo "$names" | grep -cE "^(tless_se3|tless_o[0-9]+|ycbv_o[0-9]+)$")"
act=""
echo "$names" | grep -q "^tless_se3$" && act="$act tless:1"
for n in $(echo "$names" | grep -oE "^tless_o[0-9]+$"); do act="$act tless:${n#tless_o}"; done
for n in $(echo "$names" | grep -oE "^ycbv_o[0-9]+$");  do act="$act ycbv:${n#ycbv_o}"; done
echo "ACT=$act"
td=""; for O in $(seq 1 30); do [ -f $R/checkpoints/se3_tless_obj$O/COMPLETE ] && td="$td $O"; done; echo "TDONE=$td"
yd=""; for O in $(seq 1 21); do [ -f $R/checkpoints/se3_ycbv_obj$O/COMPLETE ] && yd="$yd $O"; done; echo "YDONE=$yd"
')
  NJOBS=$(echo "$Q" | sed -n 's/^NJOBS=//p'); NJOBS=${NJOBS:-0}
  ACT=$(echo "$Q" | sed -n 's/^ACT=//p')
  TDONE=($(echo "$Q" | sed -n 's/^TDONE=//p'))
  YDONE=($(echo "$Q" | sed -n 's/^YDONE=//p'))

  tless_done=1; for O in $TLESS; do in_list "$O" "${TDONE[@]}" || tless_done=0; done
  ycbv_done=1;  for O in $YCBV;       do in_list "$O" "${YDONE[@]}" || ycbv_done=0; done
  if [ "$tless_done" = 1 ] && [ "$ycbv_done" = 1 ]; then
    echo "[$(date '+%F %T')] ALL T-LESS + YCB-V COMPLETE -> loop done" | tee -a "$LOG"; break; fi

  slots=$(( LIMIT - NJOBS )); [ "$slots" -lt 0 ] && slots=0
  phase=$([ "$tless_done" = 1 ] && echo ycbv || echo tless)
  echo "[$(date '+%F %T')] phase=$phase njobs=$NJOBS slots=$slots act=[$ACT] tdone=[${TDONE[*]-}] ydone=[${YDONE[*]-}]" | tee -a "$LOG"

  if [ "$tless_done" != 1 ]; then
    for O in $TLESS; do
      [ "$slots" -gt 0 ] || break
      in_list "$O" "${TDONE[@]}" && continue
      [[ " $ACT " == *" tless:$O "* ]] && continue
      jid=$(rsh "cd $R/scripts && qsub -N tless_o$O -v OBJ=$O hpc_tless_se3.pbs")
      echo "[$(date '+%F %T')] submitted T-LESS obj $O -> $jid" | tee -a "$LOG"
      slots=$(( slots - 1 )); ACT="$ACT tless:$O"
    done
  else
    for O in $YCBV; do
      [ "$slots" -gt 0 ] || break
      in_list "$O" "${YDONE[@]}" && continue
      [[ " $ACT " == *" ycbv:$O "* ]] && continue
      jid=$(rsh "cd $R/scripts && qsub -N ycbv_o$O -v OBJ=$O hpc_ycbv_se3.pbs")
      echo "[$(date '+%F %T')] submitted YCB-V obj $O -> $jid" | tee -a "$LOG"
      slots=$(( slots - 1 )); ACT="$ACT ycbv:$O"
    done
  fi
  sleep "$SLEEP"
done

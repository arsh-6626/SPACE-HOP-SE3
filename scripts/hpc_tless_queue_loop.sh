#!/usr/bin/env bash
# Local top-up loop for the NeST T-LESS SE(3) jobs on IIT-D HPC (scai_q, per-user limit 2).
# Keeps up to LIMIT jobs queued/running across objects 1/14/25/27, resumes any that exit
# before writing a COMPLETE marker (the PBS itself resumes from last.pth), stops when all done.
# Does NOT touch any non-tless job. Safe to re-run; it reconciles from live qstat + COMPLETE markers.
set -u
ALL="14 25 27 1"            # priority order; obj1 already running under the default name
LIMIT=2
SLEEP=${SLEEP:-180}
R=/scratch/hpc/visitor/dtu001.visitor/SPACE-HOP
HOST=iitd-hpc08
LOG=/home/kavinder/SPACE-HOP/logs/tless_hpc_queue_loop.log
mkdir -p "$(dirname "$LOG")"

rsh() {  # run $1 on the cluster login node, base64 to avoid nested-quote hell
  local b; b=$(printf '%s' "$1" | base64 -w0)
  timeout 120 ssh -o BatchMode=yes -o ConnectTimeout=30 "$HOST" "echo $b | base64 -d | bash" 2>/dev/null
}

echo "[$(date '+%F %T')] queue-loop start (objs: $ALL, limit $LIMIT)" | tee -a "$LOG"
while true; do
  # One round-trip: report NJOBS, the active object set, and the done set.
  Q=$(rsh '
R='"$R"'
names=$(qstat -u dtu001.visitor 2>/dev/null | awk "{print \$4}")
nj=$(echo "$names" | grep -cE "^tless_(se3|o[0-9]+)$")
act=""
echo "$names" | grep -q "^tless_se3$" && act="$act 1"
for n in $(echo "$names" | grep -oE "^tless_o[0-9]+$"); do act="$act ${n#tless_o}"; done
done=""
for O in 1 14 25 27; do [ -f $R/checkpoints/se3_tless_obj$O/COMPLETE ] && done="$done $O"; done
echo "NJOBS=$nj"; echo "ACTIVE=$act"; echo "DONE=$done"
')
  NJOBS=$(echo "$Q" | sed -n 's/^NJOBS=//p'); NJOBS=${NJOBS:-0}
  ACTIVE=" $(echo "$Q" | sed -n 's/^ACTIVE=//p') "
  DONE=" $(echo "$Q" | sed -n 's/^DONE=//p') "

  # all four done?
  alldone=1; for O in 1 14 25 27; do [[ "$DONE" == *" $O "* ]] || alldone=0; done
  if [ "$alldone" = 1 ]; then echo "[$(date '+%F %T')] all objects COMPLETE -> loop done" | tee -a "$LOG"; break; fi

  slots=$(( LIMIT - NJOBS )); [ "$slots" -lt 0 ] && slots=0
  echo "[$(date '+%F %T')] njobs=$NJOBS slots=$slots active=[${ACTIVE# }] done=[${DONE# }]" | tee -a "$LOG"
  for O in $ALL; do
    [ "$slots" -gt 0 ] || break
    [[ "$DONE"   == *" $O "* ]] && continue
    [[ "$ACTIVE" == *" $O "* ]] && continue
    jid=$(rsh "cd $R/scripts && qsub -N tless_o$O -v OBJ=$O hpc_tless_se3.pbs")
    echo "[$(date '+%F %T')] submitted obj $O -> $jid" | tee -a "$LOG"
    slots=$(( slots - 1 ))
    ACTIVE="$ACTIVE $O "
  done
  sleep "$SLEEP"
done

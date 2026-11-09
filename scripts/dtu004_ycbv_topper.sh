#!/bin/bash
# dtu004 YCB-V dual-queue fan-out topper (runs locally on kavi; 2-hop ssh yashdeep -> hpc-login per cycle).
# This PBS forbids node-pinning, so scai jobs are submitted UNPINNED; the PBS parks a faulted node (scai04)
# for ~25min on CUDA-guard failure so the next unpinned submit lands on a healthy node. Each cycle (ONE ssh):
#   * keep up to SCAP(=2) YCB jobs on scai_q and up to TCAP(=16) active overall, ASCENDING over
#     incomplete/inactive/UNCLAIMED objects; prefer scai_q (runs sooner), else standard/icelake;
#   * MIGRATE a standard-queued obj onto scai (unpinned) whenever scai has room;
#   * kavi mutual-exclusion: read kavi-local CLAIMED/COMPLETE, never submit those, qdel ones kavi took.
# Dead/partial jobs resume via last.pth. Pairs with kavi-local helper (descending). 10-min cycle.
set -u
SCAP="${SCAP:-2}"; TCAP="${TCAP:-16}"; PERIOD="${PERIOD:-600}"
P=/scratch/hpc/visitor/dtu001.visitor/SPACE-HOP
W=/scratch/hpc/visitor/dtu004.visitor/spacehop
LCK=~/SPACE-HOP/checkpoints
LOG=~/SPACE-HOP/logs/dtu004_ycbv_topper.log
mkdir -p ~/SPACE-HOP/logs
log(){ echo "$(date '+%F %T') $*" >> "$LOG"; }

cycle(){
  local EXCL="" KDONE=""
  for O in $(seq 1 21); do
    { [ -f "$LCK/se3_ycbv_aug_obj$O/CLAIMED" ] || [ -f "$LCK/se3_ycbv_aug_obj$O/COMPLETE" ]; } && EXCL="$EXCL $O"
    [ -f "$LCK/se3_ycbv_aug_obj$O/COMPLETE" ] && KDONE="$KDONE $O"
  done
  local REMOTE
  REMOTE=$(cat <<RS
set -u
P=$P; W=$W; SCAP=$SCAP; TCAP=$TCAP; EXCL="$EXCL"
STDSEL="select=1:ncpus=16:ngpus=1:mem=64gb:centos=icelake"
act=\$(qstat -u dtu004.visitor 2>/dev/null | awk '\$1~/pbshpc/ && \$4 ~ /^ycba_o/ {print \$4" "\$3" "\$10}')
sact=\$(printf '%s\n' "\$act" | awk '\$2=="scai_q"{c++}END{print c+0}')
tact=\$(printf '%s\n' "\$act" | grep -c 'ycba_o' || true)
for X in \$EXCL; do jid=\$(qstat -u dtu004.visitor 2>/dev/null | awk -v n=ycba_o\$X '\$4==n{print \$1}'); [ -n "\$jid" ] && { qdel \$jid 2>/dev/null; tact=\$((tact-1)); }; done
done_list=""; sub_list=""
for O in \$(seq 1 21); do
  d=\$W/checkpoints/se3_ycbv_aug_obj\$O
  [ -f \$d/COMPLETE ] && { done_list="\$done_list \$O"; continue; }
  printf '%s\n' \$EXCL | grep -qx \$O && continue
  printf '%s\n' "\$act" | awk '{print \$1}' | grep -qx ycba_o\$O && continue
  [ \$tact -ge \$TCAP ] && continue
  if [ \$sact -lt \$SCAP ]; then
    ( cd \$P && qsub -q scai_q -l select=1:ncpus=16:ngpus=1:mem=64gb -N ycba_o\$O -v OBJ=\$O scripts/hpc_ycbv_se3_dtu004.pbs >/dev/null 2>&1 ) && { sub_list="\$sub_list \$O@scai"; sact=\$((sact+1)); tact=\$((tact+1)); }
  else
    ( cd \$P && qsub -q standard -l \$STDSEL -N ycba_o\$O -v OBJ=\$O scripts/hpc_ycbv_se3_dtu004.pbs >/dev/null 2>&1 ) && { sub_list="\$sub_list \$O@std"; tact=\$((tact+1)); }
  fi
done
# migrate standard-Q -> scai (unpinned) while scai has room
while [ \$sact -lt \$SCAP ]; do
  mo=\$(printf '%s\n' "\$act" | awk '\$2=="standard" && \$3=="Q"{print \$1; exit}' | sed 's/ycba_o//')
  [ -z "\$mo" ] && break
  jid=\$(qstat -u dtu004.visitor 2>/dev/null | awk -v n=ycba_o\$mo '\$4==n{print \$1}')
  qdel \$jid 2>/dev/null
  ( cd \$P && qsub -q scai_q -l select=1:ncpus=16:ngpus=1:mem=64gb -N ycba_o\$mo -v OBJ=\$mo scripts/hpc_ycbv_se3_dtu004.pbs >/dev/null 2>&1 ) && { sub_list="\$sub_list MIG\$mo@scai"; sact=\$((sact+1)); }
  act=\$(printf '%s\n' "\$act" | grep -v "^ycba_o\$mo ")
done
echo "SACT=\$sact TACT=\$tact DTU004_DONE=[\$done_list ] SUBMITTED=[\$sub_list ]"
RS
)
  local OUT; OUT=$(timeout 90 ssh -o ConnectTimeout=15 yashdeep "ssh -o ConnectTimeout=20 hpc-login 'bash -s'" <<<"$REMOTE" 2>&1 | tail -2)
  log "EXCL=[$EXCL ] KDONE=[$KDONE ] | $OUT"
  local d4 all
  d4=$(echo "$OUT" | sed -n 's/.*DTU004_DONE=\[\(.*\) \] SUBMITTED.*/\1/p')
  all=$(printf '%s\n' $d4 $KDONE | sort -n | uniq | grep -c .)
  [ "${all:-0}" -ge 21 ] && return 9
  return 0
}

log "=== dtu004 YCB dual-queue topper (unpinned+park) start SCAP=$SCAP TCAP=$TCAP period=${PERIOD}s ==="
while :; do
  cycle; [ $? -eq 9 ] && { log "=== all 21 YCB covered (dtu004 ∪ kavi) -> exit ==="; break; }
  sleep "$PERIOD"
done

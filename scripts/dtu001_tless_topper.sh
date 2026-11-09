#!/bin/bash
# dtu001 T-LESS dual-queue fan-out topper (runs locally; direct ssh iitd-hpc, 1-hop; auth can be flaky -> retries).
# scai jobs submitted UNPINNED (node-pinning is forbidden by this PBS); the PBS parks a faulted node ~25min on
# CUDA-guard failure so the next unpinned submit lands healthy. Each cycle (ONE ssh): keep up to SCAP(=2)
# T-LESS jobs on scai_q and up to TCAP(=16) active overall over incomplete/inactive objects 1..30; prefer scai_q
# else standard/icelake; MIGRATE a standard-Q obj onto scai when room; resubmit dead/partial (resume last.pth).
set -u
SCAP="${SCAP:-2}"; TCAP="${TCAP:-16}"; PERIOD="${PERIOD:-720}"; DELAY="${DELAY:-20}"
P=/scratch/hpc/visitor/dtu001.visitor/SPACE-HOP
LOG=~/SPACE-HOP/logs/dtu001_tless_topper.log
mkdir -p ~/SPACE-HOP/logs
log(){ echo "$(date '+%F %T') $*" >> "$LOG"; }

REMOTE=$(cat <<RS
set -u
P=$P; SCAP=$SCAP; TCAP=$TCAP
STDSEL="select=1:ncpus=16:ngpus=1:mem=64gb:centos=icelake"
act=\$(qstat -u dtu001.visitor 2>/dev/null | awk '\$1~/pbshpc/ && \$4 ~ /^tlsa_o/ {print \$4" "\$3" "\$10}')
sact=\$(printf '%s\n' "\$act" | awk '\$2=="scai_q"{c++}END{print c+0}')
tact=\$(printf '%s\n' "\$act" | grep -c 'tlsa_o' || true)
done_list=""; sub_list=""
for O in \$(seq 1 30); do
  d=\$P/checkpoints/se3_tless_aug_obj\$O
  [ -f \$d/COMPLETE ] && { done_list="\$done_list \$O"; continue; }
  printf '%s\n' "\$act" | awk '{print \$1}' | grep -qx tlsa_o\$O && continue
  [ \$tact -ge \$TCAP ] && continue
  if [ \$sact -lt \$SCAP ]; then
    ( cd \$P && qsub -q scai_q -l select=1:ncpus=16:ngpus=1:mem=64gb -N tlsa_o\$O -v OBJ=\$O,TAG=_aug,AUG=1 scripts/hpc_tless_se3.pbs >/dev/null 2>&1 ) && { sub_list="\$sub_list \$O@scai"; sact=\$((sact+1)); tact=\$((tact+1)); }
  else
    ( cd \$P && qsub -q standard -l \$STDSEL -N tlsa_o\$O -v OBJ=\$O,TAG=_aug,AUG=1 scripts/hpc_tless_se3.pbs >/dev/null 2>&1 ) && { sub_list="\$sub_list \$O@std"; tact=\$((tact+1)); }
  fi
done
while [ \$sact -lt \$SCAP ]; do
  mo=\$(printf '%s\n' "\$act" | awk '\$2=="standard" && \$3=="Q"{print \$1; exit}' | sed 's/tlsa_o//')
  [ -z "\$mo" ] && break
  jid=\$(qstat -u dtu001.visitor 2>/dev/null | awk -v n=tlsa_o\$mo '\$4==n{print \$1}')
  qdel \$jid 2>/dev/null
  ( cd \$P && qsub -q scai_q -l select=1:ncpus=16:ngpus=1:mem=64gb -N tlsa_o\$mo -v OBJ=\$mo,TAG=_aug,AUG=1 scripts/hpc_tless_se3.pbs >/dev/null 2>&1 ) && { sub_list="\$sub_list MIG\$mo@scai"; sact=\$((sact+1)); }
  act=\$(printf '%s\n' "\$act" | grep -v "^tlsa_o\$mo ")
done
echo "SACT=\$sact TACT=\$tact NDONE=\$(printf '%s\n' \$done_list|grep -c .||true) SUBMITTED=[\$sub_list ]"
RS
)

log "=== dtu001 T-LESS dual-queue topper (unpinned+park): delay ${DELAY}s then SCAP=$SCAP TCAP=$TCAP period=${PERIOD}s ==="
sleep "$DELAY"
while :; do
  OUT=$(timeout 90 ssh -o ConnectTimeout=20 -o BatchMode=yes iitd-hpc "bash -s" <<<"$REMOTE" 2>&1 | tail -2)
  log "$OUT"
  echo "$OUT" | grep -q "NDONE=30" && { log "=== all 30 T-LESS COMPLETE -> exit ==="; break; }
  sleep "$PERIOD"
done

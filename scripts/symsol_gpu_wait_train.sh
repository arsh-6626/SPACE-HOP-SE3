#!/bin/bash
# Wait for local GPU headroom, then launch the SYMSOL retrain (stay local; never fall back to HPC).
# Polls device 0 free memory every 60s; once it clears THRESH MiB, hands off to the training script.
set -euo pipefail
THRESH=${THRESH:-16000}
cd /home/kavinder/SPACE-HOP
while :; do
  free=$(nvidia-smi --query-gpu=memory.free --format=csv,noheader,nounits -i 0 | tr -d ' ')
  echo "$(date +%H:%M:%S) device0 free ${free} MiB (need ${THRESH})"
  [ "${free:-0}" -ge "$THRESH" ] && break
  sleep 60
done
echo "=== GPU free -> launching training $(date) ==="
exec bash "${RUN:-scripts/symsol_train_run.sh}"

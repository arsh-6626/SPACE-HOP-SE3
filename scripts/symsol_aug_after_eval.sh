#!/bin/bash
# Wait for the in-flight test-eval to finish (keep device 0 serialised so the resident cloud jobs are never squeezed),
# surface its baseline numbers, then launch the roll-augmentation run via the GPU-headroom gate.
set -uo pipefail
cd /home/kavinder/SPACE-HOP
EVAL_PID=${EVAL_PID:-1296777}
echo "$(date) waiting for test-eval PID $EVAL_PID to finish..."
while kill -0 "$EVAL_PID" 2>/dev/null; do sleep 30; done
echo "=== $(date) test-eval finished | best.pth baseline (equiv-set LL + rotation metrics): ==="
grep -ivE "warn|deprecat" data/symsol_test_eval_best.log | tail -30
echo "=== launching roll-aug run ==="
export RUN=scripts/symsol_train_aug.sh THRESH=16000
exec bash scripts/symsol_gpu_wait_train.sh

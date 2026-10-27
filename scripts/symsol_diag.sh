#!/bin/bash
# SYMSOL coverage diagnostics -- eval only, no training, ~minutes on one GPU.
# Localises the avg-log-lik deficit on the discrete shapes (tet/cube/icosa) before we spend GPU on a fix:
#   1. ceiling   -- analytic discretisation ceiling per shape, L3-6 (CPU; fills paper tab:ceiling).
#   2. decompose -- per-level term breakdown + effective modes covered, on the converged checkpoint
#                   (fills paper tab:supp-perlevel; tells us whether mass is lost at pi0 or in the child levels).
# Usage: PY=<venv python> DATA=<prepared symsol dir> CKPT=<converged last.pth> LEVEL=5 bash scripts/symsol_diag.sh
set -euo pipefail
: "${PY:?set PY to the venv python}"; : "${DATA:?set DATA to the prepared symsol dir}"
: "${CKPT:?set CKPT to the converged checkpoint}"; LEVEL="${LEVEL:-5}"
cd "$(dirname "$0")/.."
echo "=== ceiling (CPU) ==="
$PY src/symsol_train.py ceiling --data "$DATA" --levels 3,4,5,6
echo "=== decompose (L=$LEVEL, 500 test imgs/shape) ==="
$PY src/symsol_train.py decompose --data "$DATA" --ckpt "$CKPT" --level "$LEVEL" --n 500

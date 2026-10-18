#!/usr/bin/env bash
# Full SPACE-HOP pipeline on ISRO GSAT-12R: JEPA pretraining -> pose training -> val/test evaluation.
#   PYTHON=~/miniconda3/envs/jepa/bin/python bash scripts/run_isro.sh
set -euo pipefail
REPO="$(cd "$(dirname "$0")/.." && pwd)"
PY="${PYTHON:-python}"
export WANDB_MODE="${WANDB_MODE:-disabled}"
export PYTORCH_ALLOC_CONF=expandable_segments:True
JEPA_CKPT="$REPO/checkpoints/jepa_isro/jepa_isro-latest.pth.tar"

if [[ "${SKIP_JEPA:-0}" != 1 ]]; then
  echo "== [1/3] JEPA pretraining $(date)"
  (cd "$REPO/ijepa" && "$PY" main.py --fname configs/isro_vitb16.yaml --devices cuda:0)
  # main.py runs training in a child process and does not propagate its exit status
  "$PY" - "$JEPA_CKPT" <<'PYEOF'
import sys, torch, yaml
ck = torch.load(sys.argv[1], map_location='cpu', weights_only=False)
want = yaml.safe_load(open(sys.argv[1].rsplit('/checkpoints/', 1)[0] + '/ijepa/configs/isro_vitb16.yaml'))['optimization']['epochs']
assert ck['epoch'] == want, f"JEPA stopped at epoch {ck['epoch']} / {want}"
print(f"JEPA done: epoch {ck['epoch']} loss {ck['loss']:.4f}")
PYEOF
fi

echo "== [2/3] pose training $(date)"
"$PY" "$REPO/src/train.py" --config "$REPO/configs/isro.yaml"

echo "== [3/3] evaluation $(date)"
"$PY" "$REPO/src/val.py" --config "$REPO/configs/isro.yaml" \
  --checkpoint "$REPO/checkpoints/pose_isro/best_model.pth" --splits val test --vis
echo "== done $(date)"

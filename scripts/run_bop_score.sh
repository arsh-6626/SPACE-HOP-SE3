#!/bin/bash
# Score a set of per-object BOP result CSVs with bop_toolkit (BOP19 AR = mean of VSD/MSSD/MSPD recall).
#   run_bop_score.sh <dataset> <split> <method> <csv_dir>
# e.g. run_bop_score.sh ycbv test nestse3 /home/kavinder/SPACE-HOP/results/ycbv
# Renderer: vispy + EGL (headless) from the `bopscore` conda env; bop_toolkit = ~/liepose-diffusion/lib/bop_toolkit.
set -eu
DS=$1; SPLIT=$2; METHOD=$3; CSVDIR=$4
SP=/home/kavinder/SPACE-HOP
BT=/home/kavinder/liepose-diffusion/lib/bop_toolkit
PY=/home/kavinder/miniconda3/envs/bopscore/bin/python
export BOP_PATH=/media/kavinder/hdd2/liepose_data/bop_datasets
RESDIR=$SP/results/bop_results
EVALDIR=$SP/results/bop_eval
mkdir -p "$RESDIR" "$EVALDIR"
RESCSV=$RESDIR/${METHOD}_${DS}-${SPLIT}.csv

# 1) merge per-object CSVs -> one leaderboard CSV with uniform per-image time
/home/kavinder/miniconda3/envs/jepa/bin/python "$SP"/scripts/bop_merge_csv.py --out "$RESCSV" "$CSVDIR"/*.csv

# 2) run the BOP19 AR evaluation (vispy/EGL renderer for VSD; MSSD/MSPD are renderer-free).
# eval_bop19_pose.py spawns its sub-steps as the literal `python`, so bopscore MUST be first on PATH.
export PYOPENGL_PLATFORM=egl
export PATH=/home/kavinder/miniconda3/envs/bopscore/bin:$PATH
export PYTHONPATH="$BT"
cd "$BT"
"$PY" scripts/eval_bop19_pose.py \
  --renderer_type vispy \
  --result_filenames "${METHOD}_${DS}-${SPLIT}.csv" \
  --results_path "$RESDIR" \
  --eval_path "$EVALDIR" \
  --targets_filename test_targets_bop19.json
echo "=== eval dir: $EVALDIR/${METHOD}_${DS}-${SPLIT} ==="

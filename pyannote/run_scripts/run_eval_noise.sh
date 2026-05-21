#!/bin/bash
# Pyannote Noise Robustness Threshold Sweep Script
# Evaluates selected checkpoints on noise vs non-noise cases.
# Usage: ./run_eval_noise.sh [--checkpoints <names>] [--thresholds <list>] [--case_noise_map <pairs>]
#
# Args for threshold_sweep_vivo_noise_selected_v3_case_input.py:
#   --root_dir       Project root
#   --cp_dir         Checkpoint folder
#   --data_root      Root of data/ folder
#   --output_dir     Output for results and plots
#   --checkpoints    Names: "baseline ep006.pth best_model.pth"
#   --thresholds     Threshold values: "0.05 0.25"
#   --case_noise_map Pairs: "data5=23 data14=17 data2=0 data11=0" (noise %)
#   --force_rerun    (flag) Recompute even if result.json exists

DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
cd "$DIR/../evaluation/code"

ROOT_DIR="../../"
CP_DIR="../../checkpoint/pyannote"
DATA_ROOT="../../data"
OUTPUT_DIR="../../output/pyannote/report_sweep_noise"
CHECKPOINTS="baseline best_model.pth"
THRESHOLDS="0.05 0.25"
NOISE_MAP="data5=23 data14=17 data2=0 data11=0"

echo "Starting Pyannote Noise Robustness Sweep..."
echo "  CP_DIR:     $CP_DIR"
echo "  CHECKPOINTS: $CHECKPOINTS"
echo "  THRESHOLDS:  $THRESHOLDS"
echo "  NOISE_MAP:   $NOISE_MAP"
echo "  OUTPUT_DIR:  $OUTPUT_DIR"
python threshold_sweep_vivo_noise_selected_v3_case_input.py \
    --root_dir "$ROOT_DIR" \
    --cp_dir "$CP_DIR" \
    --data_root "$DATA_ROOT" \
    --output_dir "$OUTPUT_DIR" \
    --checkpoints $CHECKPOINTS \
    --thresholds $THRESHOLDS \
    --case_noise_map $NOISE_MAP \
    "$@"

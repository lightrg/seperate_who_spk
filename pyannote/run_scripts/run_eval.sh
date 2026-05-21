#!/bin/bash
# Pyannote Evaluation Run Script (Threshold Sweep)
# Usage: ./run_eval.sh [--data_root <path>] [--checkpoint <path>] [--output_dir <path>]
#
# Args map to threshold_sweep_best_model_representative.py:
#   --data_root   Root of data/ directory
#   --checkpoint  Path to checkpoint .pth file
#   --output_dir  Where to write SUMMARY.md, tables/, plots/, runs/

DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
cd "$DIR/../evaluation/code"

# Default Evaluation Parameters
DATA_ROOT="../../data"
CKPT_PATH="../../checkpoint/pyannote/best.pth"
OUTPUT_DIR="../../output/pyannote/eval"

echo "Starting Pyannote Threshold Sweep Evaluation..."
echo "  DATA_ROOT:  $DATA_ROOT"
echo "  CKPT:       $CKPT_PATH"
echo "  OUTPUT_DIR: $OUTPUT_DIR"
python threshold_sweep_best_model_representative.py \
    --data_root "$DATA_ROOT" \
    --checkpoint "$CKPT_PATH" \
    --output_dir "$OUTPUT_DIR" \
    "$@"

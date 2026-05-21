#!/bin/bash
# Pyannote Multi-Checkpoint Evaluation Script
# Evaluates ALL checkpoints in cp_dir across all test splits and produces plots.
# Usage: ./run_eval_checkpoints.sh [--cp_dir <path>] [--data_root <path>] [--output_dir <path>] [options]
#
# Args for eval_checkpoints_visualize_optimized_resume.py:
#   --root_dir     Project root (script resolves cp/, data/ relative to this)
#   --cp_dir       Folder containing checkpoint .pth files (default: checkpoint/pyannote/)
#   --data_root    Root of data/ folder
#   --output_dir   Output for report, plots, RTTM copies
#   --splits       Splits to evaluate: test_data test_labeled (space-separated)
#   --checkpoints  "all" or specific names: "baseline ep010.pth best_model.pth"
#   --force_rerun  (flag) Recompute even if result.json already exists

DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
cd "$DIR/../evaluation/code"

ROOT_DIR="../../"
CP_DIR="../../checkpoint/pyannote"
DATA_ROOT="../../data"
OUTPUT_DIR="../../output/pyannote/report_checkpoint_eval"
SPLITS="test_data test_labeled"
CHECKPOINTS="all"

echo "Starting Pyannote Multi-Checkpoint Evaluation..."
echo "  CP_DIR:     $CP_DIR"
echo "  DATA_ROOT:  $DATA_ROOT"
echo "  SPLITS:     $SPLITS"
echo "  OUTPUT_DIR: $OUTPUT_DIR"
python eval_checkpoints_visualize_optimized_resume.py \
    --root_dir "$ROOT_DIR" \
    --cp_dir "$CP_DIR" \
    --data_root "$DATA_ROOT" \
    --output_dir "$OUTPUT_DIR" \
    --splits $SPLITS \
    --checkpoints $CHECKPOINTS \
    "$@"

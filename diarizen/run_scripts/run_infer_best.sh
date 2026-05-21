#!/bin/bash
# DiariZen Infer Best Model Script
# Usage: ./run_infer_best.sh [--audio <path>] [--output <path>] [--checkpoint <path>]
# Data structure: data/test_labeled/dataX/mixture.wav

DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
cd "$DIR/../inference/code"

# Default Parameters (override with --audio, --output, --checkpoint)
AUDIO_FILE="../../data/test_labeled/data19/mixture.wav"
OUTPUT_RTTM="../../output/diarizen/infer/infer_best.rttm"
CKPT_PATH="../../checkpoint/diari/best.pth"

echo "Starting DiariZen Best Model Inference..."
echo "  AUDIO:  $AUDIO_FILE"
echo "  OUTPUT: $OUTPUT_RTTM"
echo "  CKPT:   $CKPT_PATH"
python infer_best_model.py \
    --audio "$AUDIO_FILE" \
    --output "$OUTPUT_RTTM" \
    --checkpoint "$CKPT_PATH" \
    "$@"

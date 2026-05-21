#!/bin/bash
# DiariZen Pipeline Run Script
# Usage: ./run_pipeline.sh [--wav <path>] [--rttm <path>] [--ckpt <path>] [--n_speakers N]
# Data structure: data/test_labeled/dataX/mixture.wav + labeled/mixture.rttm

DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
cd "$DIR/../inference/code"

# Default Parameters (override any with "$@")
WAV_PATH="../../data/test_labeled/data19/mixture.wav"
RTTM_PATH="../../data/test_labeled/data19/labeled/mixture.rttm"
CKPT_PATH="../../checkpoint/diari/best.pth"
ENROLLMENT_DIR="../../data/test_labeled/data19/enrollment"
OUT_JSON="../../output/diarizen/pipeline/output.json"
N_SPEAKERS=4

echo "Starting DiariZen Full Pipeline..."
echo "  WAV:      $WAV_PATH"
echo "  CKPT:     $CKPT_PATH"
echo "  OUT_JSON: $OUT_JSON"
python pipeline_diarizen.py \
    --wav "$WAV_PATH" \
    --rttm "$RTTM_PATH" \
    --ckpt "$CKPT_PATH" \
    --enrollment_dir "$ENROLLMENT_DIR" \
    --out_json "$OUT_JSON" \
    --n_speakers "$N_SPEAKERS" \
    "$@"

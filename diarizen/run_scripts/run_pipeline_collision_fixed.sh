#!/bin/bash
# DiariZen Collision-Fixed Full Pipeline Script
# Usage: ./run_pipeline_collision_fixed.sh [--wav <path>] [--rttm <path>] [--ckpt <path>]
#
# This is the enhanced version of the diarization pipeline with:
#   - Missing confidence fix (conf_fix)
#   - Collision resolution for overlapping speaker assignments
#   - Legacy checkpoint compatibility (weights_only=False monkeypatch)
#
# --wav             Path to input WAV file
# --rttm            Path to reference RTTM (for DER evaluation)
# --ckpt            Path to DiariZen checkpoint .pth
# --enrollment_dir  Directory with per-speaker enrollment WAVs
# --out_json        Output JSON file for DER results
# --n_speakers      Number of speakers (default: 4)

DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
cd "$DIR/../evaluation/code"

# Default Parameters
WAV_PATH="../../data/test_labeled/data19/mixture.wav"
RTTM_PATH="../../data/test_labeled/data19/labeled/mixture.rttm"
CKPT_PATH="../../checkpoint/diari/best.pth"
ENROLLMENT_DIR="../../data/test_labeled/data19/enrollment"
OUT_JSON="../../output/diarizen/pipeline/output_collision_fixed.json"
N_SPEAKERS=4

echo "Starting DiariZen Collision-Fixed Pipeline..."
echo "  WAV:           $WAV_PATH"
echo "  CKPT:          $CKPT_PATH"
echo "  ENROLLMENT:    $ENROLLMENT_DIR"
echo "  OUT_JSON:      $OUT_JSON"
python diarization_pipeline_collision_fixed_missing_conf_fix.py \
    --wav "$WAV_PATH" \
    --rttm "$RTTM_PATH" \
    --ckpt "$CKPT_PATH" \
    --n_speakers "$N_SPEAKERS" \
    --enrollment_dir "$ENROLLMENT_DIR" \
    --out_json "$OUT_JSON" \
    "$@"

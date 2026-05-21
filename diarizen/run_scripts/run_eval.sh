#!/bin/bash
# DiariZen Evaluation Run Script
# Usage: ./run_eval.sh [--wav <path>] [--rttm <path>] [--ckpt <path>] [--out_json <path>] [options]
#
# Args for danh_gia.py:
#   --wav            Path to input WAV file
#   --rttm           Path to reference RTTM file
#   --utt_id         Utterance ID string (e.g. "data19_mix")
#   --out_json       Output JSON path for DER + stats
#   --enrollment_dir Folder of per-speaker WAV enrollment files (e.g. data19/enrollment/)
#   --n_speakers     Number of speakers in the audio (integer)
#   --ckpt           Path to fine-tuned checkpoint (.pth)
#   --skip_der       (flag) Skip DER computation, keep only segment output

DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
cd "$DIR/../evaluation/code"

# Default Evaluation Parameters
WAV_PATH="../../data/test_labeled/data19/mixture.wav"
RTTM_PATH="../../data/test_labeled/data19/labeled/mixture.rttm"
ENROLLMENT_DIR="../../data/test_labeled/data19/enrollment"
UTT_ID="data19_mixture"
CKPT_PATH="../../checkpoint/diari/best.pth"
OUT_JSON="../../output/diarizen/eval/eval_report.json"
N_SPEAKERS=4

echo "Starting DiariZen Evaluation..."
echo "  WAV:          $WAV_PATH"
echo "  RTTM:         $RTTM_PATH"
echo "  ENROLLMENT:   $ENROLLMENT_DIR"
echo "  UTT_ID:       $UTT_ID"
echo "  CKPT:         $CKPT_PATH"
echo "  N_SPEAKERS:   $N_SPEAKERS"
echo "  OUT_JSON:     $OUT_JSON"
python danh_gia.py \
    --wav "$WAV_PATH" \
    --rttm "$RTTM_PATH" \
    --utt_id "$UTT_ID" \
    --enrollment_dir "$ENROLLMENT_DIR" \
    --n_speakers "$N_SPEAKERS" \
    --ckpt "$CKPT_PATH" \
    --out_json "$OUT_JSON" \
    "$@"

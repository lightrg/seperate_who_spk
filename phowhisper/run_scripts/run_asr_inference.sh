#!/bin/bash
# PhoWhisper Standalone ASR Inference Script
# Usage: ./run_asr_inference.sh [--dir <data_root>] [--model <ckpt>] [--out_csv <path>]
#
# --dir     Root folder containing dataX subfolders (each with mixture.wav + labeled/*.rttm)
# --model   Path to PhoWhisper checkpoint (stage2 LoRA adapter) or HF model name
# --out_csv Output CSV with columns: sample_id, speaker, start, end, predicted_text

DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
cd "$DIR/../inference/code"

# Default Parameters
DATA_DIR="../../data/test_labeled"
MODEL_PATH="../../checkpoint/cp_phowhisper/stage2"
OUT_CSV="../../output/phowhisper/inference/asr_output.csv"

echo "Starting PhoWhisper Standalone ASR Inference..."
echo "  DATA_DIR:   $DATA_DIR"
echo "  MODEL_PATH: $MODEL_PATH"
echo "  OUT_CSV:    $OUT_CSV"
python run_asr_inference_origin.py \
    --dir "$DATA_DIR" \
    --model "$MODEL_PATH" \
    --out_csv "$OUT_CSV" \
    "$@"

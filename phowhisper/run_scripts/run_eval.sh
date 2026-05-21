#!/bin/bash
# PhoWhisper WER Evaluation Script
# Usage: ./run_eval.sh [--predictions_csv <path>] [--data_dir <path>] [--out_dir <path>]
#
# --predictions_csv  CSV from run_asr_inference.sh (columns: file_id, speaker, prediction, reference)
# --data_dir         Root of labeled test data (with dataX/labeled/mixture.rttm)
# --out_dir          Output directory for WER reports (CSV + MD per checkpoint)

DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
cd "$DIR/../evaluation/code"

# Default Parameters (run_asr_inference.sh first to generate the predictions CSV)
PREDICTIONS_CSV="../../output/phowhisper/inference/asr_output.csv"
DATA_DIR="../../data/test_labeled"
OUT_DIR="../../output/phowhisper/eval/out_report"

echo "Starting PhoWhisper WER Evaluation..."
echo "  PREDICTIONS_CSV: $PREDICTIONS_CSV"
echo "  DATA_DIR:        $DATA_DIR"
echo "  OUT_DIR:         $OUT_DIR"
python eval_wer_from_csv_v3.py \
    --predictions_csv "$PREDICTIONS_CSV" \
    --data_dir "$DATA_DIR" \
    --out_dir "$OUT_DIR" \
    "$@"

#!/bin/bash
# Qwen 7B Local Judge Evaluation Script
# Usage: ./run_eval.sh [--summary_json <path>] [--transcript <path>] [--output_json <path>]
#
# --summary_json  JSON summary produced by run_summarizer.sh (output/qwen/summary/summary.json)
# --transcript    ASR output CSV from run_asr_inference.sh OR a .jsonl transcript
# --output_json   Path to write the judge evaluation result JSON

DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
cd "$DIR/../evaluation/code"

# Default Evaluation Parameters
SUMMARY_JSON="../../output/qwen/summary/results/summary.json"
TRANSCRIPT_DATA="../../output/phowhisper/inference/asr_output.csv"
EVAL_DIR="../../output/qwen/eval/judge"
PASS1_JSON="$EVAL_DIR/judge_pass1.json"
PASS2_JSON="$EVAL_DIR/judge_pass2.json"

echo "Starting Qwen 7B Local Judge Evaluation (2 Passes)..."
echo "  SUMMARY_JSON:    $SUMMARY_JSON"
echo "  TRANSCRIPT_DATA: $TRANSCRIPT_DATA"
echo "  EVAL_DIR:        $EVAL_DIR"

echo ">> Running Pass 1..."
python judge/qwen7b_local_judge.py \
    --summary_json "$SUMMARY_JSON" \
    --transcript "$TRANSCRIPT_DATA" \
    --output_json "$PASS1_JSON" \
    "$@"

echo ">> Running Pass 2..."
python judge/qwen7b_local_judge.py \
    --summary_json "$SUMMARY_JSON" \
    --transcript "$TRANSCRIPT_DATA" \
    --output_json "$PASS2_JSON" \
    "$@"

echo ">> Aggregating Results..."
python judge/combine_qwen_judge.py --eval_dir "$EVAL_DIR"

echo ">> Done!"

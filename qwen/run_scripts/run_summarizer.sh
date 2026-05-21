#!/bin/bash
# Qwen Summarizer Script

DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
cd "$DIR/../evaluation/code"

# Default Parameters
# NOTE: --input accepts .csv (from run_asr_inference.sh) or .jsonl transcript
INPUT_DATA="../../output/phowhisper/inference/asr_output.csv"
OUTPUT_JSON="../../output/qwen/summary/results/summary.json"
OUTPUT_MD="../../output/qwen/summary/results/summary.md"

echo "Starting Qwen Summarizer..."
echo "NOTE: --input accepts .csv (from run_asr_inference.sh) or .jsonl transcript"
python inference/qwen_summarizer.py \
    --input "$INPUT_DATA" \
    --output_json "$OUTPUT_JSON" \
    --output_md "$OUTPUT_MD" \
    "$@"

#!/bin/bash
# Qwen Standalone Metric Evaluation Script (Rule-based / No LLM)
# Usage: ./run_metric.sh [--summary_json <path>] [--transcript <path>]
#
# Evaluates the structural integrity, length, and genericness of a summary.

DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
cd "$DIR/../evaluation/code"

# Default Parameters
SUMMARY_JSON="../../output/qwen/summary/results/summary.json"
TRANSCRIPT_DATA="../../output/phowhisper/inference/asr_output.csv"
OUTPUT_REPORT_JSON="../../output/qwen/eval/metric/metric_report.json"
OUTPUT_REPORT_MD="../../output/qwen/eval/metric/metric_report.md"

echo "Starting Qwen Standalone Metric Evaluation..."
echo "  SUMMARY_JSON: $SUMMARY_JSON"
echo "  TRANSCRIPT:   $TRANSCRIPT_DATA"
python metric/summary_eval_standalone.py \
    --summary_json "$SUMMARY_JSON" \
    --transcript "$TRANSCRIPT_DATA" \
    --report_json "$OUTPUT_REPORT_JSON" \
    --report_md "$OUTPUT_REPORT_MD" \
    "$@"

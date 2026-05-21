#!/bin/bash
# Full Pipeline Runner: Diarization -> ASR -> Summarization

# Default Arguments
DIAR_MODEL="diarizen"
DATA_DIR="data/test_labeled/data19"
SKIP_QWEN=false

# Parse Arguments
while [[ "$#" -gt 0 ]]; do
    case $1 in
        --diar_model) DIAR_MODEL="$2"; shift ;;
        --data_dir) DATA_DIR="$2"; shift ;;
        --skip_qwen) SKIP_QWEN=true ;;
        *) echo "Unknown parameter passed: $1"; exit 1 ;;
    esac
    shift
done

echo "========================================"
echo "Starting Full Pipeline Orchestration"
echo "Data Directory: $DATA_DIR"
echo "Diarization Model: $DIAR_MODEL"
echo "Skip Qwen: $SKIP_QWEN"
echo "========================================"

# Step 1: Diarization
WAV_FILE="${DATA_DIR}/mixture.wav"
RTTM_FILE="${DATA_DIR}/labeled/mixture.rttm"

if [ ! -f "$WAV_FILE" ]; then
    echo "Error: WAV file not found at $WAV_FILE"
    exit 1
fi

if [ ! -f "$RTTM_FILE" ]; then
    echo "Warning: RTTM file not found at $RTTM_FILE. Der evaluation will be skipped."
fi

# The output JSON from diarization
DIAR_OUT_JSON="output/${DIAR_MODEL}/pipeline/latest_run.json"

if [ "$DIAR_MODEL" == "diarizen" ]; then
    echo ">> Running DiariZen..."
    python3 diarizen/inference/code/pipeline_diarizen.py --wav "$WAV_FILE" --rttm "$RTTM_FILE" --skip_der --out_json "$DIAR_OUT_JSON"
elif [ "$DIAR_MODEL" == "pyannote" ]; then
    echo ">> Running Pyannote..."
    python3 pyannote/inference/code/pipeline_pyannote.py --wav "$WAV_FILE" --rttm "$RTTM_FILE" --skip_der --out_json "$DIAR_OUT_JSON"
else
    echo "Error: Unknown diar_model '$DIAR_MODEL'. Use 'diarizen' or 'pyannote'."
    exit 1
fi

# Step 2: PhoWhisper ASR
# Extract the newly generated RTTM from Diarization output
if [ ! -f "$DIAR_OUT_JSON" ]; then
    echo "Error: Diarization output JSON not found at $DIAR_OUT_JSON"
    exit 1
fi
HYP_RTTM=$(python3 -c "import json; print(json.load(open('$DIAR_OUT_JSON'))['hyp_rttm'])")

ASR_OUT_CSV="output/phowhisper/inference/asr_output.csv"
echo ">> Running PhoWhisper ASR..."
python3 phowhisper/inference/code/run_asr_inference_origin.py \
    --dir "$DATA_DIR" \
    --rttm "$HYP_RTTM" \
    --model "checkpoint/cp_phowhisper/stage2" \
    --out_csv "$ASR_OUT_CSV"

# Step 3: Qwen Summarization (Optional)
if [ "$SKIP_QWEN" = false ]; then
    echo ">> Running Qwen Summarization..."
    QWEN_OUT_JSON="output/qwen/summary/results/summary.json"
    QWEN_OUT_MD="output/qwen/summary/results/summary.md"
    python3 qwen/evaluation/code/inference/qwen_summarizer.py \
        --input "$ASR_OUT_CSV" \
        --output_json "$QWEN_OUT_JSON" \
        --output_md "$QWEN_OUT_MD"
    echo "Pipeline complete! Summary generated at: $QWEN_OUT_MD"
else
    echo "Pipeline complete! Qwen skipped. ASR results at: $ASR_OUT_CSV"
fi

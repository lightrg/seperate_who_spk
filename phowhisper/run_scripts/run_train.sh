#!/bin/bash
# PhoWhisper Training Run Script

DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
cd "$DIR/../train/code"

# Default Training Data Path
SELF_LABELED_ROOT="../../data"
OUTPUT_DIR="../../checkpoint/cp_phowhisper"

echo "Starting PhoWhisper Training..."
python finetune.py \
    --self_labeled_root "$SELF_LABELED_ROOT" \
    --output_dir "$OUTPUT_DIR" \
    "$@"

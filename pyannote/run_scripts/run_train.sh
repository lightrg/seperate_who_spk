#!/bin/bash
# Pyannote Training Run Script

DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
cd "$DIR/../train/code"

# Default Training Data Path
DATA_DIR="../../data"
CKPT_DIR="../../checkpoint/pyannote"

echo "Starting Pyannote Training..."
echo "Note: Configuration and paths must be set in train/code/ft_config.py"
python finetune_pyannote.py

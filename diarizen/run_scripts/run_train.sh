#!/bin/bash
# Diarizen Training Run Script

DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
cd "$DIR/../train/code"

# Default Training Data Path (Adjust as needed)
DATA_DIR="../../data"
CKPT_DIR="../../checkpoint/diari"

echo "Starting DiariZen Training..."
echo "Note: Configuration and paths must be set in train/code/ft_config.py"
python finetune_v2.py

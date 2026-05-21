# Tools & Utilities

This folder contains supporting tools for the Diarization and ASR pipeline. These tools are included in the repository and are tracked by Git for reproducibility.

## Directory Structure & File Descriptions

### 1. `tool_gen_data_mix/`
**Purpose**: Synthetic and simulated data generation for training.
*   **`meeting_simulator/`**: Contains scripts (`meeting_generator.py`, `room_simulator.py`, etc.) to mix audio files, simulate room acoustics (reverberation), create overlapping speech scenarios, and generate ground truth labels for fine-tuning EEND-EDA models.
*   **`tools/`**: Utility scripts to extract wavs and validate generated meeting overlaps (`validate_overlap.py`).
*   **`wav2vec2/`**: Scripts utilizing wav2vec2 (`align_wav2vec2.py`) to properly align phonemes and generate clean bounds for training labels.

### 2. `tool_label_vad/`
**Purpose**: Voice Activity Detection (VAD) labeling and refinement.
*   **`vad_editor_pro.py`**: A UI tool or script to manually review and refine Voice Activity Detection bounds.
*   **`Machine_labelled.py`**: Script to generate initial automated VAD labels that can later be manually corrected or used directly for large-scale data preparation.

## Usage
These tools are standalone utilities designed to be run independently when compiling datasets or evaluating data boundaries. Please refer to any internal scripts or `run.md` documentation within their respective subdirectories for specific execution details.

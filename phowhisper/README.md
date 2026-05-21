# PhoWhisper (Vietnamese ASR Model)

This module handles Automatic Speech Recognition (ASR), transcribing diarized audio segments into Vietnamese text.

## Overview

We utilize **PhoWhisper** (`vinai/PhoWhisper-large`), a Vietnamese-optimized Whisper model, fine-tuned with a **LoRA-equivalent partial fine-tuning** strategy (freezing all weights except `q_proj` / `v_proj` projections — no PEFT library required) on custom in-domain meeting data.

Training uses a **2-Stage curriculum**:
- **Stage 1**: Low-overlap data only (`overlap_ratio ≤ 0.18`) — teaches clean speech recognition.
- **Stage 2**: Harder data (`overlap_ratio ≤ 0.30`) — adds overlapping speech robustness.

Evaluation accuracy is measured using **Word Error Rate (WER)**, decomposed into Substitutions (S), Deletions (D), and Insertions (I) for diagnostic reporting.

---

## Training Pipeline (Overview)

```
Raw sources (meeting WAVs + STM/RTTM labels)
       ↓
manifest_builder.py  →  unified_manifest.jsonl
                         stage1_train / stage1_val manifests
                         stage2_train / stage2_val manifests
       ↓
finetune.py
  Stage 1: train (2 epochs) → evaluate → save best by val WER
  Stage 2: train (1 epoch)  → evaluate → save best by val WER
       ↓
checkpoint/cp_phowhisper/stage2/  (final fine-tuned weights)
```

## Inference Pipeline (Overview)

```
data/test_labeled/dataX/mixture.wav
         +
data/test_labeled/dataX/labeled/*.rttm
       ↓
run_asr_inference_origin.py
  - Loads RTTM to get speaker segments
  - Merges short segments (gap ≤ 0.5s cross-speaker, ≤ 1.5s same-speaker)
  - Chunks to HARD_MAX = 30s per segment
  - Runs PhoWhisper on each chunk
  - Retries with temperature=0.4 on hallucinated output
       ↓
output/phowhisper/inference/
  (asr_output.csv)
       ↓
eval_wer_from_csv_v3.py
  - Aligns predictions with STM ground-truth by time overlap (many-to-many)
  - Computes WER per segment, per file, per group
       ↓
output/phowhisper/eval/report_*.md + preds_*.csv
```

---

## Directory Structure & File Descriptions

### `train/code/`
Scripts for fine-tuning the PhoWhisper model.

| File | Description |
|---|---|
| `finetune.py` | Main entry point. Builds manifests, runs Stage 1 + Stage 2 training, evaluates after each stage, saves best checkpoint by val WER. **No argparse for hyperparams** — pass via CLI or edit defaults. Required: `--self_labeled_root`, `--output_dir`. |
| `config.py` | All training constants: model name, language, sample rate, overlap ratio thresholds, merge gap rules, bucket policy, stage specs. |
| `dataloader.py` | PyTorch `Dataset` / `DataCollator`. Loads audio segments from manifests, applies `truncation=True, max_length=448` to prevent label-overflow errors. |
| `manifest_builder.py` | Scans raw sources → builds unified manifest → deduplicates → exports stage1/stage2 train/val manifests. |

### `inference/code/`
Core pipeline for generating ASR predictions on new audio.

| File | Description |
|---|---|
| `run_asr_inference_origin.py` | **Primary inference script.** Reads RTTM segments, merges/chunks audio, transcribes with PhoWhisper, outputs CSV. Required: `--dir`, `--model`, `--out_csv`. Optional: `--skip_unknown`, `--trim_sec`, `--wps_threshold`, `--comp_ratio`, `--retry_temp`.<br><br>**Important Note on `--dir`:** If `--dir` points to a root folder containing multiple subfolders (e.g., `data/test_labeled/`), it will process ALL subfolders in batch mode. If it points to a specific case folder (e.g., `data/test_labeled/data19/`), it will only process that single case. |
| `config.py` | Inference constants (same structure as train config — merge gaps, overlap thresholds, bucket mapping). |
| `manifest_builder.py` | Manifest utilities shared between inference and training data preparation. |

### `evaluation/code/`
Scripts to benchmark transcription accuracy from inference outputs.

| File | Description |
|---|---|
| `eval_wer_from_csv_v3.py` | **Primary WER evaluation script.** Reads `asr_output.csv`, aligns against STM ground-truth via time overlap (many-to-many), reports Substitutions/Deletions/Insertions per file and per group. Required: `--predictions_csv`, `--data_dir`. Optional: `--out_dir`, `--report_name`. |
| `config.py` | Shared evaluation constants. |
| `manifest_builder.py` | Manifest utilities for evaluation data preparation. |

> **Design note:** `inference/code/` and `evaluation/code/` are intentionally separated. Running inference does **not** require access to ground-truth labels — preventing data leakage. Evaluation is a separate step that takes the inference CSV as input.

### `run_scripts/`
Ready-to-run Bash scripts. All paths use `../../` relative to the script's directory.

| Script | Calls | Output |
|---|---|---|
| `run_train.sh` | `finetune.py` | `checkpoint/cp_phowhisper/stage1/` and `stage2/` |
| `run_asr_inference.sh` | `run_asr_inference_origin.py` (in `inference/code/`) | `output/phowhisper/inference/asr_output.csv` |
| `run_eval.sh` | `eval_wer_from_csv_v3.py` (in `evaluation/code/`) | `output/phowhisper/eval/report_*.md` + `preds_*.csv` |

**Two-step usage:**
```bash
# Step 1: Run inference → generates asr_output.csv
./run_scripts/run_asr_inference.sh

# Step 2: Evaluate WER from the CSV
./run_scripts/run_eval.sh
```

---

## Key Implementation Notes

### Why no PEFT library?
The training environment lacked `peft` and `evaluate`. Instead:
- **Partial fine-tuning**: All Whisper weights frozen (`requires_grad=False`) except `q_proj` and `v_proj` matrices — functionally equivalent to LoRA.
- **WER metric**: Implemented via Levenshtein edit distance directly in `finetune.py`.

### Label truncation fix
Some audio/text pairs exceeded Whisper's 448-token limit, causing `ValueError` mid-training. Fixed in `dataloader.py` by adding `truncation=True, max_length=448` to the tokenizer call.

### Hallucination filtering
`run_asr_inference_origin.py` detects hallucinations (high words-per-second ratio or high zlib compression ratio) and retries with `temperature=0.4` before logging a failed segment.

---

## Historical Evaluation Results
- **Overall WER (`test_labeled`):** **11.84%** (Best Checkpoint)
- *Note:* Detailed reports comparing Base vs Stage 1 vs Stage 2 vs Best Checkpoint can be found in `output/phowhisper/eval/out_report/`. This includes WER breakdown reports per checkpoint (`report_*.md` + `preds_*.csv`).
- `inference/` — Contains ASR prediction results in CSV format (`asr_output.csv`)
- `*.png` — Training checkpoint comparison charts

---

## Requirements

Python 3.10 environment. Install from:
- `requirements/requirements_web.txt` — Full stack (Whisper + pyannote + streamlit)
- Or manually: `openai-whisper`, `transformers`, `torchaudio`, `soundfile`

## ASR Inference Methodology & Results Summary
**Methodology:** 
- **Zero Data Leakage:** The inference pipeline (`run_asr_inference_origin.py`) strictly relies on predicted `.rttm` boundaries and explicitly ignores Ground Truth (`.stm`) files during decoding.
- **Anti-Hallucination Mechanism:** Implements a retry-logic using compression ratio (`> 2.4`) and Words-Per-Second (WPS `> 8.0`) thresholds. If hallucination is detected under greedy decoding, it falls back to sampling with a temperature of `0.4`.
- **Diagnostic Evaluation:** The evaluation script uses a Diagnostic Decomposition approach, splitting errors into Insertions (I), Deletions (D), and Substitutions (S) to provide granular insights beyond a single WER number.

**Overall Results (Word Error Rate - WER):**
- **Overall WER (`test_labeled`):** **11.84%** (Best Checkpoint)
  - Substitutions (S): 5.89%
  - Deletions (D): 4.20%
  - Insertions (I): 4.08%

**WER Breakdown by Domain:**
| Domain | Total Words | WER (%) | (S) Subs (%) | (D) Dels (%) | (I) Inss (%) |
|---|---|---|---|---|---|
| `vif` | 13,597 | **6.48%** | 2.43 | 1.51 | 2.54 |
| `conan` | 3,550 | **7.58%** | 4.00 | 2.65 | 0.93 |
| `dustin_on_go` | 13,291 | **14.58%** | 5.94 | 4.93 | 3.71 |
| `chuyen_ho_chuyen_minh` | 9,872 | **17.93%** | 6.75 | 5.51 | 5.67 |
| `coi_mo` | 8,911 | **23.77%** | 10.89 | 6.39 | 6.50 |

*(For a full side-by-side comparison of Ground Truth vs. Model Predictions, please refer to the main Project Report PDF/Word).*

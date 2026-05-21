# Pyannote (Baseline Diarization Model)

This module contains the Pyannote-based speaker diarization pipeline, used as a **baseline** and comparison model against the main DiariZen architecture.

## Overview

Pyannote Audio is an open-source toolkit for speaker diarization using a segmentation + clustering approach. We finetune its internal segmentation model on custom Vietnamese meeting data and compare it against DiariZen using identical test sets and threshold-sweep evaluation scripts.

*   **Agglomerative Clustering:** Unlike traditional implementations, our finetuned Pyannote model exhibits a strong insensitivity to the clustering threshold hyperparameter. Extensive grid-search sweeps (from `0.05` to `0.45`) show identical DER scores for the best model across different thresholds. This observation indicates the model has learned robust, highly separated embeddings that do not depend heavily on manual distance threshold tuning.

Key metrics output: DER, Miss, False Alarm, Confusion, Overlap Precision/Recall/F1.

---

## Directory Structure & File Descriptions

### `train/code/`
Scripts for fine-tuning the Pyannote segmentation model.

| File | Description |
|---|---|
| `finetune_pyannote.py` | Main fine-tuning script. **No argparse** — reads all config from `ft_config.py`. Run via `run_train.sh`. |
| `ft_config.py` | **All training configuration**: learning rate, batch size, checkpoint directory, data root, Hugging Face model ID. Edit before training. |
| `ft_dataloader.py` | Dataloader utilities adapted for Pyannote's expected input format (RTTM + WAV). |

### `evaluation/code/`
Scripts to evaluate model performance and find optimal clustering thresholds.

| File | Description |
|---|---|
| `threshold_sweep_best_model_representative.py` | Grid-search script sweeping clustering thresholds (0.05–0.45) to find the optimal DER configuration. **Args**: `--root_dir`, `--data_root`, `--cp_dir`, `--output_dir`, `--checkpoint`, `--hf_model`, `--thresholds`, `--thr_start`, `--thr_end`, `--thr_step`, `--force_rerun`, `--quiet`. Outputs `SUMMARY.md` + `tables/*.csv` + per-case `result.json`. |
| `eval_checkpoints_visualize_optimized_resume.py` | Evaluates **multiple checkpoints** across all test splits with **resume support** (skips already-computed cases). Generates DER comparison plots per checkpoint. **Args**: `--root_dir`, `--cp_dir`, `--data_root`, `--output_dir`, `--splits`, `--checkpoints`, `--force_rerun`. |
| `threshold_sweep_vivo_noise_selected_v3_case_input.py` | Threshold sweep specifically on the **vivo noise test set** with configurable case-to-noise-level mapping. Compares noise vs non-noise DER for selected checkpoints. **Args**: `--root_dir`, `--cp_dir`, `--data_root`, `--output_dir`, `--checkpoints`, `--thresholds`, `--case_noise_map`, `--force_rerun`. |


### `inference/code/`
Core pipeline for generating predictions on new audio.

| File | Description |
|---|---|
| `pipeline_pyannote.py` | Full standalone Pyannote pipeline: WAV → VAD → segmentation → embedding clustering → RTTM output. **Args**: `--wav`, `--rttm`, `--ckpt`, `--out_json`, `--utt_id`, `--enrollment_dir`, `--n_speakers`, `--hf_model`, `--skip_der`. |

### `run_scripts/`
Ready-to-run Bash scripts. All paths use `../../` relative to the script's directory.

| Script | Command | Output |
|---|---|---|
| `run_train.sh` | `python finetune_pyannote.py` (no args — reads `ft_config.py`) | Checkpoint saved per `ft_config.py` |

**Important Post-Training Step:** After training finishes, the model checkpoint will be saved to `checkpoints/training/diarizen_vi_pyannote_community1_fixed/best_model.pth` (or similar depending on config). You **must** manually copy and rename this file to `checkpoint/pyannote/best.pth` before running any inference or evaluation scripts.

| `run_eval.sh` | `python threshold_sweep_best_model_representative.py --data_root --checkpoint --output_dir` | `output/pyannote/eval/` with `SUMMARY.md` + CSVs |
| `run_eval_checkpoints.sh` | `python eval_checkpoints_visualize_optimized_resume.py --cp_dir --data_root --output_dir --splits --checkpoints` | `output/pyannote/report_checkpoint_eval/` with plots + per-case JSON |
| `run_eval_noise.sh` | `python threshold_sweep_vivo_noise_selected_v3_case_input.py --checkpoints --thresholds --case_noise_map` | `output/pyannote/report_sweep_noise/` |
| `run_pipeline.sh` | `python pipeline_pyannote.py --wav --rttm --ckpt --enrollment_dir --out_json --n_speakers` | `output/pyannote/pipeline/output.json` |

---

## Historical Evaluation Results

All evaluation results are stored in `output/pyannote/`:
- `report_checkpoint_eval/` — Checkpoint comparison: baseline vs best_model (DER table in `SUMMARY.md`)
- `report_sweep_baseline/` — Threshold grid-search on baseline model
- `report_sweep_bestmodel/` — Threshold grid-search on best checkpoint
- `report_sweep_noise/` — Sweep on vivo noise test set
- `plots/` — Contains all generated visualizations:
  - `figure_latex/` — LaTeX-ready plots for paper/report (sweep, noise eval, per-checkpoint)
- `evaluation_summary.jpg` — Overall evaluation result summary image (from separate eval code)

---

## Requirements

Python 3.10 environment. Install from:
- `requirements/requirements_pyannote.txt` — Pyannote-audio standalone
- `requirements/requirements_web.txt` — If running within the web app pipeline

## Training Methodology & Results
**Methodology:** 
Pyannote is finetuned using a 3-Phase Curriculum Learning strategy, progressively unfreezing more layers and injecting harder data:
- **Phase 1 (Anchor):** Unfreeze top 4 layers (`unfreeze_top_n=4`). Train for 4 epochs (`lr=5e-5`) mostly on clean self-labeled data (80%) without augmentation to anchor the model.
- **Phase 2 (Overlap):** Unfreeze top 6 layers (`unfreeze_top_n=6`). Train for 6 epochs (`lr=2e-5`) with audio augmentation enabled. Data mix shifts to include 40% Tier-B/Tier-C overlapping/noisy data to force the model to learn complex multi-speaker segments.
- **Phase 3 (Consolidate):** Unfreeze top 6 layers. Train for 4 epochs (`lr=1e-5`) with 45% Tier-B/Tier-C data to fine-tune and consolidate the representations.

**Results (Diarization Error Rate - DER):**
| Dataset Split | Baseline DER | Best Model DER | Absolute Improvement |
|---|---|---|---|
| `test_data` | 62.17% | **17.44%** | -44.73% |
| `test_labeled` | 48.53% | **28.31%** | -20.22% |

*(For detailed charts, ablation studies, and in-depth analysis, please refer to the main Project Report PDF/Word).*

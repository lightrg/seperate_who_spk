# DiariZen (Main Diarization Model)

This module contains the primary speaker diarization system based on a finetuned EEND-EDA architecture with WavLM/Conformer backbone.

## Overview

DiariZen segments an audio recording to identify **who spoke when**, resolving overlapping speech via a custom centroid-clustering pipeline. The model is finetuned on custom Vietnamese meeting data in three phases (anchor, overlap, consolidate).

Key metrics output: DER (Diarization Error Rate), Miss, False Alarm, Confusion, Overlap Precision/Recall/F1.

---

## Directory Structure & File Descriptions

### `train/code/`
Scripts for fine-tuning the DiariZen model.

| File | Description |
|---|---|
| `finetune_v2.py` | Main training script. **No argparse** — reads all config from `ft_config.py`. Run via `run_train.sh`. |
| `ft_config.py` | **All hyperparameters and paths live here**: learning rate, batch size, checkpoint dir, data root, training phases. Edit this file before training. |
| `ft_dataloader.py` | PyTorch Dataset that loads `.wav` audio + `.rttm` speaker annotations and prepares them for EEND training. |

### `evaluation/code/`
Scripts to evaluate checkpoint quality against ground-truth RTTM files.

| File | Description |
|---|---|
| `danh_gia.py` | Primary evaluation script. Computes DER using pyannote.metrics. **Args**: `--wav`, `--rttm`, `--ckpt`, `--out_json`, `--utt_id`, `--enrollment_dir`, `--n_speakers`, `--skip_der`. Outputs a detailed `result.json`. |
| `diarization_pipeline_collision_fixed_missing_conf_fix.py` | Enhanced full pipeline with **collision resolution** (overlapping speaker assignments) and **missing confidence fix** (monkeypatches `torch.load` for legacy checkpoint compatibility). Same CLI args as `danh_gia.py` plus internal TF32/cuDNN flags for RTX GPU. Use via `run_pipeline_collision_fixed.sh`. |
| `der_eval_full_pipeline_v2_pretrained.ipynb` | Jupyter notebook for interactive evaluation of the **pretrained baseline** pipeline (v2). Mirrors `danh_gia.py` logic with cell-by-cell DER breakdown, useful for exploratory analysis and paper figure generation. |
| `infer_best_model.py` | Runs inference with the best checkpoint and outputs an RTTM prediction file. **Args**: `--audio`, `--output`, `--checkpoint`, `--n_speakers`, `--device`. |

### `inference/code/`
Core pipeline for generating diarization predictions on new audio.

| File | Description |
|---|---|
| `pipeline_diarizen.py` | Full standalone pipeline: WAV → chunked EEND inference → embedding clustering → RTTM output. **Args**: `--wav`, `--rttm`, `--ckpt`, `--out_json`, `--utt_id`, `--enrollment_dir`, `--n_speakers`, `--hf_model`, `--skip_der`. |

### `run_scripts/`
Ready-to-run Bash scripts. All paths use `../../` relative to the script's directory.

| Script | Command | Output |
|---|---|---|
| `run_train.sh` | `python finetune_v2.py` (no args — reads `ft_config.py`) | Checkpoint saved per `ft_config.py` |

**Important Post-Training Step:** After training finishes, the model checkpoint will be saved to `checkpoints/training/diarizen_vi_cuda_mdv2_multilabel_v1/best_model.pth` (or similar depending on config). You **must** manually copy and rename this file to `checkpoint/diari/best.pth` before running any inference or evaluation scripts.

| `run_eval.sh` | `python danh_gia.py --wav --rttm --ckpt --out_json` | `output/diarizen/eval/eval_report.json` |
| `run_pipeline_collision_fixed.sh` | `python diarization_pipeline_collision_fixed_missing_conf_fix.py --wav --rttm --ckpt --enrollment_dir --out_json --n_speakers` | `output/diarizen/pipeline/output_collision_fixed.json` |
| `run_infer_best.sh` | `python infer_best_model.py --audio --output --checkpoint` | `output/diarizen/infer/infer_best.rttm` |
| `run_pipeline.sh` | `python pipeline_diarizen.py --wav --rttm --ckpt --enrollment_dir --out_json --n_speakers` | `output/diarizen/pipeline/output.json` |

---

## Historical Evaluation Results

All evaluation results are stored in `output/diarizen/`:
- `report_checkpoint_eval/` — All checkpoints vs all test cases (DER breakdown)
- `report_sweep_baseline/` — Threshold grid-search on baseline model
- `report_sweep_bestmodel/` — Threshold grid-search on best checkpoint
- `report_sweep_noise/` — Sweep on vivo noise test set
- `plots/` — Contains all generated visualizations:
  - `figure_latex/` — LaTeX-ready plots for paper/report and Paper-ready PDF/PNG evaluation figures
  - `training_curves/` — Training curves (loss, DER)
- `evaluation_summary.jpg` — Overall evaluation result summary image (from separate eval code)

---

## Requirements

Python 3.10 environment. Install from:
- `requirements/requirements_diarizen.txt` — Full environment: pyannote, asteroid, espnet, diarizen (Python 3.10)
- `requirements/requirements_web.txt` — If running within the web app pipeline

## Training Methodology & Results
**Methodology:** 
The model is finetuned using a Curriculum Learning approach divided into 3 distinct phases, progressively unfreezing layers and increasing data complexity:
- **Phase 1 (Anchor):** Unfreeze the top 4 layers (`unfreeze_top_n=4`). Train with a higher learning rate (`5e-5`) on a cleaner data mix (80% self-labeled) without augmentation to establish base embeddings.
- **Phase 2 (Overlap):** Unfreeze the top 6 layers (`unfreeze_top_n=6`). Enable audio augmentation (`use_augmentation=True`) and decrease learning rate (`2e-5`). The data mix incorporates more complex/overlapping data (30% Tier-B, 10% Tier-C) to improve overlap detection.
- **Phase 3 (Consolidate):** Maintain 6 unfrozen layers but drop the learning rate (`1e-5`) with the most challenging data mix (up to 20% Tier-C) to consolidate performance on noisy/complex scenarios.

**Results (Diarization Error Rate - DER):**
| Dataset Split | Baseline DER | Best Model DER | Absolute Improvement |
|---|---|---|---|
| `test_data` | 34.69% | **13.79%** | -20.90% |
| `test_labeled` | 39.78% | **24.48%** | -15.30% |

*(For detailed charts, ablation studies, and in-depth analysis, please refer to the main Project Report PDF/Word).*

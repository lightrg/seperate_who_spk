# Speaker Diarization & Transcription — Capstone Project

> **FPT University — AIP491 Capstone Project SP26**

An end-to-end Vietnamese speaker diarization and transcription system combining four specialized models into a unified, production-ready pipeline.

---

## Full System Architecture

```text
seperate_who_sp/
├── diarizen/          # Model 1: DiariZen — Fine-tuned EEND-EDA Speaker Diarization
├── pyannote/          # Model 2: Pyannote — Baseline & Fine-tuned Speaker Diarization
├── phowhisper/        # Model 3: PhoWhisper — Vietnamese ASR (Partial-FT, No PEFT)
├── qwen/              # Model 4: Qwen — LLM Summarizer & Local Judge
├── web_app/           # Streamlit Pipeline UI: DER → ASR → Summarize
├── tool/              # Utilities: Data Generation & VAD Labeling
├── output/            # All evaluation results (JSON, CSV, MD, PNG/PDF plots)
│   ├── diarizen/      # 186 result.json + 4 SUMMARY.md + 106 plots
│   ├── pyannote/      # 164 result.json + 4 SUMMARY.md + 75 plots
│   ├── phowhisper/    # 6 CSV + 7 MD WER reports + 4 training charts
│   └── qwen/          # 23 JSON + 16 MD + 6 CSV benchmark results
├── data/              # Dataset splits
├── checkpoint/        # Model weights (.pth files)
├── requirements/      # Python 3.10 dependency files
├── full_pipeline/     # End-to-End Orchestration
│   ├── pipeline/      # Full Pipeline architecture diagrams and phase images
│   ├── config/        # Centralized path configuration (Hybrid Path Architecture)
│   ├── run_full_pipeline.py # Python orchestration script (Recommended)
│   └── run_full_pipeline.sh # Bash orchestration script
└── report/            # Capstone Project PDF Reports
```

> **Technical details:** The 300-line directory tree showing the exact location of each file, algorithm, and internal data flow of the models is detailed in [`ARCHITECTURE_FINAL.md`](ARCHITECTURE_FINAL.md).

Each model module follows a strict internal layout:
```text
<model>/
├── train/code/        # Training scripts
├── evaluation/code/   # Evaluation & benchmarking scripts
├── inference/code/    # Standalone inference pipeline
├── run_scripts/       # Bash wrappers (train / eval / infer)
└── README.md          # Module documentation
```

> **Hybrid Path Architecture (Config for Full Pipeline):** To run the **Full Pipeline** smoothly without breaking individual models, the codebase uses a hybrid approach. Inside individual model modules (`<model>/`), scripts use **relative paths** to ensure reproducibility and keep experiments independent. To stitch them together into a full pipeline, we use a centralized path configuration in `config/paths.py`. The orchestrator (`run_full_pipeline.py` and `web_app/`) uses this config to manage the unified `output/` directory and passes these dynamic paths down to the models as command-line arguments.

---

## Pipeline Overview

Audio passes through sequential stages:

| Step | Model | Task | Output |
|---|---|---|---|
| 1 | **DiariZen** or **Pyannote** | Speaker Diarization (Measured by DER*) | RTTM (who spoke when) |
| 2 | **PhoWhisper** | Vietnamese ASR | Transcript CSV |
| 3 | **Qwen 1.5B** | Summarization | Meeting Summary JSON + MD |

*(DER = Diarization Error Rate — the primary metric for speaker segmentation quality)*
*(Note: **Qwen 7B** is also included in the repository as a Local Judge, but it is strictly an isolated evaluation tool and NOT part of the execution pipeline).*

### 4 Commands to Run the Full Pipeline via Terminal (CLI)
You can run the entire system end-to-end using the `run_full_pipeline.py` script (or `run_full_pipeline.sh`) located in the `full_pipeline/` directory. The Python version is recommended as it uses a centralized `config/paths.py` architecture. Below are 4 use cases:

1. **Using DiariZen + PhoWhisper + Qwen (Full):**
   ```bash
   python full_pipeline/run_full_pipeline.py --diar_model diarizen --data_dir data/test_labeled/data19
   ```
2. **Using Pyannote + PhoWhisper + Qwen:**
   ```bash
   python full_pipeline/run_full_pipeline.py --diar_model pyannote --data_dir data/test_labeled/data19
   ```
3. **Using DiariZen + PhoWhisper (Skip Qwen Summarization):**
   ```bash
   python full_pipeline/run_full_pipeline.py --diar_model diarizen --data_dir data/test_labeled/data19 --skip_qwen
   ```
4. **Using Pyannote + PhoWhisper (Skip Qwen Summarization):**
   ```bash
   python full_pipeline/run_full_pipeline.py --diar_model pyannote --data_dir data/test_labeled/data19 --skip_qwen
   ```

---

## Web App — Quick Start

The `web_app/` provides a Streamlit UI integrating all four models end-to-end.

**Prerequisites:** Python 3.10 virtual environment with `requirements/requirements_web.txt` installed.

```bash
# Activate your venv first
source venv/bin/activate

# Run the web app
cd web_app
./run_scripts/run_web_app.sh
```

The app supports audio upload, automatic diarization, ASR transcription, and LLM-powered meeting summarization directly in the browser.

---

## Environment & Installation

> [!IMPORTANT]
> **Python 3.10 is required.** The entire pipeline depends on `pyannote.audio 3.x`, `torch 2.2`, and `transformers 4.x`, which have strict Python 3.10 compatibility requirements.

## Hugging Face Token Setup

Some Pyannote-based features require a Hugging Face access token.

```bash
# 1. Open the helper file and paste your token
vim set_hf_token.sh

# 2. Load the token into your current shell
source set_hf_token.sh

# 3. Verify it is available
echo "$HF_TOKEN"
```

`set_hf_token.sh` is ignored by Git and must never be committed.

**Option 1 — Automated setup (recommended):**
```bash
./setup_env_310.sh
```
Creates `venv/` with Python 3.10 and installs all core dependencies.

**Option 2 — Manual install per module:**
```bash
# Web App + Diarization (recommended starting point)
pip install -r requirements/requirements_web.txt

# DiariZen EEND model
pip install -r requirements/requirements_diarizen.txt

# Pyannote standalone
pip install -r requirements/requirements_pyannote.txt

# PhoWhisper ASR
pip install -r requirements/requirements_phowhisper.txt

# Qwen LLM
pip install -r requirements/requirements_qwen.txt
```

---

## Data & Checkpoints

**Data folder structure expected:**
```text
data/
├── data/                 # Train unlabelled (unlabeled wav files)
├── data_labelled/        # Train labeled (wav files + training labels)
├── test_data/            # Test unlabelled (unlabeled wav files)
├── test_labeled/         # Test labeled (wav files + RTTM labels)
├── val_data/             # Validation unlabelled (unlabeled wav files)
└── val_labeled/          # Validation labeled (wav files + validation labels)
```

**Checkpoint folder structure expected:**
```text
checkpoint/
├── diari/best.pth                  # DiariZen best checkpoint
├── pyannote/best.pth                # Pyannote best checkpoint
└── cp_phowhisper/stage2/           # PhoWhisper fine-tuned weights
```

---

## Evaluation Results

All historical evaluation results are committed to `output/` — **no re-running required** to view results.

| Model | Results Location | Key Files |
|---|---|---|
| DiariZen | `output/diarizen/` | `*/SUMMARY.md`, `*/runs/*/result.json`, `plots/` |
| Pyannote | `output/pyannote/` | `*/SUMMARY.md`, `*/runs/*/result.json`, `plots/` |
| PhoWhisper | `output/phowhisper/eval/out_report/` | `report_*.md`, `preds_report_*.csv` |
| Qwen | `output/qwen/eval/` | `judge_pass*.json`, `FINAL_JUDGE_REPORT.md` |

---

## Module Documentation

Each module has its own `README.md` with:
- Description of every source file and its role
- Exact CLI arguments for every script
- Step-by-step run instructions

| Module | README |
|---|---|
| Full Architecture | [ARCHITECTURE_FINAL.md](ARCHITECTURE_FINAL.md) |
| DiariZen | [diarizen/README.md](diarizen/README.md) |
| Pyannote | [pyannote/README.md](pyannote/README.md) |
| PhoWhisper | [phowhisper/README.md](phowhisper/README.md) |
| Qwen | [qwen/README.md](qwen/README.md) |
| Web App | [web_app/README.md](web_app/README.md) |
| Tools | [tool/README.md](tool/README.md) |

# Architecture Final — Speaker Diarization & Transcription Pipeline

> **AIP491 Capstone Project SP26 | FPT University**
> Status: **Production-Ready**
> This document combines all system knowledge from research files and detailed technical audit reports.

---

## 1. Core Models Overview
The project is a Speaker Diarization system combined with Automatic Speech Recognition (ASR) and conversational summarization via LLM, specifically optimized for Vietnamese. The entire system is built around 4 independent yet closely integrated production model modules:

1. **DiariZen (Model 1 - Module `diarizen/`)**: The primary speaker diarization model. Uses EEND-EDA architecture combined with ECAPA-TDNN Embeddings to thoroughly address overlapping speech.
2. **Pyannote (Model 2 - Module `pyannote/`)**: The baseline speaker diarization model. Uses the Segmentation and Agglomerative Clustering flow from Pyannote Audio, specifically fine-tuned for Vietnamese.
3. **PhoWhisper (Model 3 - Module `phowhisper/`)**: The Automatic Speech Recognition (ASR) model. Uses the Whisper platform combined with LoRA-equivalent partial fine-tuning (without PEFT) to accurately transcribe Vietnamese speech to text.
4. **Qwen (Model 4 - Module `qwen/`)**: The Large Language Model (LLM). The Qwen 1.5B version acts as the conversational text summarizer, while Qwen 7B serves as a Local Judge to automatically evaluate the quality of those summaries.

---

## 2. Model 1: DiariZen (Speaker Diarization)
DiariZen is the system's primary speaker diarization model, specialized in processing conversational audio with overlapping speakers.
*   **Architecture:** EEND-EDA (End-to-End Neural Diarization with Encoder-Decoder Attractors) combined with ECAPA-TDNN to extract speaker embeddings.
*   **Execution Flow (`pipeline_diarizen.py`):**
    1. Extract FBANK features (80-dim) from input Audio.
    2. Generate speaker existence probabilities per time frame (frame-level).
    3. Segment single-speaker and overlapping regions using a Dual-Threshold mechanism.
    4. Extract Speaker Embeddings and assign labels, outputting the resulting RTTM file.

---

## 3. Model 2: Pyannote (Baseline Diarization)
Pyannote serves as the baseline model to evaluate DiariZen's performance. This model has been fine-tuned on Vietnamese conversational datasets.
*   **Architecture:** Based on the standard Pyannote Audio pipeline including a Segmentation Model and Agglomerative Clustering.
*   **Token Setup:** Pyannote model loading reads the Hugging Face token from the `HF_TOKEN` environment variable, typically loaded with `source set_hf_token.sh` before running scripts.
*   **Execution Flow (`pipeline_pyannote.py`):**
    1. Run Voice Activity Detection (VAD) to remove silence.
    2. Cut audio into short segments (Speaker Segmentation).
    3. Extract Embeddings and cluster them to group identical voices, outputting RTTM.

---

## 4. Model 3: PhoWhisper (Vietnamese ASR)
PhoWhisper handles the recognition and transcription of speech into Vietnamese text.
*   **Architecture:** Based on Whisper (OpenAI) combined with a LoRA-equivalent partial fine-tuning strategy (no PEFT library required) specifically optimized for Vietnamese.
*   **Execution Flow (`run_asr_inference_origin.py`):**
    1. Receive Audio and RTTM files from the Diarization step.
    2. Cut Audio into short segments corresponding to each speaker.
    3. Pass through PhoWhisper to decode into Vietnamese text.
    4. **Anti-Hallucination Mechanism:** If nonsensical repetition (hallucination) is detected, automatically retry the recognition process with Temperature = 0.4.
    5. Output a `.csv` file containing (timestamps, speaker, text).

---

## 5. Model 4: Qwen (LLM Summarization & Judge)
Qwen is the large language model (LLM) family used in the final stage for text processing. The system uses 2 Qwen versions for 2 different purposes:
*   **Qwen/Qwen2.5-1.5B-Instruct (Conversational Summarizer - `qwen_summarizer.py`):**
    1. Read the PhoWhisper CSV output and assemble it into standard conversational format (Speaker: "Text").
    2. Use a specialized Prompt to summarize the meeting content.
    3. Write the summary results to `.json` and `.md` files.
*   **Qwen/Qwen2.5-7B-Instruct (Local Judge - `qwen7b_local_judge.py`):** Used strictly for Evaluation. Automatically reads summaries and compares them against the original text to auto-score the quality in place of human judges.

---

## 6. Execution Flow Diagram

```text
                    Audio Input (.wav)
                           │
               ┌────────────┴────────────┐
               ▼                         ▼
         DiariZen                    Pyannote
     (EEND-EDA finetuned)       (Segmentation finetuned)
      pipeline_diarizen.py       pipeline_pyannote.py
               │                         │
               └────────────┬────────────┘
                            ▼
                    RTTM Output
               (who spoke when, timestamps)
                            │
                            ▼
                       PhoWhisper
                 (Whisper Partial-FT)
              run_asr_inference_origin.py
                            │
                            ▼
                    Transcript CSV
            (speaker | start | end | text)
                            │
                            ├───────────────────────┐
                            ▼                       │
               Qwen/Qwen2.5-1.5B-Instruct           │
            (Summarizer, skippable via flag)        │
                 qwen_summarizer.py                 │
                            │                       │
                            ▼                       ▼
                     Summary JSON/MD      Qwen/Qwen2.5-7B-Instruct
                                              (Local Judge)
                                           qwen7b_local_judge.py
                                                    │
                                                    ▼
                                             Evaluation JSON
```

---

## 7. Full Repository Tree

```text
seperate_who_sp/
│
├── README.md                         # Installation guide, quick start, and results structure
├── setup_env_310.sh                  # Script to create venv and auto-install Python 3.10 environment
├── run_full_pipeline.sh              # Script to run the full pipeline (Diarization -> ASR -> Qwen)
│
├── requirements/
│   ├── requirements_web.txt          # Libraries for Streamlit Web App
│   ├── requirements_diarizen.txt     # Libraries for DiariZen (Python 3.10: inference & fine-tuning)
│   ├── requirements_pyannote.txt     # Libraries for Pyannote baseline
│   ├── requirements_phowhisper.txt   # Libraries for PhoWhisper ASR
│   └── requirements_qwen.txt         # Libraries for Qwen LLM
│
├── ─────────────── MODEL 1: DiariZen ───────────────
├── diarizen/
│   ├── README.md                     # File description, CLI arguments list, and results table
│   ├── train/code/
│   │   ├── finetune_v2.py            # EEND-EDA LoRA fine-tuning script
│   │   ├── ft_config.py              # Hyperparameter and training paths config
│   │   └── ft_dataloader.py          # PyTorch Dataset for EEND tensors
│   ├── evaluation/code/
│   │   ├── danh_gia.py               # DER evaluation from WAV and RTTM files
│   │   ├── diarization_pipeline_collision_fixed_missing_conf_fix.py # Advanced pipeline handling overlap
│   │   └── der_eval_full_pipeline_v2_pretrained.ipynb # Detailed DER analysis notebook
│   ├── inference/code/               # Standalone inference pipeline
│   │   └── infer_best_model.py       # Inference on best checkpoint
│   └── run_scripts/
│       ├── run_train.sh              # Activates finetune_v2.py
│       ├── run_eval.sh               # Evaluates on test sample dataset
│       ├── run_infer_best.sh         # Runs inference with best checkpoint
│       └── run_pipeline.sh           # Runs complete pipeline with enrollment
│
├── ─────────────── MODEL 2: Pyannote ───────────────
├── pyannote/
│   ├── README.md                     # File description, CLI arguments list, and results table
│   ├── train/code/
│   │   ├── finetune_pyannote.py      # Fine-tuning Pyannote Segmentation
│   │   ├── ft_config.py              # Training config
│   │   └── ft_dataloader.py          # Pyannote Data Loader
│   ├── evaluation/code/
│   │   ├── threshold_sweep_best_model_representative.py   # Find optimal threshold (Grid-search)
│   │   ├── eval_checkpoints_visualize_optimized_resume.py # Multi-checkpoint evaluation with resume
│   │   └── threshold_sweep_vivo_noise_selected_v3_case_input.py  # Evaluate noise robustness
│   ├── inference/code/               # Standalone inference pipeline
│   └── run_scripts/
│       ├── run_train.sh              # Activates training
│       ├── run_eval.sh               # Activates optimal threshold sweep
│       └── run_pipeline.sh           # Runs standalone Pyannote pipeline
│
├── ─────────────── MODEL 3: PhoWhisper ───────────────
├── phowhisper/
│   ├── README.md                     # Structure description, pipeline flow, CLI args, and technical notes
│   ├── train/code/
│   │   ├── finetune.py               # 2-stage Partial-FT training (equivalent to LoRA)
│   │   ├── config.py                 # Training constants: model, sample rate, stage specs, bucket policy
│   │   ├── dataloader.py             # PyTorch Dataset/DataCollator with 448 token truncation
│   │   └── manifest_builder.py       # Builds unified and stage manifests
│   ├── inference/code/               # Inference pipeline (no ground-truth required)
│   │   ├── run_asr_inference_origin.py # Main inference script: RTTM → chunk → ASR → CSV
│   │   ├── config.py                 # Inference constants (merge gap, overlap, hallucination filter)
│   │   └── manifest_builder.py       # Shared manifest utility
│   ├── evaluation/code/              # WER evaluation from CSV output (isolated step)
│   │   ├── eval_wer_from_csv_v3.py   # Many-to-many WER evaluation via time-overlap with STM
│   │   ├── config.py                 # Evaluation constants
│   │   └── manifest_builder.py       # Evaluation manifest utility
│   └── run_scripts/
│       ├── run_train.sh              # Starts PhoWhisper training (Stage 1+2)
│       ├── run_asr_inference.sh      # Runs ASR inference → asr_output.csv
│       └── run_eval.sh               # Calculates WER from CSV → Markdown report
│
├── ─────────────── MODEL 4: Qwen ───────────────
├── qwen/
│   ├── README.md                     # Qwen user guide and results summary
│   ├── inference/code/
│   │   └── qwen_infer_bridge.py      # Qwen model caller library (used by Web App)
│   ├── evaluation/code/
│   │   ├── inference/
│   │   │   └── qwen_summarizer.py    # Standalone summarizer script
│   │   ├── judge/
│   │   │   ├── qwen7b_local_judge.py # LLM-as-a-judge script
│   │   │   └── combine_qwen_judge.py # Combine multiple judge results
│   │   ├── metric/
│   │   │   └── summary_eval_standalone.py # Standard Metric evaluation
│   │   └── utils/
│   │       ├── bridge_utils.py       # Inter-process pipeline support
│   │       └── manifest_utils.py
│   └── run_scripts/
│       ├── run_summarizer.sh         # Calls summarizer script (creates MD and JSON files)
│       ├── run_metric.sh             # Activates traditional metric evaluation
│       └── run_eval.sh               # Activates auto-evaluation via LLM Judge
│
├── ─────────────── WEB APP ───────────────
├── web_app/
│   ├── README.md                     # Web App run guide and structure
│   ├── app_streamlit.py              # Streamlit Web UI entry point
│   ├── pipeline_config.py            # Config for model paths and output directories
│   ├── bridges/                      # Subprocess bridge to isolate environments
│   │   ├── der_infer_bridge.py       # Calls DiariZen speaker diarization
│   │   ├── asr_infer_bridge.py       # Calls PhoWhisper ASR
│   │   └── qwen_infer_bridge.py      # Calls Qwen text summarization
│   ├── core/
│   │   ├── der/engine.py             # In-process engine for DiariZen
│   │   ├── der/pyannote_engine.py    # In-process engine for Pyannote
│   │   ├── asr/engine.py             # ASR engine
│   │   ├── asr/config.py             # ASR config constants
│   │   ├── asr/manifest_builder.py   # Manifest builder support
│   │   └── qwen/engine.py            # Qwen summarizer engine
│   ├── utils/
│   │   ├── audio_preprocess_input.py # Audio normalization to 16kHz mono
│   │   └── asr_runner.py             # Audio slicing based on RTTM and recognition
│   ├── ui/
│   │   ├── theme.py                  # CSS theme and i18n definition
│   │   ├── components.py             # Buttons, toasts, file viewers
│   │   ├── summary_view.py           # Conversational summary display
│   │   ├── transcript_view.py        # Displays transcript per speaker
│   │   └── run_history.py            # Run history browser
│   └── run_scripts/
│       └── run_web_app.sh            # streamlit run app_streamlit.py
│
├── ─────────────── OUTPUTS ───────────────
├── output/
│   ├── diarizen/
│   │   ├── report_checkpoint_eval/           # Eval results for all checkpoints + thresholds
│   │   ├── report_sweep_baseline/            # DiariZen baseline threshold sweep
│   │   ├── report_sweep_bestmodel/           # Best checkpoint threshold sweep
│   │   ├── report_sweep_noise/               # Noise robustness threshold sweep
│   │   ├── plots/
│   │   │   ├── figure_latex/                 # Images and PDFs ready for LaTeX reporting
│   │   │   └── training_curves/              # Loss/DER curves
│   │   ├── eval/                             # (Receives results on new run_eval.sh)
│   │   ├── infer/                            # (Receives results on new run_infer_best.sh)
│   │   ├── pipeline/                         # (Receives results on new run_pipeline.sh)
│   │   └── evaluation_summary.jpg            # Overall eval chart (from standalone eval code)
│   ├── pyannote/
│   │   ├── report_checkpoint_eval/           # Baseline vs optimized checkpoint comparison
│   │   ├── report_sweep_baseline/            # Pyannote baseline threshold sweep
│   │   ├── report_sweep_bestmodel/           # Best checkpoint threshold sweep
│   │   ├── report_sweep_noise/               # Noise robustness threshold sweep
│   │   ├── plots/
│   │   │   └── figure_latex/                 # LaTeX formatted charts
│   │   ├── eval/                             # (Receives results on new run_eval.sh)
│   │   ├── pipeline/                         # (Receives results on new run_pipeline.sh)
│   │   └── evaluation_summary.jpg            # Overall eval chart (from standalone eval code)
│   ├── phowhisper/
│   │   ├── eval/out_report/                  # WER analysis: report_*.md files + CSV
│   │   ├── inference/                        # Contains ASR prediction results (asr_output.csv)
│   │   └── *.png                             # Checkpoint accuracy comparison charts
│   └── qwen/
│       ├── eval/                             # Evaluation reports
│       │   ├── judge/                        # Qwen 7B Judge results (all 5 cases: data2, data19, data26, data52, data55)
│       │   └── metric/                       # Metric results (ROUGE, BERTScore for 5 cases)
│       ├── summary/                          # Summary reports
│       │   ├── benchmark/                    # Model version benchmarks and bench_standard.csv
│       │   └── results/                      # (Receives results on new run_summarizer.sh)
│
├── ─────────────── TOOLS ───────────────
├── tool/
│   ├── README.md
│   ├── tool_gen_data_mix/            # Synthetic data generation tool (meeting room + speaker mixing)
│   └── tool_label_vad/               # Auto VAD labeling UI
│
├── ─────────────── DATA & CHECKPOINTS ───────────────
├── data/                             # 30GB audio data
│   ├── data/                         # Train unlabelled (unlabeled wav files)
│   ├── data_labelled/                # Train labeled (wav files + training labels)
│   ├── test_data/                    # Test unlabelled (unlabeled wav files)
│   ├── test_labeled/                 # Test labeled (wav files + RTTM labels)
│   ├── val_data/                     # Validation unlabelled (unlabeled wav files)
│   └── val_labeled/                  # Validation labeled (wav files + validation labels)
├── checkpoint/                       # Model weights
│   ├── diari/best.pth                # Best DiariZen weights
│   ├── pyannote/best.pth              # Best Pyannote weights
│   └── cp_phowhisper/stage2/         # PhoWhisper fine-tuned weights
```

---

## 8. Utility Run Scripts Cheat Sheet

*(Note: To see detailed CLI instructions, please view the '4 Commands to Run the Full Pipeline' section in README.md. The table below is solely a technical reference of which scripts call which Python files).*

### Full Pipeline
| Script | Called File | Purpose |
|---|---|---|
| `run_full_pipeline.sh` | Runs sequentially: Diarization → ASR → Summarization | Automates End-to-End full project |

### DiariZen
| Script | Calls | Key Args |
|---|---|---|
| `run_train.sh` | `finetune_v2.py` | *(reads ft_config.py)* |
| `run_eval.sh` | `danh_gia.py` | `--wav --rttm --ckpt --out_json --utt_id --enrollment_dir --n_speakers` |
| `run_infer_best.sh` | `infer_best_model.py` | `--audio --output --checkpoint --n_speakers --device` |
| `run_pipeline.sh` | `pipeline_diarizen.py` | `--wav --rttm --ckpt --enrollment_dir --n_speakers` |
| `run_pipeline_collision_fixed.sh` | `diarization_pipeline_collision_fixed_missing_conf_fix.py` | `--wav --rttm --ckpt --enrollment_dir --out_json --n_speakers` |

### Pyannote
| Script | Calls | Key Args |
|---|---|---|
| `run_train.sh` | `finetune_pyannote.py` | *(reads ft_config.py)* |
| `run_eval.sh` | `threshold_sweep_best_model_representative.py` | `--data_root --checkpoint --output_dir` |
| `run_eval_checkpoints.sh` | `eval_checkpoints_visualize_optimized_resume.py` | `--cp_dir --data_root --output_dir --splits --checkpoints` |
| `run_eval_noise.sh` | `threshold_sweep_vivo_noise_selected_v3_case_input.py` | `--checkpoints --thresholds --case_noise_map` |
| `run_pipeline.sh` | `pipeline_pyannote.py` | `--wav --rttm --ckpt --enrollment_dir --n_speakers` |

### PhoWhisper
| Script | Calls | Key Args |
|---|---|---|
| `run_train.sh` | `finetune.py` (in `train/code/`) | `--self_labeled_root --output_dir` |
| `run_asr_inference.sh` | `run_asr_inference_origin.py` (in `inference/code/`) | `--dir --model --out_csv` |
| `run_eval.sh` | `eval_wer_from_csv_v3.py` (in `evaluation/code/`) | `--predictions_csv --data_dir --out_dir` |

### Qwen
| Script | Calls | Key Args |
|---|---|---|
| `run_summarizer.sh` | `inference/qwen_summarizer.py` | `--input --output_json --output_md` |
| `run_metric.sh` | `metric/summary_eval_standalone.py` | `--summary_json --transcript --report_json` |
| `run_eval.sh` | `judge/qwen7b_local_judge.py` | `--summary_json --transcript --output_json` |


### Web App
| Script | Calls | Notes |
|---|---|---|
| `run_web_app.sh` | `streamlit run app_streamlit.py` | Full DER→ASR→LLM UI |

---

## 9. Evaluation Results Summary

| Model | Metric | Result Status |
|---|---|---|
| **DiariZen** | DER (Diarization Error Rate) | Generated 186 detailed result files across all checkpoints and threshold sweeps. |
| **Pyannote** | DER | Generated 164 detailed comparison result files between baseline and optimized models. |
| **PhoWhisper** | WER (Word Error Rate) | Includes 7 CSV reports + 6 Markdown reports evaluating across stage1, stage2, and best checkpoint. |
| **Qwen** | Summary Quality | Outputs summary quality evaluation files (`data2_judge.json`, etc.) totaling 23 JSON, 16 MD, and 6 CSV files across benchmark folders. |

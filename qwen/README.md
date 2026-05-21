# Qwen (LLM Summarization & Evaluation)

This module uses **Qwen/Qwen2.5-1.5B-Instruct** for conversational summarization and **Qwen/Qwen2.5-7B-Instruct** as an automated LLM Judge to evaluate the summaries.

## Overview

Qwen is the final stage of the pipeline. It operates in two modes:
1. **Summarization**: Takes a cleaned transcript (from PhoWhisper ASR) and generates a structured meeting summary with speaker insights, key points, and action items.
2. **Local Judge Evaluation**: A 7B Instruct model scores the generated summaries based on schema compliance, evidence grounding, and length quality — without human intervention.

>  **Important**: `qwen/inference/code/qwen_infer_bridge.py` is a **library module** used internally by the web app bridge. It has **no CLI** and cannot be run directly. For standalone execution, use `run_summarizer.sh` → `qwen_summarizer.py`.

---

## Directory Structure & File Descriptions

>  **Architectural Note: Why are there two Inference scripts?**
> - `qwen/inference/code/qwen_infer_bridge.py`: This is a **Python Library Module** meant for Production. It is called dynamically by the **Web App** via memory import to generate summaries on the fly. It cannot be run via command-line.
> - `qwen/evaluation/code/inference/qwen_summarizer.py`: This is a **Standalone CLI Script**. It was built for **Evaluation** purposes (hence its location) so researchers could batch-generate summary outputs from terminal commands, save them to disk, and then feed them into the Local Judge for scoring.

### `inference/code/`
| File | Description |
|---|---|
| `qwen_infer_bridge.py` | **Library module only** — no CLI. Called by `web_app/bridges/qwen_infer_bridge.py` to run summarization within the web app subprocess pipeline. |

### `evaluation/code/`
Core standalone scripts for summarization and evaluation, cleanly separated by tasks.

#### `inference/`
| File | Description |
|---|---|
| `qwen_summarizer.py` | **Primary standalone summarizer.** Normalizes transcript blocks and generates final JSON + Markdown summary. **Args**: `--input` (`.csv` from PhoWhisper, `.jsonl`, or bracket-format `.txt`), `--output_json`, `--output_md`. |

#### `judge/`
| File | Description |
|---|---|
| `qwen7b_local_judge.py` | 7B LLM-as-a-judge for evaluating summary quality (faithfulness/conciseness). **Args**: `--summary_json`, `--transcript`. |
| `combine_qwen_judge.py` | Utility script to aggregate multiple `judge_report.json` files into a single unified report. |

#### `metric/`
| File | Description |
|---|---|
| `summary_eval_standalone.py` | Standalone rule-based summary evaluator computing ROUGE and BERTScore metrics (no LLM). |
| `final_judge_template.json` | Final output template containing aggregated evaluations from Qwen 7B. |

#### `utils/`
| File | Description |
|---|---|
| `bridge_utils.py` | Shared utilities for inter-process communication. |
| `manifest_utils.py` | Utilities to load and convert transcript manifest formats. |

### `run_scripts/`
Ready-to-run Bash scripts. All paths use `../../` relative to the script's directory.

| Script | Calls | Default Output |
|---|---|---|
| `run_summarizer.sh` | `inference/qwen_summarizer.py` | `output/qwen/summary/results/summary.json` + `.md` |
| `run_metric.sh` | `metric/summary_eval_standalone.py` | `output/qwen/eval/metric/metric_report.json` + `.md` |
| `run_eval.sh` | `judge/qwen7b_local_judge.py` (2 Passes) + `combine_qwen_judge.py` | Runs Qwen 7B to evaluate summary quality against transcript. Produces `judge_pass*.json` for the new run, and `combine_qwen_judge.py` automatically aggregates all `*judge*.json` files (including existing benchmark files like `data19_qwen7b_judge.json`) into `FINAL_JUDGE_REPORT.json` / `.md`. |

---

## Historical Evaluation Results

All results are stored in `output/qwen/`:
- `eval/` — Automatic evaluation reports
  - `metric/` — Standalone metric evaluations (ROUGE, BERTScore for all 5 cases: data2, data19, data26, data52, data55)
  - `judge/` — Qwen 7B Judge evaluations (contains full judge results for 5 data cases)
- `summary/` — Summary Reports
  - `results/` — (receives new `run_summarizer.sh` outputs, contains 5 data cases)
  - `benchmark/` — Benchmarks of different model versions (`bench_1p5b_base`, `bench_1p5b_4bit`, etc.) and the aggregated `bench_standard.csv` file

---

## Requirements

Python 3.10 environment. Install from:
- `requirements/requirements_qwen.txt` — Transformers + bitsandbytes + accelerate + peft

---

## Fine-tuning Workflow & Hardware Optimizations

*(Note: These steps are strictly optional. The current models are already fully trained and ready for inference. Only follow this section if you intend to fine-tune the Qwen models further on custom datasets. The actual `dataset_builder.py` and `finetune.py` scripts used for this are maintained in a separate training repository and are not included here).*

### 1. Data Preparation
For the LLM to learn summarization, it needs explicit targets. In each dataset folder (e.g., `data3/labeled/`), alongside the raw `transcript.jsonl`, create a `summary_target.md` containing the ideal summary (Meeting Overview, Speaker Summaries, etc.).
Run the dataset builder to merge all transcripts and targets into a single JSONL:
```bash
python dataset_builder.py --root_dir data_labeled/Tong_hop_data_labelled --out_jsonl out_dataset/train_dataset.jsonl
```

### 2. Training Execution
Execute the fine-tuning script on the built dataset:
```bash
python finetune.py --train_data out_dataset/train_dataset.jsonl --output_dir output_qwen_adapter
```

### 3. Hardware Optimizations & Fixes (Historical Context)
- **VRAM Limitations:** Qwen 2.5 7B has a massive vocabulary size (151k). To avoid OOM (Out Of Memory) errors even on 24GB VRAM (e.g., RTX 3090 Ti) with `max_seq_length=8192`, we configured `per_device_train_batch_size=1` and `gradient_accumulation_steps=8` to heavily slice the CrossEntropyLoss matrix computation.
- **8-Bit Optimization:** The training automatically uses `Paged_8bit` optimizers and Gradient Checkpointing to allow training on standard consumer GPUs (like the RTX 4060 8GB).
- **Dependency Conflicts:** The code implements dynamic monkey-patching in memory to bypass version conflicts between `peft`, `accelerate`, and older `transformers` without needing to break/upgrade the strictly versioned `.venv`.

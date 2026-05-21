# Web App Pipeline (Streamlit)

This module provides a unified Streamlit UI to run the entire system pipeline sequentially: **Diarization (DER) -> ASR (Whisper) -> Summarization (Qwen)**.

## Architecture & Flow

1.  **Audio Input**: The user uploads or selects an audio file via the Streamlit interface.
2.  **Diarization (DER)**: 
    *   Calls `diarizen` or `pyannote` inference engines (configured via `pipeline_config.py`).
    *   Outputs RTTM (Speaker Segmentation).
3.  **ASR (PhoWhisper)**:
    *   Reads the generated RTTM.
    *   Slices audio and runs ASR on each chunk using PhoWhisper.
    *   Outputs a transcript CSV.
4.  **Summarization & Formatting (Qwen)**:
    *   Passes the transcript to the local Qwen LLM.
    *   Outputs a normalized transcript and a formatted summary markdown.

## Directory Structure & File Descriptions

To maintain modularity and ease of maintenance, the web app code is split into logical components:

*   **`app_streamlit.py`**: The main entry point of the Streamlit application. Manages the overall user interface, audio upload handling, and the sequential execution flow.
*   **`pipeline_config.py`**: Configuration variables defining default paths to models, output directories, and active pipeline options. **Note:** This file utilizes the project's central **Hybrid Path Architecture** by importing centralized paths from the root `config/paths.py` to ensure outputs are saved in the unified global `output/` directory.
*   **`run_scripts/`**: Contains the launch script (`run_web_app.sh`) for macOS/Linux users.
*   **`windows_scripts/`**: Legacy folder for Windows PowerShell environments. Contains older `.bat` files originally used for testing components on Windows. `pipeline.txt` is a historical command log for reference only — not maintained. `run_streamlit.ps1` (launch script), `install_extra_web.ps1` (dependency installer), and `pipeline.txt` (command reference). These are **not** used by the main Linux/Mac pipeline.

### `bridges/` (Subprocess Bridges)
These files act as internal APIs. They use Python's `subprocess` module to call the standalone models directly, isolating dependency environments.
*   **`der_infer_bridge.py`**: Initiates the Diarization (DiariZen or Pyannote) subprocess to get RTTM segmentations.
*   **`asr_infer_bridge.py`**: Initiates the Whisper ASR subprocess to convert speech into text segments.
*   **`qwen_infer_bridge.py`**: Initiates the Qwen LLM subprocess to format the transcript into a readable meeting summary.

### `utils/` (Utility Scripts)
*   **`audio_preprocess_input.py`**: Responsible for checking and standardizing uploaded audio (converting sample rates to 16kHz, mono channel) before it enters the pipeline.
*   **`asr_runner.py`**: Slices the original audio according to the RTTM timestamps and iteratively processes each slice through the local Whisper model.

### `ui/` (User Interface Components)
*   Modular files (`theme.py` for CSS, `components.py` for buttons/alerts, `summary_view.py` for final markdown rendering, `transcript_view.py` for displaying the transcript per speaker, `run_history.py` for displaying historical runs) used to build the Streamlit frontend cleanly.
    *   **Note on `transcript_view.py`**: This file is not imported directly by `app_streamlit.py`. Instead, it is imported and utilized by `summary_view.py` to render the interactive transcript view alongside the summary.

### `core/`
*   Contains in-process engine modules. These are loaded directly by `app_streamlit.py` when models share the same Python environment, avoiding subprocess overhead. Use this path for single-environment deployments.
| File | Description |
|---|---|
| `der/engine.py` | In-process engine for DiariZen |
| `der/pyannote_engine.py` | In-process engine for Pyannote |
| `asr/engine.py` | ASR engine |
| `asr/config.py` | ASR config constants |
| `asr/manifest_builder.py` | Manifest builder support |
| `qwen/engine.py` | Qwen summarizer engine |

## Running the Web App

Ensure your Python 3.10 environment has the web requirements installed.

### On macOS / Linux
Run the application using the script in the `run_scripts` folder:
```bash
./run_scripts/run_web_app.sh
```

### On Windows
You can use the provided PowerShell script:
```powershell
cd windows_scripts
.\run_streamlit.ps1
```

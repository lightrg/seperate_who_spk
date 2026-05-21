from pathlib import Path
import os

import sys

# Add project root to path so we can import from config
WEB_APP_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = WEB_APP_DIR.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from full_pipeline.config.paths import (
    get_diarizen_checkpoint,
    get_pyannote_checkpoint,
    get_phowhisper_checkpoint,
    OUTPUT_DIR
)

ROOT_DIR = WEB_APP_DIR

# Default paths to the live DER and ASR pipeline entrypoints.
# The interface is designed to run inside one environment, using the current
# Python interpreter for both DER and ASR subprocess execution.
DER_SCRIPT_PATH = Path(os.environ.get(
    "DER_SCRIPT_PATH",
    ROOT_DIR / "core" / "der" / "engine.py",
)).resolve()
PYANNOTE_SCRIPT_PATH = Path(os.environ.get(
    "PYANNOTE_SCRIPT_PATH",
    ROOT_DIR / "core" / "der" / "pyannote_engine.py",
)).resolve()
ASR_SCRIPT_PATH = Path(os.environ.get(
    "ASR_SCRIPT_PATH",
    ROOT_DIR / "utils" / "asr_runner.py",
)).resolve()

QWEN_SCRIPT_PATH = Path(os.environ.get(
    "QWEN_SCRIPT_PATH",
    ROOT_DIR / "core" / "qwen" / "engine.py",
)).resolve()
QWEN_NORMALIZE_MODEL_DEFAULT = "Qwen/Qwen2.5-1.5B-Instruct"
QWEN_SUMMARY_MODEL_DEFAULT = "Qwen/Qwen2.5-1.5B-Instruct"

# Default checkpoint paths for DER and ASR (Global Checkpoint Folder)
DER_CHECKPOINT_DEFAULT = get_diarizen_checkpoint()
PYANNOTE_CHECKPOINT_DEFAULT = get_pyannote_checkpoint()
ASR_CHECKPOINT_DEFAULT = get_phowhisper_checkpoint()

# Default output folder for interface-run results.
# Points to the centralized repo-level output/ directory (NOT web_app/output/).
DEFAULT_OUTPUT_DIR = OUTPUT_DIR
DEFAULT_OUTPUT_DIR.mkdir(exist_ok=True)

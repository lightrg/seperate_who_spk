from pathlib import Path
import os

ROOT_DIR = Path(__file__).resolve().parent

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
DER_CHECKPOINT_DEFAULT = str(ROOT_DIR.parent / "checkpoint" / "diari" / "best.pth")
PYANNOTE_CHECKPOINT_DEFAULT = str(ROOT_DIR.parent / "checkpoint" / "pyannote" / "best.pth")
ASR_CHECKPOINT_DEFAULT = str(ROOT_DIR.parent / "checkpoint" / "cp_phowhisper" / "stage2")

# Default output folder for interface-run results.
# Points to the centralized repo-level output/ directory (NOT web_app/output/).
DEFAULT_OUTPUT_DIR = ROOT_DIR.parent / "output"
DEFAULT_OUTPUT_DIR.mkdir(exist_ok=True)

from pathlib import Path

# Thư mục gốc của project (nằm ở cha của thư mục full_pipeline)
ROOT_DIR = Path(__file__).resolve().parents[2]

# Các thư mục cố định
DATA_DIR = ROOT_DIR / "data"
CHECKPOINT_DIR = ROOT_DIR / "checkpoint"
OUTPUT_DIR = ROOT_DIR / "output"

def get_pipeline_output_dirs(model_name: str, run_id: str = "latest_run"):
    """
    Tạo và trả về các đường dẫn output chuẩn cho luồng orchestration (diarization/asr/llm).
    VD: output/diarizen/pipeline/latest_run/...
    """
    base = OUTPUT_DIR / model_name / "pipeline" / run_id
    base.mkdir(parents=True, exist_ok=True)
    return {
        "base": base,
        "json": base / f"{run_id}.json",
        "csv": base / f"{run_id}_asr.csv",
        "md": base / f"{run_id}_summary.md",
        "md_final": base / f"{run_id}_final_summary.md", # Dùng nếu LLM xuất file format khác
    }

def get_shared_outputs():
    """
    Dùng cho web_app và các chức năng cần thư mục chung.
    """
    return {
        "diarization": OUTPUT_DIR / "diarization",
        "asr": OUTPUT_DIR / "asr",
        "llm": OUTPUT_DIR / "llm",
        "merged": OUTPUT_DIR / "merged"
    }

# Các Helper truy xuất Checkpoint gốc
def get_diarizen_checkpoint():
    return str(CHECKPOINT_DIR / "diari" / "best.pth")

def get_pyannote_checkpoint():
    return str(CHECKPOINT_DIR / "pyannote" / "best.pth")

def get_phowhisper_checkpoint():
    return str(CHECKPOINT_DIR / "cp_phowhisper" / "stage2")

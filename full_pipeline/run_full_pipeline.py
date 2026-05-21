import argparse
import subprocess
import sys
import json
from pathlib import Path

# Import từ centralized paths
from config.paths import (
    get_pipeline_output_dirs,
    get_phowhisper_checkpoint,
    ROOT_DIR
)

def run_subprocess(cmd):
    """Utility chạy subprocess và hiển thị log real-time."""
    print(f"Executing: {' '.join(cmd)}")
    process = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, cwd=ROOT_DIR)
    for line in process.stdout:
        print(line, end='')
    process.wait()
    if process.returncode != 0:
        print(f"Error executing command. Return code: {process.returncode}")
        sys.exit(process.returncode)

def main():
    parser = argparse.ArgumentParser(description="Full Pipeline Runner: Diarization -> ASR -> Summarization")
    group = parser.add_argument_group("Pipeline Config")
    group.add_argument("--diar_model", type=str, default="diarizen", choices=["diarizen", "pyannote"], help="Diarization model to use")
    group.add_argument("--data_dir", type=str, default="data/test_labeled/data19", help="Data directory containing mixture.wav")
    group.add_argument("--skip_qwen", action="store_true", help="Skip Qwen summarization")
    args = parser.parse_args()

    print("========================================")
    print("Starting Full Pipeline Orchestration (Pythonic)")
    print(f"Data Directory: {args.data_dir}")
    print(f"Diarization Model: {args.diar_model}")
    print(f"Skip Qwen: {args.skip_qwen}")
    print("========================================")

    data_dir = ROOT_DIR / args.data_dir
    wav_file = data_dir / "mixture.wav"
    rttm_file = data_dir / "labeled/mixture.rttm"

    if not wav_file.exists():
        print(f"Error: WAV file not found at {wav_file}")
        sys.exit(1)
        
    if not rttm_file.exists():
        print(f"Warning: RTTM file not found at {rttm_file}. Der evaluation will be skipped.")

    # Lấy output paths tập trung
    out_dirs = get_pipeline_output_dirs(model_name=args.diar_model, run_id="latest_run")
    diar_out_json = out_dirs["json"]

    # Step 1: Diarization
    if args.diar_model == "diarizen":
        print(">> Running DiariZen...")
        cmd = ["python3", "diarizen/inference/code/pipeline_diarizen.py"]
    else:
        print(">> Running Pyannote...")
        cmd = ["python3", "pyannote/inference/code/pipeline_pyannote.py"]

    cmd.extend([
        "--wav", str(wav_file),
        "--rttm", str(rttm_file),
        "--skip_der",
        "--out_json", str(diar_out_json)
    ])
    run_subprocess(cmd)

    # Step 2: PhoWhisper ASR
    if not diar_out_json.exists():
        print(f"Error: Diarization output JSON not found at {diar_out_json}")
        sys.exit(1)

    with open(diar_out_json, "r") as f:
        hyp_rttm = json.load(f).get("hyp_rttm")

    if not hyp_rttm:
        print("Error: Could not extract hyp_rttm from diarization JSON.")
        sys.exit(1)

    asr_out_csv = out_dirs["csv"]
    print(">> Running PhoWhisper ASR...")
    asr_cmd = [
        "python3", "phowhisper/inference/code/run_asr_inference_origin.py",
        "--dir", str(data_dir),
        "--rttm", str(hyp_rttm),
        "--model", get_phowhisper_checkpoint(),
        "--out_csv", str(asr_out_csv)
    ]
    run_subprocess(asr_cmd)

    # Step 3: Qwen Summarization
    if not args.skip_qwen:
        print(">> Running Qwen Summarization...")
        qwen_out_json = out_dirs["json"].parent / "latest_run_summary.json"
        qwen_out_md = out_dirs["md"]
        
        qwen_cmd = [
            "python3", "qwen/evaluation/code/inference/qwen_summarizer.py",
            "--input", str(asr_out_csv),
            "--output_json", str(qwen_out_json),
            "--output_md", str(qwen_out_md)
        ]
        run_subprocess(qwen_cmd)
        print(f"Pipeline complete! Summary generated at: {qwen_out_md}")
    else:
        print(f"Pipeline complete! Qwen skipped. ASR results at: {asr_out_csv}")

if __name__ == "__main__":
    main()

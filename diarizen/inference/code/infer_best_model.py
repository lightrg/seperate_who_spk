import argparse
import os
import sys
from pathlib import Path
import torch
import torchaudio
import numpy as np
from pyannote.core import Annotation, Segment
import warnings

warnings.filterwarnings("ignore")

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
CODE_DIR = PROJECT_ROOT / "code_DiariZen"

sys.path.insert(0, str(CODE_DIR))

try:
    from ft_config import FinetuneConfig
    from finetune_v2 import DiariZenWrapper
except ImportError as e:
    print(f"Error importing DiariZen modules: {e}")
    print("Make sure you're running from the correct directory")
    sys.exit(1)

def load_best_model(checkpoint_path: str, device: str = "auto") -> DiariZenWrapper:

    print(f"Loading best model from: {checkpoint_path}")

    cfg = FinetuneConfig()

    model = DiariZenWrapper(cfg)

    checkpoint = torch.load(checkpoint_path, map_location='cpu')

    if 'model_state_dict' in checkpoint:
        model.load_state_dict(checkpoint['model_state_dict'])
    else:
        model.load_state_dict(checkpoint)

    if device == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"

    model.to(device)
    model.eval()

    print(f"Model loaded successfully on {device}")
    return model

def process_audio_to_rttm(model: DiariZenWrapper, audio_path: str, output_path: str, n_speakers: int = 4):

    print(f"Processing audio: {audio_path}")

    waveform, sample_rate = torchaudio.load(audio_path)

    if waveform.shape[0] > 1:
        waveform = torch.mean(waveform, dim=0, keepdim=True)

    if sample_rate != 16000:
        resampler = torchaudio.transforms.Resample(sample_rate, 16000)
        waveform = resampler(waveform)
        sample_rate = 16000

    audio = waveform.squeeze().numpy()

    print(f"Audio loaded: {audio.shape}, sample_rate: {sample_rate}")

    with torch.no_grad():
        predictions = model.infer(audio, n_speakers=n_speakers)

    annotation = predictions_to_annotation(predictions, audio.shape[0] / sample_rate)

    save_rttm(annotation, output_path, Path(audio_path).stem)

    print(f"RTTM saved to: {output_path}")

def predictions_to_annotation(predictions, duration):

    annotation = Annotation()

    for pred in predictions:
        start = pred['start']
        end = pred['end']
        speaker = pred['speaker']

        start = max(0, start)
        end = min(duration, end)

        if end > start:
            annotation[Segment(start, end)] = speaker

    return annotation

def save_rttm(annotation, output_path, uri="audio"):

    with open(output_path, 'w', encoding='utf-8') as f:
        for segment, track in annotation.itertracks():
            start = segment.start
            duration = segment.end - segment.start
            speaker = track

            line = f"SPEAKER {uri} 1 {start:.3f} {duration:.3f} <NA> <NA> {speaker} <NA>\n"
            f.write(line)

def main():
    parser = argparse.ArgumentParser(description="Load best DER model and process audio to RTTM")
    parser.add_argument("--audio", type=str, required=True, help="Path to input audio file")
    parser.add_argument("--output", type=str, required=True, help="Path to output RTTM file")
    parser.add_argument("--checkpoint", type=str,
                       default=str(PROJECT_ROOT / "code_DiariZen" / "checkpoints" / "training" / "diarizen_vi_cuda_mdv2_multilabel_v1" / "best_model.pth"),
                       help="Path to model checkpoint")
    parser.add_argument("--n_speakers", type=int, default=4, help="Number of speakers")
    parser.add_argument("--device", type=str, default="auto", help="Device to run on (auto/cuda/cpu)")

    args = parser.parse_args()

    if not os.path.exists(args.audio):
        print(f"Error: Audio file not found: {args.audio}")
        return

    if not os.path.exists(args.checkpoint):
        print(f"Error: Checkpoint not found: {args.checkpoint}")
        return

    output_dir = os.path.dirname(args.output)
    if output_dir and not os.path.exists(output_dir):
        os.makedirs(output_dir, exist_ok=True)

    try:
        model = load_best_model(args.checkpoint, args.device)

        process_audio_to_rttm(model, args.audio, args.output, args.n_speakers)

        print(" Inference completed successfully!")

    except Exception as e:
        print(f" Error during inference: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()
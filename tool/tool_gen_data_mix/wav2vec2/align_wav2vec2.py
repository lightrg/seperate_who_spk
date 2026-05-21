import torch
import torchaudio
import numpy as np
from pathlib import Path
from typing import List, Dict, Any, Tuple
import argparse
from tqdm import tqdm
import json

class Wav2Vec2Aligner:

    def __init__(
        self, model_name: str = "nguyenvulebinh/wav2vec2-base-vietnamese-250h"
    ):

        print(f"Loading model: {model_name}")

        try:
            from transformers import Wav2Vec2Processor, Wav2Vec2ForCTC
        except ImportError:
            raise ImportError(
                "transformers not installed. Run: pip install transformers torch torchaudio"
            )

        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        print(f"Using device: {self.device}")

        self.processor = Wav2Vec2Processor.from_pretrained(model_name)
        self.model = Wav2Vec2ForCTC.from_pretrained(model_name).to(self.device)
        self.model.eval()

        print(" Model loaded successfully")

    def align_audio_text(
        self, audio_path: Path, text: str, sample_rate: int = 16000
    ) -> List[Dict[str, Any]]:

        waveform, sr = torchaudio.load(audio_path)

        if sr != sample_rate:
            resampler = torchaudio.transforms.Resample(sr, sample_rate)
            waveform = resampler(waveform)

        if waveform.shape[0] > 1:
            waveform = torch.mean(waveform, dim=0, keepdim=True)

        input_values = self.processor(
            waveform.squeeze().numpy(), sampling_rate=sample_rate, return_tensors="pt"
        ).input_values.to(self.device)

        with torch.no_grad():
            logits = self.model(input_values).logits

        predicted_ids = torch.argmax(logits, dim=-1)

        transcription = self.processor.batch_decode(predicted_ids)[0]

        words = self._align_with_text(
            logits=logits[0],
            reference_text=text,
            audio_duration=waveform.shape[1] / sample_rate,
        )

        return words

    def _align_with_text(
        self, logits: torch.Tensor, reference_text: str, audio_duration: float
    ) -> List[Dict[str, Any]]:

        probs = torch.softmax(logits, dim=-1)
        predicted_ids = torch.argmax(probs, dim=-1)

        tokens = self.processor.tokenizer.convert_ids_to_tokens(predicted_ids)

        cleaned_tokens = []
        cleaned_frames = []

        prev_token = None
        for i, token in enumerate(tokens):

            if token in ["<pad>", "<unk>", "<s>", "</s>"] or token == prev_token:
                continue
            if token.startswith("##"):

                if cleaned_tokens:
                    cleaned_tokens[-1] += token[2:]
                else:
                    cleaned_tokens.append(token[2:])
                    cleaned_frames.append(i)
            else:
                cleaned_tokens.append(token)
                cleaned_frames.append(i)
            prev_token = token

        ref_words = reference_text.upper().split()

        words = []
        frame_per_second = len(tokens) / audio_duration

        if not ref_words:
            return []

        tokens_per_word = len(cleaned_tokens) / len(ref_words) if ref_words else 1

        for i, word in enumerate(ref_words):

            start_token_idx = int(i * tokens_per_word)
            end_token_idx = int((i + 1) * tokens_per_word)

            start_token_idx = min(start_token_idx, len(cleaned_frames) - 1)
            end_token_idx = min(end_token_idx, len(cleaned_frames))

            if start_token_idx < len(cleaned_frames):
                start_frame = cleaned_frames[start_token_idx]
                end_frame = (
                    cleaned_frames[end_token_idx - 1]
                    if end_token_idx > 0
                    else start_frame
                )

                start_time = start_frame / frame_per_second
                end_time = end_frame / frame_per_second

                words.append(
                    {
                        "word": word,
                        "start": round(start_time, 3),
                        "end": round(end_time, 3),
                        "duration": round(end_time - start_time, 3),
                    }
                )

        return words

def align_dataset(
    audio_dir: Path,
    prompts_file: Path,
    output_dir: Path,
    model_name: str = "nguyenvulebinh/wav2vec2-base-vietnamese-250h",
    corpus_type: str = "vivos",
):

    print("Loading prompts...")
    prompts = {}
    with open(prompts_file, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            parts = line.split(" ", 1)
            if len(parts) == 2:
                prompts[parts[0]] = parts[1]

    print(f"Loaded {len(prompts)} transcriptions")

    aligner = Wav2Vec2Aligner(model_name)

    audio_files = list(audio_dir.rglob("*.wav"))
    print(f"Found {len(audio_files)} audio files")

    stats = {"total": len(audio_files), "success": 0, "failed": 0, "skipped": 0}

    output_dir.mkdir(parents=True, exist_ok=True)

    for audio_path in tqdm(audio_files, desc="Aligning"):
        if corpus_type == "common_voice":

            try:
                filename = audio_path.stem
                if not filename.startswith("common_voice_vi_"):

                    continue

                numeric_part = filename.replace("common_voice_vi_", "")

                index = numeric_part[-3:]
                speaker_id = numeric_part[:-3]

                audio_id = f"speaker{speaker_id}_{index}"
            except Exception:

                continue
        else:

            audio_id = audio_path.stem

        if audio_id not in prompts:

            stats["skipped"] += 1
            continue

        try:

            words = aligner.align_audio_text(
                audio_path=audio_path, text=prompts[audio_id]
            )

            speaker_id = audio_path.parent.name
            output_speaker_dir = output_dir / speaker_id
            output_speaker_dir.mkdir(parents=True, exist_ok=True)

            output_file = output_speaker_dir / f"{audio_path.stem}.json"

            with open(output_file, "w", encoding="utf-8") as f:
                json.dump(
                    {"file": audio_path.name, "audio_id": audio_id, "words": words},
                    f,
                    ensure_ascii=False,
                    indent=2,
                )

            stats["success"] += 1

        except Exception as e:
            print(f"\nError processing {audio_id}: {e}")
            stats["failed"] += 1

    print("\n" + "=" * 50)
    print("Alignment Summary")
    print("=" * 50)
    print(f"Total files: {stats['total']}")
    print(f"Successfully aligned: {stats['success']}")
    print(f"Skipped (no prompt): {stats['skipped']}")
    print(f"Failed: {stats['failed']}")
    print(f"Output directory: {output_dir.absolute()}")
    print("=" * 50)

def main():
    parser = argparse.ArgumentParser(
        description="Wav2Vec2-based forced alignment for Vietnamese"
    )

    project_root = Path(__file__).parent.parent

    parser.add_argument(
        "--audio-dir",
        type=Path,
        default=project_root / "data/val/waves",
        help="Directory containing audio files",
    )
    parser.add_argument(
        "--prompts-file",
        type=Path,
        default=project_root / "data/val/prompts.txt",
        help="Path to prompts.txt",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=project_root / "data/val/aligned_wav2vec2",
        help="Output directory for aligned JSON files",
    )
    parser.add_argument(
        "--model",
        type=str,
        default="nguyenvulebinh/wav2vec2-base-vietnamese-250h",
        help="Wav2Vec2 model name from HuggingFace",
    )

    parser.add_argument(
        "--corpus-type",
        type=str,
        choices=["vivos", "common_voice", "lsvsc"],
        default="vivos",
        help="Corpus type (vivos, common_voice, or lsvsc)",
    )

    args = parser.parse_args()

    if not args.audio_dir.exists():
        print(f"Error: Audio directory not found: {args.audio_dir}")
        return 1

    if not args.prompts_file.exists():
        print(f"Error: Prompts file not found: {args.prompts_file}")
        return 1

    align_dataset(
        audio_dir=args.audio_dir,
        prompts_file=args.prompts_file,
        output_dir=args.output_dir,
        model_name=args.model,
        corpus_type=args.corpus_type,
    )

    print("\n Alignment complete!")
    print("\nNext steps:")
    print("  python validate_alignment.py --input-dir", args.output_dir)

    return 0

if __name__ == "__main__":
    exit(main())
import os
import json
import dataclasses
from pathlib import Path
from typing import List, Dict, Optional, Tuple

@dataclasses.dataclass
class AudioSample:
    sample_id: str
    parent_audio_id: str
    audio_path: str
    transcript_path: str
    split_name: str
    source_type: str
    bucket: str
    class_name: str
    group_name: str
    data_name: str
    episode_index: str
    start: float
    end: float
    duration: float
    text: str
    speaker: str = "Unknown"
    overlap_ratio: float = 0.0
    is_robustness: bool = False
    parent_clean_id: Optional[str] = None
    base_utt_ids: List[str] = dataclasses.field(default_factory=list)

def scan_segments(target_dir: Path, data_name: str) -> List[AudioSample]:

    audio_path = target_dir / "mixture.wav"
    transcript_path = target_dir / "labeled" / "transcript.jsonl"

    if not audio_path.exists() or not transcript_path.exists():
        return []

    samples = []
    with open(transcript_path, "r", encoding="utf-8") as f:
        for idx, line in enumerate(f):
            try:
                d = json.loads(line)
                start = float(d.get("start", 0))
                end = float(d.get("end", 0))
                samples.append(AudioSample(
                    sample_id=f"{data_name}_{idx}",
                    parent_audio_id=data_name,
                    audio_path=str(audio_path.absolute()),
                    transcript_path=str(transcript_path.absolute()),
                    split_name="test",
                    source_type="custom",
                    bucket="real",
                    class_name="general",
                    group_name="general",
                    data_name=data_name,
                    episode_index="1",
                    start=start,
                    end=end,
                    duration=max(0, end - start),
                    text=d.get("text", ""),
                    speaker=d.get("speaker", "Unknown")
                ))
            except Exception:
                continue
    return samples

def build_manifest(input_dir: str, output_manifest: str):
    path = Path(input_dir)
    data_name = path.name
    print(f"[V9] Scanning {input_dir} (detected name: {data_name})...")

    samples = scan_segments(path, data_name)
    if not samples:

        for sub in path.iterdir():
            if sub.is_dir():
                samples.extend(scan_segments(sub, sub.name))

    if not samples:
        raise ValueError(f"No valid mixture.wav and transcript.jsonl found in {input_dir}")

    os.makedirs(os.path.dirname(output_manifest), exist_ok=True)
    with open(output_manifest, "w", encoding="utf-8") as f:
        for s in samples:
            f.write(json.dumps(dataclasses.asdict(s), ensure_ascii=False) + "\n")

    print(f"[V9] Created manifest with {len(samples)} samples at {output_manifest}")
    return len(samples)

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    build_manifest(args.input, args.output)
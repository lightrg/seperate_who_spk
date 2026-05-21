
import os
import json
import random
import numpy as np
import soundfile as sf
from pathlib import Path
from typing import List, Dict, Tuple, Optional, Union
from dataclasses import dataclass, asdict
import logging

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)

@dataclass
class Segment:

    speaker_id: str
    audio_path: Path
    duration: float
    segment_id: str = ""
    audio_data: Optional[np.ndarray] = None

    def __post_init__(self):

        if not self.segment_id and self.audio_path:
            self.segment_id = Path(self.audio_path).stem

    def load_audio(self, sample_rate: int = 16000) -> np.ndarray:

        if self.audio_data is None:
            self.audio_data, sr = sf.read(self.audio_path)
            if sr != sample_rate:

                self.audio_data = resample_audio(self.audio_data, sr, sample_rate)
        return self.audio_data

@dataclass
class TimelineEvent:

    start_time: float
    end_time: float
    speaker_id: str
    segment_path: Optional[Path] = None
    event_type: str = "speech"

    @property
    def duration(self) -> float:
        return self.end_time - self.start_time

    def to_rttm(self, file_id: str) -> str:

        if self.event_type == "silence":
            return ""

        return (
            f"SPEAKER {file_id} 1 {self.start_time:.3f} {self.duration:.3f} "
            f"<NA> <NA> {self.speaker_id} <NA> <NA>"
        )

@dataclass
class MeetingMetadata:

    meeting_id: str
    duration: float
    num_speakers: int
    speakers: List[str]
    num_events: int
    silence_ratio: float
    overlap_ratio: float
    single_speaker_ratio: float
    has_noise: bool
    has_reverb: bool
    snr_db: Optional[float] = None
    rt60: Optional[float] = None

    def to_dict(self) -> Dict:
        return asdict(self)

def load_audio(
    path: Union[str, Path], sample_rate: int = 16000
) -> Tuple[np.ndarray, int]:

    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Audio file not found: {path}")

    audio, sr = sf.read(path)

    if len(audio.shape) > 1:
        audio = np.mean(audio, axis=1)

    if sr != sample_rate:
        audio = resample_audio(audio, sr, sample_rate)
        sr = sample_rate

    return audio, sr

def save_audio(
    audio: np.ndarray, path: Union[str, Path], sample_rate: int = 16000
) -> None:

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    max_val = np.max(np.abs(audio))
    if max_val > 1.0:
        audio = audio / max_val * 0.95

    sf.write(path, audio, sample_rate, subtype="PCM_16")
    logger.debug(f"Saved audio: {path}")

def resample_audio(audio: np.ndarray, orig_sr: int, target_sr: int) -> np.ndarray:

    if orig_sr == target_sr:
        return audio

    duration = len(audio) / orig_sr
    target_length = int(duration * target_sr)

    indices = np.linspace(0, len(audio) - 1, target_length)
    resampled = np.interp(indices, np.arange(len(audio)), audio)

    return resampled

def get_audio_duration(path: Union[str, Path]) -> float:

    info = sf.info(path)
    return info.duration

def export_rttm(
    events: List[TimelineEvent], file_id: str, output_path: Union[str, Path]
) -> None:

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    lines = []
    for event in events:
        if event.event_type != "silence":
            line = event.to_rttm(file_id)
            if line:
                lines.append(line)

    lines.sort(key=lambda x: float(x.split()[3]))

    with open(output_path, "w") as f:
        f.write("\n".join(lines) + "\n")

    logger.info(f"Exported RTTM: {output_path} ({len(lines)} segments)")

def parse_rttm(rttm_path: Union[str, Path]) -> List[TimelineEvent]:

    events = []

    with open(rttm_path, "r") as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) >= 8 and parts[0] == "SPEAKER":
                start_time = float(parts[3])
                duration = float(parts[4])
                speaker_id = parts[7]

                events.append(
                    TimelineEvent(
                        start_time=start_time,
                        end_time=start_time + duration,
                        speaker_id=speaker_id,
                        event_type="speech",
                    )
                )

    return events

def validate_rttm(events: List[TimelineEvent], total_duration: float) -> Dict:

    if not events:
        return {"valid": False, "error": "No events"}

    speaker_times = {}
    for event in events:
        if event.speaker_id not in speaker_times:
            speaker_times[event.speaker_id] = 0.0
        speaker_times[event.speaker_id] += event.duration

    overlap_time = calculate_overlap_time(events)

    speech_regions = merge_overlapping_events(events)
    total_speech_time = sum(e.duration for e in speech_regions)
    silence_time = total_duration - total_speech_time

    return {
        "valid": True,
        "total_duration": total_duration,
        "num_events": len(events),
        "num_speakers": len(speaker_times),
        "speaker_times": speaker_times,
        "total_speech_time": total_speech_time,
        "overlap_time": overlap_time,
        "silence_time": silence_time,
        "silence_ratio": silence_time / total_duration,
        "overlap_ratio": overlap_time / total_duration,
    }

def merge_overlapping_events(events: List[TimelineEvent]) -> List[TimelineEvent]:

    if not events:
        return []

    sorted_events = sorted(events, key=lambda x: x.start_time)

    merged = []
    current = TimelineEvent(
        start_time=sorted_events[0].start_time,
        end_time=sorted_events[0].end_time,
        speaker_id="MERGED",
        event_type="speech",
    )

    for event in sorted_events[1:]:
        if event.start_time <= current.end_time:

            current.end_time = max(current.end_time, event.end_time)
        else:

            merged.append(current)
            current = TimelineEvent(
                start_time=event.start_time,
                end_time=event.end_time,
                speaker_id="MERGED",
                event_type="speech",
            )

    merged.append(current)
    return merged

def calculate_overlap_time(events: List[TimelineEvent]) -> float:

    if len(events) < 2:
        return 0.0

    sorted_events = sorted(events, key=lambda x: x.start_time)
    max_time = max(e.end_time for e in events)

    resolution = 0.01
    num_frames = int(max_time / resolution) + 1
    speaker_count = np.zeros(num_frames, dtype=np.int32)

    for event in events:
        start_frame = int(event.start_time / resolution)
        end_frame = int(event.end_time / resolution)
        speaker_count[start_frame:end_frame] += 1

    overlap_frames = np.sum(speaker_count >= 2)
    overlap_time = overlap_frames * resolution

    return overlap_time

def find_gaps(
    events: List[TimelineEvent], total_duration: float, min_gap: float = 0.1
) -> List[Tuple[float, float]]:

    if not events:
        return [(0.0, total_duration)]

    merged = merge_overlapping_events(events)
    sorted_events = sorted(merged, key=lambda x: x.start_time)

    gaps = []

    if sorted_events[0].start_time > min_gap:
        gaps.append((0.0, sorted_events[0].start_time))

    for i in range(len(sorted_events) - 1):
        gap_start = sorted_events[i].end_time
        gap_end = sorted_events[i + 1].start_time
        if gap_end - gap_start >= min_gap:
            gaps.append((gap_start, gap_end))

    if total_duration - sorted_events[-1].end_time > min_gap:
        gaps.append((sorted_events[-1].end_time, total_duration))

    return gaps

def scan_vivos_speaker(vivos_path: Path, speaker_id: str) -> List[Segment]:

    speaker_path = vivos_path / speaker_id
    if not speaker_path.exists():
        logger.warning(f"Speaker path not found: {speaker_path}")
        return []

    segments = []
    for audio_file in speaker_path.glob("*.wav"):
        try:
            duration = get_audio_duration(audio_file)
            segments.append(
                Segment(speaker_id=speaker_id, audio_path=audio_file, duration=duration)
            )
        except Exception as e:
            logger.warning(f"Error loading {audio_file}: {e}")

    logger.info(f"Found {len(segments)} segments for {speaker_id}")
    return segments

def scan_all_speakers(
    vivos_path: Path, speaker_ids: List[str]
) -> Dict[str, List[Segment]]:

    all_segments = {}

    for speaker_id in speaker_ids:
        segments = scan_vivos_speaker(vivos_path, speaker_id)
        if segments:
            all_segments[speaker_id] = segments

    total = sum(len(segs) for segs in all_segments.values())
    logger.info(f"Total: {total} segments from {len(all_segments)} speakers")

    return all_segments

def set_random_seed(seed: int) -> None:

    random.seed(seed)
    np.random.seed(seed)
    logger.info(f"Random seed set to {seed}")

def random_choice_weighted(items: List, weights: List[float]) -> any:

    total = sum(weights)
    weights = [w / total for w in weights]
    return random.choices(items, weights=weights, k=1)[0]

def random_range(min_val: float, max_val: float) -> float:

    return random.uniform(min_val, max_val)

def ensure_dir(path: Union[str, Path]) -> Path:

    path = Path(path)
    path.mkdir(parents=True, exist_ok=True)
    return path

def generate_meeting_id() -> str:

    import uuid
    from datetime import datetime

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    short_uuid = str(uuid.uuid4())[:8]
    return f"meeting_{timestamp}_{short_uuid}"

_alignment_cache: Optional[Dict[str, Dict]] = None
_prompts_cache: Optional[Dict[str, str]] = None

def _get_data_base_path() -> Path:

    import platform

    if platform.system() == "Windows":
        return Path("D:/cachyos/du_an1/data")
    else:
        linux_paths = [
            Path("/mnt/sda1/du_an1/data"),
            Path("/run/media/phuong/HDD/cachyos/du_an1/data"),
            Path("/mnt/Data/cachyos/du_an1/data"),
            Path("/mnt/d/cachyos/du_an1/data"),
            Path.home() / "cachyos/du_an1/data",
        ]
        for p in linux_paths:
            if p.exists():
                return p
        return Path("/mnt/sda1/du_an1/data")

def load_alignment_data(alignment_dir: Union[str, Path] = None) -> Dict[str, Dict]:

    global _alignment_cache

    if _alignment_cache is not None:
        return _alignment_cache

    alignments = {}

    if alignment_dir is not None:

        dirs_to_scan = [Path(alignment_dir)]
    else:

        base = _get_data_base_path()
        dirs_to_scan = [
            base / "vivos" / "aligned_wav2vec2",
            base / "LSVSC" / "aligned_wav2vec2",
            base / "cv_corpus" / "aligned_wav2vec2",
        ]

    for adir in dirs_to_scan:
        if not adir.exists():
            logger.warning(f"Alignment directory not found: {adir}")
            continue

        try:
            json_files = list(adir.rglob("*.json"))
            count = 0
            for json_file in json_files:
                try:
                    with open(json_file, "r", encoding="utf-8") as f:
                        data = json.load(f)
                        segment_id = data.get("file", json_file.stem)
                        if segment_id.endswith(".wav"):
                            segment_id = segment_id[:-4]
                        alignments[segment_id] = data
                        count += 1
                except Exception as e:
                    logger.warning(f"Failed to load {json_file}: {e}")

            logger.info(f"Loaded {count} alignments from {adir}")
        except Exception as e:
            logger.error(f"Failed to scan alignments in {adir}: {e}")

    logger.info(f"Total alignments loaded: {len(alignments)}")
    _alignment_cache = alignments
    return alignments

def load_prompts(prompts_path: Union[str, Path] = None) -> Dict[str, str]:

    global _prompts_cache

    if _prompts_cache is not None:
        return _prompts_cache

    alignments = load_alignment_data()
    prompts = {}
    if alignments:
        for segment_id, data in alignments.items():
            words = data.get("words", [])
            if words:
                text = " ".join(w["word"] for w in words)
                prompts[segment_id] = text
        logger.info(f"Loaded {len(prompts)} prompts from alignment data")

    if prompts_path is not None:
        prompts_files = [Path(prompts_path)]
    else:
        base = _get_data_base_path()
        prompts_files = [
            base / "vivos" / "prompts.txt",
            base / "LSVSC" / "prompts.txt",
            base / "cv_corpus" / "prompts.txt",
        ]

    for pf in prompts_files:
        if not pf.exists():
            logger.warning(f"Prompts file not found: {pf}")
            continue

        try:
            count = 0
            with open(pf, "r", encoding="utf-8") as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    if "\t" in line:
                        parts = line.split("\t", 1)
                    else:
                        parts = line.split(" ", 1)

                    if len(parts) == 2:
                        segment_id, text = parts
                        segment_id = segment_id.strip()

                        if segment_id.startswith("speaker") and "_" in segment_id:
                            spk, num = segment_id.split("_", 1)
                            spk_id = spk.replace("speaker", "")
                            segment_id = f"common_voice_vi_{spk_id}{num}"

                        prompts[segment_id] = text.strip()
                        count += 1
            logger.info(f"Loaded {count} prompts from {pf}")
        except Exception as e:
            logger.error(f"Failed to load prompts from {pf}: {e}")

    _prompts_cache = prompts
    logger.info(f"Total prompts loaded: {len(prompts)}")
    return prompts

def get_alignment(segment_id: str, alignments: Dict[str, Dict] = None) -> Dict:

    if alignments is None:
        alignments = load_alignment_data()

    return alignments.get(segment_id, {})

def get_words(segment_id: str, alignments: Dict[str, Dict] = None) -> List[Dict]:

    alignment = get_alignment(segment_id, alignments)
    return alignment.get("words", [])

def get_transcript(segment_id: str, prompts: Dict[str, str] = None) -> str:

    if prompts is None:
        prompts = load_prompts()

    return prompts.get(segment_id, "")

def get_word_text(segment_id: str, alignments: Dict[str, Dict] = None) -> str:

    words = get_words(segment_id, alignments)
    return " ".join(w["word"] for w in words)

def save_json(data: Dict, path: Union[str, Path]) -> None:

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)

    logger.debug(f"Saved JSON: {path}")

def load_json(path: Union[str, Path]) -> Dict:

    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)

def format_duration(seconds: float) -> str:

    hours = int(seconds // 3600)
    minutes = int((seconds % 3600) // 60)
    secs = seconds % 60

    if hours > 0:
        return f"{hours:02d}:{minutes:02d}:{secs:05.2f}"
    else:
        return f"{minutes:02d}:{secs:05.2f}"

def print_progress(current: int, total: int, prefix: str = "") -> None:

    bar_length = 40
    progress = current / total
    filled = int(bar_length * progress)
    bar = "█" * filled + "░" * (bar_length - filled)
    print(f"\r{prefix} |{bar}| {current}/{total} ({progress*100:.1f}%)", end="")
    if current == total:
        print()

if __name__ == "__main__":

    print("Testing utils.py...")

    event = TimelineEvent(
        start_time=1.5, end_time=4.2, speaker_id="VIVOSSPK04", event_type="speech"
    )
    print(f"Event duration: {event.duration:.2f}s")
    print(f"RTTM line: {event.to_rttm('meeting_001')}")

    print(f"Format 3661.5s: {format_duration(3661.5)}")
    print(f"Format 125.3s: {format_duration(125.3)}")

    events = [
        TimelineEvent(0, 5, "A", event_type="speech"),
        TimelineEvent(3, 8, "B", event_type="speech"),
        TimelineEvent(10, 15, "A", event_type="speech"),
    ]
    overlap = calculate_overlap_time(events)
    print(f"Overlap time: {overlap:.2f}s (expected ~2s)")

    gaps = find_gaps(events, total_duration=20.0)
    print(f"Gaps: {gaps}")

    print("\n✅ All tests passed!")
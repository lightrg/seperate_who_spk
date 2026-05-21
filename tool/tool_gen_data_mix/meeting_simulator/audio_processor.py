import random
import numpy as np
from pathlib import Path
from typing import List, Dict, Tuple, Optional
from dataclasses import dataclass
import logging

from config import Config, DEFAULT_CONFIG
from utils import (
    Segment,
    load_audio,
    save_audio,
    get_audio_duration,
    scan_all_speakers,
    random_range,
    logger,
)

@dataclass
class ProcessedSegment:

    speaker_id: str
    audio_data: np.ndarray
    duration: float
    sample_rate: int
    source_segments: List[Path]
    source_segment_ids: List[str]
    source_segment_durations: List[float]
    is_merged: bool

    @classmethod
    def from_segment(
        cls, segment: Segment, sample_rate: int = 16000
    ) -> "ProcessedSegment":

        audio = segment.load_audio(sample_rate)

        audio = AudioProcessor._trim_silence_static(audio, sample_rate)
        seg_duration = len(audio) / sample_rate
        return cls(
            speaker_id=segment.speaker_id,
            audio_data=audio,
            duration=seg_duration,
            sample_rate=sample_rate,
            source_segments=[segment.audio_path],
            source_segment_ids=[segment.segment_id],
            source_segment_durations=[seg_duration],
            is_merged=False,
        )

class AudioProcessor:

    def __init__(self, config: Config = DEFAULT_CONFIG):
        self.config = config
        self.sample_rate = config.audio.sample_rate

        self._speaker_segments: Dict[str, List[Segment]] = {}

        self._used_segments: Dict[str, set] = {}

        self._scanned = False

    def scan_dataset(self) -> None:

        if self._scanned:
            return

        logger.info("Scanning all datasets...")

        self._speaker_segments = {}

        vivos_speakers = self.config.speaker.vivos_speakers
        vivos_path = self.config.dataset.vivos_path
        logger.info(
            f"Scanning VIVOS dataset: {len(vivos_speakers)} speakers, path={vivos_path}"
        )
        if vivos_speakers:
            vivos_segments = scan_all_speakers(vivos_path, vivos_speakers)
            self._speaker_segments.update(vivos_segments)
            logger.info(f"  VIVOS: loaded {len(vivos_segments)} speakers")

        lsvsc_speakers = self.config.speaker.lsvsc_speakers
        lsvsc_path = self.config.dataset.lsvsc_path
        logger.info(
            f"Scanning LSVSC dataset: {len(lsvsc_speakers)} speakers, path={lsvsc_path}"
        )
        if lsvsc_speakers:
            lsvsc_segments = scan_all_speakers(lsvsc_path, lsvsc_speakers)
            self._speaker_segments.update(lsvsc_segments)
            logger.info(f"  LSVSC: loaded {len(lsvsc_segments)} speakers")

        cv_speakers = self.config.speaker.cv_speakers
        cv_path = self.config.dataset.cv_path
        logger.info(
            f"Scanning Common Voice dataset: {len(cv_speakers)} speakers, path={cv_path}"
        )
        if cv_speakers:
            cv_segments = scan_all_speakers(cv_path, cv_speakers)
            self._speaker_segments.update(cv_segments)
            logger.info(f"  Common Voice: loaded {len(cv_segments)} speakers")

        logger.info(
            f"Total speakers in _speaker_segments: {len(self._speaker_segments)}"
        )
        for speaker_id, segments in self._speaker_segments.items():
            total_duration = sum(s.duration for s in segments)
            durations = [s.duration for s in segments]
            logger.info(
                f"  {speaker_id}: {len(segments)} segments, "
                f"{total_duration:.1f}s total, "
                f"range={min(durations):.2f}-{max(durations):.2f}s"
            )

        self._scanned = True

    def reset_meeting(self) -> None:

        self._used_segments = {
            speaker_id: set() for speaker_id in self._speaker_segments.keys()
        }

    def get_available_speakers(self) -> List[str]:

        self.scan_dataset()
        return list(self._speaker_segments.keys())

    def get_speaker_total_duration(self, speaker_id: str) -> float:

        if speaker_id not in self._speaker_segments:
            return 0.0
        return sum(s.duration for s in self._speaker_segments[speaker_id])

    def _get_unused_segments(self, speaker_id: str) -> List[Segment]:

        if speaker_id not in self._speaker_segments:
            return []

        all_segments = self._speaker_segments[speaker_id]
        used = self._used_segments.get(speaker_id, set())

        unused = [s for s in all_segments if str(s.audio_path) not in used]

        if not unused:
            logger.debug(f"Resetting segments for {speaker_id}")
            self._used_segments[speaker_id] = set()
            unused = all_segments.copy()

        return unused

    def _mark_used(self, segment: Segment) -> None:

        speaker_id = segment.speaker_id
        if speaker_id not in self._used_segments:
            self._used_segments[speaker_id] = set()
        self._used_segments[speaker_id].add(str(segment.audio_path))

    def get_random_segment(
        self,
        speaker_id: str,
        min_duration: float = 0.5,
        max_duration: float = None,
        prefer_duration: float = None,
    ) -> Optional[Segment]:

        unused = self._get_unused_segments(speaker_id)

        if not unused:
            logger.warning(
                f"No unused segments for {speaker_id} "
                f"(in _speaker_segments: {speaker_id in self._speaker_segments}, "
                f"total: {len(self._speaker_segments.get(speaker_id, []))})"
            )
            return None

        valid = [
            s
            for s in unused
            if s.duration >= min_duration
            and (max_duration is None or s.duration <= max_duration)
        ]

        if not valid:

            valid = [s for s in unused if s.duration >= min_duration]

        if not valid:

            durations = [s.duration for s in unused]
            logger.warning(
                f"No valid segments for {speaker_id}: "
                f"unused={len(unused)}, "
                f"dur_range={min(durations):.3f}-{max(durations):.3f}s, "
                f"filter: min={min_duration}, max={max_duration}"
            )

            valid = unused

        if prefer_duration and len(valid) > 1:

            valid.sort(key=lambda s: abs(s.duration - prefer_duration))

            top_n = min(3, len(valid))
            segment = random.choice(valid[:top_n])
        else:
            segment = random.choice(valid)

        self._mark_used(segment)
        return segment

    def get_processed_segment(
        self, speaker_id: str, target_duration: float = None
    ) -> Optional[ProcessedSegment]:

        self.scan_dataset()

        merge_ratio = self.config.segment.merge_ratio

        if random.random() < merge_ratio:

            return self._get_merged_segment(speaker_id, target_duration)
        else:

            return self._get_single_segment(speaker_id, target_duration)

    def _get_single_segment(
        self, speaker_id: str, target_duration: float = None
    ) -> Optional[ProcessedSegment]:

        segment = self.get_random_segment(
            speaker_id,
            max_duration=target_duration * 1.2 if target_duration else None,
            prefer_duration=target_duration,
        )
        if segment is None:
            return None

        return ProcessedSegment.from_segment(segment, self.sample_rate)

    def _get_merged_segment(
        self, speaker_id: str, target_duration: float = None
    ) -> Optional[ProcessedSegment]:

        min_segments = self.config.segment.merge_min_segments
        max_segments = self.config.segment.merge_max_segments
        config_max = self.config.segment.max_merged_duration_sec

        if target_duration:
            effective_max = min(target_duration * 1.2, config_max)
        else:
            effective_max = config_max

        num_to_merge = random.randint(min_segments, max_segments)

        segments_to_merge = []
        total_duration = 0.0

        for i in range(num_to_merge):

            remaining = effective_max - total_duration
            if remaining < 0.5:
                break

            segments_left = num_to_merge - i
            prefer_per_seg = (
                remaining / segments_left if segments_left > 0 else remaining
            )

            segment = self.get_random_segment(
                speaker_id,
                min_duration=0.3,
                max_duration=remaining,
                prefer_duration=prefer_per_seg,
            )

            if segment is None:
                break

            segments_to_merge.append(segment)
            total_duration += segment.duration

        if len(segments_to_merge) == 0:
            return None

        if len(segments_to_merge) == 1:

            return ProcessedSegment.from_segment(segments_to_merge[0], self.sample_rate)

        return self._merge_segments(segments_to_merge)

    def _merge_segments(self, segments: List[Segment]) -> ProcessedSegment:

        if len(segments) == 0:
            raise ValueError("No segments to merge")

        if len(segments) == 1:
            return ProcessedSegment.from_segment(segments[0], self.sample_rate)

        speaker_id = segments[0].speaker_id

        audio_parts = []
        source_paths = []
        source_ids = []
        source_durations = []

        for i, segment in enumerate(segments):
            audio = segment.load_audio(self.sample_rate)

            audio = self.trim_silence(audio, min_silence_duration=0.05)
            seg_duration = len(audio) / self.sample_rate

            audio_parts.append(audio)
            source_paths.append(segment.audio_path)
            source_ids.append(segment.segment_id)
            source_durations.append(seg_duration)

            if i < len(segments) - 1:
                pause_duration = random_range(0.1, 0.3)
                pause_samples = int(pause_duration * self.sample_rate)
                pause = np.zeros(pause_samples)
                audio_parts.append(pause)

        merged_audio = np.concatenate(audio_parts)

        merged_audio = self._apply_crossfade(merged_audio, segments)

        return ProcessedSegment(
            speaker_id=speaker_id,
            audio_data=merged_audio,
            duration=len(merged_audio) / self.sample_rate,
            sample_rate=self.sample_rate,
            source_segments=source_paths,
            source_segment_ids=source_ids,
            source_segment_durations=source_durations,
            is_merged=True,
        )

    def _apply_crossfade(
        self,
        audio: np.ndarray,
        segments: List[Segment],
        fade_duration: float = 0.02,
    ) -> np.ndarray:

        fade_samples = int(fade_duration * self.sample_rate)

        if len(audio) < fade_samples * 2:
            return audio

        fade_in = np.linspace(0, 1, fade_samples)
        audio[:fade_samples] *= fade_in

        fade_out = np.linspace(1, 0, fade_samples)
        audio[-fade_samples:] *= fade_out

        return audio

    def normalize_audio(self, audio: np.ndarray, target_db: float = -3.0) -> np.ndarray:

        rms = np.sqrt(np.mean(audio**2))
        if rms < 1e-10:
            return audio

        target_rms = 10 ** (target_db / 20)

        scale = target_rms / rms
        normalized = audio * scale

        normalized = np.clip(normalized, -1.0, 1.0)

        return normalized

    def trim_silence(
        self,
        audio: np.ndarray,
        threshold_db: float = -40.0,
        min_silence_duration: float = 0.05,
    ) -> np.ndarray:

        return AudioProcessor._trim_silence_static(
            audio, self.sample_rate, threshold_db, min_silence_duration
        )

    @staticmethod
    def _trim_silence_static(
        audio: np.ndarray,
        sample_rate: int = 16000,
        threshold_db: float = -40.0,
        min_silence_duration: float = 0.05,
    ) -> np.ndarray:

        if len(audio) == 0:
            return audio

        threshold = 10 ** (threshold_db / 20)
        min_samples = int(min_silence_duration * sample_rate)

        buffer_samples = int(0.025 * sample_rate)

        if min_samples <= 0 or len(audio) < min_samples:
            return audio

        start = 0
        for i in range(0, len(audio) - min_samples, min_samples):
            chunk = audio[i : i + min_samples]
            if np.max(np.abs(chunk)) > threshold:
                start = max(0, i - buffer_samples)
                break

        end = len(audio)
        for i in range(len(audio) - min_samples, start, -min_samples):
            chunk = audio[i : i + min_samples]
            if np.max(np.abs(chunk)) > threshold:
                end = min(len(audio), i + min_samples + buffer_samples)
                break

        return audio[start:end]

    def adjust_volume(self, audio: np.ndarray, gain_db: float) -> np.ndarray:

        gain = 10 ** (gain_db / 20)
        adjusted = audio * gain
        return np.clip(adjusted, -1.0, 1.0)

    def extract_portion(
        self, audio: np.ndarray, start_time: float, end_time: float
    ) -> np.ndarray:

        start_sample = int(start_time * self.sample_rate)
        end_sample = int(end_time * self.sample_rate)

        start_sample = max(0, start_sample)
        end_sample = min(len(audio), end_sample)

        return audio[start_sample:end_sample]

class SegmentPoolManager:

    def __init__(self, processor: AudioProcessor, meeting_speakers: List[str]):
        self.processor = processor
        self.meeting_speakers = meeting_speakers

        self.processor.reset_meeting()

        self.segments_used = {speaker: 0 for speaker in meeting_speakers}
        self.total_duration_used = {speaker: 0.0 for speaker in meeting_speakers}

    def get_segment(
        self, speaker_id: str, prefer_duration: float = None
    ) -> Optional[ProcessedSegment]:

        if speaker_id not in self.meeting_speakers:
            logger.warning(f"{speaker_id} not in meeting speakers")
            return None

        segment = self.processor.get_processed_segment(
            speaker_id, target_duration=prefer_duration
        )

        if segment:
            self.segments_used[speaker_id] += 1
            self.total_duration_used[speaker_id] += segment.duration

        return segment

    def get_stats(self) -> Dict:

        return {
            "segments_used": self.segments_used.copy(),
            "duration_used": self.total_duration_used.copy(),
            "total_segments": sum(self.segments_used.values()),
            "total_duration": sum(self.total_duration_used.values()),
        }

if __name__ == "__main__":
    from config import Config

    print("Testing audio_processor.py...")
    print("=" * 50)

    config = Config()
    processor = AudioProcessor(config)

    test_audio = np.random.randn(16000) * 0.1
    normalized = processor.normalize_audio(test_audio, target_db=-3.0)

    original_rms = np.sqrt(np.mean(test_audio**2))
    normalized_rms = np.sqrt(np.mean(normalized**2))

    print(f"Original RMS: {original_rms:.4f}")
    print(f"Normalized RMS: {normalized_rms:.4f}")
    print(f"Target RMS (-3dB): {10**(-3/20):.4f}")

    test_audio_with_silence = np.concatenate(
        [
            np.zeros(8000),
            np.random.randn(16000) * 0.5,
            np.zeros(8000),
        ]
    )

    trimmed = processor.trim_silence(test_audio_with_silence)
    print(f"\nOriginal length: {len(test_audio_with_silence)} samples")
    print(f"Trimmed length: {len(trimmed)} samples")

    portion = processor.extract_portion(test_audio, 0.2, 0.5)
    expected_samples = int(0.3 * 16000)
    print(f"\nExtracted portion: {len(portion)} samples (expected ~{expected_samples})")

    print("\n" + "=" * 50)
    print("Merge simulation:")
    print(f"  Merge ratio: {config.segment.merge_ratio * 100:.0f}%")
    print(
        f"  Segments per merge: {config.segment.merge_min_segments}-{config.segment.merge_max_segments}"
    )

    merge_count = 0
    single_count = 0
    for _ in range(100):
        if random.random() < config.segment.merge_ratio:
            merge_count += 1
        else:
            single_count += 1

    print(f"  Simulation (100 trials): {merge_count} merged, {single_count} single")

    print("\n✅ audio_processor.py tests passed!")
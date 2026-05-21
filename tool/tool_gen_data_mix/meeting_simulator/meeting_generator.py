import random
import numpy as np
from pathlib import Path
from typing import List, Dict, Tuple, Optional
from dataclasses import dataclass, field
from datetime import datetime
import json
import time

from config import Config, DEFAULT_CONFIG
from utils import (
    TimelineEvent,
    save_audio,
    format_duration,
    ensure_dir,
    generate_meeting_id,
    logger,
)
from audio_processor import AudioProcessor, ProcessedSegment, SegmentPoolManager
from room_simulator import RoomSimulator, create_room_for_meeting
from timeline_generator import TimelineGenerator, TimeSlot, EventType
from noise_augmentor import NoiseAugmentor, add_noise_to_meeting

@dataclass
class MeetingSegment:

    speaker_id: str
    audio: np.ndarray
    start_time: float
    end_time: float
    source_files: List[Path] = field(default_factory=list)
    source_segment_ids: List[str] = field(
        default_factory=list
    )
    source_segment_durations: List[float] = field(
        default_factory=list
    )
    is_overlap: bool = False
    overlap_with: Optional[str] = None

    @property
    def duration(self) -> float:
        return self.end_time - self.start_time

@dataclass
class GeneratedMeeting:

    meeting_id: str
    audio: np.ndarray
    sample_rate: int
    duration: float

    speakers: List[str]
    timeline_events: List[TimelineEvent]
    segments: List[MeetingSegment]

    room_config: Optional[Dict] = None
    noise_config: Optional[Dict] = None

    statistics: Dict = field(default_factory=dict)

    def get_rttm_content(self) -> str:

        lines = []
        for event in self.timeline_events:

            line = (
                f"SPEAKER {self.meeting_id} 1 "
                f"{event.start_time:.3f} {event.duration:.3f} "
                f"<NA> <NA> {event.speaker_id} <NA> <NA>"
            )
            lines.append(line)
        return "\n".join(lines)

    def get_asr_content(self, prompts: Dict[str, str] = None) -> str:

        from utils import load_alignment_data, get_words, load_prompts, get_transcript

        alignments = load_alignment_data()

        if prompts is None:
            prompts = load_prompts()

        entries = []

        for seg in self.segments:
            if not seg.source_segment_ids:
                continue

            num_sources = len(seg.source_segment_ids)
            slot_duration = seg.end_time - seg.start_time
            seg_duration = slot_duration / num_sources

            current_time = seg.start_time

            for i, seg_id in enumerate(seg.source_segment_ids):

                words = get_words(seg_id, alignments) if alignments else []

                if words:

                    source_duration = words[-1]["end"] if words else 0

                    for word_data in words:

                        if source_duration > 0:
                            rel_start = word_data["start"] / source_duration
                            rel_end = word_data["end"] / source_duration
                        else:
                            rel_start = 0
                            rel_end = 1

                        word_start = current_time + (rel_start * seg_duration)
                        word_end = current_time + (rel_end * seg_duration)

                        word_start = max(seg.start_time, min(word_start, seg.end_time))
                        word_end = max(word_start, min(word_end, seg.end_time))

                        is_word_overlap = False
                        for other_seg in self.segments:
                            if other_seg.speaker_id == seg.speaker_id:
                                continue

                            ov_start = max(word_start, other_seg.start_time)
                            ov_end = min(word_end, other_seg.end_time)

                            if ov_end - ov_start > 0.05:
                                is_word_overlap = True
                                break

                        entries.append(
                            {
                                "start": word_start,
                                "end": word_end,
                                "speaker": seg.speaker_id,
                                "text": word_data["word"],
                                "is_overlap": is_word_overlap,
                            }
                        )

                else:

                    text = get_transcript(seg_id, prompts) if prompts else ""
                    if not text:

                        logger.debug(f"Missing text for segment {seg_id}")
                        text = "<UNTRANSCRIBED>"

                    end_time = min(current_time + seg_duration, seg.end_time)

                    is_entry_overlap = False
                    for other_seg in self.segments:
                        if other_seg.speaker_id == seg.speaker_id:
                            continue

                        ov_start = max(current_time, other_seg.start_time)
                        ov_end = min(end_time, other_seg.end_time)

                        if ov_end - ov_start > 0.05:
                            is_entry_overlap = True
                            break

                    entries.append(
                        {
                            "start": current_time,
                            "end": end_time,
                            "speaker": seg.speaker_id,
                            "text": text,
                            "is_overlap": is_entry_overlap,
                        }
                    )

                current_time += seg_duration

        entries.sort(key=lambda x: (x["start"], x["speaker"]))

        lines = []
        for e in entries:
            overlap_tag = "[OVERLAP] " if e["is_overlap"] else ""
            line = f"{e['start']:.3f} {e['end']:.3f} {e['speaker']} {overlap_tag}{e['text']}"
            lines.append(line)

        return "\n".join(lines)

    def _get_segment_text(self, seg) -> str:

        from utils import load_alignment_data, get_words, load_prompts, get_transcript

        alignments = load_alignment_data()
        prompts = load_prompts()

        segment_words = []
        for seg_id in seg.source_segment_ids:
            words = get_words(seg_id, alignments) if alignments else []
            if words:
                segment_words.extend([w["word"] for w in words])
            else:
                text = get_transcript(seg_id, prompts) if prompts else ""
                if text:
                    segment_words.extend(text.split())
                else:
                    logger.debug(f"Missing text for segment {seg_id}")
                    segment_words.append("<UNTRANSCRIBED>")

        if not segment_words:
            segment_words.append("<UNTRANSCRIBED>")

        return " ".join(segment_words).upper()

    def get_stm_content(self) -> str:

        from utils import load_alignment_data, get_words, load_prompts, get_transcript

        alignments = load_alignment_data()
        prompts = load_prompts()

        seg_word_times = {}

        for seg in self.segments:
            if not seg.source_segment_ids:
                continue

            num_sources = len(seg.source_segment_ids)
            slot_duration = seg.end_time - seg.start_time
            per_source_duration = slot_duration / num_sources
            current_time = seg.start_time

            word_list = []

            for seg_id in seg.source_segment_ids:
                words = get_words(seg_id, alignments) if alignments else []

                if words:

                    source_duration = words[-1]["end"] if words else 0

                    for word_data in words:
                        if source_duration > 0:
                            rel_start = word_data["start"] / source_duration
                            rel_end = word_data["end"] / source_duration
                        else:
                            rel_start = 0
                            rel_end = 1

                        abs_start = current_time + (rel_start * per_source_duration)
                        abs_end = current_time + (rel_end * per_source_duration)

                        abs_start = max(seg.start_time, min(abs_start, seg.end_time))
                        abs_end = max(abs_start, min(abs_end, seg.end_time))

                        word_list.append(
                            (abs_start, abs_end, word_data["word"].upper())
                        )
                else:

                    text = get_transcript(seg_id, prompts) if prompts else ""
                    if not text:
                        text = "<UNTRANSCRIBED>"

                    words_split = text.split()
                    if words_split:
                        word_dur = per_source_duration / len(words_split)
                        wt = current_time
                        for w in words_split:
                            word_list.append((wt, wt + word_dur, w.upper()))
                            wt += word_dur

                current_time += per_source_duration

            if word_list:
                seg_word_times[id(seg)] = word_list

        change_points = set()
        active_segs = []

        for seg in self.segments:
            if id(seg) not in seg_word_times:
                continue
            active_segs.append(seg)
            change_points.add(seg.start_time)
            change_points.add(seg.end_time)

        if not change_points:
            return ""

        sorted_points = sorted(change_points)

        lines = []
        for i in range(len(sorted_points) - 1):
            interval_start = sorted_points[i]
            interval_end = sorted_points[i + 1]

            if interval_end - interval_start < 0.01:
                continue

            active_speakers = []
            seen_speakers = set()
            for seg in active_segs:

                if seg.start_time <= interval_start and seg.end_time >= interval_end:
                    if seg.speaker_id not in seen_speakers:
                        active_speakers.append(seg)
                        seen_speakers.add(seg.speaker_id)

            if not active_speakers:
                continue

            def _get_words_in_interval(seg, iv_start, iv_end):

                words_in = []
                for w_start, w_end, w_text in seg_word_times[id(seg)]:

                    w_mid = (w_start + w_end) / 2.0
                    if iv_start <= w_mid < iv_end:
                        words_in.append(w_text)
                return " ".join(words_in) if words_in else ""

            if len(active_speakers) == 1:

                seg = active_speakers[0]
                transcript = _get_words_in_interval(seg, interval_start, interval_end)
                if not transcript:
                    continue

                speaker_label = seg.speaker_id

                line = (
                    f"{self.meeting_id} 1 {speaker_label} "
                    f"{interval_start:.3f} {interval_end:.3f} {transcript}"
                )
                lines.append(line)
            else:

                active_speakers.sort(key=lambda s: s.speaker_id)

                for seg in active_speakers:
                    seg_text = _get_words_in_interval(seg, interval_start, interval_end)
                    if seg_text:

                        line = (
                            f"{self.meeting_id} 1 {seg.speaker_id} "
                            f"{interval_start:.3f} {interval_end:.3f} {seg_text}"
                        )
                        lines.append(line)

        lines.sort(key=lambda x: (float(x.split()[3]), x.split()[2]))

        return "\n".join(lines)

    def get_ctm_content(self) -> str:

        from utils import load_alignment_data, get_words, load_prompts, get_transcript

        alignments = load_alignment_data()
        prompts = load_prompts()

        entries = []

        for seg in self.segments:
            if not seg.source_segment_ids:
                continue

            num_sources = len(seg.source_segment_ids)
            slot_duration = seg.end_time - seg.start_time
            seg_duration = slot_duration / num_sources

            current_time = seg.start_time

            for seg_id in seg.source_segment_ids:

                words = get_words(seg_id, alignments) if alignments else []

                if words:

                    source_duration = words[-1]["end"] if words else 0

                    for word_data in words:

                        if source_duration > 0:
                            rel_start = word_data["start"] / source_duration
                            rel_end = word_data["end"] / source_duration
                        else:
                            rel_start = 0
                            rel_end = 1

                        word_start = current_time + (rel_start * seg_duration)
                        word_end = current_time + (rel_end * seg_duration)

                        word_start = max(seg.start_time, min(word_start, seg.end_time))
                        word_end = max(word_start, min(word_end, seg.end_time))

                        word_duration = word_end - word_start

                        entries.append(
                            {
                                "start": word_start,
                                "duration": word_duration,
                                "word": word_data["word"].upper(),
                            }
                        )

                else:

                    text = get_transcript(seg_id, prompts) if prompts else ""
                    if text:
                        words_list = text.split()
                        if words_list:

                            word_duration = seg_duration / len(words_list)
                            word_time = current_time

                            for word in words_list:
                                entries.append(
                                    {
                                        "start": word_time,
                                        "duration": word_duration,
                                        "word": word.upper(),
                                    }
                                )
                                word_time += word_duration

                current_time += seg_duration

        entries.sort(key=lambda x: x["start"])

        lines = []
        for e in entries:

            line = (
                f"{self.meeting_id} 1 {e['start']:.2f} {e['duration']:.2f} {e['word']}"
            )
            lines.append(line)

        return "\n".join(lines)

    def get_metadata(self) -> Dict:

        segment_details = []
        for idx, seg in enumerate(self.segments, 1):
            segment_details.append(
                {
                    "segment_id": idx,
                    "start_time": round(seg.start_time, 3),
                    "end_time": round(seg.end_time, 3),
                    "duration": round(seg.duration, 3),
                    "speaker_id": seg.speaker_id,
                }
            )

        return {
            "meeting_id": self.meeting_id,
            "duration": self.duration,
            "sample_rate": self.sample_rate,
            "num_speakers": len(self.speakers),
            "speakers": self.speakers,
            "num_segments": len(self.segments),
            "room_config": self.room_config,
            "noise_config": self.noise_config,
            "statistics": self.statistics,
            "generated_at": datetime.now().isoformat(),
            "segments": segment_details,
        }

class MeetingGenerator:

    def __init__(self, config: Config = DEFAULT_CONFIG):
        self.config = config
        self.sample_rate = config.audio.sample_rate

        logger.info("Initializing MeetingGenerator...")

        self.audio_processor = AudioProcessor(config)
        self.audio_processor.scan_dataset()

        self.timeline_generator = TimelineGenerator(config)

        self.noise_augmentor = NoiseAugmentor(config)

        self.room_simulator: Optional[RoomSimulator] = None

        self.segment_pool: Optional[SegmentPoolManager] = None

        logger.info("MeetingGenerator initialized")

    def generate(
        self,
        duration: float = None,
        num_speakers: int = None,
        seed: Optional[int] = None,
        speaker_ids: List[str] = None,
    ) -> GeneratedMeeting:

        start_time = time.time()

        if seed is not None:
            random.seed(seed)
            np.random.seed(seed)

        if duration is None:
            min_dur = self.config.meeting.min_duration_minutes * 60
            max_dur = self.config.meeting.max_duration_minutes * 60
            duration = random.uniform(min_dur, max_dur)

        if num_speakers is None:
            num_speakers = random.randint(
                self.config.meeting.min_speakers, self.config.meeting.max_speakers
            )

        meeting_id = generate_meeting_id()

        logger.info(f"=" * 60)
        logger.info(f"Generating meeting: {meeting_id}")
        logger.info(f"  Duration: {format_duration(duration)}")
        logger.info(f"  Speakers: {num_speakers}")

        if speaker_ids is not None:

            speakers = speaker_ids[:num_speakers]
            if len(speakers) < num_speakers:
                logger.warning(
                    f"Not enough speaker_ids provided ({len(speakers)} < {num_speakers})"
                )
        else:

            available_speakers = self.audio_processor.get_available_speakers()
            if len(available_speakers) < num_speakers:
                num_speakers = len(available_speakers)

            speakers = random.sample(available_speakers, num_speakers)

        logger.info(f"  Selected speakers: {speakers}")

        self.segment_pool = SegmentPoolManager(self.audio_processor, speakers)

        self.room_simulator = create_room_for_meeting(speakers, self.config)
        room_config = (
            self.room_simulator.get_room_info() if self.room_simulator else None
        )

        timeline_slots = self.timeline_generator.generate(
            total_duration=duration, speakers=speakers
        )

        segments = self._generate_segments(timeline_slots)

        segments = self._merge_same_speaker_overlaps(segments)

        self._detect_segment_overlaps(segments)

        mixed_audio = self._mix_segments(segments, duration)

        actual_audio_duration = len(mixed_audio) / self.sample_rate

        segments = self._clip_segments_to_duration(segments, actual_audio_duration)

        noisy_audio, noise_meta = add_noise_to_meeting(mixed_audio, self.config)

        final_audio = self._normalize_audio(noisy_audio)

        timeline_events = self._create_timeline_events(segments)

        statistics = self._calculate_statistics(segments, actual_audio_duration)

        elapsed = time.time() - start_time
        logger.info(f"  Generation completed in {elapsed:.2f}s")
        logger.info(f"=" * 60)

        return GeneratedMeeting(
            meeting_id=meeting_id,
            audio=final_audio,
            sample_rate=self.sample_rate,
            duration=len(final_audio) / self.sample_rate,
            speakers=speakers,
            timeline_events=timeline_events,
            segments=segments,
            room_config=room_config,
            noise_config=noise_meta,
            statistics=statistics,
        )

    def _merge_same_speaker_overlaps(
        self, segments: List[MeetingSegment]
    ) -> List[MeetingSegment]:

        if not segments:
            return segments

        sorted_segs = sorted(segments, key=lambda s: s.start_time)

        merged = [sorted_segs[0]]
        merge_count = 0

        for seg in sorted_segs[1:]:
            prev = merged[-1]

            if seg.speaker_id == prev.speaker_id and seg.start_time < prev.end_time:

                overlap_amount = prev.end_time - seg.start_time
                logger.debug(
                    f"Merging same-speaker overlap: {seg.speaker_id} "
                    f"({prev.start_time:.3f}-{prev.end_time:.3f}) + "
                    f"({seg.start_time:.3f}-{seg.end_time:.3f}), "
                    f"overlap={overlap_amount:.3f}s"
                )

                prev.end_time = max(prev.end_time, seg.end_time)

                overlap_samples = int(overlap_amount * self.sample_rate)
                if overlap_samples < len(seg.audio):

                    new_audio_part = seg.audio[overlap_samples:]
                    prev.audio = np.concatenate([prev.audio, new_audio_part])
                else:

                    pass

                prev.source_files.extend(seg.source_files)
                prev.source_segment_ids.extend(seg.source_segment_ids)
                prev.source_segment_durations.extend(seg.source_segment_durations)

                merge_count += 1
            else:
                merged.append(seg)

        if merge_count > 0:
            logger.info(f"Merged {merge_count} same-speaker overlapping segment(s)")

        return merged

    def _detect_segment_overlaps(self, segments: List[MeetingSegment]) -> None:

        sorted_segments = sorted(segments, key=lambda s: s.start_time)

        for i in range(len(sorted_segments)):
            seg_i = sorted_segments[i]

            for j in range(i + 1, len(sorted_segments)):
                seg_j = sorted_segments[j]

                if seg_j.start_time >= seg_i.end_time:
                    break

                if seg_i.speaker_id == seg_j.speaker_id:
                    continue

                overlap_start = max(seg_i.start_time, seg_j.start_time)
                overlap_end = min(seg_i.end_time, seg_j.end_time)

                if overlap_end - overlap_start > 0.01:
                    sorted_segments[i].is_overlap = True
                    sorted_segments[j].is_overlap = True

                    if not sorted_segments[i].overlap_with:
                        sorted_segments[i].overlap_with = seg_j.speaker_id
                    if not sorted_segments[j].overlap_with:
                        sorted_segments[j].overlap_with = seg_i.speaker_id

    def _generate_segments(
        self, timeline_slots: List[TimeSlot]
    ) -> List[MeetingSegment]:

        segments = []

        if self.room_simulator and self.room_simulator.room_config:
            logger.info("Pre-computing RIRs for all speakers...")
            for speaker_id in self.room_simulator.speaker_positions:
                self.room_simulator.compute_rir(speaker_id)
            logger.info("RIRs computed and cached")

        slots_to_process = [
            s for s in timeline_slots if s.event_type != EventType.SILENCE
        ]
        total_slots = len(slots_to_process)

        logger.info(f"Generating audio for {total_slots} slots...")

        cumulative_shift = 0.0
        group_orig_end = 0.0
        group_actual_end = 0.0

        anchor_orig_start = 0.0
        anchor_orig_dur = 1.0
        anchor_actual_start = 0.0
        anchor_actual_dur = 1.0

        for idx, slot in enumerate(timeline_slots):
            if slot.event_type == EventType.SILENCE:
                continue

            if idx % 50 == 0:
                progress = (idx / len(timeline_slots)) * 100
                logger.info(
                    f"  Progress: {progress:.0f}% ({idx}/{len(timeline_slots)} slots)"
                )

            original_start = slot.start
            original_end = slot.end
            is_new_group = False

            if original_start < group_orig_end - 1e-3:

                offset = original_start - anchor_orig_start
                ratio = (
                    anchor_actual_dur / anchor_orig_dur if anchor_orig_dur > 0 else 1.0
                )
                slot.start = anchor_actual_start + offset * ratio

            else:

                is_new_group = True

                gap_original = max(0.0, original_start - group_orig_end)

                gap_actual = gap_original * 0.45

                slot.start = group_actual_end + gap_actual

                cumulative_shift = slot.start - original_start

                anchor_orig_start = original_start
                anchor_orig_dur = original_end - original_start
                anchor_actual_start = slot.start

            if slot.start < 0:
                slot.start = 0.0

            if self.config.timeline.overlap_ratio == 0.0:
                max_end = max((s.end_time for s in segments), default=0.0)
                if slot.start < max_end:
                    slot.start = max_end

            if slot.event_type == EventType.SINGLE_SPEAKER:
                segment = self._generate_single_speaker_segment(slot)
                if segment:
                    segments.append(segment)
                    slot.end = segment.end_time

                    if is_new_group:
                        anchor_actual_dur = segment.end_time - segment.start_time

                    group_orig_end = max(group_orig_end, original_end)
                    group_actual_end = max(group_actual_end, segment.end_time)

            elif slot.event_type == EventType.OVERLAP:
                overlap_segments = self._generate_overlap_segments(slot)
                if overlap_segments:
                    segments.extend(overlap_segments)
                    max_actual_end = max(
                        (s.end_time for s in overlap_segments), default=slot.start
                    )
                    slot.end = max_actual_end

                    if is_new_group:
                        anchor_actual_dur = max_actual_end - slot.start

                    group_orig_end = max(group_orig_end, original_end)
                    group_actual_end = max(group_actual_end, max_actual_end)

        logger.info(f"Generated {len(segments)} audio segments")
        return segments

    def _generate_single_speaker_segment(
        self, slot: TimeSlot
    ) -> Optional[MeetingSegment]:

        speaker_id = slot.speakers[0]

        processed = self.segment_pool.get_segment(
            speaker_id, prefer_duration=slot.duration
        )

        if processed is None:
            return None

        audio = processed.audio_data
        actual_duration = len(audio) / self.sample_rate

        if actual_duration < slot.duration:
            target_samples = int(slot.duration * self.sample_rate)
            max_pad_seconds = 0.15
            max_pad_samples = int(max_pad_seconds * self.sample_rate)
            needed_padding = target_samples - len(audio)
            if needed_padding > max_pad_samples:

                padding = np.zeros(max_pad_samples)
                audio = np.concatenate([audio, padding])
                actual_duration = len(audio) / self.sample_rate
            else:
                audio = self._adjust_audio_duration(audio, target_samples)
                actual_duration = slot.duration

        if self.room_simulator and speaker_id in self.room_simulator.speaker_positions:
            audio = self.room_simulator.apply_reverb(
                audio, speaker_id, dry_wet_ratio=random.uniform(0.3, 0.6)
            )
            audio = self.room_simulator.apply_distance_attenuation(audio, speaker_id)

        actual_end_time = slot.start + actual_duration

        return MeetingSegment(
            speaker_id=speaker_id,
            audio=audio,
            start_time=slot.start,
            end_time=actual_end_time,
            source_files=processed.source_segments,
            source_segment_ids=processed.source_segment_ids,
            source_segment_durations=processed.source_segment_durations,
            is_overlap=False,
        )

    def _generate_overlap_segments(self, slot: TimeSlot) -> List[MeetingSegment]:

        segments = []

        for i, speaker_id in enumerate(slot.speakers):

            processed = self.segment_pool.get_segment(
                speaker_id, prefer_duration=slot.duration
            )

            if processed is None:
                continue

            audio = processed.audio_data
            actual_duration = len(audio) / self.sample_rate

            if actual_duration < slot.duration:
                target_samples = int(slot.duration * self.sample_rate)
                max_pad_seconds = 0.15
                max_pad_samples = int(max_pad_seconds * self.sample_rate)
                needed_padding = target_samples - len(audio)
                if needed_padding > max_pad_samples:

                    padding = np.zeros(max_pad_samples)
                    audio = np.concatenate([audio, padding])
                    actual_duration = len(audio) / self.sample_rate
                else:
                    audio = self._adjust_audio_duration(audio, target_samples)
                    actual_duration = slot.duration

            if (
                self.room_simulator
                and speaker_id in self.room_simulator.speaker_positions
            ):
                audio = self.room_simulator.apply_reverb(
                    audio, speaker_id, dry_wet_ratio=random.uniform(0.3, 0.6)
                )
                audio = self.room_simulator.apply_distance_attenuation(
                    audio, speaker_id
                )

            audio = audio * random.uniform(0.7, 0.9)

            other_speaker = slot.speakers[1] if i == 0 else slot.speakers[0]

            actual_end_time = slot.start + actual_duration

            segments.append(
                MeetingSegment(
                    speaker_id=speaker_id,
                    audio=audio,
                    start_time=slot.start,
                    end_time=actual_end_time,
                    source_files=processed.source_segments,
                    source_segment_ids=processed.source_segment_ids,
                    source_segment_durations=processed.source_segment_durations,
                    is_overlap=True,
                    overlap_with=other_speaker,
                )
            )

        return segments

    def _adjust_audio_duration(
        self, audio: np.ndarray, target_samples: int
    ) -> np.ndarray:

        current_samples = len(audio)

        if current_samples == target_samples:
            return audio

        elif current_samples < target_samples:

            padding = np.zeros(target_samples - current_samples)
            return np.concatenate([audio, padding])

        else:

            return audio

    def _clip_segments_to_duration(
        self, segments: List[MeetingSegment], total_duration: float
    ) -> List[MeetingSegment]:

        clipped_segments = []
        eps = 0.001

        for seg in segments:

            if seg.start_time >= total_duration - eps:
                continue

            if seg.end_time > total_duration:
                new_end_time = total_duration

                new_duration = new_end_time - seg.start_time
                new_samples = int(new_duration * self.sample_rate)
                trimmed_audio = seg.audio[:new_samples]

                clipped_seg = MeetingSegment(
                    speaker_id=seg.speaker_id,
                    audio=trimmed_audio,
                    start_time=seg.start_time,
                    end_time=new_end_time,
                    source_files=seg.source_files,
                    source_segment_ids=seg.source_segment_ids,
                    source_segment_durations=seg.source_segment_durations,
                    is_overlap=seg.is_overlap,
                    overlap_with=seg.overlap_with,
                )
                clipped_segments.append(clipped_seg)
            else:

                clipped_segments.append(seg)

        return clipped_segments

    def _mix_segments(
        self, segments: List[MeetingSegment], total_duration: float
    ) -> np.ndarray:

        max_end_sample = int(total_duration * self.sample_rate)
        for segment in segments:
            seg_end = int(segment.start_time * self.sample_rate) + len(segment.audio)
            if seg_end > max_end_sample:
                max_end_sample = seg_end

        total_samples = max_end_sample
        mixed = np.zeros(total_samples)

        for segment in segments:
            start_sample = int(segment.start_time * self.sample_rate)
            end_sample = start_sample + len(segment.audio)

            start_sample = max(0, start_sample)
            end_sample = min(total_samples, end_sample)

            audio_len = end_sample - start_sample
            if audio_len <= 0:
                continue

            segment_audio = segment.audio[:audio_len]

            mixed[start_sample:end_sample] += segment_audio

        return mixed

    def _normalize_audio(
        self, audio: np.ndarray, target_peak: float = 0.95
    ) -> np.ndarray:

        max_val = np.max(np.abs(audio))

        if max_val > 0:
            audio = audio / max_val * target_peak

        return audio

    def _create_timeline_events(
        self, segments: List[MeetingSegment]
    ) -> List[TimelineEvent]:

        events = []

        for segment in segments:
            events.append(
                TimelineEvent(
                    start_time=segment.start_time,
                    end_time=segment.end_time,
                    speaker_id=segment.speaker_id,
                    event_type="speech",
                )
            )

        events.sort(key=lambda e: (e.start_time, e.speaker_id))

        return events

    def _calculate_statistics(
        self, segments: List[MeetingSegment], total_duration: float
    ) -> Dict:

        points = []
        for seg in segments:
            points.append((seg.start_time, 1))
            points.append((seg.end_time, -1))

        points.sort(key=lambda x: (x[0], -x[1]))

        single_time = 0.0
        overlap_time = 0.0
        silence_time = 0.0

        if not points:
            silence_time = total_duration
        else:
            active_count = 0
            last_time = 0.0

            for t, change in points:
                if t > total_duration:
                    t = total_duration

                duration = t - last_time
                if duration > 0:
                    if active_count == 0:
                        silence_time += duration
                    elif active_count == 1:
                        single_time += duration
                    else:
                        overlap_time += duration

                active_count += change
                last_time = t

            if last_time < total_duration:
                silence_time += total_duration - last_time

        speaker_times = {}
        for seg in segments:
            if seg.speaker_id not in speaker_times:
                speaker_times[seg.speaker_id] = 0.0
            speaker_times[seg.speaker_id] += seg.duration

        overlap_count = 0
        active_count = 0
        for t, change in points:
            prev_count = active_count
            active_count += change
            if prev_count == 1 and active_count >= 2:
                overlap_count += 1

        return {
            "total_duration": total_duration,
            "num_segments": len(segments),
            "single_speaker_ratio": (
                single_time / total_duration if total_duration > 0 else 0
            ),
            "silence_ratio": silence_time / total_duration if total_duration > 0 else 0,
            "overlap_ratio": overlap_time / total_duration if total_duration > 0 else 0,
            "speaker_times": speaker_times,
            "overlap_count": overlap_count,
        }

    def save_meeting(
        self, meeting: GeneratedMeeting, output_dir: Path
    ) -> Dict[str, Path]:

        output_dir = Path(output_dir)
        ensure_dir(output_dir)

        audio_path = output_dir / f"{meeting.meeting_id}.wav"
        rttm_path = output_dir / f"{meeting.meeting_id}.rttm"
        meta_path = output_dir / f"{meeting.meeting_id}.json"
        asr_path = output_dir / f"{meeting.meeting_id}.txt"
        stm_path = output_dir / f"{meeting.meeting_id}.stm"
        ctm_path = output_dir / f"{meeting.meeting_id}.ctm"

        save_audio(meeting.audio, audio_path, meeting.sample_rate)
        logger.info(f"Saved audio: {audio_path}")

        with open(rttm_path, "w") as f:
            f.write(meeting.get_rttm_content())
        logger.info(f"Saved RTTM: {rttm_path}")

        with open(asr_path, "w", encoding="utf-8") as f:
            f.write(meeting.get_asr_content())
        logger.info(f"Saved ASR: {asr_path}")

        with open(stm_path, "w", encoding="utf-8") as f:
            f.write(meeting.get_stm_content())
        logger.info(f"Saved STM: {stm_path}")

        with open(ctm_path, "w", encoding="utf-8") as f:
            f.write(meeting.get_ctm_content())
        logger.info(f"Saved CTM: {ctm_path}")

        with open(meta_path, "w", encoding="utf-8") as f:
            json.dump(meeting.get_metadata(), f, indent=2, ensure_ascii=False)
        logger.info(f"Saved metadata: {meta_path}")

        return {
            "audio": audio_path,
            "rttm": rttm_path,
            "asr": asr_path,
            "stm": stm_path,
            "ctm": ctm_path,
            "metadata": meta_path,
        }

class BatchMeetingGenerator:

    def __init__(self, config: Config = DEFAULT_CONFIG):
        self.config = config
        self.generator = MeetingGenerator(config)

    def generate_batch(
        self,
        num_meetings: int,
        output_dir: Path,
        duration_range: Tuple[float, float] = None,
        speakers_range: Tuple[int, int] = None,
    ) -> List[Dict]:

        output_dir = Path(output_dir)
        ensure_dir(output_dir)

        duration_range = duration_range or (
            self.config.meeting.min_duration_minutes * 60,
            self.config.meeting.max_duration_minutes * 60,
        )

        speakers_range = speakers_range or (
            self.config.meeting.min_speakers,
            self.config.meeting.max_speakers,
        )

        results = []

        logger.info(f"Generating {num_meetings} meetings...")

        for i in range(num_meetings):
            logger.info(f"\nMeeting {i+1}/{num_meetings}")

            try:

                duration = random.uniform(*duration_range)
                num_speakers = random.randint(*speakers_range)

                meeting = self.generator.generate(
                    duration=duration, num_speakers=num_speakers
                )

                paths = self.generator.save_meeting(meeting, output_dir)

                results.append(
                    {
                        "success": True,
                        "meeting_id": meeting.meeting_id,
                        "paths": {k: str(v) for k, v in paths.items()},
                        "statistics": meeting.statistics,
                    }
                )

            except Exception as e:
                logger.error(f"Failed to generate meeting: {e}")
                results.append({"success": False, "error": str(e)})

        success_count = sum(1 for r in results if r.get("success"))
        logger.info(f"\nBatch complete: {success_count}/{num_meetings} successful")

        return results

if __name__ == "__main__":
    print("Testing meeting_generator.py...")
    print("=" * 60)

    config = Config()

    print("\n1. Testing MeetingGenerator initialization...")

    try:
        generator = MeetingGenerator(config)
        print("  ✅ MeetingGenerator initialized")

        speakers = generator.audio_processor.get_available_speakers()
        print(f"  Available speakers: {speakers}")

        if len(speakers) >= 3:

            print("\n2. Generating test meeting (60 seconds, 3 speakers)...")
            meeting = generator.generate(
                duration=60.0, num_speakers=3, seed=42
            )

            print(f"\n  Generated meeting:")
            print(f"    ID: {meeting.meeting_id}")
            print(f"    Duration: {format_duration(meeting.duration)}")
            print(f"    Speakers: {meeting.speakers}")
            print(f"    Segments: {len(meeting.segments)}")
            print(f"    Timeline events: {len(meeting.timeline_events)}")

            print(f"\n  Statistics:")
            stats = meeting.statistics
            print(f"    Single speaker: {stats['single_speaker_ratio']*100:.1f}%")
            print(f"    Silence: {stats['silence_ratio']*100:.1f}%")
            print(f"    Overlap: {stats['overlap_ratio']*100:.1f}%")

            print(f"\n  Speaker times:")
            for spk, time_val in stats["speaker_times"].items():
                pct = time_val / meeting.duration * 100
                print(f"    {spk}: {format_duration(time_val)} ({pct:.1f}%)")

            print(f"\n3. Saving meeting...")
            output_dir = Path("output/test_meetings")
            paths = generator.save_meeting(meeting, output_dir)

            print(f"  Saved files:")
            for name, path in paths.items():
                if Path(path).exists():
                    size = Path(path).stat().st_size
                    print(f"    ✅ {Path(path).name}: {size:,} bytes")
                else:
                    print(f"    ❌ {Path(path).name}: NOT FOUND")
        else:
            print(f"\n  ️ Not enough speakers in VIVOS. Found: {speakers}")
            print("  Skipping meeting generation test.")

    except Exception as e:
        print(f"  ❌ Error: {e}")
        import traceback

        traceback.print_exc()

    print("\n" + "=" * 60)
    print("✅ meeting_generator.py test completed!")
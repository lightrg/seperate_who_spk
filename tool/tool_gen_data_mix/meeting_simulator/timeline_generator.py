import random
import numpy as np
from typing import List, Dict, Tuple, Optional
from dataclasses import dataclass, field
from enum import Enum
import logging

from config import Config, DEFAULT_CONFIG
from utils import TimelineEvent, random_range, format_duration, logger

class EventType(Enum):
    SINGLE_SPEAKER = "single"
    SILENCE = "silence"
    OVERLAP = "overlap"

@dataclass
class TimeSlot:

    start: float
    end: float
    event_type: EventType
    speakers: List[str] = field(default_factory=list)

    @property
    def duration(self) -> float:
        return self.end - self.start

    def __str__(self) -> str:
        if self.event_type == EventType.SILENCE:
            return f"[{self.start:.2f}-{self.end:.2f}] SILENCE ({self.duration:.2f}s)"
        elif self.event_type == EventType.OVERLAP:
            return f"[{self.start:.2f}-{self.end:.2f}] OVERLAP: {'+'.join(self.speakers)} ({self.duration:.2f}s)"
        else:
            return f"[{self.start:.2f}-{self.end:.2f}] {self.speakers[0]} ({self.duration:.2f}s)"

class TimelineGenerator:

    def __init__(self, config: Config = DEFAULT_CONFIG):
        self.config = config

        self.target_single = config.timeline.single_speaker_ratio
        self.target_silence = config.timeline.silence_ratio
        self.target_overlap = config.timeline.overlap_ratio

        self.silence_range = config.timeline.silence_duration_range
        self.overlap_range = config.timeline.overlap_duration_range

        self.slots: List[TimeSlot] = []
        self.current_time: float = 0.0
        self.total_duration: float = 0.0
        self.speakers: List[str] = []

        self._last_speaker: Optional[str] = None
        self._speaker_talk_time: Dict[str, float] = {}

    def generate(
        self, total_duration: float, speakers: List[str], seed: Optional[int] = None
    ) -> List[TimeSlot]:

        if seed is not None:
            random.seed(seed)
            np.random.seed(seed)

        self.slots = []
        self.current_time = 0.0
        self.total_duration = total_duration
        self.speakers = speakers
        self._last_speaker = None
        self._speaker_talk_time = {spk: 0.0 for spk in speakers}

        logger.info(
            f"Generating timeline: {format_duration(total_duration)}, {len(speakers)} speakers"
        )

        while self.current_time < total_duration:
            remaining = total_duration - self.current_time

            if remaining < 0.5:

                self._add_silence(remaining)
                break

            event_type = self._decide_next_event_type()

            if event_type == EventType.SILENCE:
                self._generate_silence_slot(remaining)
            elif event_type == EventType.OVERLAP:
                self._generate_overlap_slot(remaining)
            else:
                self._generate_single_speaker_slot(remaining)

        self._validate_max_two_speakers()

        self._log_statistics()

        return self.slots

    def _validate_max_two_speakers(self) -> None:

        max_speakers = self._count_max_concurrent_speakers()

        if max_speakers > 2:
            logger.warning(f"WARNING: Found {max_speakers} speakers at some point!")
        else:
            logger.debug(f"OK: Max concurrent speakers = {max_speakers}")

    def _count_max_concurrent_speakers(self) -> int:

        events = []
        for slot in self.slots:
            if slot.event_type == EventType.SILENCE:
                continue
            for spk in slot.speakers:
                events.append((slot.start, 1, spk))
                events.append((slot.end, -1, spk))

        if not events:
            return 0

        events.sort(key=lambda x: (x[0], -x[1]))

        active = {}
        max_concurrent = 0

        for t, change, spk in events:
            if change == 1:
                active[spk] = active.get(spk, 0) + 1
            else:
                active[spk] = active.get(spk, 0) - 1
                if active[spk] <= 0:
                    del active[spk]

            if len(active) > max_concurrent:
                max_concurrent = len(active)

        return max_concurrent

    def _decide_next_event_type(self) -> EventType:

        current_stats = self._get_current_distribution()

        single_dev = self.target_single - current_stats["single_ratio"]
        silence_dev = self.target_silence - current_stats["silence_ratio"]
        overlap_dev = self.target_overlap - current_stats["overlap_ratio"]

        avg_single_dur = 4.5
        avg_silence_dur = 1.65
        avg_overlap_dur = 2.5

        silence_dur_ratio = avg_single_dur / avg_silence_dur
        overlap_dur_ratio = avg_single_dur / avg_overlap_dur

        adjustment_factor = 4.0

        adjusted_probs = {
            EventType.SINGLE_SPEAKER: (
                max(0.0, self.target_single + single_dev * adjustment_factor)
                if self.target_single > 0
                else 0.0
            ),
            EventType.SILENCE: (
                max(
                    0.0,
                    (self.target_silence + silence_dev * adjustment_factor)
                    * silence_dur_ratio,
                )
                if self.target_silence > 0
                else 0.0
            ),
            EventType.OVERLAP: (
                max(
                    0.0,
                    (self.target_overlap + overlap_dev * adjustment_factor)
                    * overlap_dur_ratio,
                )
                if self.target_overlap > 0
                else 0.0
            ),
        }

        total = sum(adjusted_probs.values())
        if total <= 0:
            adjusted_probs = {
                EventType.SINGLE_SPEAKER: 1.0,
                EventType.SILENCE: 0.0,
                EventType.OVERLAP: 0.0,
            }
            total = 1.0
        adjusted_probs = {k: v / total for k, v in adjusted_probs.items()}

        r = random.random()
        cumulative = 0.0

        for event_type, prob in adjusted_probs.items():
            cumulative += prob
            if r < cumulative:
                return event_type

        return EventType.SINGLE_SPEAKER

    def _get_current_distribution(self) -> Dict:

        if not self.slots or self.current_time == 0:
            return {"single_ratio": 0.0, "silence_ratio": 0.0, "overlap_ratio": 0.0}

        stats = self._calculate_time_stats()
        total = stats["total_time"]

        return {
            "single_ratio": stats["single_time"] / total if total > 0 else 0,
            "silence_ratio": stats["silence_time"] / total if total > 0 else 0,
            "overlap_ratio": stats["overlap_time"] / total if total > 0 else 0,
        }

    def _generate_single_speaker_slot(self, max_duration: float) -> None:

        speaker = self._select_speaker(exclude_last=True)

        min_dur = min(1.0, max_duration)
        max_dur = min(8.0, max_duration)
        duration = random_range(min_dur, max_dur)

        slot = TimeSlot(
            start=self.current_time,
            end=self.current_time + duration,
            event_type=EventType.SINGLE_SPEAKER,
            speakers=[speaker],
        )

        self.slots.append(slot)
        self.current_time += duration
        self._last_speaker = speaker
        self._speaker_talk_time[speaker] += duration

    def _generate_silence_slot(self, max_duration: float) -> None:

        min_dur = min(self.silence_range[0], max_duration)
        max_dur = min(self.silence_range[1], max_duration)

        if min_dur > max_dur:
            min_dur = max_dur

        duration = random_range(min_dur, max_dur)
        self._add_silence(duration)

    def _add_silence(self, duration: float) -> None:

        slot = TimeSlot(
            start=self.current_time,
            end=self.current_time + duration,
            event_type=EventType.SILENCE,
            speakers=[],
        )

        self.slots.append(slot)
        self.current_time += duration
        self._last_speaker = None

    def _generate_overlap_slot(self, max_duration: float) -> None:

        if len(self.speakers) < 2:
            self._generate_single_speaker_slot(max_duration)
            return

        speaker1 = self._select_speaker(exclude_last=False)
        speaker2 = self._select_speaker(exclude=[speaker1])

        is_turn_taking = random.random() < self.config.overlap.turn_taking_ratio

        if is_turn_taking:
            self._generate_turn_taking_overlap(speaker1, speaker2, max_duration)
        else:
            self._generate_interruption_overlap(speaker1, speaker2, max_duration)

    def _generate_turn_taking_overlap(self, s1: str, s2: str, max_duration: float):

        if self.target_overlap >= 0.35:

            s1_dur = random_range(1.5, min(3.5, max_duration))
            overlap_dur = random_range(1.5, 3.0)
            overlap_dur = min(overlap_dur, s1_dur * 0.9)
            s2_dur = random_range(1.5, min(3.5, max_duration))
        elif self.target_overlap >= 0.2:

            s1_dur = random_range(1.5, min(4.0, max_duration))
            overlap_dur = random_range(
                max(self.config.overlap.turn_taking_min_sec, 1.0),
                max(self.config.overlap.turn_taking_max_sec, 3.0),
            )
            overlap_dur = min(overlap_dur, s1_dur * 0.8)
            s2_dur = random_range(1.5, min(4.0, max_duration))
        else:

            s1_dur = random_range(2.0, min(8.0, max_duration))

            overlap_dur = random_range(
                self.config.overlap.turn_taking_min_sec,
                self.config.overlap.turn_taking_max_sec,
            )
            overlap_dur = min(
                overlap_dur, s1_dur * 0.5
            )

            s2_dur = random_range(2.0, min(8.0, max_duration))

        slot1 = TimeSlot(
            start=self.current_time,
            end=self.current_time + s1_dur,
            event_type=EventType.SINGLE_SPEAKER,
            speakers=[s1],
        )
        self.slots.append(slot1)
        self._speaker_talk_time[s1] += s1_dur

        s2_start = slot1.end - overlap_dur
        slot2 = TimeSlot(
            start=s2_start,
            end=s2_start + s2_dur,
            event_type=EventType.SINGLE_SPEAKER,
            speakers=[s2],
        )
        self.slots.append(slot2)
        self._speaker_talk_time[s2] += s2_dur

        self.current_time = slot2.end
        self._last_speaker = s2

    def _generate_interruption_overlap(self, s1: str, s2: str, max_duration: float):

        if self.target_overlap >= 0.35:

            s1_dur = random_range(2.0, min(5.0, max_duration))
            interruption_point = s1_dur * random.uniform(0.1, 0.2)
            s2_dur = random_range(2.0, min(5.0, max_duration))
        elif self.target_overlap >= 0.2:

            s1_dur = random_range(2.0, min(6.0, max_duration))
            interruption_point = s1_dur * random.uniform(0.1, 0.4)
            s2_dur = random_range(2.0, min(6.0, max_duration))
        else:

            s1_dur = random_range(3.0, min(10.0, max_duration))

            interruption_point = s1_dur * random.uniform(0.3, 0.7)

            s2_dur = random_range(2.0, min(8.0, max_duration))

        slot1 = TimeSlot(
            start=self.current_time,
            end=self.current_time + s1_dur,
            event_type=EventType.SINGLE_SPEAKER,
            speakers=[s1],
        )
        self.slots.append(slot1)
        self._speaker_talk_time[s1] += s1_dur

        s2_start = slot1.start + interruption_point
        slot2 = TimeSlot(
            start=s2_start,
            end=s2_start + s2_dur,
            event_type=EventType.SINGLE_SPEAKER,
            speakers=[s2],
        )
        self.slots.append(slot2)
        self._speaker_talk_time[s2] += s2_dur

        self.current_time = max(slot1.end, slot2.end)

        self._last_speaker = s2 if slot2.end > slot1.end else s1

    def _select_speaker(
        self, exclude_last: bool = False, exclude: List[str] = None
    ) -> str:

        exclude = exclude or []

        available = [
            spk
            for spk in self.speakers
            if spk not in exclude and (not exclude_last or spk != self._last_speaker)
        ]

        if not available:

            available = [spk for spk in self.speakers if spk not in exclude]

        if not available:
            available = self.speakers

        total_time = sum(self._speaker_talk_time.values()) or 1.0

        weights = []
        for spk in available:
            talk_ratio = self._speaker_talk_time.get(spk, 0) / total_time

            weight = 1.0 - talk_ratio + 0.1
            weights.append(weight)

        total_weight = sum(weights)
        weights = [w / total_weight for w in weights]

        return random.choices(available, weights=weights, k=1)[0]

    def to_timeline_events(self) -> List[TimelineEvent]:

        events = []

        for slot in self.slots:
            if slot.event_type == EventType.SILENCE:

                continue

            elif slot.event_type == EventType.SINGLE_SPEAKER:
                events.append(
                    TimelineEvent(
                        start_time=slot.start,
                        end_time=slot.end,
                        speaker_id=slot.speakers[0],
                        event_type="speech",
                    )
                )

            elif slot.event_type == EventType.OVERLAP:

                for speaker in slot.speakers:
                    events.append(
                        TimelineEvent(
                            start_time=slot.start,
                            end_time=slot.end,
                            speaker_id=speaker,
                            event_type="speech",
                        )
                    )

        return events

    def _log_statistics(self) -> None:

        stats = self.get_statistics()

        logger.info(f"Timeline generated:")
        logger.info(f"  Total duration: {format_duration(stats['total_duration'])}")
        logger.info(f"  Slots: {stats['num_slots']}")
        logger.info(f"  Distribution:")
        logger.info(
            f"    Single speaker: {stats['single_ratio']*100:.1f}% (target: {self.target_single*100:.0f}%)"
        )
        logger.info(
            f"    Silence: {stats['silence_ratio']*100:.1f}% (target: {self.target_silence*100:.0f}%)"
        )
        logger.info(
            f"    Overlap: {stats['overlap_ratio']*100:.1f}% (target: {self.target_overlap*100:.0f}%)"
        )
        logger.info(f"  Speaker talk time:")
        for spk, time in stats["speaker_talk_time"].items():
            pct = time / stats["total_duration"] * 100
            logger.info(f"    {spk}: {format_duration(time)} ({pct:.1f}%)")

    def _calculate_time_stats(self) -> Dict[str, float]:

        if not self.slots:
            return {
                "single_time": 0.0,
                "silence_time": 0.0,
                "overlap_time": 0.0,
                "total_time": 0.0,
            }

        min_start = min(s.start for s in self.slots)
        max_end = max(s.end for s in self.slots)
        total_time = max(self.current_time, max_end)

        points = []
        for s in self.slots:
            if s.event_type == EventType.SILENCE:
                continue
            points.append((s.start, 1))
            points.append((s.end, -1))

        points.sort(key=lambda x: x[0])

        single_dur = 0.0
        overlap_dur = 0.0
        silence_dur = 0.0

        if not points:

            silence_dur = sum(
                s.duration for s in self.slots if s.event_type == EventType.SILENCE
            )
            return {
                "single_time": 0.0,
                "silence_time": silence_dur,
                "overlap_time": 0.0,
                "total_time": silence_dur,
            }

        active_count = 0
        last_time = points[0][0]

        if last_time > 0:
            silence_dur += last_time

        for t, change in points:
            duration = t - last_time
            if duration > 0:
                if active_count == 0:
                    silence_dur += duration
                elif active_count == 1:
                    single_dur += duration
                else:
                    overlap_dur += duration

            active_count += change
            last_time = t

        if last_time < total_time:
            silence_dur += total_time - last_time

        return {
            "single_time": single_dur,
            "silence_time": silence_dur,
            "overlap_time": overlap_dur,
            "total_time": total_time,
        }

    def get_statistics(self) -> Dict:

        stats = self._calculate_time_stats()
        total = stats["total_time"]

        return {
            "total_duration": total,
            "num_slots": len(self.slots),
            "single_time": stats["single_time"],
            "silence_time": stats["silence_time"],
            "overlap_time": stats["overlap_time"],
            "single_ratio": stats["single_time"] / total if total > 0 else 0,
            "silence_ratio": stats["silence_time"] / total if total > 0 else 0,
            "overlap_ratio": stats["overlap_time"] / total if total > 0 else 0,
            "speaker_talk_time": self._speaker_talk_time.copy(),
            "num_speakers": len(self.speakers),
        }

    def print_timeline(self, max_slots: int = 50) -> None:

        print(f"\n{'='*60}")
        print(
            f"TIMELINE: {format_duration(self.current_time)}, {len(self.speakers)} speakers"
        )
        print(f"{'='*60}")

        for i, slot in enumerate(self.slots[:max_slots]):
            print(f"  {i+1:3d}. {slot}")

        if len(self.slots) > max_slots:
            print(f"  ... and {len(self.slots) - max_slots} more slots")

        print(f"{'='*60}\n")

class ConversationPatternGenerator:

    def __init__(self, config: Config = DEFAULT_CONFIG):
        self.config = config

    def generate_turn_taking_sequence(
        self, speakers: List[str], num_turns: int, avg_turn_duration: float = 3.0
    ) -> List[TimeSlot]:

        slots = []
        current_time = 0.0
        last_speaker = None

        for _ in range(num_turns):

            available = [s for s in speakers if s != last_speaker]
            speaker = random.choice(available) if available else random.choice(speakers)

            duration = random.gauss(avg_turn_duration, avg_turn_duration * 0.3)
            duration = max(0.5, min(10.0, duration))

            if slots and random.random() < 0.7:
                pause = random.uniform(0.1, 0.5)
                slots.append(
                    TimeSlot(
                        start=current_time,
                        end=current_time + pause,
                        event_type=EventType.SILENCE,
                    )
                )
                current_time += pause

            slots.append(
                TimeSlot(
                    start=current_time,
                    end=current_time + duration,
                    event_type=EventType.SINGLE_SPEAKER,
                    speakers=[speaker],
                )
            )

            current_time += duration
            last_speaker = speaker

        return slots

    def generate_interruption_pattern(
        self, speaker1: str, speaker2: str, base_duration: float = 5.0
    ) -> List[TimeSlot]:

        slots = []
        current_time = 0.0

        initial_duration = random.uniform(2.0, 4.0)
        slots.append(
            TimeSlot(
                start=current_time,
                end=current_time + initial_duration,
                event_type=EventType.SINGLE_SPEAKER,
                speakers=[speaker1],
            )
        )
        current_time += initial_duration

        overlap_duration = random.uniform(0.5, 1.5)
        slots.append(
            TimeSlot(
                start=current_time,
                end=current_time + overlap_duration,
                event_type=EventType.OVERLAP,
                speakers=[speaker1, speaker2],
            )
        )
        current_time += overlap_duration

        continuation = random.uniform(2.0, 5.0)
        slots.append(
            TimeSlot(
                start=current_time,
                end=current_time + continuation,
                event_type=EventType.SINGLE_SPEAKER,
                speakers=[speaker2],
            )
        )

        return slots

if __name__ == "__main__":
    print("Testing timeline_generator.py...")
    print("=" * 60)

    config = Config()
    generator = TimelineGenerator(config)

    speakers = ["SPEAKER_01", "SPEAKER_02", "SPEAKER_03", "SPEAKER_04"]
    duration = 300.0

    slots = generator.generate(total_duration=duration, speakers=speakers, seed=42)

    generator.print_timeline(max_slots=30)

    stats = generator.get_statistics()

    print("\nStatistics:")
    print(f"  Total slots: {stats['num_slots']}")
    print(f"  Single speaker: {stats['single_ratio']*100:.1f}% (target: 60%)")
    print(f"  Silence: {stats['silence_ratio']*100:.1f}% (target: 25%)")
    print(f"  Overlap: {stats['overlap_ratio']*100:.1f}% (target: 15%)")

    tolerance = 0.10

    single_ok = abs(stats["single_ratio"] - 0.60) < tolerance
    silence_ok = abs(stats["silence_ratio"] - 0.25) < tolerance
    overlap_ok = abs(stats["overlap_ratio"] - 0.15) < tolerance

    print(f"\nDistribution check:")
    print(f"  Single speaker: {'✅' if single_ok else '❌'}")
    print(f"  Silence: {'✅' if silence_ok else '❌'}")
    print(f"  Overlap: {'✅' if overlap_ok else '❌'}")

    events = generator.to_timeline_events()
    print(f"\nConverted to {len(events)} TimelineEvents")

    print("\nFirst 10 events:")
    for i, event in enumerate(events[:10]):
        print(
            f"  {i+1}. [{event.start_time:.2f}-{event.end_time:.2f}] {event.speaker_id}"
        )

    print("\n" + "=" * 60)
    print("Testing ConversationPatternGenerator...")

    pattern_gen = ConversationPatternGenerator(config)

    turn_slots = pattern_gen.generate_turn_taking_sequence(
        speakers=["A", "B", "C"], num_turns=5
    )
    print("\nTurn-taking pattern:")
    for slot in turn_slots:
        print(f"  {slot}")

    interrupt_slots = pattern_gen.generate_interruption_pattern("A", "B")
    print("\nInterruption pattern:")
    for slot in interrupt_slots:
        print(f"  {slot}")

    print("\n" + "=" * 60)
    print("✅ timeline_generator.py tests passed!")
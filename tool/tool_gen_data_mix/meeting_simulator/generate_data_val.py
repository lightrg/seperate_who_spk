import argparse
import random
import time
from pathlib import Path
from copy import deepcopy
from dataclasses import dataclass, field
from typing import List, Dict

from config import (
    Config,
    AudioConfig,
    SpeakerConfig,
    MeetingConfig,
    OverlapConfig,
    SegmentConfig,
    SilenceConfig,
    NoiseConfig,
    ReverbConfig,
    TimelineConfig,
    DatasetConfig,
)
from meeting_generator import MeetingGenerator
from utils import ensure_dir, format_duration, logger
import utils as _utils_module

def _preload_test_data_caches():

    _utils_module._alignment_cache = None
    _utils_module._prompts_cache = None

    test_alignment_dir = TEST_DATA_PATH / "aligned_wav2vec2"
    logger.info(f"Pre-loading alignments from: {test_alignment_dir}")
    _utils_module.load_alignment_data(alignment_dir=test_alignment_dir)

    test_prompts_file = TEST_DATA_PATH / "prompts.txt"
    logger.info(f"Pre-loading prompts from: {test_prompts_file}")
    _utils_module.load_prompts(prompts_path=test_prompts_file)

TEST_SPEAKERS = [
    f"VIVOSSPK{i:02d}" for i in range(1, 47) if i not in (4, 7, 8, 44, 45, 46)
]

TEST_DATA_PATH = Path("/mnt/sda1/du_an1/data/val")

@dataclass
class TestSpeakerConfig(SpeakerConfig):

    vivos_speakers: List[str] = field(default_factory=lambda: TEST_SPEAKERS)
    lsvsc_speakers: List[str] = field(default_factory=list)
    cv_speakers: List[str] = field(default_factory=list)

    @property
    def all_speakers(self) -> List[str]:
        return self.vivos_speakers

    @property
    def datasets(self) -> Dict[str, List[str]]:
        return {"VIVOS_TEST": self.vivos_speakers}

def create_test_config() -> Config:

    config = Config(
        audio=AudioConfig(),
        speaker=TestSpeakerConfig(),
        meeting=MeetingConfig(

            min_duration_minutes=28.0,
            max_duration_minutes=35.0,
            min_speakers=2,
            max_speakers=4,
        ),
        overlap=OverlapConfig(
            turn_taking_ratio=0.40,
            turn_taking_min_sec=0.5,
            turn_taking_max_sec=2.0,
            interruption_ratio=0.60,
            interruption_min_sec=1.5,
            interruption_max_sec=5.0,
        ),
        segment=SegmentConfig(
            merge_ratio=0.70,
            merge_min_segments=2,
            merge_max_segments=3,
        ),
        silence=SilenceConfig(),
        noise=NoiseConfig(
            snr_min_db=20.0,
            snr_max_db=35.0,
            noise_probability=0.50,
        ),
        reverb=ReverbConfig(
            reverb_probability=0.40,
        ),
        timeline=TimelineConfig(
            single_speaker_ratio=0.60,
            silence_ratio=0.35,
            overlap_ratio=0.05,
        ),
        dataset=DatasetConfig(
            total_hours=10.0,
            num_sets=3,

            vivos_path=TEST_DATA_PATH / "waves",
            lsvsc_path=Path("/dev/null"),
            cv_path=Path("/dev/null"),
            output_path=TEST_DATA_PATH / "output",
        ),
    )
    return config

def calculate_meetings_needed(config: Config) -> dict:

    total_hours = config.dataset.total_hours
    total_minutes = total_hours * 60

    avg_meeting_minutes = (
        config.meeting.min_duration_minutes + config.meeting.max_duration_minutes
    ) / 2

    minutes_per_set = total_minutes / 3
    meetings_per_set = int(minutes_per_set / avg_meeting_minutes)

    return {
        2: meetings_per_set,
        3: meetings_per_set,
        4: meetings_per_set,
    }

def generate_full_dataset(
    output_dir: Path = None,
    seed: int = None,
    dry_run: bool = False,
    num_speakers_filter: list = None,
):

    config = create_test_config()
    output_dir = (
        Path(output_dir)
        if output_dir
        else config.dataset.output_path / "overlap_dataset"
    )

    overlap_levels = [0.0, 0.05, 0.1, 0.15, 0.2, 0.25, 0.3]
    hours_per_level = 6

    print("=" * 60)
    print("MEETING DATASET GENERATOR - TEST DATA")
    print(
        f"({len(overlap_levels)} OVERLAP LEVELS × {hours_per_level}h = {len(overlap_levels) * hours_per_level}h TOTAL)"
    )
    print("=" * 60)
    print(
        f"Target: {len(overlap_levels)} levels × {hours_per_level}h "
        f"= {len(overlap_levels) * hours_per_level} hours total"
    )
    print(
        f"Meeting duration: {config.meeting.min_duration_minutes}"
        f"-{config.meeting.max_duration_minutes} min"
    )
    print(f"\nOutput: {output_dir}")

    all_speakers = config.speaker.all_speakers
    if len(all_speakers) < 4:
        print(f"❌ Error: Not enough speakers ({len(all_speakers)} < 4)")
        return []

    print(f"Total available speakers: {len(all_speakers)}")
    for dataset_name, speakers in config.speaker.datasets.items():
        if speakers:
            print(f"  📁 {dataset_name}: {len(speakers)} speakers")

            for i in range(0, len(speakers), 5):
                chunk = speakers[i : i + 5]
                print(f"      {', '.join(chunk)}")

    if dry_run:
        print("\n[DRY RUN] Không generate, chỉ hiển thị plan")

        config.dataset.total_hours = hours_per_level
        plan = calculate_meetings_needed(config)
        if num_speakers_filter:
            plan = {k: v for k, v in plan.items() if k in num_speakers_filter}

        print(f"\nPlan PER OVERLAP LEVEL ({hours_per_level}h):")
        for num_speakers, count in plan.items():
            print(f"  - {num_speakers} speakers: {count} meetings")
        print(f"  Total meetings per level: {sum(plan.values())}")
        print(
            f"  Total meetings all levels: {sum(plan.values()) * len(overlap_levels)}"
        )
        return

    if seed is not None:
        random.seed(seed)
        print(f"\nRandom seed: {seed}")

    ensure_dir(output_dir)

    _preload_test_data_caches()

    print("\nInitializing generator (Scanning VIVOS test data)...")
    generator = MeetingGenerator(config)

    all_channels = generator.audio_processor.get_available_speakers()
    print(f"Successfully loaded {len(all_channels)} speakers.")

    all_results = []
    start_time = time.time()

    for overlap in overlap_levels:
        overlap_percent = int(overlap * 100)
        overlap_dir = output_dir / f"overlap_{overlap_percent}"
        ensure_dir(overlap_dir)

        generator.config.dataset.total_hours = hours_per_level
        generator.config.timeline.overlap_ratio = overlap

        silence_ratio = generator.config.timeline.silence_ratio
        single_ratio = max(0.1, 1.0 - silence_ratio - overlap)
        generator.config.timeline.single_speaker_ratio = single_ratio

        from timeline_generator import TimelineGenerator

        generator.timeline_generator = TimelineGenerator(generator.config)

        plan = calculate_meetings_needed(generator.config)
        if num_speakers_filter:
            plan = {k: v for k, v in plan.items() if k in num_speakers_filter}

        print(f"\n{'#'*60}")
        print(f"--- OVERLAP LEVEL: {overlap_percent}% ({hours_per_level} hours) ---")
        print(
            f"Ratios - Single: {single_ratio*100:.1f}%, "
            f"Silence: {silence_ratio*100:.1f}%, "
            f"Overlap: {overlap*100:.1f}%"
        )
        print(f"{'#'*60}")

        for num_speakers, num_meetings in plan.items():
            print(f"\n{'='*60}")
            print(
                f"Generating Set: Overlap {overlap_percent}% | "
                f"{num_speakers} speakers ({num_meetings} meetings)"
            )
            print("=" * 60)

            for i in range(num_meetings):
                try:
                    print(
                        f"\n[{i+1}/{num_meetings}] Generating meeting "
                        f"(Overlap {overlap_percent}%, {num_speakers} spks)..."
                    )

                    if len(all_channels) < num_speakers:
                        print(
                            f"  ️  Skip: Not enough speakers "
                            f"({len(all_channels)} < {num_speakers})"
                        )
                        continue

                    selected_speakers = random.sample(all_channels, num_speakers)

                    set_dir = overlap_dir / f"{num_speakers}speakers"
                    ensure_dir(set_dir)

                    print(f"  🔒 Selected speakers: {selected_speakers}")
                    print(f"  💾 Output: {set_dir}")

                    meeting = generator.generate(
                        num_speakers=num_speakers,
                        duration=random.uniform(
                            generator.config.meeting.min_duration_minutes * 60,
                            generator.config.meeting.max_duration_minutes * 60,
                        ),
                        speaker_ids=selected_speakers,
                    )

                    paths = generator.save_meeting(meeting, set_dir)

                    print(f"  ✅ {meeting.meeting_id}")
                    print(f"     Duration: {format_duration(meeting.duration)}")
                    print(f"     Segments: {len(meeting.segments)}")

                    all_results.append(
                        {
                            "success": True,
                            "meeting_id": meeting.meeting_id,
                            "overlap_level": overlap,
                            "num_speakers": num_speakers,
                            "duration": meeting.duration,
                            "speakers": selected_speakers,
                            "statistics": meeting.statistics,
                        }
                    )

                except Exception as e:
                    print(f"  ❌ Error: {e}")
                    import traceback

                    traceback.print_exc()
                    all_results.append({"success": False, "error": str(e)})

    elapsed = time.time() - start_time
    success_results = [r for r in all_results if r.get("success")]
    total_duration = sum(r["duration"] for r in success_results)

    print("\n" + "=" * 60)
    print("GENERATION COMPLETE")
    print("=" * 60)
    print(f"Success: {len(success_results)}/{len(all_results)}")
    print(
        f"Total audio: {format_duration(total_duration)} "
        f"({total_duration/3600:.2f} hours)"
    )
    print(f"Time elapsed: {format_duration(elapsed)}")
    print(f"Output: {output_dir}")

    if total_duration > 0:
        print(f"\n--- OVERALL STATISTICS ---")
        total_overlap_time = 0.0
        total_silence_time = 0.0
        total_single_speaker_time = 0.0
        for r in success_results:
            stats = r.get("statistics", {})
            dur = r["duration"]
            total_overlap_time += dur * stats.get("overlap_ratio", 0)
            total_silence_time += dur * stats.get("silence_ratio", 0)
            total_single_speaker_time += dur * stats.get("single_speaker_ratio", 0)

        print(f"Single speaker: {total_single_speaker_time/total_duration*100:.1f}%")
        print(f"Silence:        {total_silence_time/total_duration*100:.1f}%")
        print(f"Overlap:        {total_overlap_time/total_duration*100:.1f}%")

    return all_results

def generate_test(output_dir: Path = None):

    config = create_test_config()
    output_dir = (
        Path(output_dir) if output_dir else TEST_DATA_PATH / "output" / "test_quick"
    )

    print("=" * 60)
    print("TEST: Generating 1 meeting (60 seconds) with 20% overlap")
    print("=" * 60)

    config.timeline.overlap_ratio = 0.2
    config.timeline.single_speaker_ratio = max(
        0.1, 1.0 - config.timeline.silence_ratio - 0.2
    )

    _preload_test_data_caches()

    generator = MeetingGenerator(config)

    all_channels = generator.audio_processor.get_available_speakers()
    selected_speakers = random.sample(all_channels, 3)

    print(f"\n🔒 Selected speakers: {selected_speakers}")

    meeting = generator.generate(
        duration=60.0,
        num_speakers=3,
        seed=42,
        speaker_ids=selected_speakers,
    )

    paths = generator.save_meeting(meeting, output_dir)
    print(f"\n✅ Generated: {meeting.meeting_id}")
    print(f"Duration: {format_duration(meeting.duration)}")
    print(f"Speakers: {meeting.speakers}")
    print(f"\nStatistics:")
    stats = meeting.statistics
    print(f"  Single speaker: {stats['single_speaker_ratio']*100:.1f}%")
    print(f"  Silence: {stats['silence_ratio']*100:.1f}%")
    print(f"  Overlap: {stats['overlap_ratio']*100:.1f}%")
    print(f"\nOutput files:")
    for name, path in paths.items():
        print(f"  {name}: {path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Generate meeting audio dataset from VIVOS test data (4 overlap levels)"
    )
    parser.add_argument(
        "--mode",
        choices=["full", "test", "plan"],
        default="test",
        help="Mode: full (40h), test (1 meeting), plan (show plan only)",
    )
    parser.add_argument(
        "--output",
        type=str,
        default=None,
        help="Output directory",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed",
    )
    parser.add_argument(
        "--speakers",
        type=int,
        nargs="+",
        default=None,
        help="Only generate for specific num_speakers (e.g. 2 3)",
    )

    args = parser.parse_args()
    out_path = Path(args.output) if args.output else None

    if args.mode == "full":
        generate_full_dataset(
            out_path, seed=args.seed, num_speakers_filter=args.speakers
        )
    elif args.mode == "test":
        generate_test(out_path)
    elif args.mode == "plan":
        generate_full_dataset(out_path, dry_run=True, num_speakers_filter=args.speakers)
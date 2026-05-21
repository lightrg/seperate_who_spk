import argparse
import random
import time
from pathlib import Path
from copy import deepcopy

from config import Config, DEFAULT_CONFIG
from meeting_generator import MeetingGenerator
from utils import ensure_dir, format_duration, logger

def calculate_meetings_needed(config: Config) -> dict:

    total_hours = config.dataset.total_hours
    total_minutes = total_hours * 60

    avg_meeting_minutes = (
        config.meeting.min_duration_minutes +
        config.meeting.max_duration_minutes
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
    num_speakers_filter: list = None
):

    config = deepcopy(DEFAULT_CONFIG)
    output_dir = Path(output_dir) if output_dir else config.dataset.output_path / "overlap_dataset"

    overlap_levels = [0.1, 0.2, 0.3, 0.4, 0.5]
    hours_per_level = 40.0

    print("=" * 60)
    print("MEETING DATASET GENERATOR (5 OVERLAP LEVELS)")
    print("=" * 60)
    print(f"Target: {len(overlap_levels)} levels × {hours_per_level}h = {len(overlap_levels) * hours_per_level} hours total")
    print(f"Meeting duration: {config.meeting.min_duration_minutes}-{config.meeting.max_duration_minutes} min")
    print(f"\nOutput: {output_dir}")

    all_speakers = config.speaker.all_speakers
    if len(all_speakers) < 4:
        print(f"❌ Error: Not enough speakers configured ({len(all_speakers)} < 4)")
        return []

    print(f"Total available speakers: {len(all_speakers)}")
    for dataset_name, speakers in config.speaker.datasets.items():
        if speakers:
            print(f"  📁 {dataset_name}: {len(speakers)} speakers — {', '.join(speakers)}")

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
        print(f"  Total meetings all levels: {sum(plan.values()) * len(overlap_levels)}")
        return

    if seed is not None:
        random.seed(seed)
        print(f"\nRandom seed: {seed}")

    ensure_dir(output_dir)

    print("\nInitializing generator (Scanning VIVOS, LSVSC, CV)...")
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
        print(f"Ratios - Single: {single_ratio*100:.1f}%, Silence: {silence_ratio*100:.1f}%, Overlap: {overlap*100:.1f}%")
        print(f"{'#'*60}")

        for num_speakers, num_meetings in plan.items():
            print(f"\n{'='*60}")
            print(f"Generating Set: Overlap {overlap_percent}% | {num_speakers} speakers ({num_meetings} meetings)")
            print("=" * 60)

            for i in range(num_meetings):
                try:
                    print(f"\n[{i+1}/{num_meetings}] Generating meeting (Overlap {overlap_percent}%, {num_speakers} spks)...")

                    if len(all_channels) < num_speakers:
                        print(f"  ️  Skip: Not enough speakers ({len(all_channels)} < {num_speakers})")
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
                            generator.config.meeting.max_duration_minutes * 60
                        ),
                        speaker_ids=selected_speakers
                    )

                    paths = generator.save_meeting(meeting, set_dir)

                    print(f"  ✅ {meeting.meeting_id}")
                    print(f"     Duration: {format_duration(meeting.duration)}")
                    print(f"     Segments: {len(meeting.segments)}")

                    all_results.append({
                        "success": True,
                        "meeting_id": meeting.meeting_id,
                        "overlap_level": overlap,
                        "num_speakers": num_speakers,
                        "duration": meeting.duration,
                        "speakers": selected_speakers,
                        "statistics": meeting.statistics
                    })

                except Exception as e:
                    print(f"  ❌ Error: {e}")
                    import traceback
                    traceback.print_exc()
                    all_results.append({
                        "success": False,
                        "error": str(e)
                    })

    elapsed = time.time() - start_time
    success_results = [r for r in all_results if r.get("success")]
    total_duration = sum(r["duration"] for r in success_results)

    print("\n" + "=" * 60)
    print("GENERATION COMPLETE")
    print("=" * 60)
    print(f"Success: {len(success_results)}/{len(all_results)}")
    print(f"Total audio: {format_duration(total_duration)} ({total_duration/3600:.2f} hours)")
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

    config = deepcopy(DEFAULT_CONFIG)
    output_dir = Path(output_dir) if output_dir else Path("output/test_overlap")

    print("=" * 60)
    print("TEST: Generating 1 meeting (60 seconds) with 20% overlap")
    print("=" * 60)

    config.timeline.overlap_ratio = 0.2
    config.timeline.single_speaker_ratio = max(0.1, 1.0 - config.timeline.silence_ratio - 0.2)

    generator = MeetingGenerator(config)

    all_channels = generator.audio_processor.get_available_speakers()
    selected_speakers = random.sample(all_channels, 3)

    print(f"\n🔒 Selected speakers: {selected_speakers}")

    meeting = generator.generate(
        duration=60.0,
        num_speakers=3,
        seed=42,
        speaker_ids=selected_speakers
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
        description="Generate meeting audio dataset with 5 overlap levels"
    )
    parser.add_argument(
        "--mode",
        choices=["full", "test", "plan"],
        default="test",
        help="Mode: full (200h), test (1 meeting), plan (show plan only)"
    )
    parser.add_argument(
        "--output",
        type=str,
        default=None,
        help="Output directory"
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed"
    )
    parser.add_argument(
        "--speakers",
        type=int,
        nargs="+",
        default=None,
        help="Only generate for specific num_speakers (e.g. 2 3)"
    )

    args = parser.parse_args()
    out_path = Path(args.output) if args.output else None

    if args.mode == "full":
        generate_full_dataset(out_path, seed=args.seed, num_speakers_filter=args.speakers)
    elif args.mode == "test":
        generate_test(out_path)
    elif args.mode == "plan":
        generate_full_dataset(out_path, dry_run=True, num_speakers_filter=args.speakers)
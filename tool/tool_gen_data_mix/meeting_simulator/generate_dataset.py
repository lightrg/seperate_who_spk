import argparse
import random
import time
from pathlib import Path

from config import Config, DEFAULT_CONFIG
from meeting_generator import MeetingGenerator, BatchMeetingGenerator
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

def filter_speakers_by_dataset(config: Config, datasets: list = None) -> Config:

    if not datasets:
        return config

    datasets_lower = [d.lower() for d in datasets]

    filtered_speakers = []
    for ds in datasets_lower:
        if ds in ["vivos", "v"]:
            filtered_speakers.extend(config.speaker.vivos_speakers)
        elif ds in ["lsvsc", "l"]:
            filtered_speakers.extend(config.speaker.lsvsc_speakers)
        elif ds in ["cv", "common_voice", "commonvoice", "c"]:
            filtered_speakers.extend(config.speaker.cv_speakers)

    if not filtered_speakers:
        raise ValueError(f"No speakers found for datasets: {datasets}")

    from copy import deepcopy
    new_config = deepcopy(config)

    if "vivos" in datasets_lower or "v" in datasets_lower:
        pass
    else:
        new_config.speaker.vivos_speakers = []

    if "lsvsc" in datasets_lower or "l" in datasets_lower:
        pass
    else:
        new_config.speaker.lsvsc_speakers = []

    if any(x in datasets_lower for x in ["cv", "common_voice", "commonvoice", "c"]):
        pass
    else:
        new_config.speaker.cv_speakers = []

    return new_config

def generate_full_dataset(
    output_dir: Path = None,
    seed: int = None,
    dry_run: bool = False,
    datasets: list = None
):

    config = DEFAULT_CONFIG
    output_dir = output_dir or config.dataset.output_path

    if not datasets:
        print("\n" + "=" * 60)
        print("🔄 AUTO MODE: Running all datasets sequentially")
        print("=" * 60)
        print("Will generate:")
        print("  1. VIVOS dataset")
        print("  2. LSVSC dataset")
        print("  3. Common Voice dataset")
        print()

        if dry_run:
            print("[DRY RUN] Would run each dataset separately")
            return

        all_results = []
        for dataset_name in ["vivos", "lsvsc", "cv"]:
            print("\n" + "🔹" * 30)
            print(f"Starting dataset: {dataset_name.upper()}")
            print("🔹" * 30 + "\n")

            results = generate_full_dataset(
                output_dir=output_dir,
                seed=seed,
                dry_run=False,
                datasets=[dataset_name]
            )
            all_results.extend(results)

        return all_results

    config = filter_speakers_by_dataset(config, datasets)
    print(f"\n🔍 Filtering speakers by datasets: {', '.join(datasets)}")

    meetings_plan = calculate_meetings_needed(config)
    total_meetings = sum(meetings_plan.values())

    print("=" * 60)
    print("MEETING DATASET GENERATOR")
    print("=" * 60)
    print(f"\nTarget: {config.dataset.total_hours} hours")
    print(f"Meeting duration: {config.meeting.min_duration_minutes}-{config.meeting.max_duration_minutes} min")

    print(f"\nSpeakers (Total: {len(config.speaker.all_speakers)}):")
    for dataset_name, speakers in config.speaker.datasets.items():
        print(f"  📁 {dataset_name}: {len(speakers)} speakers")
        print(f"     {', '.join(speakers)}")

    print(f"\nPlan:")
    for num_speakers, count in meetings_plan.items():
        print(f"  - {num_speakers} speakers: {count} meetings")
    print(f"  Total: {total_meetings} meetings")
    print(f"\nOutput: {output_dir}")

    if dry_run:
        print("\n[DRY RUN] Không generate, chỉ hiển thị plan")
        return

    if seed:
        random.seed(seed)
        print(f"\nRandom seed: {seed}")

    print("\nInitializing generator...")
    generator = MeetingGenerator(config)

    ensure_dir(output_dir)

    all_results = []
    start_time = time.time()

    active_datasets = [
        (name, speakers)
        for name, speakers in config.speaker.datasets.items()
        if speakers
    ]

    if not active_datasets:
        print("\n❌ Error: No active datasets with speakers!")
        return []

    print(f"\n📊 Active datasets: {[name for name, _ in active_datasets]}")
    if datasets:
        print(f"️  Each meeting will use speakers from ONLY ONE dataset (no mixing)")

    for num_speakers, num_meetings in meetings_plan.items():
        print(f"\n{'='*60}")
        print(f"Generating Set: {num_speakers} speakers ({num_meetings} meetings)")
        print("=" * 60)

        for i in range(num_meetings):
            try:
                print(f"\n[{i+1}/{num_meetings}] Generating meeting...")

                dataset_name, dataset_speakers = random.choice(active_datasets)

                if len(dataset_speakers) < num_speakers:

                    available_datasets = [
                        (n, s) for n, s in active_datasets
                        if len(s) >= num_speakers
                    ]
                    if not available_datasets:
                        print(f"  ️  Skip: No dataset has {num_speakers} speakers")
                        continue
                    dataset_name, dataset_speakers = random.choice(available_datasets)

                selected_speakers = random.sample(dataset_speakers, num_speakers)

                dataset_folder_name = dataset_name.lower().replace(" ", "_")
                set_dir = output_dir / dataset_folder_name / f"{num_speakers}speakers"
                ensure_dir(set_dir)

                print(f"  📁 Dataset: {dataset_name}")
                print(f"  👥 Speakers: {', '.join(selected_speakers)}")
                print(f"  💾 Output: {set_dir}")

                meeting = generator.generate(
                    num_speakers=num_speakers,
                    duration=random.uniform(
                        config.meeting.min_duration_minutes * 60,
                        config.meeting.max_duration_minutes * 60
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
                    "num_speakers": num_speakers,
                    "duration": meeting.duration,
                    "statistics": meeting.statistics
                })

            except Exception as e:
                print(f"  ❌ Error: {e}")
                all_results.append({
                    "success": False,
                    "error": str(e)
                })

    elapsed = time.time() - start_time
    success_results = [r for r in all_results if r.get("success")]
    success_count = len(success_results)
    total_duration = sum(r["duration"] for r in success_results)

    total_overlap_time = 0.0
    total_silence_time = 0.0
    total_single_speaker_time = 0.0
    speaker_talk_times = {}

    for r in success_results:
        stats = r.get("statistics", {})
        duration = r["duration"]

        total_overlap_time += duration * stats.get("overlap_ratio", 0)
        total_silence_time += duration * stats.get("silence_ratio", 0)
        total_single_speaker_time += duration * stats.get("single_speaker_ratio", 0)

        for speaker_id, talk_time in stats.get("speaker_times", {}).items():
            if speaker_id not in speaker_talk_times:
                speaker_talk_times[speaker_id] = 0.0
            speaker_talk_times[speaker_id] += talk_time

    print("\n" + "=" * 60)
    print("GENERATION COMPLETE")
    print("=" * 60)
    print(f"Success: {success_count}/{len(all_results)}")
    print(f"Total audio: {format_duration(total_duration)} ({total_duration/3600:.2f} hours)")
    print(f"Time elapsed: {format_duration(elapsed)}")
    print(f"Output: {output_dir}")

    if total_duration > 0:
        print(f"\n--- OVERALL STATISTICS ---")
        print(f"Single speaker: {total_single_speaker_time/total_duration*100:.1f}% ({format_duration(total_single_speaker_time)})")
        print(f"Silence:        {total_silence_time/total_duration*100:.1f}% ({format_duration(total_silence_time)})")
        print(f"Overlap:        {total_overlap_time/total_duration*100:.1f}% ({format_duration(total_overlap_time)})")

        print(f"\n--- SPEAKER TALK TIME ---")

        dataset_speakers = {
            "VIVOS": config.speaker.vivos_speakers,
            "LSVSC": config.speaker.lsvsc_speakers,
            "Common Voice": config.speaker.cv_speakers
        }

        total_speech = sum(speaker_talk_times.values())

        for dataset_name, dataset_speaker_list in dataset_speakers.items():
            dataset_total = sum(
                speaker_talk_times.get(spk, 0)
                for spk in dataset_speaker_list
            )
            if dataset_total > 0:
                dataset_pct = dataset_total / total_speech * 100 if total_speech > 0 else 0
                print(f"\n  📁 {dataset_name} ({format_duration(dataset_total)}, {dataset_pct:.1f}%):")

                dataset_speakers_sorted = [
                    (spk, speaker_talk_times.get(spk, 0))
                    for spk in dataset_speaker_list
                    if spk in speaker_talk_times
                ]
                dataset_speakers_sorted.sort(key=lambda x: x[1], reverse=True)

                for speaker_id, talk_time in dataset_speakers_sorted:
                    pct = talk_time / total_speech * 100 if total_speech > 0 else 0
                    print(f"     {speaker_id}: {format_duration(talk_time)} ({pct:.1f}%)")

    return all_results

def generate_test(output_dir: Path = None):

    config = DEFAULT_CONFIG
    output_dir = output_dir or Path("output/test")

    print("=" * 60)
    print("TEST: Generating 1 meeting (60 seconds)")
    print("=" * 60)

    generator = MeetingGenerator(config)

    meeting = generator.generate(
        duration=60.0,
        num_speakers=3,
        seed=42
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
        description="Generate meeting audio dataset"
    )
    parser.add_argument(
        "--mode",
        choices=["full", "test", "plan"],
        default="test",
        help="Mode: full (20h), test (1 meeting), plan (show plan only)"
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
        "--datasets",
        type=str,
        nargs="+",
        default=None,
        help="Datasets to use: vivos, lsvsc, cv (default: all)"
    )

    args = parser.parse_args()

    output_dir = Path(args.output) if args.output else None

    if args.mode == "full":
        generate_full_dataset(output_dir, seed=args.seed, datasets=args.datasets)
    elif args.mode == "test":
        generate_test(output_dir)
    elif args.mode == "plan":
        generate_full_dataset(output_dir, dry_run=True, datasets=args.datasets)
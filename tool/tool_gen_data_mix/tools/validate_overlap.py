import sys
import json
from pathlib import Path
from collections import defaultdict

def validate_overlap(dataset_dir: Path):
    if not dataset_dir.exists():
        print(f"Error: {dataset_dir} does not exist.")
        return

    overlap_dirs = list(dataset_dir.glob("overlap_*"))
    overlap_dirs.sort(key=lambda x: int(x.name.split("_")[1]))

    print(f"{'='*60}")
    print(f"OVERLAP VALIDATION REPORT")
    print(f"{'='*60}")

    for o_dir in overlap_dirs:
        target_overlap_pct = int(o_dir.name.split("_")[1])

        json_files = list(o_dir.rglob("*.json"))
        if not json_files:
            continue

        total_meeting_dur = 0.0
        total_overlap_dur = 0.0
        total_single_dur = 0.0
        total_silence_dur = 0.0

        meeting_count = len(json_files)

        for jf in json_files:
            with open(jf, "r", encoding="utf-8") as f:
                data = json.load(f)

            meeting_dur = data["duration"]
            total_meeting_dur += meeting_dur

            segments = data.get("segments", [])
            points = []
            for seg in segments:
                points.append((seg["start_time"], 1))
                points.append((seg["end_time"], -1))

            points.sort(key=lambda x: (x[0], -x[1]))

            active = 0
            last_t = 0.0

            m_overlap = 0.0
            m_single = 0.0
            m_silence = 0.0

            for t, change in points:
                dur = t - last_t
                if dur > 0:
                    if active == 0:
                        m_silence += dur
                    elif active == 1:
                        m_single += dur
                    else:
                        m_overlap += dur

                active += change
                last_t = t

            if last_t < meeting_dur:
                m_silence += meeting_dur - last_t

            total_overlap_dur += m_overlap
            total_single_dur += m_single
            total_silence_dur += m_silence

        if total_meeting_dur > 0:
            actual_overlap_pct = (total_overlap_dur / total_meeting_dur) * 100
            actual_single_pct = (total_single_dur / total_meeting_dur) * 100
            actual_silence_pct = (total_silence_dur / total_meeting_dur) * 100

            status = (
                "✅ PASSED"
                if abs(actual_overlap_pct - target_overlap_pct) < 3.0
                else "️ WARNING"
            )
            if target_overlap_pct == 0 and actual_overlap_pct > 0.0:
                status = "❌ FAILED (Must be exact 0%)"
            elif target_overlap_pct == 0 and actual_overlap_pct == 0.0:
                status = "✅ PERFECT"

            print(f"\n[{o_dir.name.upper()}] - Target: {target_overlap_pct}%")
            print(f"  Total Meetings: {meeting_count}")
            print(f"  Total Duration: {total_meeting_dur/3600:.2f} hours")
            print(f"  Actual Overlap: {actual_overlap_pct:.2f}% {status}")
            print(f"  Actual Single : {actual_single_pct:.2f}%")
            print(f"  Actual Silence: {actual_silence_pct:.2f}%")

    print(f"\n{'='*60}")

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--dir", type=str, default="/mnt/sda1/du_an1/data/val/output/overlap_dataset"
    )
    args = parser.parse_args()

    validate_overlap(Path(args.dir))
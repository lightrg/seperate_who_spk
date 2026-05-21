import wave
import json
import sys

def validate_meeting(base_path):
    wav_path = base_path + ".wav"
    json_path = base_path + ".json"
    rttm_path = base_path + ".rttm"
    txt_path = base_path + ".txt"

    with wave.open(wav_path, "rb") as wf:
        frames = wf.getnframes()
        rate = wf.getframerate()
        wav_duration = frames / float(rate)
        print(f"WAV Duration: {wav_duration:.3f}s ({wav_duration/60:.2f} min)")
        print(f"Sample Rate: {rate} Hz")
        print(f"Channels: {wf.getnchannels()}")

    with open(json_path, "r", encoding="utf-8") as f:
        data = json.load(f)
        json_duration = data["duration"]
        overlap_ratio = data["statistics"]["overlap_ratio"]
        print(f"\nJSON Duration: {json_duration:.3f}s")
        print(f"JSON overlap_ratio: {overlap_ratio:.4f} ({overlap_ratio*100:.2f}%)")

    max_rttm_end = 0
    with open(rttm_path, "r", encoding="utf-8") as f:
        for line in f:
            parts = line.strip().split()
            start = float(parts[3])
            dur = float(parts[4])
            end = start + dur
            if end > max_rttm_end:
                max_rttm_end = end
    print(f"\nRTTM Max End Time: {max_rttm_end:.3f}s")

    max_txt_end = 0
    overlap_count = 0
    total_lines = 0
    with open(txt_path, "r", encoding="utf-8") as f:
        for line in f:
            total_lines += 1
            parts = line.strip().split()
            if len(parts) >= 2:
                end = float(parts[1])
                if end > max_txt_end:
                    max_txt_end = end
            if "[OVERLAP]" in line:
                overlap_count += 1
    print(f"TXT Max End Time: {max_txt_end:.3f}s")
    print(f"\nTXT Total Lines: {total_lines}")
    print(
        f"TXT Lines with [OVERLAP]: {overlap_count} ({100*overlap_count/total_lines:.1f}%)"
    )

    print(f"\n=== VALIDATION SUMMARY ===")
    print(f"WAV Duration:     {wav_duration:.3f}s")
    print(f"JSON Duration:    {json_duration:.3f}s")
    print(f"RTTM Max End:     {max_rttm_end:.3f}s")
    print(f"TXT Max End:      {max_txt_end:.3f}s")

    rttm_diff = max_rttm_end - wav_duration
    txt_diff = max_txt_end - wav_duration

    print(
        f'\nRTTM exceeds WAV by: {rttm_diff:.3f}s {"[ERROR]" if rttm_diff > 0.01 else "[OK]"}'
    )
    print(
        f'TXT exceeds WAV by:  {txt_diff:.3f}s {"[ERROR]" if txt_diff > 0.01 else "[OK]"}'
    )

    events = []
    with open(rttm_path, "r", encoding="utf-8") as f:
        for line in f:
            parts = line.strip().split()
            start = float(parts[3])
            dur = float(parts[4])
            events.append((start, 1))
            events.append((start + dur, -1))

    events.sort(key=lambda x: (x[0], -x[1]))

    active = 0
    prev_time = 0
    overlap_time = 0
    speech_time = 0

    for time, delta in events:
        if active >= 2:
            overlap_time += time - prev_time
        if active >= 1:
            speech_time += time - prev_time
        active += delta
        prev_time = time

    actual_overlap_ratio = overlap_time / wav_duration if wav_duration > 0 else 0
    print(f"\nActual overlap time from RTTM: {overlap_time:.2f}s")
    print(
        f"Actual overlap ratio: {actual_overlap_ratio:.4f} ({actual_overlap_ratio*100:.2f}%)"
    )
    print(f"JSON overlap ratio:   {overlap_ratio:.4f} ({overlap_ratio*100:.2f}%)")
    ratio_diff = abs(actual_overlap_ratio - overlap_ratio)
    print(
        f'Ratio difference:     {ratio_diff:.4f} {"[WARNING]" if ratio_diff > 0.02 else "[OK]"}'
    )

if __name__ == "__main__":
    import argparse
    import os

    parser = argparse.ArgumentParser(description="Validate meeting output (wav, json, rttm, txt)")
    parser.add_argument("base_path", help="Base path to the meeting files (without extension)")

    args = parser.parse_args()

    if not os.path.exists(args.base_path + ".wav"):
         print(f"Error: .wav file not found at {args.base_path}.wav")
         sys.exit(1)

    validate_meeting(args.base_path)
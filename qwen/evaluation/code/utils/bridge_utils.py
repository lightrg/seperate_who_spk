import csv
import json
import os

def bridge_asr_diarization(csv_path: str, transcript_jsonl: str, output_txt: str):

    print(f"[V9] Bridging ASR and Diarization labels...")

    speakers_map = {}
    if os.path.exists(transcript_jsonl):
        with open(transcript_jsonl, "r", encoding="utf-8") as f:
            for line in f:
                try:
                    d = json.loads(line)
                    speakers_map[round(float(d["start"]), 2)] = d.get("speaker", "Unknown")
                except: continue

    lines = []
    with open(csv_path, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            start_val = float(row["start"])
            end_val = float(row["end"])
            text = row["predicted_text"]

            spk = speakers_map.get(round(start_val, 2), "Unknown")
            lines.append(f"[{start_val:.1f}s - {end_val:.1f}s] {spk}: {text}")

    if not lines:
        raise ValueError("No segments found to bridge.")

    os.makedirs(os.path.dirname(output_txt), exist_ok=True)
    with open(output_txt, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))

    print(f"[V9] Bridged input created at {output_txt}")
    return output_txt

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--csv", required=True)
    parser.add_argument("--jsonl", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    bridge_asr_diarization(args.csv, args.jsonl, args.output)
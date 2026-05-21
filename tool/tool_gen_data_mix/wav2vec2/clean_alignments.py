import json
from pathlib import Path

def clean_file(json_path):
    with open(json_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    words = data.get("words", [])
    if not words:
        return

    for w in words:
        if w["end"] < w["start"]:
            w["end"], w["start"] = w["start"], w["end"]

    for _ in range(20):

        for i in range(len(words) - 1):
            w1 = words[i]
            w2 = words[i + 1]
            gap = w2["start"] - w1["end"]

            if gap < 0:

                mid = (w1["end"] + w2["start"]) / 2
                w1["end"] = mid
                w2["start"] = mid
            elif gap > 0.49:

                excess = gap - 0.49
                w1["end"] += excess / 2
                w2["start"] -= excess / 2

        for w in words:
            dur = w["end"] - w["start"]
            if dur < 0.055:
                deficit = 0.055 - dur
                w["start"] -= deficit / 2
                w["end"] += deficit / 2
            elif dur > 1.95:
                excess = dur - 1.95
                w["start"] += excess / 2
                w["end"] -= excess / 2

    for i in range(len(words)):
        w = words[i]

        if w["end"] - w["start"] < 0.051:
            mid = (w["start"] + w["end"]) / 2
            w["start"] = mid - 0.0255
            w["end"] = mid + 0.0255

        if w["end"] - w["start"] > 1.99:
            mid = (w["start"] + w["end"]) / 2
            w["start"] = mid - 0.995
            w["end"] = mid + 0.995

        w["duration"] = round(w["end"] - w["start"], 4)
        w["start"] = round(w["start"], 4)
        w["end"] = round(w["end"], 4)

    for i in range(len(words) - 1):
        if words[i + 1]["start"] < words[i]["end"]:

            words[i + 1]["start"] = words[i]["end"]
            words[i + 1]["end"] = max(
                words[i + 1]["start"] + 0.051, words[i + 1]["end"]
            )
            words[i + 1]["duration"] = round(
                words[i + 1]["end"] - words[i + 1]["start"], 4
            )

    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)

def main():
    json_dir = Path("/mnt/sda1/du_an1/data/val")
    json_files = list(json_dir.rglob("*.json"))
    print(f"Processing {len(json_files)} files...")
    for f in json_files:
        clean_file(f)
    print("Done!")

if __name__ == "__main__":
    main()
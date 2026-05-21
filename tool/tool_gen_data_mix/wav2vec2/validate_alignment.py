

import json
import argparse
from pathlib import Path
from typing import List, Dict, Any
import statistics
from collections import defaultdict

class AlignmentValidator:

    def __init__(self):
        self.stats = defaultdict(list)
        self.errors = []
        self.warnings = []

    def validate_file(self, json_path: Path) -> Dict[str, Any]:

        with open(json_path, "r", encoding="utf-8") as f:
            data = json.load(f)

        words = data.get("words", [])

        if not words:
            self.warnings.append(f"{json_path.name}: No words found")
            return {"valid": False, "reason": "no_words"}

        results = {
            "file": json_path.name,
            "num_words": len(words),
            "total_duration": 0,
            "avg_word_duration": 0,
            "gaps": [],
            "overlaps": [],
            "very_short_words": [],
            "very_long_words": [],
        }

        if words:
            results["total_duration"] = words[-1]["end"] - words[0]["start"]

        for i, word in enumerate(words):
            duration = word["duration"]

            self.stats["word_durations"].append(duration)

            if duration < 0.05:
                results["very_short_words"].append(
                    {"word": word["word"], "duration": duration, "position": i}
                )

            if duration > 2.0:
                results["very_long_words"].append(
                    {"word": word["word"], "duration": duration, "position": i}
                )

            if i < len(words) - 1:
                next_word = words[i + 1]
                gap = next_word["start"] - word["end"]

                if gap > 0.5:
                    results["gaps"].append(
                        {"after_word": word["word"], "gap_duration": gap, "position": i}
                    )
                    self.stats["gaps"].append(gap)

                if gap < 0:
                    results["overlaps"].append(
                        {
                            "words": f"{word['word']} -> {next_word['word']}",
                            "overlap": abs(gap),
                            "position": i,
                        }
                    )
                    self.errors.append(
                        f"{json_path.name}: Overlap at position {i} "
                        f"({word['word']} -> {next_word['word']})"
                    )

        if words:
            results["avg_word_duration"] = results["total_duration"] / len(words)

        return results

    def validate_directory(self, json_dir: Path) -> Dict[str, Any]:

        json_files = list(json_dir.rglob("*.json"))

        if not json_files:
            print(f"No JSON files found in {json_dir}")
            return {}

        print(f"Validating {len(json_files)} alignment files...")

        all_results = []

        for json_file in json_files:
            try:
                result = self.validate_file(json_file)
                all_results.append(result)
            except Exception as e:
                self.errors.append(f"Error validating {json_file.name}: {e}")

        return self.generate_summary(all_results)

    def generate_summary(self, results: List[Dict[str, Any]]) -> Dict[str, Any]:

        summary = {
            "total_files": len(results),
            "total_words": sum(r["num_words"] for r in results),
            "files_with_gaps": [r["file"] for r in results if r["gaps"]],
            "files_with_overlaps": [r["file"] for r in results if r["overlaps"]],
            "files_with_short_words": [
                r["file"] for r in results if r["very_short_words"]
            ],
            "files_with_long_words": [
                r["file"] for r in results if r["very_long_words"]
            ],
            "total_errors": len(self.errors),
            "total_warnings": len(self.warnings),
        }

        if self.stats["word_durations"]:
            summary["word_duration_stats"] = {
                "mean": statistics.mean(self.stats["word_durations"]),
                "median": statistics.median(self.stats["word_durations"]),
                "stdev": (
                    statistics.stdev(self.stats["word_durations"])
                    if len(self.stats["word_durations"]) > 1
                    else 0
                ),
                "min": min(self.stats["word_durations"]),
                "max": max(self.stats["word_durations"]),
            }

        if self.stats["gaps"]:
            summary["gap_stats"] = {
                "mean": statistics.mean(self.stats["gaps"]),
                "median": statistics.median(self.stats["gaps"]),
                "max": max(self.stats["gaps"]),
            }

        return summary

    def print_report(self, summary: Dict[str, Any], verbose: bool = False):

        print("\n" + "=" * 60)
        print("Alignment Validation Report")
        print("=" * 60)

        print(f"\nFiles processed: {summary['total_files']}")
        print(f"Total words aligned: {summary['total_words']}")

        print(f"\n--- Issues Found ---")
        print(f"Files with gaps: {len(summary['files_with_gaps'])}")
        if verbose and summary["files_with_gaps"]:
            print("  List of files with gaps:")
            for f in summary["files_with_gaps"]:
                print(f"    - {f}")

        print(f"Files with overlaps: {len(summary['files_with_overlaps'])}")
        if verbose and summary["files_with_overlaps"]:
            print("  List of files with overlaps:")
            for f in summary["files_with_overlaps"]:
                print(f"    - {f}")

        print(
            f"Files with very short words (<50ms): {len(summary['files_with_short_words'])}"
        )
        if verbose and summary["files_with_short_words"]:
            print("  List of files with very short words:")
            for f in summary["files_with_short_words"]:
                print(f"    - {f}")

        print(
            f"Files with very long words (>2s): {len(summary['files_with_long_words'])}"
        )
        if verbose and summary["files_with_long_words"]:
            print("  List of files with very long words:")
            for f in summary["files_with_long_words"]:
                print(f"    - {f}")

        print(f"Total errors: {summary['total_errors']}")
        print(f"Total warnings: {summary['total_warnings']}")

        if "word_duration_stats" in summary:
            stats = summary["word_duration_stats"]
            print(f"\n--- Word Duration Statistics ---")
            print(f"Mean: {stats['mean']:.3f}s")
            print(f"Median: {stats['median']:.3f}s")
            print(f"Std Dev: {stats['stdev']:.3f}s")
            print(f"Min: {stats['min']:.3f}s")
            print(f"Max: {stats['max']:.3f}s")

        if "gap_stats" in summary:
            stats = summary["gap_stats"]
            print(f"\n--- Gap Statistics ---")
            print(f"Mean gap: {stats['mean']:.3f}s")
            print(f"Median gap: {stats['median']:.3f}s")
            print(f"Max gap: {stats['max']:.3f}s")

        if self.errors:
            print(f"\n--- Sample Errors (first 10) ---")
            for error in self.errors[:10]:
                print(f"  • {error}")
            if len(self.errors) > 10:
                print(f"  ... and {len(self.errors) - 10} more")

        if self.warnings:
            print(f"\n--- Sample Warnings (first 10) ---")
            for warning in self.warnings[:10]:
                print(f"  • {warning}")
            if len(self.warnings) > 10:
                print(f"  ... and {len(self.warnings) - 10} more")

        print("\n" + "=" * 60)

        quality_score = 100
        quality_score -= min(len(summary["files_with_overlaps"]) * 2, 30)
        quality_score -= min(len(summary["files_with_gaps"]) * 0.5, 20)
        quality_score -= min(summary["total_errors"] * 1, 30)

        print(f"\nOverall Quality Score: {max(quality_score, 0):.1f}/100")

        if quality_score >= 90:
            print(" Excellent alignment quality!")
        elif quality_score >= 70:
            print(" Good alignment quality with minor issues")
        elif quality_score >= 50:
            print(" Fair alignment quality with some issues")
        else:
            print(" Poor alignment quality - review recommended")

        print("=" * 60)

def main():
    parser = argparse.ArgumentParser(description="Validate forced alignment quality")
    project_root = Path(__file__).parent.parent
    parser.add_argument(
        "--input-dir",
        type=Path,
        default=project_root / "data/val",
        help="Directory containing alignment JSON files",
    )
    parser.add_argument(
        "--output-report", type=Path, help="Optional: Save report to JSON file"
    )
    parser.add_argument(
        "--verbose", action="store_true", help="Print details of files with issues"
    )

    args = parser.parse_args()

    if not args.input_dir.exists():
        print(f"Error: Input directory not found: {args.input_dir}")
        return 1

    validator = AlignmentValidator()
    summary = validator.validate_directory(args.input_dir)

    if summary:
        validator.print_report(summary, verbose=args.verbose)

        if args.output_report:
            with open(args.output_report, "w", encoding="utf-8") as f:
                json.dump(summary, f, ensure_ascii=False, indent=2)
            print(f"\n Report saved to: {args.output_report}")

    return 0

if __name__ == "__main__":
    exit(main())
import os
import json
import glob
import argparse
from pathlib import Path
from collections import defaultdict

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--eval_dir", type=str, default="output/qwen/eval", help="Directory containing judge_pass*.json files")
    args = parser.add_argument_group()
    args = parser.parse_args()

    search_dir = Path(args.eval_dir)
    report_files = [f for f in search_dir.rglob("*judge*.json") if "FINAL" not in f.name]
    
    # Group reports by summary_json
    grouped_reports = defaultdict(list)
    
    for file_path in report_files:
        with open(file_path, "r", encoding="utf-8") as f:
            try:
                data = json.load(f)
            except:
                continue
                
        summary_path = data.get("summary_json", str(file_path))
        grouped_reports[summary_path].append(data)
            
    print(f"Found {len(report_files)} report files across {len(grouped_reports)} unique summaries/domains.")
    
    # Aggregate scores for each domain
    domain_avg_scores = {}
    all_issues = []
    
    for summary_path, passes in grouped_reports.items():
        domain_scores = {
            "main_summary_score": [],
            "speaker_summary_score": [],
            "faithfulness_score": [],
            "conciseness_score": [],
            "total_score": []
        }
        
        for data in passes:
            overall = data.get("judge_output", {}).get("overall", {})
            for k in domain_scores.keys():
                if k in overall:
                    domain_scores[k].append(overall[k])
                    
            issues = data.get("judge_output", {}).get("issues", [])
            all_issues.extend(issues)
            
        # Average passes for this domain
        avg_for_this_domain = {}
        for k, v in domain_scores.items():
            if v:
                avg_for_this_domain[k] = sum(v) / len(v)
            else:
                avg_for_this_domain[k] = 0.0
                
        domain_avg_scores[summary_path] = avg_for_this_domain
        
    # Global average across all domains
    global_scores = {
        "main_summary_score": [],
        "speaker_summary_score": [],
        "faithfulness_score": [],
        "conciseness_score": [],
        "total_score": []
    }
    
    for avg_scores in domain_avg_scores.values():
        for k in global_scores.keys():
            global_scores[k].append(avg_scores[k])
            
    final_global_avg = {}
    for k, v in global_scores.items():
        if v:
            final_global_avg[k] = sum(v) / len(v)
        else:
            final_global_avg[k] = 0.0
            
    # Remove exact duplicate issues to keep the list clean
    unique_issues = list(dict.fromkeys(all_issues))
    
    final_report = {
        "num_domains_evaluated": len(grouped_reports),
        "total_passes_analyzed": len(report_files),
        "global_average_scores": final_global_avg,
        "domain_average_scores": domain_avg_scores,
        "aggregated_issues": unique_issues,
        "files_analyzed": [str(p) for p in report_files]
    }
    
    out_json = search_dir / "FINAL_JUDGE_REPORT.json"
    with open(out_json, "w", encoding="utf-8") as f:
        json.dump(final_report, f, ensure_ascii=False, indent=2)
        
    out_md = search_dir / "FINAL_JUDGE_REPORT.md"
    with open(out_md, "w", encoding="utf-8") as f:
        f.write("# Qwen 7B Judge - Aggregated Final Report\n\n")
        f.write(f"**Number of domains evaluated:** {len(grouped_reports)}\n")
        f.write(f"**Total passes analyzed:** {len(report_files)}\n\n")
        f.write("## 1. Global Average Scores\n")
        f.write(f"- **Main Summary Score:** {final_global_avg.get('main_summary_score', 0):.2f} / 3.0\n")
        f.write(f"- **Speaker Summary Score:** {final_global_avg.get('speaker_summary_score', 0):.2f} / 3.0\n")
        f.write(f"- **Faithfulness Score:** {final_global_avg.get('faithfulness_score', 0):.2f} / 3.0\n")
        f.write(f"- **Conciseness Score:** {final_global_avg.get('conciseness_score', 0):.2f} / 3.0\n")
        f.write(f"- **Total Score:** {final_global_avg.get('total_score', 0):.2f} / 12.0\n\n")
        
        f.write("## 2. Aggregated Issues\n")
        if unique_issues:
            for issue in unique_issues:
                f.write(f"- {issue}\n")
        else:
            f.write("- No critical issues found.\n")
            
    print(f"Aggregated reports saved to {out_json} and {out_md}")

if __name__ == "__main__":
    main()

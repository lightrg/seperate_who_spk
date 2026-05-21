from __future__ import annotations

import argparse
import dataclasses
import json
import math
import re
import sys
import statistics
from pathlib import Path
sys.path.append(str(Path(__file__).resolve().parent.parent))
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

GENERIC_PHRASES = [
    "không đủ bằng chứng",
    "đoạn trao đổi khác",
    "đoạn trao đổi chính",
    "đang làm gì trong cuộc hội thoại",
    "tham gia trao đổi về",
]

REQUIRED_TOP_LEVEL_KEYS = [
    "meeting_overview",
    "conversation_main_summary",
    "meeting_type",
    "segments",
    "action_items",
    "speaker_insights",
    "speaker_main_summaries",
    "cleaned_transcript",
]

@dataclasses.dataclass
class TranscriptLine:
    start: float
    end: float
    speaker: str
    text: str

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Evaluate summary JSON without modifying generator code.")
    p.add_argument("--summary_json", required=True, help="Generated summary JSON to evaluate.")
    p.add_argument("--transcript", default=None, help="Optional original transcript.jsonl or plain transcript text.")
    p.add_argument("--reference_json", default=None, help="Optional human reference JSON for lexical/semantic comparison.")
    p.add_argument("--report_json", default=None, help="Path to save machine-readable report JSON.")
    p.add_argument("--report_md", default=None, help="Path to save human-readable markdown report.")
    p.add_argument("--compute_bertscore", action="store_true", help="Try to compute BERTScore if bert_score is installed.")
    return p.parse_args()

def load_json(path: str | Path) -> Dict[str, Any]:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)

def safe_float(x: Any, default: float = 0.0) -> float:
    try:
        return float(x)
    except Exception:
        return default

def tokenize_for_rouge(text: str) -> List[str]:
    text = text.lower().strip()
    text = re.sub(r"\s+", " ", text)
    return text.split()

def rouge_l_f1(pred: str, ref: str) -> float:
    a = tokenize_for_rouge(pred)
    b = tokenize_for_rouge(ref)
    if not a or not b:
        return 0.0
    dp = [[0] * (len(b) + 1) for _ in range(len(a) + 1)]
    for i in range(1, len(a) + 1):
        for j in range(1, len(b) + 1):
            if a[i - 1] == b[j - 1]:
                dp[i][j] = dp[i - 1][j - 1] + 1
            else:
                dp[i][j] = max(dp[i - 1][j], dp[i][j - 1])
    lcs = dp[-1][-1]
    prec = lcs / len(a) if a else 0.0
    rec = lcs / len(b) if b else 0.0
    if prec + rec == 0:
        return 0.0
    return 2 * prec * rec / (prec + rec)

def load_transcript(path: Optional[str]) -> List[TranscriptLine]:
    if not path:
        return []
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(f"Transcript not found: {p}")

    lines: List[TranscriptLine] = []
    if p.suffix.lower() == ".jsonl":
        with open(p, "r", encoding="utf-8") as f:
            for raw in f:
                raw = raw.strip()
                if not raw:
                    continue
                obj = json.loads(raw)
                start = safe_float(obj.get("start", obj.get("start_time", 0.0)))
                end = safe_float(obj.get("end", obj.get("end_time", start)))
                speaker = str(obj.get("speaker", obj.get("speaker_id", "unknown")))
                text = str(obj.get("text", obj.get("sentence", ""))).strip()
                if text:
                    lines.append(TranscriptLine(start=start, end=end, speaker=speaker, text=text))
        return lines

    pattern = re.compile(r"\[(?P<start>[\d.]+)s?\s*-\s*(?P<end>[\d.]+)s?\]\s*(?P<speaker>[^:]+):\s*(?P<text>.+)")
    with open(p, "r", encoding="utf-8") as f:
        for raw in f:
            m = pattern.match(raw.strip())
            if not m:
                continue
            lines.append(
                TranscriptLine(
                    start=safe_float(m.group("start")),
                    end=safe_float(m.group("end")),
                    speaker=m.group("speaker").strip(),
                    text=m.group("text").strip(),
                )
            )
    return lines

def span_intersects(a0: float, a1: float, b0: float, b1: float) -> bool:
    return not (a1 < b0 or b1 < a0)

def avg_words(texts: Sequence[str]) -> float:
    vals = [len(t.split()) for t in texts if t and t.strip()]
    return statistics.mean(vals) if vals else 0.0

def text_has_generic_phrase(text: str) -> bool:
    lo = text.lower()
    return any(phrase in lo for phrase in GENERIC_PHRASES)

def evaluate_schema(summary: Dict[str, Any]) -> Dict[str, Any]:
    missing = [k for k in REQUIRED_TOP_LEVEL_KEYS if k not in summary]
    return {
        "required_keys_present": len(missing) == 0,
        "missing_required_keys": missing,
        "schema_pass_rate": (len(REQUIRED_TOP_LEVEL_KEYS) - len(missing)) / len(REQUIRED_TOP_LEVEL_KEYS),
    }

def evaluate_lengths(summary: Dict[str, Any]) -> Dict[str, Any]:
    convo = str(summary.get("conversation_main_summary", "")).strip()
    speaker_summaries = summary.get("speaker_main_summaries", []) or []
    sms = [str(x.get("main_summary", "")).strip() for x in speaker_summaries if isinstance(x, dict)]
    return {
        "conversation_main_summary_words": len(convo.split()),
        "speaker_main_summary_count": len(sms),
        "speaker_main_summary_avg_words": avg_words(sms),
        "speaker_main_summary_max_words": max((len(x.split()) for x in sms), default=0),
    }

def evaluate_evidence(summary: Dict[str, Any], transcript: List[TranscriptLine]) -> Dict[str, Any]:
    speaker_items = summary.get("speaker_main_summaries", []) or []
    total = len(speaker_items)
    supported = 0
    valid_span_items = 0
    speaker_present = 0
    per_item = []

    transcript_by_speaker: Dict[str, List[TranscriptLine]] = defaultdict(list)
    for line in transcript:
        transcript_by_speaker[line.speaker].append(line)

    for item in speaker_items:
        if not isinstance(item, dict):
            continue
        speaker = str(item.get("speaker", ""))
        spans = item.get("evidence_spans", []) or []
        has_valid_span = False
        has_support = False
        if speaker in transcript_by_speaker:
            speaker_present += 1
        for sp in spans:
            if not isinstance(sp, (list, tuple)) or len(sp) != 2:
                continue
            s0, s1 = safe_float(sp[0]), safe_float(sp[1])
            if s1 >= s0:
                has_valid_span = True
            for line in transcript_by_speaker.get(speaker, []):
                if span_intersects(s0, s1, line.start, line.end):
                    has_support = True
                    break
            if has_support:
                break
        if has_valid_span:
            valid_span_items += 1
        if has_support:
            supported += 1
        per_item.append({
            "speaker": speaker,
            "has_valid_span": has_valid_span,
            "supported_by_transcript": has_support,
        })

    return {
        "speaker_summary_items": total,
        "valid_evidence_span_rate": valid_span_items / total if total else 0.0,
        "evidence_support_rate": supported / total if total else 0.0,
        "speaker_presence_rate": speaker_present / total if total else 0.0,
        "items": per_item,
    }

def evaluate_genericness(summary: Dict[str, Any]) -> Dict[str, Any]:
    speaker_items = summary.get("speaker_main_summaries", []) or []
    texts = [str(x.get("main_summary", "")).strip() for x in speaker_items if isinstance(x, dict)]
    generic = sum(1 for t in texts if text_has_generic_phrase(t))
    low_conf = sum(1 for x in speaker_items if isinstance(x, dict) and str(x.get("confidence", "")).lower() == "low")
    return {
        "generic_speaker_summary_rate": generic / len(texts) if texts else 0.0,
        "low_confidence_rate": low_conf / len(texts) if texts else 0.0,
    }

def normalize_reference_speaker_map(reference: Dict[str, Any]) -> Dict[str, str]:
    out: Dict[str, str] = {}
    for item in reference.get("speaker_main_summaries", []) or []:
        if isinstance(item, dict) and item.get("speaker"):
            out[str(item["speaker"])] = str(item.get("main_summary", "")).strip()
    return out

def evaluate_against_reference(summary: Dict[str, Any], reference: Dict[str, Any], compute_bertscore: bool) -> Dict[str, Any]:
    results: Dict[str, Any] = {}

    pred_main = str(summary.get("conversation_main_summary", "")).strip()
    ref_main = str(reference.get("conversation_main_summary", "")).strip()
    results["conversation_main_rouge_l_f1"] = rouge_l_f1(pred_main, ref_main) if ref_main else None

    pred_speakers = {str(x.get("speaker")): str(x.get("main_summary", "")).strip()
                     for x in summary.get("speaker_main_summaries", []) if isinstance(x, dict) and x.get("speaker")}
    ref_speakers = normalize_reference_speaker_map(reference)
    common = sorted(set(pred_speakers) & set(ref_speakers))
    if common:
        vals = [rouge_l_f1(pred_speakers[s], ref_speakers[s]) for s in common]
        results["speaker_main_rouge_l_f1_avg"] = sum(vals) / len(vals)
        results["speaker_main_common_count"] = len(common)
    else:
        results["speaker_main_rouge_l_f1_avg"] = None
        results["speaker_main_common_count"] = 0

    if compute_bertscore:
        try:
            from bert_score import score as bertscore_score
            candidates = []
            references = []
            labels = []
            if pred_main and ref_main:
                candidates.append(pred_main)
                references.append(ref_main)
                labels.append("conversation_main")
            for s in common:
                candidates.append(pred_speakers[s])
                references.append(ref_speakers[s])
                labels.append(f"speaker::{s}")
            if candidates:
                _, _, f1 = bertscore_score(candidates, references, lang="vi", verbose=False)
                f1_vals = [float(x) for x in f1]
                results["bertscore_f1_avg"] = sum(f1_vals) / len(f1_vals)
                results["bertscore_breakdown"] = dict(zip(labels, f1_vals))
            else:
                results["bertscore_f1_avg"] = None
                results["bertscore_breakdown"] = {}
        except Exception as e:
            results["bertscore_f1_avg"] = None
            results["bertscore_error"] = str(e)
    return results

def aggregate_score(parts: Dict[str, Any]) -> Dict[str, Any]:
    schema = parts["schema"]["schema_pass_rate"]
    evidence = parts["evidence"]["evidence_support_rate"]
    generic_penalty = 1.0 - parts["genericness"]["generic_speaker_summary_rate"]
    length = parts["lengths"]
    compact = 1.0
    if length["speaker_main_summary_avg_words"] > 30:
        compact = max(0.0, 1.0 - (length["speaker_main_summary_avg_words"] - 30) / 50)

    final = 0.35 * schema + 0.35 * evidence + 0.15 * generic_penalty + 0.15 * compact
    if parts.get("reference"):
        ref = parts["reference"]
        rouge = ref.get("conversation_main_rouge_l_f1")
        if isinstance(rouge, (float, int)):
            final = 0.25 * final + 0.35 * float(rouge) + 0.40 * (ref.get("speaker_main_rouge_l_f1_avg") or 0.0)
    return {
        "summary_quality_score": round(final, 4),
        "interpretation": (
            "good" if final >= 0.75 else
            "usable" if final >= 0.55 else
            "needs_review"
        ),
    }

def make_markdown(report: Dict[str, Any]) -> str:
    lines = ["# Summary Evaluation Report", ""]
    lines.append(f"- Overall score: **{report['aggregate']['summary_quality_score']:.4f}**")
    lines.append(f"- Interpretation: **{report['aggregate']['interpretation']}**")
    lines.append("")

    s = report["schema"]
    lines.append("## Schema")
    lines.append(f"- required_keys_present: `{s['required_keys_present']}`")
    lines.append(f"- schema_pass_rate: `{s['schema_pass_rate']:.3f}`")
    if s["missing_required_keys"]:
        lines.append(f"- missing_required_keys: `{', '.join(s['missing_required_keys'])}`")
    lines.append("")

    l = report["lengths"]
    lines.append("## Length / Conciseness")
    lines.append(f"- conversation_main_summary_words: `{l['conversation_main_summary_words']}`")
    lines.append(f"- speaker_main_summary_avg_words: `{l['speaker_main_summary_avg_words']:.2f}`")
    lines.append(f"- speaker_main_summary_max_words: `{l['speaker_main_summary_max_words']}`")
    lines.append("")

    e = report["evidence"]
    lines.append("## Evidence")
    lines.append(f"- valid_evidence_span_rate: `{e['valid_evidence_span_rate']:.3f}`")
    lines.append(f"- evidence_support_rate: `{e['evidence_support_rate']:.3f}`")
    lines.append(f"- speaker_presence_rate: `{e['speaker_presence_rate']:.3f}`")
    lines.append("")

    g = report["genericness"]
    lines.append("## Genericness / Confidence")
    lines.append(f"- generic_speaker_summary_rate: `{g['generic_speaker_summary_rate']:.3f}`")
    lines.append(f"- low_confidence_rate: `{g['low_confidence_rate']:.3f}`")
    lines.append("")

    if report.get("reference"):
        r = report["reference"]
        lines.append("## Reference Comparison")
        for key, value in r.items():
            if isinstance(value, (float, int)):
                lines.append(f"- {key}: `{value:.4f}`")
            elif value is not None:
                lines.append(f"- {key}: `{value}`")
        lines.append("")
    return "\n".join(lines)

def main() -> None:
    args = parse_args()
    summary = load_json(args.summary_json)
    transcript = load_transcript(args.transcript)

    report: Dict[str, Any] = {
        "inputs": {
            "summary_json": str(Path(args.summary_json).resolve()),
            "transcript": str(Path(args.transcript).resolve()) if args.transcript else None,
            "reference_json": str(Path(args.reference_json).resolve()) if args.reference_json else None,
        },
        "schema": evaluate_schema(summary),
        "lengths": evaluate_lengths(summary),
        "evidence": evaluate_evidence(summary, transcript),
        "genericness": evaluate_genericness(summary),
    }

    if args.reference_json:
        reference = load_json(args.reference_json)
        report["reference"] = evaluate_against_reference(summary, reference, args.compute_bertscore)

    report["aggregate"] = aggregate_score(report)

    out_json = args.report_json or str(Path(args.summary_json).with_suffix(".eval.json"))
    out_md = args.report_md or str(Path(args.summary_json).with_suffix(".eval.md"))

    with open(out_json, "w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=2)
    with open(out_md, "w", encoding="utf-8") as f:
        f.write(make_markdown(report))

    print(f"Saved JSON report: {out_json}")
    print(f"Saved Markdown report: {out_md}")
    print(f"Overall score: {report['aggregate']['summary_quality_score']:.4f} ({report['aggregate']['interpretation']})")

if __name__ == "__main__":
    main()
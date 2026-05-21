"""
eval_wer_from_csv.py  (v5 -- segment-level WER, time-overlap ref assignment)
─────────────────────────────────────────────────────────────────────────────
STM và RTTM hoàn toàn độc lập.

Pipeline:
    run_asr_inference.py  →  predictions.csv
                                  ↓
    eval_wer_from_csv.py
        1. Load STM per file (ground truth text + timestamp)
        2. Với mỗi RTTM segment trong CSV:
              a. Hallucination / no-speech → predicted_text="" → DROP
              b. Tìm TẤT CẢ STM overlap đủ ngưỡng (many-to-many, v5 fix)
                 - Rule: overlap >= 0.2s OR >= 20% stm_duration
                 - Ưu tiên same-speaker; fallback bất kỳ speaker
                 - Proportional word slice theo vùng overlap
              c. Nếu không match STM nào → ref="" → tính insertion
        3. Tính WER segment-level, cộng dồn theo file và theo group
        4. Report Markdown + CSV

Quyết định thiết kế:
  - v5 BUG FIX: đổi ownership model từ STM→RTTM (1-to-1) sang RTTM→STM
    (many-to-many). STM dài (79–193s, slide thuyết trình) có thể được
    match bởi nhiều RTTM chunks 27s — không còn tranh giành owner.
  - Overlap rule OR: abs >= 0.2s HOẶC ratio >= 20% stm_duration.
  - Same-speaker ưu tiên; fallback any-speaker (WER đo ASR, không đo diarization).
  - Proportional slice: mỗi RTTM chỉ lấy phần text tương ứng vùng overlap.
  - Hallucination DROP hoàn toàn.
  - RTTM không match STM → ref="" → insertion.
"""

from __future__ import annotations

import sys
sys.stdout.reconfigure(encoding="utf-8")

import argparse
import csv
import logging
import unicodedata
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np

try:
    from jiwer import process_words as _jiwer_process_words
    _USE_JIWER = True
except ImportError:
    _USE_JIWER = False
    print("[WARN] jiwer chưa cài -- pip install jiwer")

from manifest_builder import load_text_segments, TextSegment

logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
LOGGER = logging.getLogger("eval_wer_from_csv")

# ── Constants ────────────────────────────────────────────────────────────────

# Rule match (v4): overlap_sec >= ABS_THRESHOLD  HOẶC  overlap_sec >= RATIO * stm_duration
#
# Tại sao dùng OR thay vì chỉ 1 rule:
#   - ABS (0.2s): cứu trường hợp STM rất dài (10-15s) mà RTTM ngắn chỉ
#     overlap 1-2s → ratio thấp dù thực tế có overlap thật.
#   - RATIO (0.20): bảo vệ STM ngắn (< 1s) khỏi false match khi RTTM
#     chỉ chạm nhẹ 0.2s vào STM 0.3s (ratio=67% → đúng, không miss).
#     Nếu chỉ dùng ABS thì STM 0.3s bị match bởi RTTM chỉ overlap 0.21s.
OVERLAP_ABS_THRESHOLD: float   = 0.20   # giây tuyệt đối
OVERLAP_RATIO_THRESHOLD: float = 0.20   # tỉ lệ so với stm_duration


# ── Text helpers ─────────────────────────────────────────────────────────────

def normalize_vi_text(text: str) -> str:
    text = unicodedata.normalize("NFC", str(text)).lower()
    text = "".join(ch if (ch.isalpha() or ch.isdigit() or ch.isspace()) else " " for ch in text)
    return " ".join(text.split())


# ── WER ──────────────────────────────────────────────────────────────────────

def compute_wer_details(r: list, h: list) -> Tuple[int, int, int, int, int]:
    """Trả về (total_edits, ref_length, S, D, I)."""
    if _USE_JIWER:
        ref_str, hyp_str = " ".join(r), " ".join(h)
        if not ref_str and not hyp_str:
            return 0, 0, 0, 0, 0
        if not ref_str:
            return len(h), 0, 0, 0, len(h)
        if not hyp_str:
            return len(r), len(r), 0, len(r), 0
        out = _jiwer_process_words(ref_str, hyp_str)
        S, D, I = out.substitutions, out.deletions, out.insertions
        return S + D + I, len(r), S, D, I

    # Fallback numpy DP
    d = np.zeros((len(r) + 1, len(h) + 1), dtype=np.int32)
    for i in range(len(r) + 1): d[i][0] = i
    for j in range(len(h) + 1): d[0][j] = j
    for i in range(1, len(r) + 1):
        for j in range(1, len(h) + 1):
            cost = 0 if r[i-1] == h[j-1] else 1
            d[i][j] = min(d[i-1][j] + 1, d[i][j-1] + 1, d[i-1][j-1] + cost)
    i, j = len(r), len(h)
    S = D = I = 0
    while i > 0 or j > 0:
        if i > 0 and j > 0 and r[i-1] == h[j-1]:
            i -= 1; j -= 1
        elif i > 0 and j > 0 and d[i][j] == d[i-1][j-1] + 1:
            S += 1; i -= 1; j -= 1
        elif i > 0 and d[i][j] == d[i-1][j] + 1:
            D += 1; i -= 1
        else:
            I += 1; j -= 1
    return d[len(r)][len(h)], len(r), S, D, I


# ── Overlap helpers ───────────────────────────────────────────────────────────

def overlap_len(a_start: float, a_end: float,
                b_start: float, b_end: float) -> float:
    return max(0.0, min(a_end, b_end) - max(a_start, b_start))


# ── STM cache ────────────────────────────────────────────────────────────────

def build_stm_cache(data_dir: Path) -> Dict[str, List[TextSegment]]:
    """
    Load STM segments (có timestamp) cho từng data_name.
    Trả về dict: data_name → List[TextSegment] đã sort theo start.

    Giữ nguyên timestamp để dùng cho time-overlap matching.
    Không concat thành chuỗi ở bước này.
    """
    cache: Dict[str, List[TextSegment]] = {}
    for d in sorted(data_dir.glob("data*")):
        if not d.is_dir():
            continue
        labeled_dir = d / "labeled"
        if not labeled_dir.exists():
            continue
        segs, _ = load_text_segments(labeled_dir)
        if not segs:
            continue
        segs_sorted = sorted(segs, key=lambda s: s.start)
        # Normalize text ngay khi load, tránh normalize lại nhiều lần
        normalized = []
        for seg in segs_sorted:
            nt = normalize_vi_text(seg.text)
            if nt:
                normalized.append(TextSegment(
                    start=seg.start,
                    end=seg.end,
                    text=nt,
                    speaker=getattr(seg, "speaker", ""),
                ))
        if normalized:
            cache[d.name] = normalized
    LOGGER.info("Đã load STM cho %d file.", len(cache))
    return cache


# ── Time-overlap ref assignment ───────────────────────────────────────────────
#
# Chiến lược (giống eval_manifest_patch._align_rttm_to_stm nhưng đơn giản hơn):
#
# Bước 1: với mỗi STM segment, tìm RTTM segment nào có overlap lớn nhất
#         → đó là "chủ sở hữu" của STM segment này.
#         Ngưỡng: overlap >= OVERLAP_RATIO_THRESHOLD * stm_duration.
#
# Bước 2: với mỗi RTTM segment, gom các STM mà nó sở hữu.
#         Slice words theo tỉ lệ overlap / stm_duration (proportional).
#         Nếu RTTM cover >= 80% STM → lấy toàn bộ words (tránh cắt vụn).
#
# Bất biến: mỗi STM word xuất hiện đúng 1 lần trong toàn bộ ref,
#           không inflate total_words.

def _is_overlap_sufficient(
    ov: float,
    stm_dur: float,
    abs_threshold: float  = OVERLAP_ABS_THRESHOLD,
    ratio_threshold: float = OVERLAP_RATIO_THRESHOLD,
) -> bool:
    """
    Rule v4: overlap đủ nếu >= abs_threshold (giây) HOẶC >= ratio * stm_duration.

    OR logic:
      - ABS cứu STM dài bị undercount bởi ratio (RTTM ngắn overlap 1-2s
        vào STM 10-15s → ratio thấp nhưng overlap thật đủ).
      - RATIO bảo vệ STM ngắn khỏi false match quá dễ (STM 0.3s không
        nên bị match bởi RTTM chỉ chạm 0.21s -- ratio=70% sẽ đúng hơn).
    """
    return (ov >= abs_threshold) or (ov >= ratio_threshold * stm_dur)


def assign_ref_to_rttm(
    rttm_segments: List[dict],          # sorted by rttm_start
    stm_segments:  List[TextSegment],   # sorted by start, text đã normalize
    overlap_ratio_threshold: float = OVERLAP_RATIO_THRESHOLD,
    overlap_abs_threshold:   float = OVERLAP_ABS_THRESHOLD,
) -> List[Optional[str]]:
    """
    Trả về list[Optional[str]] cùng length với rttm_segments.
    None  → segment bị drop (không nên xảy ra ở đây, xử lý bên ngoài)
    ""    → RTTM không match STM nào → ref rỗng → insertion
    str   → ref text đã normalize, sẵn sàng tính WER

    Thay đổi v5 so với v4:
    ─────────────────────
    BUG FIX CHÍNH (v4→v5): đổi chiều ownership từ STM→RTTM sang RTTM→STM.

    Vấn đề v4: Bước 1 assign mỗi STM cho đúng 1 RTTM owner (max overlap).
    Khi STM rất dài (79–193s, như slide thuyết trình) bị nhiều RTTM chunks
    27s cùng overlap, chỉ 1 chunk được làm owner → các chunk còn lại
    không có STM nào → ref='' → WER bị inflate (vd: 158% cho data_30p).

    Fix v5: đảo chiều — với mỗi RTTM segment, tìm TẤT CẢ STM overlap đủ
    ngưỡng, không cần tranh giành ownership. Nhiều RTTM có thể cùng tham
    chiếu 1 STM — proportional slice đảm bảo mỗi RTTM chỉ lấy phần text
    tương ứng với vùng overlap của nó.

    Các fix từ v4 vẫn giữ:
    - Rule overlap: ABS (0.2s) OR RATIO (20%).
    - Ưu tiên same-speaker (fix 1A).
    - Fallback any-speaker (fix 1B).
    """
    n_rttm = len(rttm_segments)

    if not stm_segments:
        return [""] * n_rttm

    # ── Bước 1: với mỗi RTTM, tìm TẤT CẢ STM overlap đủ ngưỡng ─────────────
    # rttm_to_stm[ri] = list of si có overlap đủ, sorted theo stm.start
    # Không dùng ownership (1-to-1) nữa — nhiều RTTM có thể share 1 STM.
    rttm_to_stm: Dict[int, List[int]] = {ri: [] for ri in range(n_rttm)}

    for ri, row in enumerate(rttm_segments):
        rs           = float(row["rttm_start"])
        re           = float(row["rttm_end"])
        rttm_speaker = row.get("speaker", "")

        same_spk_matches: List[Tuple[float, int]] = []   # (ov, si) same-speaker
        any_spk_matches:  List[Tuple[float, int]] = []   # (ov, si) any-speaker

        for si, stm in enumerate(stm_segments):
            ov      = overlap_len(rs, re, stm.start, stm.end)
            stm_dur = max(stm.end - stm.start, 1e-6)
            if ov <= 0 or not _is_overlap_sufficient(
                ov, stm_dur, overlap_abs_threshold, overlap_ratio_threshold
            ):
                continue

            stm_speaker = getattr(stm, "speaker", "")
            any_spk_matches.append((ov, si))
            if stm_speaker and rttm_speaker and stm_speaker == rttm_speaker:
                same_spk_matches.append((ov, si))

        # Ưu tiên same-speaker; fallback any-speaker
        chosen = same_spk_matches if same_spk_matches else any_spk_matches
        # Sort theo thứ tự thời gian của STM (không phải overlap lớn nhất)
        rttm_to_stm[ri] = [si for _, si in sorted(chosen, key=lambda x: stm_segments[x[1]].start)]

    # ── Bước 2: build ref per RTTM segment ───────────────────────────────────
    refs: List[str] = []

    for ri, row in enumerate(rttm_segments):
        rs = float(row["rttm_start"])
        re = float(row["rttm_end"])

        owned = rttm_to_stm[ri]   # list of si, sorted theo stm.start

        if not owned:
            refs.append("")  # false positive → sẽ bị tính insertion
            continue

        word_chunks: List[str] = []
        for si in owned:
            stm = stm_segments[si]
            words = stm.text.split()
            if not words:
                continue

            stm_dur = max(stm.end - stm.start, 1e-6)
            ov      = overlap_len(rs, re, stm.start, stm.end)
            ratio   = min(ov / stm_dur, 1.0)

            # Nếu RTTM cover >= 80% STM → lấy toàn bộ, tránh cắt vụn
            if ratio >= 0.80:
                word_chunks.append(stm.text)
                continue

            # Proportional slice:
            # Xác định vùng overlap trong timeline của STM
            ov_start    = max(rs, stm.start)
            ov_end      = min(re, stm.end)
            start_ratio = (ov_start - stm.start) / stm_dur
            end_ratio   = (ov_end   - stm.start) / stm_dur

            w_start = int(start_ratio * len(words))
            w_end   = max(w_start + 1, int(end_ratio * len(words)))
            w_end   = min(w_end, len(words))

            sliced = words[w_start:w_end]
            if sliced:
                word_chunks.append(" ".join(sliced))

        refs.append(" ".join(word_chunks).strip())

    return refs


# ── Main ─────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description=(
            "Calculate segment-level WER from CSV predictions. "
            "STM and RTTM are completely independent — ref is assigned via time-overlap."
        )
    )
    parser.add_argument("--predictions_csv", required=True,
                        help="CSV output from run_asr_inference.py")
    parser.add_argument("--data_dir", required=True,
                        help="Root directory containing data* (with labeled/)")
    parser.add_argument("--report_name", default="report_wer_breakdown")
    parser.add_argument("--out_dir",     default="out_report")
    parser.add_argument(
        "--overlap_threshold", type=float, default=OVERLAP_RATIO_THRESHOLD,
        help=(
            f"Minimum overlap ratio (overlap_sec >= threshold * stm_duration). "
            f"Used in OR condition with --overlap_abs. "
            f"Default={OVERLAP_RATIO_THRESHOLD}"
        ),
    )
    parser.add_argument(
        "--overlap_abs", type=float, default=OVERLAP_ABS_THRESHOLD,
        help=(
            f"Absolute overlap threshold (seconds). "
            f"Match if overlap >= abs OR overlap >= ratio * stm_dur. "
            f"Default={OVERLAP_ABS_THRESHOLD}s"
        ),
    )
    args = parser.parse_args()

    # ── Load predictions CSV ─────────────────────────────────────────────────
    pred_path = Path(args.predictions_csv)
    if not pred_path.exists():
        LOGGER.error("Không tìm thấy file: %s", pred_path)
        return

    with open(pred_path, "r", encoding="utf-8") as f:
        pred_rows = list(csv.DictReader(f))
    LOGGER.info("Đọc %d segment predictions.", len(pred_rows))

    required_cols = {"parent_audio_id", "data_name", "rttm_start", "rttm_end", "predicted_text"}
    missing = required_cols - set(pred_rows[0].keys() if pred_rows else [])
    if missing:
        LOGGER.error("CSV thiếu cột: %s", missing)
        return

    # ── Load STM (ground truth) ──────────────────────────────────────────────
    # Đây là nơi duy nhất trong toàn pipeline đọc ground truth.
    # STM không được dùng ở run_asr_inference.py — đảm bảo không leakage.
    stm_cache = build_stm_cache(Path(args.data_dir))

    # ── Group predictions theo file ──────────────────────────────────────────
    file_groups: Dict[str, List[dict]] = defaultdict(list)
    for row in pred_rows:
        key = row.get("parent_audio_id") or row.get("audio_path", "unknown")
        file_groups[key].append(row)

    # Sort mỗi group theo rttm_start
    for key in file_groups:
        file_groups[key].sort(key=lambda r: float(r.get("rttm_start", 0)))

    LOGGER.info("Tìm thấy %d file cần đánh giá.", len(file_groups))
    LOGGER.info(
        "Config: overlap_abs=%.2fs | overlap_ratio=%.0f%% (match nếu abs OR ratio đủ ngưỡng)",
        args.overlap_abs, args.overlap_threshold * 100,
    )

    # ── Đánh giá ─────────────────────────────────────────────────────────────
    stats_by_group = defaultdict(lambda: {
        "total_edits": 0, "total_words": 0, "S": 0, "D": 0, "I": 0,
        "n_files": 0, "n_segments_evaluated": 0,
        "n_segments_dropped_hallucination": 0,
        "n_segments_no_stm_match": 0,
    })
    global_stats = {
        "total_edits": 0, "total_words": 0, "S": 0, "D": 0, "I": 0,
    }

    total_segments             = 0
    total_dropped_hallucination = 0
    total_no_stm_match         = 0
    total_evaluated            = 0
    skipped_no_stm_file        = 0

    result_rows:    List[dict] = []   # per-file summary
    segment_rows:   List[dict] = []   # per-segment detail

    for parent_audio_id, segments in sorted(file_groups.items()):
        meta      = segments[0]
        data_name = meta.get("data_name", "unknown")
        group     = meta.get("group_name", data_name)
        bucket    = meta.get("bucket", "")

        total_segments += len(segments)

        stm_segs = stm_cache.get(data_name)
        if not stm_segs:
            LOGGER.warning("Không có STM cho '%s' — bỏ qua file.", data_name)
            skipped_no_stm_file += 1
            continue

        # ── Lọc hallucination / no-speech ────────────────────────────────────
        # predicted_text == "" nghĩa là run_asr_inference đã detect
        # silence hoặc hallucination sau retry → drop hoàn toàn,
        # không đếm vào WER (cả ref lẫn hyp).
        valid_segments   = [r for r in segments if normalize_vi_text(r.get("predicted_text", ""))]
        dropped_hall     = len(segments) - len(valid_segments)
        total_dropped_hallucination += dropped_hall

        # ── Assign ref qua time-overlap ───────────────────────────────────────
        refs = assign_ref_to_rttm(
            valid_segments, stm_segs,
            overlap_ratio_threshold=args.overlap_threshold,
            overlap_abs_threshold=args.overlap_abs,
        )

        # ── Tính WER segment-level ────────────────────────────────────────────
        file_edits = file_words = file_S = file_D = file_I = 0
        file_no_match = 0

        for row, ref in zip(valid_segments, refs):
            hyp = normalize_vi_text(row.get("predicted_text", ""))
            # ref đã normalize trong build_stm_cache
            r_words = ref.split()
            h_words = hyp.split()

            if not ref:
                file_no_match += 1

            edits, length, S, D, I = compute_wer_details(r_words, h_words)

            file_edits += edits
            file_words += length
            file_S     += S
            file_D     += D
            file_I     += I
            total_evaluated += 1

            segment_rows.append({
                "parent_audio_id":  parent_audio_id,
                "data_name":        data_name,
                "group_name":       group,
                "rttm_start":       row.get("rttm_start"),
                "rttm_end":         row.get("rttm_end"),
                "speaker":          row.get("speaker", ""),
                "reference_text":   ref,
                "predicted_text":   hyp,
                "ref_words":        length,
                "edits":            edits,
                "Substitutions":    S,
                "Deletions":        D,
                "Insertions":       I,
                "WER_PERCENT":      round(edits / max(length, 1) * 100, 2),
                "stm_matched":      bool(ref),
                "was_retried":      row.get("was_retried", False),
            })

        total_no_stm_match += file_no_match

        # Cộng dồn vào global và group
        for st in (stats_by_group[group], global_stats):
            st["total_edits"] += file_edits
            st["total_words"] += file_words
            st["S"]           += file_S
            st["D"]           += file_D
            st["I"]           += file_I

        grp_st = stats_by_group[group]
        grp_st["n_files"]                       += 1
        grp_st["n_segments_evaluated"]           += len(valid_segments)
        grp_st["n_segments_dropped_hallucination"] += dropped_hall
        grp_st["n_segments_no_stm_match"]        += file_no_match

        result_rows.append({
            "parent_audio_id":              parent_audio_id,
            "data_name":                    data_name,
            "group_name":                   group,
            "bucket":                       bucket,
            "n_segments_total":             len(segments),
            "n_segments_dropped_halluc":    dropped_hall,
            "n_segments_evaluated":         len(valid_segments),
            "n_segments_no_stm_match":      file_no_match,
            "ref_words":                    file_words,
            "edits":                        file_edits,
            "Substitutions":                file_S,
            "Deletions":                    file_D,
            "Insertions":                   file_I,
            "WER_PERCENT":                  round(file_edits / max(file_words, 1) * 100, 2),
        })

    if not result_rows:
        LOGGER.error("Không có file nào được đánh giá.")
        return

    # ── Ghi output ───────────────────────────────────────────────────────────
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    csv_file_out = out_dir / f"preds_{args.report_name}.csv"
    csv_seg_out  = out_dir / f"segments_{args.report_name}.csv"
    md_out       = out_dir / f"{args.report_name}.md"

    with open(csv_file_out, "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=result_rows[0].keys())
        writer.writeheader()
        writer.writerows(result_rows)

    with open(csv_seg_out, "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=segment_rows[0].keys())
        writer.writeheader()
        writer.writerows(segment_rows)

    # ── Markdown report ───────────────────────────────────────────────────────
    g  = global_stats
    tw = max(g["total_words"], 1)
    g_wer = g["total_edits"] / tw * 100
    g_s   = g["S"] / tw * 100
    g_d   = g["D"] / tw * 100
    g_i   = g["I"] / tw * 100

    hall_pct    = total_dropped_hallucination / max(total_segments, 1) * 100
    no_match_pct = total_no_stm_match / max(total_evaluated, 1) * 100

    md  = "# Bao Cao WER -- ASR Diagnostic (Segment-Level, Time-Overlap Ref)\n\n"
    md += "> WER tinh o cap segment. STM va RTTM hoan toan doc lap.\n"
    md += "> Ref duoc gan qua time-overlap matching (khong align boundary).\n"
    md += f"> Overlap rule (v5): abs>={args.overlap_abs:.2f}s OR ratio>={args.overlap_threshold:.0%} stm_duration.\n"
    md += "> Many-to-many matching: nhieu RTTM co the share 1 STM (fix STM dai).\n"
    md += "> Same-speaker uu tien; fallback bat ky speaker.\n\n"

    md += "## 1. Ket Qua Tong Quan\n"
    md += f"- **WER TONG:** `{g_wer:.2f}%`\n"
    md += f"  - Substitutions (S): `{g_s:.2f}%`\n"
    md += f"  - Deletions     (D): `{g_d:.2f}%`\n"
    md += f"  - Insertions    (I): `{g_i:.2f}%`\n\n"

    md += "## 2. Segment Statistics\n"
    md += f"- Tong segment tu RTTM        : `{total_segments:,}`\n"
    md += (f"- Dropped (hallucination/silence): `{total_dropped_hallucination:,}` "
           f"({hall_pct:.1f}%) -- KHONG tinh vao WER\n")
    md += (f"- Evaluated                   : `{total_evaluated:,}`\n")
    md += (f"  - Co ref tu STM             : `{total_evaluated - total_no_stm_match:,}`\n")
    md += (f"  - Khong match STM (ref='')  : `{total_no_stm_match:,}` "
           f"({no_match_pct:.1f}%) -- tinh la insertion\n")
    md += f"- File bo qua (khong co STM)  : `{skipped_no_stm_file}`\n\n"

    md += "## 3. Breakdown theo Dataset\n\n"
    md += ("| Dataset | Files | Seg eval | Dropped | No-match | Ref words "
           "| WER (%) | S (%) | D (%) | I (%) |\n")
    md += ("|---------|-------|----------|---------|----------|-----------|"
           "---------|-------|-------|-------|\n")
    for grp, st in sorted(stats_by_group.items()):
        _tw = max(st["total_words"], 1)
        md += (
            f"| {grp} "
            f"| {st['n_files']} "
            f"| {st['n_segments_evaluated']:,} "
            f"| {st['n_segments_dropped_hallucination']:,} "
            f"| {st['n_segments_no_stm_match']:,} "
            f"| {st['total_words']:,} "
            f"| **{st['total_edits'] / _tw * 100:.2f}** "
            f"| {st['S'] / _tw * 100:.2f} "
            f"| {st['D'] / _tw * 100:.2f} "
            f"| {st['I'] / _tw * 100:.2f} |\n"
        )

    md += "\n## 4. Chi Tiet\n"
    md += f"- Per-file WER   : `preds_{args.report_name}.csv`\n"
    md += f"- Per-segment    : `segments_{args.report_name}.csv`\n\n"

    md += "### Pipeline\n```\n"
    md += "run_asr_inference.py\n"
    md += "  Input : RTTM -> audio boundary  [STM KHONG duoc load]\n"
    md += "  Output: predictions.csv  {rttm_start, rttm_end, predicted_text, ...}\n\n"
    md += "eval_wer_from_csv.py\n"
    md += "  Input 1: predictions.csv\n"
    md += "            - predicted_text='' (halluc/silence) -> DROP\n"
    md += "            - con lai -> tim STM match qua time-overlap\n"
    md += "  Input 2: STM -> timestamps + text  [khong gap RTTM luc inference]\n"
    md += f"            - Match neu overlap >= {args.overlap_abs:.2f}s (abs) OR >= {args.overlap_threshold:.0%} stm_duration (ratio)\n"
    md += "            - Many-to-many: nhieu RTTM chunk co the match 1 STM dai\n"
    md += "            - Uu tien same-speaker; fallback bat ky speaker\n"
    md += "            - Proportional word slice (moi RTTM lay phan text tuong ung)\n"
    md += "            - Khong match -> ref='' -> tinh insertion\n"
    md += "  Output : WER segment-level, cong don theo file va group\n```\n"

    with open(md_out, "w", encoding="utf-8") as f:
        f.write(md)

    # ── Console summary ───────────────────────────────────────────────────────
    LOGGER.info("-" * 65)
    LOGGER.info("WER TONG   : %.2f%%  (S=%.2f%%  D=%.2f%%  I=%.2f%%)",
                g_wer, g_s, g_d, g_i)
    LOGGER.info("Segment    : %d total | %d dropped (%.1f%%) | %d evaluated",
                total_segments, total_dropped_hallucination, hall_pct, total_evaluated)
    LOGGER.info("No-STM-match (ref=''): %d / %d evaluated (%.1f%%)",
                total_no_stm_match, total_evaluated, no_match_pct)
    LOGGER.info("File skipped (no STM): %d", skipped_no_stm_file)
    LOGGER.info("jiwer: %s", "ON" if _USE_JIWER else "OFF (fallback DP)")
    LOGGER.info("CSV file-level : %s", csv_file_out)
    LOGGER.info("CSV seg-level  : %s", csv_seg_out)
    LOGGER.info("Markdown       : %s", md_out)


if __name__ == "__main__":
    main()

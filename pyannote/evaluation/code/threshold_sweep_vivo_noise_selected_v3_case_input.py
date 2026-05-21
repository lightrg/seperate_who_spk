from __future__ import annotations

import argparse
import contextlib
import importlib.util
import json
import os
import re
import sys
import traceback
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

import matplotlib.pyplot as plt
import matplotlib.patheffects as pe
import numpy as np
import pandas as pd

EVAL_SCRIPT_DEFAULT = "danh_gia_check_point_md_v2_fixed.py"
HF_MODEL_DEFAULT = "BUT-FIT/diarizen-wavlm-large-s80-md-v2"
DEFAULT_THRESHOLDS = [0.05, 0.25]
DEFAULT_CHECKPOINTS = ["baseline", "ep006_phase2_top6_overlap.pth", "best_model.pth"]
DEFAULT_CASES = ["data5", "data14", "data62", "data2", "data11", "data56"]
NOISE_MAP = {
    "data5": 23,
    "data14": 17,
    "data62": 23,
    "data2": 0,
    "data11": 0,
    "data56": 0,
}
AUDIO_CANDIDATES = ["mixture.wav", "audio.wav"]

@dataclass(frozen=True)
class CheckpointSpec:
    label: str
    ckpt_path: Optional[Path]
    order_key: Tuple[int, int, str]

@dataclass
class SampleSpec:
    split: str
    case_id: str
    case_dir: Path
    wav_path: Path
    rttm_path: Path
    enrollment_dir: Optional[Path]
    n_speakers: int
    group_name: str
    overlap_count: Optional[float]
    metadata: Dict[str, Any]
    ref_segments: int
    noise_pct: Optional[float] = None
    duration_sec: Optional[float] = None

META_ID_KEYS = [
    "case_id", "data_id", "data_name", "case_name", "id", "uid", "item", "sample", "sample_id", "folder", "dirname", "name",
]
META_CLASS_KEYS = [
    "class", "class_name", "category", "label", "scene", "noise", "condition", "type", "group", "domain", "index",
]
META_OVERLAP_KEYS = [
    "overlap_count", "overlap", "overlap_pct", "overlap_percent", "overlap_percentage", "ovl", "ovl_count",
]
META_SPK_KEYS = [
    "n_speakers", "num_speakers", "speaker_count", "spk", "speakers",
]

PAPER_QUAL_COLORS = [
    "#4477AA", "#66CCEE", "#228833", "#CCBB44", "#EE6677", "#AA3377",
    "#332288", "#88CCEE", "#44AA99", "#117733", "#999933", "#DDCC77",
    "#CC6677", "#882255", "#AA4499", "#EE7733", "#0077BB", "#33BBEE",
    "#009988", "#CC3311", "#EE3377",
]

CHECKPOINT_STYLE = {
    "baseline": {"linestyle": "-", "marker": "D", "linewidth": 2.8, "markersize": 6.2},
    "ep006_phase2_top6_overlap": {"linestyle": (0, (6, 1.8)), "marker": "o", "linewidth": 2.6, "markersize": 6.0},
    "best_model": {"linestyle": "-", "marker": "s", "linewidth": 3.0, "markersize": 6.4},
}

def natural_key(text: str) -> List[Any]:
    return [int(t) if t.isdigit() else t.lower() for t in re.split(r"(\d+)", text)]

def safe_mkdir(path: Path) -> Path:
    path.mkdir(parents=True, exist_ok=True)
    return path

def slugify(text: str) -> str:
    text = str(text).strip().lower()
    text = re.sub(r"[^a-z0-9]+", "_", text)
    return text.strip("_") or "unknown"

def read_text(path: Path) -> str:
    return path.read_text(encoding="utf-8", errors="ignore")

def count_rttm_segments(rttm_path: Path) -> int:
    if not rttm_path.exists():
        return 0
    count = 0
    with rttm_path.open("r", encoding="utf-8", errors="ignore") as f:
        for line in f:
            if line.strip().startswith("SPEAKER"):
                count += 1
    return count

def unique_speakers_from_rttm(rttm_path: Path) -> int:
    speakers = set()
    if not rttm_path.exists():
        return 0
    with rttm_path.open("r", encoding="utf-8", errors="ignore") as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) >= 8 and parts[0] == "SPEAKER":
                speakers.add(parts[7])
    return len(speakers)

def _first_existing(d: Dict[str, Any], keys: Sequence[str]) -> Any:
    for k in keys:
        if k in d and d[k] not in (None, ""):
            return d[k]
    return None

def _as_float_or_none(x: Any) -> Optional[float]:
    if x is None:
        return None
    if isinstance(x, (int, float)):
        return float(x)
    s = str(x).strip()
    if not s:
        return None
    m = re.search(r"-?\d+(?:\.\d+)?", s)
    return float(m.group()) if m else None

def overlap_bucket_label(x: Any) -> str:
    val = _as_float_or_none(x)
    if val is None:
        return "unknown"
    if abs(val - round(val)) < 1e-6:
        return f"ov{int(round(val))}"
    return f"ov{val:g}"

def parse_case_noise_map(items: Sequence[str]) -> Dict[str, float]:
    mapping: Dict[str, float] = {}
    for item in items:
        s = str(item).strip()
        if not s:
            continue
        if "=" not in s:
            raise ValueError(f"Invalid case specification '{item}'. Use case_id=noise_pct, e.g. data5=23")
        case_id, noise = s.split("=", 1)
        case_id = case_id.strip()
        noise_val = _as_float_or_none(noise)
        if not case_id or noise_val is None:
            raise ValueError(f"Invalid case specification '{item}'. Use case_id=noise_pct, e.g. data5=23")
        mapping[case_id] = float(noise_val)
    return mapping

def find_audio_file(case_dir: Path) -> Optional[Path]:
    for name in AUDIO_CANDIDATES:
        p = case_dir / name
        if p.exists():
            return p
    wavs = [p for p in case_dir.glob("*.wav") if p.is_file()]
    if wavs:
        return sorted(wavs, key=lambda p: natural_key(p.name))[0]
    wavs = [
        p for p in case_dir.rglob("*.wav")
        if p.is_file() and "enrollment" not in p.parts and "labeled" not in p.parts
    ]
    if wavs:
        return sorted(wavs, key=lambda p: natural_key(str(p)))[0]
    return None

def find_rttm_file(case_dir: Path) -> Optional[Path]:
    labeled = case_dir / "labeled"
    if labeled.exists():
        rttms = sorted(labeled.rglob("*.rttm"), key=lambda p: natural_key(str(p)))
        if rttms:
            return rttms[0]
    rttms = [p for p in case_dir.rglob("*.rttm") if p.is_file()]
    if rttms:
        return sorted(rttms, key=lambda p: natural_key(str(p)))[0]
    return None

def find_enrollment_dir(case_dir: Path) -> Optional[Path]:
    d = case_dir / "enrollment"
    return d if d.exists() and d.is_dir() else None

def infer_n_speakers(case_dir: Path, rttm_path: Path, enrollment_dir: Optional[Path]) -> int:
    if enrollment_dir and enrollment_dir.exists():
        speakers = [p for p in enrollment_dir.iterdir() if p.is_dir()]
        if speakers:
            return len(speakers)
    n_ref = unique_speakers_from_rttm(rttm_path)
    if n_ref > 0:
        return n_ref
    return 3

def _extract_case_id_from_record(rec: Dict[str, Any], known_case_ids: Sequence[str]) -> Optional[str]:
    val = _first_existing(rec, META_ID_KEYS)
    if val is not None:
        val = str(val).strip()
        if val in known_case_ids:
            return val
    blob = " ".join(f"{k}={v}" for k, v in rec.items())
    for cid in known_case_ids:
        if re.search(rf"\b{re.escape(cid)}\b", blob):
            return cid
    return None

def _normalize_record(rec: Dict[str, Any]) -> Dict[str, Any]:
    out = dict(rec)
    cls = _first_existing(out, META_CLASS_KEYS)
    if cls is not None:
        out["_class_name"] = str(cls).strip()
    ov = _as_float_or_none(_first_existing(out, META_OVERLAP_KEYS))
    if ov is not None:
        out["_overlap_count"] = ov
    spk = _as_float_or_none(_first_existing(out, META_SPK_KEYS))
    if spk is not None:
        out["_n_speakers"] = int(round(spk))
    return out

def _split_table_line(line: str, delimiter: str) -> List[str]:
    return [part.strip().lstrip("#").strip() for part in line.split(delimiter)]

def parse_metadata_file(metadata_path: Path, known_case_ids: Sequence[str]) -> Dict[str, Dict[str, Any]]:
    result: Dict[str, Dict[str, Any]] = {}
    if not metadata_path.exists():
        return result

    text = read_text(metadata_path)
    lines = [ln.strip() for ln in text.splitlines() if ln.strip()]

    for line in lines:
        if not line.startswith("{"):
            continue
        try:
            rec = json.loads(line)
        except Exception:
            continue
        if not isinstance(rec, dict):
            continue
        rec = _normalize_record(rec)
        cid = _extract_case_id_from_record(rec, known_case_ids)
        if cid:
            result[cid] = rec
    if result:
        return result

    if len(lines) >= 2:
        header_line = lines[0]
        delimiters = ["|", "\t", ",", ";"]
        for delimiter in delimiters:
            if delimiter not in header_line:
                continue
            header = _split_table_line(header_line, delimiter)
            if len(header) < 2:
                continue
            any_row = False
            for line in lines[1:]:
                if line.startswith("#") or delimiter not in line:
                    continue
                parts = _split_table_line(line, delimiter)
                if len(parts) != len(header):
                    continue
                rec = _normalize_record(dict(zip(header, parts)))
                cid = _extract_case_id_from_record(rec, known_case_ids)
                if cid:
                    result[cid] = rec
                    any_row = True
            if any_row:
                return result

    for line in lines:
        cid = None
        for candidate in known_case_ids:
            if re.search(rf"\b{re.escape(candidate)}\b", line):
                cid = candidate
                break
        if cid is None:
            continue
        rec: Dict[str, Any] = {"_raw": line}
        for token in re.split(r"[|,;]", line):
            token = token.strip()
            if not token:
                continue
            if "=" in token:
                k, v = token.split("=", 1)
                rec[k.strip().lstrip("#").strip()] = v.strip()
            elif ":" in token:
                k, v = token.split(":", 1)
                rec[k.strip().lstrip("#").strip()] = v.strip()
        rec = _normalize_record(rec)
        result[cid] = rec
    return result

def discover_samples(data_root: Path, splits: Sequence[str], verbose: bool = True) -> List[SampleSpec]:
    samples: List[SampleSpec] = []
    for split in splits:
        split_dir = data_root / split
        if not split_dir.exists():
            if verbose:
                print(f"[WARN] Missing split dir: {split_dir}")
            continue
        case_dirs = sorted([p for p in split_dir.iterdir() if p.is_dir()], key=lambda p: natural_key(p.name))
        metadata_map = parse_metadata_file(split_dir / "metadata.txt", [p.name for p in case_dirs])
        for case_dir in case_dirs:
            wav_path = find_audio_file(case_dir)
            rttm_path = find_rttm_file(case_dir)
            if wav_path is None or rttm_path is None:
                if verbose:
                    print(f"[WARN] Skip {split}/{case_dir.name}: missing wav or rttm")
                continue
            enrollment_dir = find_enrollment_dir(case_dir)
            meta = metadata_map.get(case_dir.name, {})
            n_speakers = int(meta.get("_n_speakers") or infer_n_speakers(case_dir, rttm_path, enrollment_dir))
            overlap_count = _as_float_or_none(meta.get("_overlap_count"))
            ref_segments = count_rttm_segments(rttm_path)
            samples.append(
                SampleSpec(
                    split=split,
                    case_id=case_dir.name,
                    case_dir=case_dir,
                    wav_path=wav_path,
                    rttm_path=rttm_path,
                    enrollment_dir=enrollment_dir,
                    n_speakers=n_speakers,
                    group_name="vivo" if split == "test_data" else str(meta.get("_class_name") or "unknown").strip(),
                    overlap_count=overlap_count,
                    metadata=meta,
                    ref_segments=ref_segments,
                    noise_pct=NOISE_MAP.get(case_dir.name),
                )
            )
    return samples

def discover_checkpoints(cp_dir: Path, requested: Sequence[str]) -> List[CheckpointSpec]:
    pths = sorted(cp_dir.glob("*.pth"), key=lambda p: natural_key(p.name))
    file_map = {p.name: p for p in pths}
    specs: List[CheckpointSpec] = []

    def parse_order(name: str) -> Tuple[int, int, str]:
        lower = name.lower()
        if lower == "baseline":
            return (0, 0, "baseline")
        m = re.search(r"ep(\d+)", lower)
        if m:
            return (1, int(m.group(1)), lower)
        if lower == "best_model.pth":
            return (2, 9998, lower)
        return (3, 9999, lower)

    def append_spec(label: str, path: Optional[Path]):
        specs.append(CheckpointSpec(label=label, ckpt_path=path, order_key=parse_order(label if path is None else path.name)))

    for item in requested:
        if item.lower() == "baseline":
            append_spec("baseline", None)
            continue
        p = Path(item)
        if p.exists():
            append_spec(p.stem, p)
            continue
        if item in file_map:
            append_spec(Path(item).stem, file_map[item])
            continue
        if f"{item}.pth" in file_map:
            p2 = file_map[f"{item}.pth"]
            append_spec(p2.stem, p2)
            continue
        raise FileNotFoundError(f"Checkpoint not found: {item}")

    dedup: Dict[str, CheckpointSpec] = {}
    for spec in specs:
        dedup.setdefault(spec.label, spec)
    return sorted(dedup.values(), key=lambda s: s.order_key)

def setup_plot_style() -> None:
    plt.rcParams.update({
        "figure.facecolor": "white",
        "axes.facecolor": "white",
        "axes.grid": True,
        "grid.alpha": 0.9,
        "grid.linestyle": "--",
        "grid.linewidth": 0.6,
        "grid.color": "#D9D9D9",
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.linewidth": 1.0,
        "axes.edgecolor": "#4A4A4A",
        "font.size": 11,
        "axes.titlesize": 16,
        "axes.labelsize": 12.5,
        "legend.fontsize": 10.0,
        "legend.title_fontsize": 11.0,
        "xtick.labelsize": 10.5,
        "ytick.labelsize": 10.5,
        "savefig.bbox": "tight",
        "savefig.facecolor": "white",
        "lines.solid_capstyle": "round",
        "lines.solid_joinstyle": "round",
        "axes.axisbelow": True,
    })

def _metric_value_format(metric: str) -> str:
    metric = metric.lower()
    if metric in {"pred_segments", "ref_segments"}:
        return "{:.0f}"
    if metric.startswith("ov_"):
        return "{:.3f}"
    return "{:.1f}"

def _apply_line_polish(line, style):
    line.set_markeredgecolor("white")
    line.set_markeredgewidth(style.get("markeredgewidth", 1.0))
    line.set_path_effects([
        pe.Stroke(linewidth=style["linewidth"] + 1.4, foreground="white", alpha=0.95),
        pe.Normal(),
    ])

def _annotate_last(ax, xs, ys, metric, color, idx):
    if not xs:
        return
    x = xs[-1]
    y = ys[-1]
    if y is None or (isinstance(y, float) and np.isnan(y)):
        return
    fmt = _metric_value_format(metric)
    offset_cycle = [8, -10, 12, -14, 16, -18]
    y_offset = offset_cycle[idx % len(offset_cycle)]
    ax.annotate(
        fmt.format(float(y)),
        (x, y),
        textcoords="offset points",
        xytext=(4, y_offset),
        ha="left",
        fontsize=8.0,
        fontweight="bold",
        color="#111111",
        bbox=dict(boxstyle="round,pad=0.18", facecolor="white", edgecolor=color, linewidth=0.7, alpha=0.95),
        clip_on=False,
        zorder=20,
    )

def _checkpoint_style(label: str) -> Dict[str, Any]:
    style = dict(CHECKPOINT_STYLE.get(label, {"linestyle": "-", "marker": "o", "linewidth": 2.5, "markersize": 6.0}))
    if label == "baseline":
        style["color"] = "#222222"
    elif label == "best_model":
        style["color"] = "#D55E00"
    else:
        style["color"] = "#228833"
    style.setdefault("markeredgewidth", 1.0)
    style.setdefault("alpha", 0.98)
    style.setdefault("zorder", 5)
    return style

def plot_threshold_lines_per_checkpoint(
    df: pd.DataFrame,
    metrics: Sequence[Tuple[str, str]],
    title_prefix: str,
    out_dir: Path,
    legend_title: str,
    footnote: Optional[str] = None,
    max_cols: int = 2,
) -> None:
    if df.empty:
        return
    checkpoints = list(dict.fromkeys(df["checkpoint_label"].astype(str).tolist()))
    for ckpt in checkpoints:
        sdf_ckpt = df[df["checkpoint_label"].astype(str) == ckpt].copy()
        if sdf_ckpt.empty:
            continue
        thresholds = sorted(sdf_ckpt["low_threshold"].dropna().astype(float).unique().tolist())
        groups = sorted(sdf_ckpt["sample_label"].dropna().astype(str).unique().tolist(), key=natural_key)
        n_metrics = len(metrics)
        ncols = min(max_cols, n_metrics)
        nrows = int(np.ceil(n_metrics / ncols))
        fig, axes = plt.subplots(nrows, ncols, figsize=(max(14, 4.9 * ncols), max(4.8, 4.0 * nrows)), squeeze=False)
        axes = axes.flatten()

        color_map = {group: PAPER_QUAL_COLORS[i % len(PAPER_QUAL_COLORS)] for i, group in enumerate(groups)}
        base_style = _checkpoint_style(str(ckpt))

        for ax_idx, (metric, ylabel) in enumerate(metrics):
            ax = axes[ax_idx]
            for gi, group in enumerate(groups):
                sdf = sdf_ckpt[sdf_ckpt["sample_label"].astype(str) == str(group)].copy().sort_values("low_threshold")
                val_map = {float(r["low_threshold"]): r[metric] for _, r in sdf.iterrows()}
                xs = thresholds
                ys = [float(val_map.get(t)) if pd.notna(val_map.get(t)) else np.nan for t in thresholds]
                style = dict(base_style)
                style["color"] = color_map[group]
                line, = ax.plot(xs, ys, label=str(group), **style)
                _apply_line_polish(line, style)
                _annotate_last(ax, xs, ys, metric, style["color"], gi)
            ax.set_title(ylabel, fontweight="bold")
            ax.set_ylabel(ylabel)
            ax.set_xlabel("Low threshold")
            ax.set_xticks(thresholds)
            ax.set_xticklabels([f"{x:.2f}" for x in thresholds])
            ax.margins(x=0.04)

        for ax in axes[n_metrics:]:
            ax.axis("off")

        handles, labels = axes[0].get_legend_handles_labels()
        if handles:
            fig.legend(
                handles, labels, title=legend_title, loc="upper center", bbox_to_anchor=(0.5, -0.03),
                ncol=min(3, max(2, int(np.ceil(len(labels) / 3)))), frameon=True, fancybox=True, edgecolor="#D0D0D0"
            )

        fig.suptitle(f"{title_prefix} — {ckpt}", fontsize=20, fontweight="bold", y=1.02)
        if footnote:
            fig.text(0.01, -0.07, footnote, ha="left", va="top", fontsize=10, color="#555555")
        fig.tight_layout()
        safe_mkdir(out_dir)
        out_path = out_dir / f"{slugify(ckpt)}_{slugify(title_prefix)}.png"
        fig.savefig(out_path, dpi=240)
        plt.close(fig)

def write_summary_markdown(out_path: Path, raw_df: pd.DataFrame, summary_df: pd.DataFrame, failed_df: pd.DataFrame, checkpoints: Sequence[CheckpointSpec], thresholds: Sequence[float]) -> None:
    lines: List[str] = []
    lines.append("# Vivo Selected Cases Low-Threshold Sweep Summary")
    lines.append("")
    lines.append("This report keeps the original diarization logic unchanged and sweeps only the low-threshold hyperparameter.")
    lines.append("")
    lines.append(f"Checkpoints: {', '.join(c.label for c in checkpoints)}")
    lines.append("")
    lines.append(f"Thresholds: {', '.join(f'{x:.2f}' for x in thresholds)}")
    lines.append("")

    if not summary_df.empty:
        lines.append("## Summary by sample, checkpoint, and low-threshold")
        lines.append("")
        lines.append(summary_df.to_markdown(index=False))
        lines.append("")

    if not failed_df.empty:
        lines.append("## Failed runs after retry")
        lines.append("")
        lines.append(failed_df.to_markdown(index=False))
        lines.append("")

    lines.append("## Run counts")
    lines.append("")
    lines.append(f"- Total successful runs: {len(raw_df)}")
    lines.append(f"- Unique checkpoints: {raw_df['checkpoint_label'].nunique() if not raw_df.empty else 0}")
    lines.append(f"- Unique selected cases: {raw_df['case_id'].nunique() if not raw_df.empty else 0}")
    lines.append(f"- Unique thresholds: {raw_df['low_threshold'].nunique() if not raw_df.empty else 0}")
    lines.append("")
    out_path.write_text("\n".join(lines), encoding="utf-8")

def load_original_eval_module(eval_script: Path):
    module_name = f"orig_eval_{slugify(eval_script.stem)}"
    old_argv = sys.argv[:]
    try:
        sys.argv = [str(eval_script)]
        spec = importlib.util.spec_from_file_location(module_name, str(eval_script))
        if spec is None or spec.loader is None:
            raise ImportError(f"Cannot create import spec for {eval_script}")
        mod = importlib.util.module_from_spec(spec)
        sys.modules[module_name] = mod
        spec.loader.exec_module(mod)
        return mod
    finally:
        sys.argv = old_argv

def init_checkpoint_model(mod, hf_model: str, checkpoint: CheckpointSpec):
    from diarizen.pipelines.inference import DiariZenPipeline

    print(f"[LOAD] checkpoint={checkpoint.label}")
    print(f"[LOAD] hf_model={hf_model}")
    model = DiariZenPipeline.from_pretrained(hf_model)
    model = mod._load_finetuned_checkpoint_into_model(model, None if checkpoint.ckpt_path is None else str(checkpoint.ckpt_path))
    if hasattr(model, "to"):
        model = model.to(mod.DEVICE)
    if hasattr(model, "eval"):
        model.eval()
    if getattr(mod.DEVICE, "type", "cpu") == "cuda":
        mod.torch.cuda.empty_cache()
        print(
            f"[LOAD] VRAM: {mod.torch.cuda.memory_allocated() / 1024**2:.0f} MB allocated / "
            f"{mod.torch.cuda.memory_reserved() / 1024**2:.0f} MB reserved"
        )
    return model

def evaluate_sample_inprocess(mod, model, sample: SampleSpec, checkpoint: CheckpointSpec, out_json: Path, low_threshold: float) -> Dict[str, Any]:
    mod.WAV_PATH = str(sample.wav_path)
    mod.CLEAN_RTTM_PATH = str(sample.rttm_path)
    mod.UTT_ID = f"{checkpoint.label}__thr{int(round(low_threshold * 100)):03d}__{sample.split}__{sample.case_id}"
    mod.OUT_JSON = str(out_json)
    mod.ENROLLMENT_DIR = None if sample.enrollment_dir is None else str(sample.enrollment_dir)
    mod.FIXED_N_CLUSTERS = int(sample.n_speakers)
    mod.VBX_PREMERGE_MAX_CLUSTERS = int(sample.n_speakers) + 1
    mod.FORENSIC_DIR = os.path.join(os.path.dirname(mod.OUT_JSON), f"forensics_{mod.UTT_ID}")
    mod.LOW_THRESHOLD = float(low_threshold)

    ref = mod.load_rttm(mod.CLEAN_RTTM_PATH, mod.UTT_ID)
    wav, sr = mod.torchaudio.load(mod.WAV_PATH)
    if wav.shape[0] > 1:
        wav = wav.mean(dim=0, keepdim=True)
    if sr != mod.TARGET_SR:
        wav = mod.torchaudio.functional.resample(wav, sr, mod.TARGET_SR)
    wav = wav.to(mod.DEVICE)

    resampler = mod.get_resampler()
    ecapa = model
    total_dur = wav.shape[1] / mod.TARGET_SR

    chunk_outputs = mod.infer_all_chunk_probs(model, wav, chunk_sec=mod.CHUNK_SEC)
    if not chunk_outputs:
        raise RuntimeError("No chunk outputs. Abort.")
    if mod.DEVICE.type == "cuda":
        mod.torch.cuda.empty_cache()

    centroids, high_thr_used, n_anchors, p1_stats, init_meta = mod.initialize_phase1_centroids(
        chunk_outputs, wav, ecapa, resampler, mod.FIXED_N_CLUSTERS
    )
    if centroids is None:
        raise RuntimeError("Phase 1 failed: not enough valid centroids to initialize. Abort.")

    embs, sinfo, collect_stats = mod.collect_assign_embeddings(chunk_outputs, wav, ecapa, resampler, threshold=float(low_threshold))
    n_emb = len(embs)
    result = None
    os.makedirs(mod.FORENSIC_DIR, exist_ok=True)

    if n_emb == 0:
        result = {
            "low_threshold": float(low_threshold),
            "high_threshold_used": (float(high_thr_used) if high_thr_used is not None else None),
            "num_assign_embeddings": 0,
            "skipped": True,
            "skip_reason": "no_embeddings",
            "collect_stats": collect_stats,
            "centroid_init": init_meta,
        }
    else:
        _, _, _, embed_details0 = mod.assign_to_centroids(embs, sinfo, centroids)
        purified_centroids, purify_stats = mod.purify_centroids(embs, sinfo, embed_details0, centroids, mod.FIXED_N_CLUSTERS)
        base_mapping, track_meta, assign_stats, embed_details = mod.assign_to_centroids(embs, sinfo, purified_centroids)
        force_assign_track_meta = mod.build_force_assign_track_meta(
            embs, sinfo, purified_centroids,
            chunk_outputs=chunk_outputs,
            wav_tensor=wav,
            ecapa=ecapa,
            resampler=resampler,
            threshold=float(low_threshold),
            mapped_keys=set(base_mapping.keys()),
        )
        final_mapping, final_track_meta, prop_stats = mod.propagate_missing_tracks(
            chunk_outputs, base_mapping, track_meta, mod.FIXED_N_CLUSTERS,
            force_assign_track_meta=force_assign_track_meta,
            threshold=float(low_threshold),
        )

        raw_hyp, raw_rows = mod.build_raw_hypothesis(chunk_outputs, final_mapping, final_track_meta, float(low_threshold), mod.UTT_ID)
        sf_hyp, sf_rows, sf_stats = mod.apply_silence_filter(raw_rows, wav, mod.UTT_ID)
        raw_hyp, raw_rows = sf_hyp, sf_rows

        low_tag = f"{int(round(float(low_threshold) * 100)):03d}"
        vbx_hyp, vbx_rows, vbx_stats = mod.apply_real_vbx_single_speaker_backbone(
            raw_rows, mod.UTT_ID, mod.WAV_PATH, f"low{low_tag}",
            ecapa=ecapa, wav_tensor=wav, resampler=resampler,
        )
        final_stage = mod.build_stage_report("post_real_vbx", ref, vbx_hyp, total_dur)
        hyp, final_rows = vbx_hyp, vbx_rows

        der, miss, fa, conf = final_stage["der"], final_stage["miss"], final_stage["fa"], final_stage["conf"]

        hyp_rttm_path = os.path.join(mod.FORENSIC_DIR, f"hyp_low{low_tag}.rttm")
        raw_rttm_path = os.path.join(mod.FORENSIC_DIR, f"raw_low{low_tag}.rttm")
        cluster_name_map = mod.build_cluster_name_map(init_meta)
        hyp = mod.remap_rttm_labels(hyp, cluster_name_map)
        raw_hyp = mod.remap_rttm_labels(raw_hyp, cluster_name_map)
        mod.save_rttm(hyp, hyp_rttm_path)
        mod.save_rttm(raw_hyp, raw_rttm_path)

        result = {
            "low_threshold": float(low_threshold),
            "high_threshold_used": (float(high_thr_used) if high_thr_used is not None else None),
            "num_assign_embeddings": int(n_emb),
            "collect_stats": collect_stats,
            "assign_stats_final": assign_stats,
            "centroid_purify_stats": purify_stats,
            "force_assign_meta_debug": mod.LAST_FORCE_ASSIGN_META_DEBUG,
            "silence_filter_stats": sf_stats,
            "propagate_stats": prop_stats,
            "real_vbx_stats": vbx_stats,
            "mapped_tracks_before_propagate": int(len(base_mapping)),
            "mapped_tracks_after_propagate": int(len(final_mapping)),
            "raw_segment_rows": int(len(raw_rows)),
            "vbx_segment_rows": int(len(vbx_rows)),
            "final_segment_rows": int(len(final_rows)),
            "artifacts": {
                "raw_rttm": raw_rttm_path,
                "hyp_rttm": hyp_rttm_path,
            },
            "final_overlap_metrics": final_stage["overlap_metrics"],
            "skipped": False,
            "der": (float(der) if not mod.SKIP_DER else None),
            "miss": (float(miss) if not mod.SKIP_DER else None),
            "fa": (float(fa) if not mod.SKIP_DER else None),
            "conf": (float(conf) if not mod.SKIP_DER else None),
        }

    os.makedirs(os.path.dirname(mod.OUT_JSON), exist_ok=True)
    with open(mod.OUT_JSON, "w", encoding="utf-8") as f:
        json.dump(result, f, indent=2, ensure_ascii=False)
    return result

def row_from_result(checkpoint: CheckpointSpec, sample: SampleSpec, low_threshold: float, run_dir: Path, data: Dict[str, Any]) -> Dict[str, Any]:
    overlap = (((data or {}).get("final_overlap_metrics") or {}).get("overlap") or {})
    sample_label = format_sample_label(sample)
    artifacts = (data or {}).get("artifacts") or {}
    return {
        "checkpoint_label": checkpoint.label,
        "sample_label": sample_label,
        "split": sample.split,
        "case_id": sample.case_id,
        "group_name": sample.group_name,
        "noise_pct": sample.noise_pct,
        "n_speakers": int(sample.n_speakers),
        "overlap_count": sample.overlap_count,
        "overlap_bucket": overlap_bucket_label(sample.overlap_count),
        "low_threshold": float(low_threshold),
        "ref_segments": int(sample.ref_segments),
        "pred_segments": int((data or {}).get("final_segment_rows") or 0),
        "raw_segments": int((data or {}).get("raw_segment_rows") or 0),
        "vbx_segments": int((data or {}).get("vbx_segment_rows") or 0),
        "der": (data or {}).get("der"),
        "miss": (data or {}).get("miss"),
        "fa": (data or {}).get("fa"),
        "conf": (data or {}).get("conf"),
        "ov_precision": overlap.get("precision"),
        "ov_recall": overlap.get("recall"),
        "ov_f1": overlap.get("f1"),
        "skipped": bool((data or {}).get("skipped", False)),
        "skip_reason": (data or {}).get("skip_reason"),
        "run_json": str(run_dir / "result.json"),
        "stdout_log": str(run_dir / "stdout.log"),
        "stderr_log": str(run_dir / "stderr.log"),
        "hyp_rttm": artifacts.get("hyp_rttm"),
        "raw_rttm": artifacts.get("raw_rttm"),
    }

def format_sample_label(sample: SampleSpec) -> str:
    noise_txt = f"{int(sample.noise_pct)}% noise" if sample.noise_pct is not None else "noise unknown"
    ov_txt = overlap_bucket_label(sample.overlap_count)
    return f"{sample.case_id} | {ov_txt} | {noise_txt}"

def load_result_json(path: Path) -> Dict[str, Any]:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Low-threshold sweep on selected vivo cases for baseline, ep006, and best_model, with explicit case=noise inputs.")
    parser.add_argument("--root_dir", type=str, default=".")
    parser.add_argument("--eval_script", type=str, default=EVAL_SCRIPT_DEFAULT)
    parser.add_argument("--cp_dir", type=str, default="cp")
    parser.add_argument("--data_root", type=str, default="data")
    parser.add_argument("--output_dir", type=str, default="threshold_sweep_vivo_noise_report")
    parser.add_argument("--checkpoints", nargs="+", default=DEFAULT_CHECKPOINTS)
    parser.add_argument("--thresholds", nargs="+", default=[str(x) for x in DEFAULT_THRESHOLDS])
    parser.add_argument("--case_noise_map", nargs="+", default=[f"{k}={int(v)}" for k, v in NOISE_MAP.items()], help="Selected test_data cases as case_id=noise_pct, e.g. data5=23 data14=17")
    parser.add_argument("--hf_model", type=str, default=HF_MODEL_DEFAULT)
    parser.add_argument("--force_rerun", action="store_true")
    parser.add_argument("--quiet", action="store_true")
    return parser.parse_args()

def main() -> None:
    args = parse_args()
    root_dir = Path(args.root_dir).resolve()
    eval_script = Path(args.eval_script)
    if not eval_script.is_absolute():
        eval_script = root_dir / eval_script
    eval_script = eval_script.resolve()
    cp_dir = Path(args.cp_dir)
    if not cp_dir.is_absolute():
        cp_dir = root_dir / cp_dir
    cp_dir = cp_dir.resolve()
    data_root = Path(args.data_root)
    if not data_root.is_absolute():
        data_root = root_dir / data_root
    data_root = data_root.resolve()
    output_dir = Path(args.output_dir)
    if not output_dir.is_absolute():
        output_dir = root_dir / output_dir
    output_dir = output_dir.resolve()

    if not eval_script.exists():
        raise FileNotFoundError(f"eval_script not found: {eval_script}")
    if not cp_dir.exists():
        raise FileNotFoundError(f"cp_dir not found: {cp_dir}")
    if not data_root.exists():
        raise FileNotFoundError(f"data_root not found: {data_root}")

    runs_root = safe_mkdir(output_dir / "runs")
    plots_dir = safe_mkdir(output_dir / "plots")
    tables_dir = safe_mkdir(output_dir / "tables")

    setup_plot_style()
    thresholds = [round(float(_as_float_or_none(x)), 4) for x in args.thresholds]
    checkpoints = discover_checkpoints(cp_dir, args.checkpoints)
    case_noise_map = parse_case_noise_map(args.case_noise_map)
    all_samples = discover_samples(data_root, ["test_data"], verbose=not args.quiet)
    sample_map = {s.case_id: s for s in all_samples}
    samples: List[SampleSpec] = []
    for cid, noise_pct in case_noise_map.items():
        if cid not in sample_map:
            raise FileNotFoundError(f"Selected case not found in test_data: {cid}")
        s = sample_map[cid]
        s.noise_pct = float(noise_pct)
        samples.append(s)

    if not args.quiet:
        print(f"[INFO] eval_script : {eval_script}")
        print(f"[INFO] checkpoints : {[c.label for c in checkpoints]}")
        print(f"[INFO] thresholds  : {thresholds}")
        print(f"[INFO] output_dir  : {output_dir}")
        print("[INFO] selected vivo cases:")
        for sample in samples:
            print(
                f"  - {sample.case_id}: noise={sample.noise_pct}% | ov={sample.overlap_count} | "
                f"n_spk={sample.n_speakers}"
            )

    mod = load_original_eval_module(eval_script)
    rows: List[Dict[str, Any]] = []
    failed_first_pass: List[Tuple[CheckpointSpec, SampleSpec, float, str]] = []

    total = len(checkpoints) * len(samples) * len(thresholds)
    progress = 0

    for checkpoint in checkpoints:
        model = init_checkpoint_model(mod, args.hf_model, checkpoint)
        for sample in samples:
            for thr in thresholds:
                progress += 1
                thr_slug = f"thr_{int(round(thr * 100)):03d}"
                ckpt_slug = slugify(checkpoint.label)
                run_dir = safe_mkdir(runs_root / ckpt_slug / thr_slug / sample.split / sample.case_id)
                out_json = run_dir / "result.json"
                stdout_path = run_dir / "stdout.log"
                stderr_path = run_dir / "stderr.log"
                if out_json.exists() and not args.force_rerun:
                    if not args.quiet:
                        print(f"[{progress}/{total}] SKIP cache | {checkpoint.label} | {sample.case_id} | thr={thr:.2f}")
                    continue
                if not args.quiet:
                    print(f"[{progress}/{total}] RUN | {checkpoint.label} | {sample.case_id} | thr={thr:.2f}")
                try:
                    with stdout_path.open("w", encoding="utf-8", errors="ignore") as stdout_f, stderr_path.open("w", encoding="utf-8", errors="ignore") as stderr_f:
                        with contextlib.redirect_stdout(stdout_f), contextlib.redirect_stderr(stderr_f):
                            evaluate_sample_inprocess(mod, model, sample, checkpoint, out_json, low_threshold=thr)
                except Exception as exc:
                    tb = traceback.format_exc()
                    if out_json.exists():
                        try:
                            out_json.unlink()
                        except Exception:
                            pass
                    with stderr_path.open("a", encoding="utf-8", errors="ignore") as stderr_f:
                        stderr_f.write("\n\n[FIRST PASS ERROR]\n")
                        stderr_f.write(tb)
                    failed_first_pass.append((checkpoint, sample, thr, str(exc)))
                    if not args.quiet:
                        print(f"[WARN] first-pass fail | {checkpoint.label} | {sample.case_id} | thr={thr:.2f} | {exc}")
        model = None
        if getattr(mod.DEVICE, "type", "cpu") == "cuda":
            mod.torch.cuda.empty_cache()

    failed_final: List[Dict[str, Any]] = []
    if failed_first_pass:
        if not args.quiet:
            print(f"[INFO] Retry pass for {len(failed_first_pass)} failed runs.")
        grouped: Dict[str, List[Tuple[CheckpointSpec, SampleSpec, float, str]]] = {}
        for item in failed_first_pass:
            grouped.setdefault(item[0].label, []).append(item)

        for ckpt_label, items in grouped.items():
            checkpoint = items[0][0]
            try:
                model = init_checkpoint_model(mod, args.hf_model, checkpoint)
            except Exception as exc:
                tb = traceback.format_exc()
                for checkpoint, sample, thr, first_err in items:
                    failed_final.append({
                        "checkpoint_label": checkpoint.label,
                        "sample_label": format_sample_label(sample),
                        "split": sample.split,
                        "case_id": sample.case_id,
                        "noise_pct": sample.noise_pct,
                        "low_threshold": thr,
                        "stage": "retry_model_init",
                        "first_error": first_err,
                        "final_error": str(exc),
                        "traceback": tb,
                    })
                continue

            for checkpoint, sample, thr, first_err in items:
                thr_slug = f"thr_{int(round(thr * 100)):03d}"
                ckpt_slug = slugify(checkpoint.label)
                run_dir = safe_mkdir(runs_root / ckpt_slug / thr_slug / sample.split / sample.case_id)
                out_json = run_dir / "result.json"
                stdout_path = run_dir / "stdout.log"
                stderr_path = run_dir / "stderr.log"
                try:
                    with stdout_path.open("a", encoding="utf-8", errors="ignore") as stdout_f, stderr_path.open("a", encoding="utf-8", errors="ignore") as stderr_f:
                        stdout_f.write("\n\n[RETRY PASS]\n")
                        stderr_f.write("\n\n[RETRY PASS]\n")
                        with contextlib.redirect_stdout(stdout_f), contextlib.redirect_stderr(stderr_f):
                            evaluate_sample_inprocess(mod, model, sample, checkpoint, out_json, low_threshold=thr)
                except Exception as exc:
                    tb = traceback.format_exc()
                    if out_json.exists():
                        try:
                            out_json.unlink()
                        except Exception:
                            pass
                    with stderr_path.open("a", encoding="utf-8", errors="ignore") as stderr_f:
                        stderr_f.write("\n\n[RETRY PASS ERROR]\n")
                        stderr_f.write(tb)
                    failed_final.append({
                        "checkpoint_label": checkpoint.label,
                        "sample_label": format_sample_label(sample),
                        "split": sample.split,
                        "case_id": sample.case_id,
                        "noise_pct": sample.noise_pct,
                        "low_threshold": thr,
                        "stage": "retry_sample",
                        "first_error": first_err,
                        "final_error": str(exc),
                        "traceback": tb,
                    })
                    if not args.quiet:
                        print(f"[WARN] retry fail | {checkpoint.label} | {sample.case_id} | thr={thr:.2f} | {exc}")
            model = None
            if getattr(mod.DEVICE, "type", "cpu") == "cuda":
                mod.torch.cuda.empty_cache()

    for checkpoint in checkpoints:
        for sample in samples:
            for thr in thresholds:
                thr_slug = f"thr_{int(round(thr * 100)):03d}"
                ckpt_slug = slugify(checkpoint.label)
                run_dir = runs_root / ckpt_slug / thr_slug / sample.split / sample.case_id
                out_json = run_dir / "result.json"
                if not out_json.exists():
                    continue
                try:
                    data = load_result_json(out_json)
                    rows.append(row_from_result(checkpoint, sample, thr, run_dir, data))
                except Exception as exc:
                    failed_final.append({
                        "checkpoint_label": checkpoint.label,
                        "sample_label": format_sample_label(sample),
                        "split": sample.split,
                        "case_id": sample.case_id,
                        "noise_pct": sample.noise_pct,
                        "low_threshold": thr,
                        "stage": "collect",
                        "first_error": "",
                        "final_error": str(exc),
                        "traceback": traceback.format_exc(),
                    })

    raw_df = pd.DataFrame(rows)
    if raw_df.empty:
        raise RuntimeError("No successful runs were produced.")

    raw_df = raw_df.sort_values(["checkpoint_label", "sample_label", "low_threshold"]).reset_index(drop=True)
    summary_df = raw_df[[
        "checkpoint_label", "sample_label", "case_id", "noise_pct", "low_threshold",
        "der", "miss", "fa", "conf", "ov_precision", "ov_recall", "ov_f1", "pred_segments", "ref_segments"
    ]].copy()
    failed_df = pd.DataFrame(failed_final)

    raw_csv = tables_dir / "raw_case_results.csv"
    summary_csv = tables_dir / "summary_by_checkpoint_case_threshold.csv"
    failed_csv = tables_dir / "failed_runs.csv"
    raw_df.to_csv(raw_csv, index=False, encoding="utf-8-sig")
    summary_df.to_csv(summary_csv, index=False, encoding="utf-8-sig")
    if not failed_df.empty:
        failed_df.to_csv(failed_csv, index=False, encoding="utf-8-sig")

    footnote = "Cases fixed to data5 | ov0 | 23% noise, data14 | ov10 | 17% noise, and data62 | ov5 | 23% noise. Evaluator logic unchanged."
    plot_threshold_lines_per_checkpoint(
        summary_df,
        metrics=[("der", "DER %"), ("miss", "Miss %"), ("fa", "FA %"), ("conf", "Conf %")],
        title_prefix="Low-Threshold Sweep — DER Components on Selected Vivo Cases",
        out_dir=plots_dir,
        legend_title="Selected case | ov | noise",
        footnote=footnote,
        max_cols=2,
    )
    plot_threshold_lines_per_checkpoint(
        summary_df,
        metrics=[("ov_precision", "Overlap Precision"), ("ov_recall", "Overlap Recall"), ("ov_f1", "Overlap F1"), ("pred_segments", "Predicted Segments")],
        title_prefix="Low-Threshold Sweep — Overlap Metrics and Segment Count on Selected Vivo Cases",
        out_dir=plots_dir,
        legend_title="Selected case | ov | noise",
        footnote=footnote,
        max_cols=2,
    )

    write_summary_markdown(output_dir / "SUMMARY.md", raw_df, summary_df, failed_df, checkpoints, thresholds)

    manifest = {
        "checkpoints": [{"label": c.label, "ckpt_path": None if c.ckpt_path is None else str(c.ckpt_path)} for c in checkpoints],
        "thresholds": thresholds,
        "cases": [
            {
                "case_id": s.case_id,
                "noise_pct": s.noise_pct,
                "overlap_count": s.overlap_count,
                "n_speakers": s.n_speakers,
            }
            for s in samples
        ],
        "raw_csv": str(raw_csv),
        "summary_csv": str(summary_csv),
        "failed_csv": str(failed_csv) if failed_csv.exists() else None,
        "plots_dir": str(plots_dir),
    }
    (output_dir / "manifest.json").write_text(json.dumps(manifest, indent=2, ensure_ascii=False), encoding="utf-8")

    print("\n[DONE] Selected vivo low-threshold sweep complete.")
    print(f"[DONE] Raw CSV      : {raw_csv}")
    print(f"[DONE] Summary CSV  : {summary_csv}")
    print(f"[DONE] Failed CSV   : {failed_csv if failed_csv.exists() else 'none'}")
    print(f"[DONE] Plots dir    : {plots_dir}")
    print(f"[DONE] Summary MD   : {output_dir / 'SUMMARY.md'}")

if __name__ == "__main__":
    try:
        main()
    except Exception:
        traceback.print_exc()
        sys.exit(1)
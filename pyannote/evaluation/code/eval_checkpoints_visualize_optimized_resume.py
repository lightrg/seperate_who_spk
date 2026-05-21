from __future__ import annotations

import argparse
import contextlib
import importlib.util
import json
import os
import re
import shutil
import subprocess
import sys
import traceback
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

import matplotlib.pyplot as plt
import matplotlib.patheffects as pe
import numpy as np
import pandas as pd

EVAL_SCRIPT_DEFAULT = "danh_gia_check_point_md_v2_fixed.py"
HF_MODEL_DEFAULT = "BUT-FIT/diarizen-wavlm-large-s80-md-v2"

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
    duration_sec: Optional[float] = None

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

AUDIO_CANDIDATES = ["mixture.wav", "audio.wav"]

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
                if line.startswith("#"):
                    continue
                if delimiter not in line:
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

        if "_class_name" not in rec:

            parts = [p.strip() for p in re.split(r"[|,;\t ]+", line) if p.strip()]
            filtered = [p for p in parts if p != cid and not re.fullmatch(r"data\d+", p.lower())]
            for p in filtered:
                if re.fullmatch(r"\d+(?:\.\d+)?", p):
                    continue
                if p.lower() in {"overlap", "count", "spk", "speaker", "speakers"}:
                    continue
                rec["_class_name"] = p
                break

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
        case_dirs = sorted(
            [p for p in split_dir.iterdir() if p.is_dir()],
            key=lambda p: natural_key(p.name),
        )
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
            if split == "test_data":
                group_name = "vivo"
            else:
                group_name = str(meta.get("_class_name") or "unknown").strip()
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
                    group_name=group_name,
                    overlap_count=overlap_count,
                    metadata=meta,
                    ref_segments=ref_segments,
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

    req_lower = [r.lower() for r in requested]
    if not requested or req_lower == ["all"]:
        append_spec("baseline", None)
        for p in pths:
            append_spec(p.stem, p)
    else:
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
                append_spec(item, file_map[f"{item}.pth"])
                continue
            raise FileNotFoundError(f"Checkpoint not found: {item}")

    dedup: Dict[str, CheckpointSpec] = {}
    for spec in specs:
        dedup.setdefault(spec.label, spec)
    return sorted(dedup.values(), key=lambda s: s.order_key)

def build_command(
    python_exe: Path,
    eval_script: Path,
    sample: SampleSpec,
    checkpoint: CheckpointSpec,
    out_json: Path,
    hf_model: str,
) -> List[str]:
    cmd = [
        str(python_exe),
        str(eval_script),
        "--wav", str(sample.wav_path),
        "--rttm", str(sample.rttm_path),
        "--utt_id", f"{checkpoint.label}__{sample.split}__{sample.case_id}",
        "--out_json", str(out_json),
        "--n_speakers", str(sample.n_speakers),
        "--hf_model", hf_model,
    ]
    if sample.enrollment_dir is not None and sample.enrollment_dir.exists():
        cmd.extend(["--enrollment_dir", str(sample.enrollment_dir)])
    if checkpoint.ckpt_path is not None:
        cmd.extend(["--ckpt", str(checkpoint.ckpt_path)])
    return cmd

def load_result_json(path: Path) -> Dict[str, Any]:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)

def copy_artifact_if_exists(src: Optional[str], dst: Path) -> Optional[Path]:
    if not src:
        return None
    src_path = Path(src)
    if not src_path.exists():
        return None
    safe_mkdir(dst.parent)
    shutil.copy2(src_path, dst)
    return dst

def run_one(
    python_exe: Path,
    eval_script: Path,
    checkpoint: CheckpointSpec,
    sample: SampleSpec,
    runs_root: Path,
    rttm_root: Path,
    hf_model: str,
    skip_existing: bool = True,
) -> Dict[str, Any]:
    ckpt_slug = slugify(checkpoint.label)
    run_dir = safe_mkdir(runs_root / ckpt_slug / sample.split / sample.case_id)
    out_json = run_dir / "result.json"
    stdout_path = run_dir / "stdout.log"
    stderr_path = run_dir / "stderr.log"

    if skip_existing and out_json.exists():
        data = load_result_json(out_json)
    else:
        cmd = build_command(python_exe, eval_script, sample, checkpoint, out_json, hf_model)
        proc = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
        stdout_path.write_text(proc.stdout, encoding="utf-8", errors="ignore")
        stderr_path.write_text(proc.stderr, encoding="utf-8", errors="ignore")
        if proc.returncode != 0:
            raise RuntimeError(
                f"Evaluation failed for {checkpoint.label} | {sample.split}/{sample.case_id}\n"
                f"CMD: {' '.join(cmd)}\n"
                f"STDOUT:\n{proc.stdout[-6000:]}\nSTDERR:\n{proc.stderr[-6000:]}"
            )
        if not out_json.exists():
            raise FileNotFoundError(f"Expected output json not found: {out_json}")
        data = load_result_json(out_json)

    artifacts = data.get("artifacts", {}) if isinstance(data, dict) else {}
    hyp_copy = copy_artifact_if_exists(
        artifacts.get("hyp_rttm"),
        rttm_root / ckpt_slug / sample.split / sample.case_id / "hyp.rttm",
    )
    raw_copy = copy_artifact_if_exists(
        artifacts.get("raw_rttm"),
        rttm_root / ckpt_slug / sample.split / sample.case_id / "raw.rttm",
    )

    overlap = (((data or {}).get("final_overlap_metrics") or {}).get("overlap") or {})
    row = {
        "checkpoint_label": checkpoint.label,
        "checkpoint_file": None if checkpoint.ckpt_path is None else checkpoint.ckpt_path.name,
        "checkpoint_path": None if checkpoint.ckpt_path is None else str(checkpoint.ckpt_path),
        "split": sample.split,
        "case_id": sample.case_id,
        "group_name": sample.group_name,
        "n_speakers": int(sample.n_speakers),
        "overlap_count": sample.overlap_count,
        "overlap_bucket": overlap_bucket_label(sample.overlap_count),
        "wav_path": str(sample.wav_path),
        "ref_rttm_path": str(sample.rttm_path),
        "enrollment_dir": None if sample.enrollment_dir is None else str(sample.enrollment_dir),
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
        "run_json": str(out_json),
        "stdout_log": str(stdout_path),
        "stderr_log": str(stderr_path),
        "copied_hyp_rttm": None if hyp_copy is None else str(hyp_copy),
        "copied_raw_rttm": None if raw_copy is None else str(raw_copy),
    }
    return row

def sample_to_payload(sample: SampleSpec) -> Dict[str, Any]:
    return {
        "split": sample.split,
        "case_id": sample.case_id,
        "case_dir": str(sample.case_dir),
        "wav_path": str(sample.wav_path),
        "rttm_path": str(sample.rttm_path),
        "enrollment_dir": None if sample.enrollment_dir is None else str(sample.enrollment_dir),
        "n_speakers": int(sample.n_speakers),
        "group_name": sample.group_name,
        "overlap_count": sample.overlap_count,
        "metadata": sample.metadata,
        "ref_segments": int(sample.ref_segments),
        "duration_sec": sample.duration_sec,
    }

def sample_from_payload(payload: Dict[str, Any]) -> SampleSpec:
    enrollment = payload.get("enrollment_dir")
    return SampleSpec(
        split=str(payload["split"]),
        case_id=str(payload["case_id"]),
        case_dir=Path(payload["case_dir"]),
        wav_path=Path(payload["wav_path"]),
        rttm_path=Path(payload["rttm_path"]),
        enrollment_dir=None if not enrollment else Path(enrollment),
        n_speakers=int(payload["n_speakers"]),
        group_name=str(payload.get("group_name", "unknown")),
        overlap_count=payload.get("overlap_count"),
        metadata=dict(payload.get("metadata") or {}),
        ref_segments=int(payload.get("ref_segments", 0)),
        duration_sec=payload.get("duration_sec"),
    )

def checkpoint_to_payload(checkpoint: CheckpointSpec) -> Dict[str, Any]:
    return {
        "label": checkpoint.label,
        "ckpt_path": None if checkpoint.ckpt_path is None else str(checkpoint.ckpt_path),
        "order_key": list(checkpoint.order_key),
    }

def checkpoint_from_payload(payload: Dict[str, Any]) -> CheckpointSpec:
    ckpt_path = payload.get("ckpt_path")
    order_key = payload.get("order_key") or [3, 9999, str(payload.get("label", "unknown"))]
    return CheckpointSpec(
        label=str(payload["label"]),
        ckpt_path=None if not ckpt_path else Path(ckpt_path),
        order_key=(int(order_key[0]), int(order_key[1]), str(order_key[2])),
    )

def row_from_result_data(
    checkpoint: CheckpointSpec,
    sample: SampleSpec,
    run_dir: Path,
    rttm_root: Path,
    data: Dict[str, Any],
) -> Dict[str, Any]:
    stdout_path = run_dir / "stdout.log"
    stderr_path = run_dir / "stderr.log"
    artifacts = data.get("artifacts", {}) if isinstance(data, dict) else {}
    ckpt_slug = slugify(checkpoint.label)
    hyp_copy = copy_artifact_if_exists(
        artifacts.get("hyp_rttm"),
        rttm_root / ckpt_slug / sample.split / sample.case_id / "hyp.rttm",
    )
    raw_copy = copy_artifact_if_exists(
        artifacts.get("raw_rttm"),
        rttm_root / ckpt_slug / sample.split / sample.case_id / "raw.rttm",
    )
    overlap = (((data or {}).get("final_overlap_metrics") or {}).get("overlap") or {})
    return {
        "checkpoint_label": checkpoint.label,
        "checkpoint_file": None if checkpoint.ckpt_path is None else checkpoint.ckpt_path.name,
        "checkpoint_path": None if checkpoint.ckpt_path is None else str(checkpoint.ckpt_path),
        "split": sample.split,
        "case_id": sample.case_id,
        "group_name": sample.group_name,
        "n_speakers": int(sample.n_speakers),
        "overlap_count": sample.overlap_count,
        "overlap_bucket": overlap_bucket_label(sample.overlap_count),
        "wav_path": str(sample.wav_path),
        "ref_rttm_path": str(sample.rttm_path),
        "enrollment_dir": None if sample.enrollment_dir is None else str(sample.enrollment_dir),
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
        "stdout_log": str(stdout_path),
        "stderr_log": str(stderr_path),
        "copied_hyp_rttm": None if hyp_copy is None else str(hyp_copy),
        "copied_raw_rttm": None if raw_copy is None else str(raw_copy),
    }

def gather_cached_rows(
    checkpoints: Sequence[CheckpointSpec],
    samples: Sequence[SampleSpec],
    runs_root: Path,
    rttm_root: Path,
    quiet: bool = False,
) -> Tuple[pd.DataFrame, List[Dict[str, Any]]]:
    rows: List[Dict[str, Any]] = []
    missing: List[Dict[str, Any]] = []
    for ckpt in checkpoints:
        ckpt_slug = slugify(ckpt.label)
        for sample in samples:
            run_dir = runs_root / ckpt_slug / sample.split / sample.case_id
            out_json = run_dir / "result.json"
            if not out_json.exists():
                missing.append({
                    "checkpoint_label": ckpt.label,
                    "split": sample.split,
                    "case_id": sample.case_id,
                    "group_name": sample.group_name,
                    "n_speakers": sample.n_speakers,
                    "reason": "missing_result_json",
                })
                continue
            try:
                data = load_result_json(out_json)
                rows.append(row_from_result_data(ckpt, sample, run_dir, rttm_root, data))
            except Exception as exc:
                missing.append({
                    "checkpoint_label": ckpt.label,
                    "split": sample.split,
                    "case_id": sample.case_id,
                    "group_name": sample.group_name,
                    "n_speakers": sample.n_speakers,
                    "reason": f"bad_result_json: {exc}",
                })
                if not quiet:
                    print(f"[WARN] Failed to read cached result: {out_json} | {exc}")
    return pd.DataFrame(rows), missing

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

    print(f"[WORKER] Loading model for checkpoint: {checkpoint.label}")
    print(f"[WORKER] HF model id: {hf_model}")
    model = DiariZenPipeline.from_pretrained(hf_model)
    model = mod._load_finetuned_checkpoint_into_model(
        model,
        None if checkpoint.ckpt_path is None else str(checkpoint.ckpt_path),
    )
    if hasattr(model, "to"):
        model = model.to(mod.DEVICE)
    if hasattr(model, "eval"):
        model.eval()
    if getattr(mod.DEVICE, "type", "cpu") == "cuda":
        mod.torch.cuda.empty_cache()
        print(
            f"[WORKER] VRAM after model load: "
            f"{mod.torch.cuda.memory_allocated() / 1024**2:.0f} MB allocated / "
            f"{mod.torch.cuda.memory_reserved() / 1024**2:.0f} MB reserved"
        )
    return model

def evaluate_sample_inprocess(mod, model, sample: SampleSpec, checkpoint: CheckpointSpec, out_json: Path) -> Dict[str, Any]:
    mod.WAV_PATH = str(sample.wav_path)
    mod.CLEAN_RTTM_PATH = str(sample.rttm_path)
    mod.UTT_ID = f"{checkpoint.label}__{sample.split}__{sample.case_id}"
    mod.OUT_JSON = str(out_json)
    mod.ENROLLMENT_DIR = None if sample.enrollment_dir is None else str(sample.enrollment_dir)
    mod.FIXED_N_CLUSTERS = int(sample.n_speakers)
    mod.VBX_PREMERGE_MAX_CLUSTERS = int(sample.n_speakers) + 1
    mod.FORENSIC_DIR = os.path.join(os.path.dirname(mod.OUT_JSON), f"forensics_{mod.UTT_ID}")

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
    print(f"File Duration: {total_dur:.2f}s")

    print(f"\nRunning EEND-EDA chunk inference... (chunk={mod.CHUNK_SEC}s)")
    chunk_outputs = mod.infer_all_chunk_probs(model, wav, chunk_sec=mod.CHUNK_SEC)
    if not chunk_outputs:
        raise RuntimeError("No chunk outputs. Abort.")
    print(f"Chunks inferred: {len(chunk_outputs)}")
    if mod.DEVICE.type == "cuda":
        mod.torch.cuda.empty_cache()

    print("\n[Phase 1] Initializing speaker centroids...")
    centroids, high_thr_used, n_anchors, p1_stats, init_meta = mod.initialize_phase1_centroids(
        chunk_outputs, wav, ecapa, resampler, mod.FIXED_N_CLUSTERS
    )
    if centroids is None:
        raise RuntimeError("Phase 1 failed: not enough valid centroids to initialize. Abort.")

    p1_primary = int((p1_stats or {}).get("primary_segments", 0)) if isinstance(p1_stats, dict) else 0
    p1_subseg = int((p1_stats or {}).get("subseg_used_segments", 0)) if isinstance(p1_stats, dict) else 0
    high_thr_display = f"{float(high_thr_used):.2f}" if high_thr_used is not None else "N/A"
    print(
        f"  Init mode={init_meta.get('mode')} | shape={centroids.shape} | high_thr={high_thr_display} | "
        f"primary={p1_primary} | subseg_used={p1_subseg}"
    )
    if init_meta.get("speaker_seed_info"):
        for item in init_meta["speaker_seed_info"]:
            print(
                f"    - idx={item['index']} | seed={item['seed_type']} | name={item['seed_name']} | "
                f"phase1_match={item.get('matched_phase1_cluster')}"
            )

    if mod.DEVICE.type == "cuda":
        mod.torch.cuda.empty_cache()

    low_thr = float(mod.LOW_THRESHOLD)
    print("\n[Phase 2] Running main flow with fixed low threshold...")
    print(f"  n_clusters={mod.FIXED_N_CLUSTERS} | high_thr={high_thr_display} | low_threshold={low_thr:.2f}")
    print("-" * 170)
    print(
        f"{'LOW_THR':<8} | {'Emb':<5} | {'P/A/F':<12} | {'Assigned':<8} | "
        f"{'Prop':<5} | {'GlobProp':<9} | {'Unmap':<5} | {'Purify':<6} | {'VBxRel':<6} | "
        f"{'DER%':<8} | {'Miss%':<8} | {'FA%':<8} | {'Conf%':<8} | {'OvF1':<8}"
    )
    print("-" * 170)

    result = None
    os.makedirs(mod.FORENSIC_DIR, exist_ok=True)

    embs, sinfo, collect_stats = mod.collect_assign_embeddings(
        chunk_outputs, wav, ecapa, resampler, threshold=low_thr
    )

    n_emb = len(embs)
    tiers_str = f"{collect_stats['primary']}/{collect_stats['aux']}/{collect_stats['fallback']}"
    if n_emb == 0:
        print(
            f"{low_thr:<8.2f} | {0:<5d} | {tiers_str:<12} | {'SKIP':<8} | "
            f"{'-':<5} | {'-':<5} | {'-':<6} | {'-':<6} | {'NA':<7} | {'NA':<8} | {'NA':<8} | {'NA':<8} | {'NA':<8} | {'NA':<8}"
        )
        result = {
            "low_threshold": float(low_thr),
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
            threshold=low_thr,
            mapped_keys=set(base_mapping.keys()),
        )
        final_mapping, final_track_meta, prop_stats = mod.propagate_missing_tracks(
            chunk_outputs, base_mapping, track_meta, mod.FIXED_N_CLUSTERS,
            force_assign_track_meta=force_assign_track_meta,
            threshold=low_thr,
        )

        raw_hyp, raw_rows = mod.build_raw_hypothesis(chunk_outputs, final_mapping, final_track_meta, low_thr, mod.UTT_ID)
        sf_hyp, sf_rows, sf_stats = mod.apply_silence_filter(raw_rows, wav, mod.UTT_ID)
        raw_hyp, raw_rows = sf_hyp, sf_rows

        low_tag = f"{int(round(low_thr * 100)):03d}"
        vbx_hyp, vbx_rows, vbx_stats = mod.apply_real_vbx_single_speaker_backbone(
            raw_rows, mod.UTT_ID, mod.WAV_PATH, f"low{low_tag}",
            ecapa=ecapa, wav_tensor=wav, resampler=resampler,
        )
        final_stage = mod.build_stage_report("post_real_vbx", ref, vbx_hyp, total_dur)
        hyp, final_rows = vbx_hyp, vbx_rows

        der, miss, fa, conf = final_stage["der"], final_stage["miss"], final_stage["fa"], final_stage["conf"]
        ov_f1 = final_stage["overlap_metrics"]["overlap"]["f1"]

        hyp_rttm_path = os.path.join(mod.FORENSIC_DIR, f"hyp_low{low_tag}.rttm")
        raw_rttm_path = os.path.join(mod.FORENSIC_DIR, f"raw_low{low_tag}.rttm")
        cluster_name_map = mod.build_cluster_name_map(init_meta)
        if cluster_name_map:
            print(f"  [INFO] Remapping RTTM labels: {cluster_name_map}")
        hyp = mod.remap_rttm_labels(hyp, cluster_name_map)
        raw_hyp = mod.remap_rttm_labels(raw_hyp, cluster_name_map)
        mod.save_rttm(hyp, hyp_rttm_path)
        mod.save_rttm(raw_hyp, raw_rttm_path)

        if mod.SKIP_DER:
            der_str = miss_str = fa_str = conf_str = "N/A"
        else:
            der_str = f"{der:<8.2f}"
            miss_str = f"{miss:<8.2f}"
            fa_str = f"{fa:<8.2f}"
            conf_str = f"{conf:<8.2f}"
        print(
            f"{low_thr:<8.2f} | {n_emb:<5d} | {tiers_str:<12} | "
            f"{len(base_mapping):<8d} | {prop_stats['propagated_tracks']:<5d} | "
            f"{prop_stats.get('global_propagated_tracks', 0):<9d} | "
            f"{prop_stats['unmapped_tracks']:<5d} | {purify_stats['selected_total']:<6d} | {vbx_stats['relabelled_intervals']:<6d} | "
            f"{der_str} | {miss_str} | {fa_str} | {conf_str} | {ov_f1:<8.3f}"
        )

        result = {
            "low_threshold": float(low_thr),
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

    print("-" * 160)
    if result and not result.get("skipped", False):
        if not mod.SKIP_DER:
            print(
                f"\nFinal DER: {result['der']:.2f}% @ low_thr={result['low_threshold']:.2f} "
                f"(Miss={result['miss']:.2f}%, FA={result['fa']:.2f}%, Conf={result['conf']:.2f}%)"
            )
        else:
            print(
                f"\nFinal DER: N/A (DER computation skipped) @ low_thr={result['low_threshold']:.2f}"
            )
        ov = result["final_overlap_metrics"]["overlap"]
        print(
            f"Final overlap: Precision={ov['precision']:.3f} Recall={ov['recall']:.3f} F1={ov['f1']:.3f}"
        )

    os.makedirs(os.path.dirname(mod.OUT_JSON), exist_ok=True)
    with open(mod.OUT_JSON, "w", encoding="utf-8") as f:
        json.dump(result, f, indent=2, ensure_ascii=False)

    print(f"\nResults saved to: {mod.OUT_JSON}")
    return result

def write_failure_manifest(path: Path, failures: List[Dict[str, Any]]) -> None:
    safe_mkdir(path.parent)
    path.write_text(json.dumps(failures, indent=2, ensure_ascii=False), encoding="utf-8")

def worker_main(worker_manifest: Path) -> int:
    manifest = json.loads(worker_manifest.read_text(encoding="utf-8"))
    checkpoint = checkpoint_from_payload(manifest["checkpoint"])
    samples = [sample_from_payload(x) for x in manifest.get("samples", [])]
    eval_script = Path(manifest["eval_script"])
    runs_root = Path(manifest["runs_root"])
    failed_manifest = Path(manifest["failed_manifest"])
    hf_model = str(manifest["hf_model"])

    failures: List[Dict[str, Any]] = []
    mod = None
    model = None
    try:
        mod = load_original_eval_module(eval_script)
        model = init_checkpoint_model(mod, hf_model, checkpoint)
    except Exception as exc:
        tb = traceback.format_exc()
        for sample in samples:
            failures.append({
                "checkpoint_label": checkpoint.label,
                "split": sample.split,
                "case_id": sample.case_id,
                "group_name": sample.group_name,
                "n_speakers": sample.n_speakers,
                "stage": "worker_model_init",
                "error": str(exc),
                "traceback": tb,
            })
        write_failure_manifest(failed_manifest, failures)
        return 0

    try:
        for idx, sample in enumerate(samples, start=1):
            run_dir = safe_mkdir(runs_root / slugify(checkpoint.label) / sample.split / sample.case_id)
            out_json = run_dir / "result.json"
            stdout_path = run_dir / "stdout.log"
            stderr_path = run_dir / "stderr.log"
            try:
                with stdout_path.open("w", encoding="utf-8", errors="ignore") as stdout_f, stderr_path.open("w", encoding="utf-8", errors="ignore") as stderr_f:
                    with contextlib.redirect_stdout(stdout_f), contextlib.redirect_stderr(stderr_f):
                        print(f"[WORKER {idx}/{len(samples)}] {checkpoint.label} | {sample.split}/{sample.case_id}")
                        evaluate_sample_inprocess(mod, model, sample, checkpoint, out_json)
            except Exception as exc:
                if out_json.exists():
                    try:
                        out_json.unlink()
                    except Exception:
                        pass
                tb = traceback.format_exc()
                with stderr_path.open("a", encoding="utf-8", errors="ignore") as stderr_f:
                    stderr_f.write("\n\n[WORKER ERROR]\n")
                    stderr_f.write(tb)
                failures.append({
                    "checkpoint_label": checkpoint.label,
                    "split": sample.split,
                    "case_id": sample.case_id,
                    "group_name": sample.group_name,
                    "n_speakers": sample.n_speakers,
                    "stage": "worker_sample",
                    "error": str(exc),
                    "traceback": tb,
                    "stdout_log": str(stdout_path),
                    "stderr_log": str(stderr_path),
                })
    finally:
        write_failure_manifest(failed_manifest, failures)
        if mod is not None and getattr(mod, "DEVICE", None) is not None and getattr(mod.DEVICE, "type", None) == "cuda":
            try:
                mod.torch.cuda.empty_cache()
            except Exception:
                pass
        model = None
    return 0

def write_worker_manifest(
    manifest_path: Path,
    eval_script: Path,
    checkpoint: CheckpointSpec,
    samples: Sequence[SampleSpec],
    runs_root: Path,
    hf_model: str,
) -> Path:
    payload = {
        "eval_script": str(eval_script),
        "checkpoint": checkpoint_to_payload(checkpoint),
        "runs_root": str(runs_root),
        "hf_model": hf_model,
        "failed_manifest": str(manifest_path.with_name(manifest_path.stem + "_failed.json")),
        "samples": [sample_to_payload(s) for s in samples],
    }
    safe_mkdir(manifest_path.parent)
    manifest_path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
    return manifest_path

def run_checkpoint_worker(
    python_exe: Path,
    this_script: Path,
    manifest_path: Path,
    quiet: bool = False,
) -> Tuple[int, str, str, Path]:
    cmd = [str(python_exe), str(this_script), "--worker_manifest", str(manifest_path)]
    if quiet:
        cmd.append("--quiet")
    proc = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
    failed_manifest = Path(json.loads(manifest_path.read_text(encoding="utf-8"))["failed_manifest"])
    worker_log = manifest_path.with_suffix(".worker.log")
    worker_err = manifest_path.with_suffix(".worker.err.log")
    worker_log.write_text(proc.stdout, encoding="utf-8", errors="ignore")
    worker_err.write_text(proc.stderr, encoding="utf-8", errors="ignore")
    return proc.returncode, proc.stdout, proc.stderr, failed_manifest

def load_failure_manifest(path: Path) -> List[Dict[str, Any]]:
    if not path.exists():
        return []
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return []

def pending_samples_for_checkpoint(
    checkpoint: CheckpointSpec,
    samples: Sequence[SampleSpec],
    runs_root: Path,
    force_rerun: bool,
) -> List[SampleSpec]:
    if force_rerun:
        return list(samples)
    pending: List[SampleSpec] = []
    ckpt_slug = slugify(checkpoint.label)
    for sample in samples:
        out_json = runs_root / ckpt_slug / sample.split / sample.case_id / "result.json"
        if not out_json.exists():
            pending.append(sample)
    return pending

def run_checkpoint_batch(
    python_exe: Path,
    this_script: Path,
    eval_script: Path,
    checkpoint: CheckpointSpec,
    samples: Sequence[SampleSpec],
    runs_root: Path,
    rttm_root: Path,
    manifests_dir: Path,
    hf_model: str,
    quiet: bool,
    failed_records: List[Dict[str, Any]],
) -> None:
    pending = list(samples)
    if not pending:
        return

    manifest_path = manifests_dir / f"{slugify(checkpoint.label)}.json"
    write_worker_manifest(manifest_path, eval_script, checkpoint, pending, runs_root, hf_model)
    rc, worker_stdout, worker_stderr, failed_manifest = run_checkpoint_worker(python_exe, this_script, manifest_path, quiet=quiet)

    if rc != 0:
        if not quiet:
            print(f"[WARN] Worker crashed for {checkpoint.label}; falling back to per-sample subprocess runs.")
        failures = [
            {
                "checkpoint_label": checkpoint.label,
                "split": s.split,
                "case_id": s.case_id,
                "group_name": s.group_name,
                "n_speakers": s.n_speakers,
                "stage": "worker_process_crash",
                "error": f"worker_exit_code={rc}",
            }
            for s in pending
        ]
    else:
        failures = load_failure_manifest(failed_manifest)

    retry_samples: List[SampleSpec] = []
    if failures:
        lookup = {(s.split, s.case_id): s for s in pending}
        for rec in failures:
            sample = lookup.get((str(rec.get("split")), str(rec.get("case_id"))))
            if sample is not None:
                retry_samples.append(sample)

    if retry_samples and not quiet:
        print(f"[INFO] Retrying {len(retry_samples)} failed samples individually for checkpoint {checkpoint.label}.")

    for sample in retry_samples:
        try:
            run_one(
                python_exe=python_exe,
                eval_script=eval_script,
                checkpoint=checkpoint,
                sample=sample,
                runs_root=runs_root,
                rttm_root=rttm_root,
                hf_model=hf_model,
                skip_existing=False,
            )
        except Exception as exc:
            failed_records.append({
                "checkpoint_label": checkpoint.label,
                "split": sample.split,
                "case_id": sample.case_id,
                "group_name": sample.group_name,
                "n_speakers": sample.n_speakers,
                "reason": str(exc),
            })

def write_failed_runs_csv(path: Path, failed_records: Sequence[Dict[str, Any]]) -> None:
    df = pd.DataFrame(list(failed_records))
    if df.empty:
        if path.exists():
            path.unlink()
        return
    df.to_csv(path, index=False, encoding="utf-8-sig")
def aggregate_mean(df: pd.DataFrame, group_cols: Sequence[str]) -> pd.DataFrame:
    metric_cols = [
        "der", "miss", "fa", "conf",
        "ov_precision", "ov_recall", "ov_f1",
        "pred_segments", "raw_segments", "vbx_segments", "ref_segments",
    ]
    existing = [c for c in metric_cols if c in df.columns]
    out = (
        df.groupby(list(group_cols), dropna=False, observed=False)
          .agg(
              n_cases=("case_id", "nunique"),
              **{col: (col, "mean") for col in existing},
          )
          .reset_index()
    )
    return out

def sort_checkpoint_labels(labels: Iterable[str]) -> List[str]:
    def key_fn(label: str) -> Tuple[int, int, str]:
        if label == "baseline":
            return (0, 0, label)
        m = re.search(r"ep(\d+)", label.lower())
        if m:
            return (1, int(m.group(1)), label.lower())
        if label.lower() == "best_model":
            return (2, 9998, label.lower())
        return (3, 9999, label.lower())
    return sorted(set(labels), key=key_fn)

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
        "legend.fontsize": 10.5,
        "legend.title_fontsize": 11.5,
        "xtick.labelsize": 10.5,
        "ytick.labelsize": 10.5,
        "savefig.bbox": "tight",
        "savefig.facecolor": "white",
        "lines.solid_capstyle": "round",
        "lines.solid_joinstyle": "round",
        "axes.axisbelow": True,
    })

PAPER_QUAL_COLORS = [
    "#4477AA", "#66CCEE", "#228833", "#CCBB44", "#EE6677", "#AA3377",
    "#332288", "#88CCEE", "#44AA99", "#117733", "#999933", "#DDCC77",
    "#CC6677", "#882255", "#AA4499", "#EE7733", "#0077BB", "#33BBEE",
    "#009988", "#CC3311", "#EE3377",
]

PHASE_COLOR_POOLS = {
    "phase1": ["#4477AA", "#66CCEE", "#004488", "#77AADD", "#0077BB"],
    "phase2": ["#228833", "#44AA99", "#009988", "#EE7733", "#CCBB44", "#999933"],
    "phase3": ["#AA3377", "#CC6677", "#882255", "#AA4499", "#EE6677"],
    "other":  ["#332288", "#88CCEE", "#117733", "#CC3311", "#BBBBBB"],
}

PHASE_LINESTYLES = {
    "phase1": "-",
    "phase2": (0, (6, 1.8)),
    "phase3": (0, (4, 1.4, 1.2, 1.4)),
    "other": (0, (2.5, 1.4)),
}

PHASE_MARKERS = {
    "phase1": ["o", "s", "^", "D", "P"],
    "phase2": ["o", "s", "^", "D", "P", "X"],
    "phase3": ["o", "s", "^", "D", "P"],
    "other": ["o", "s", "^", "D"],
}

COLOR_CYCLE = PAPER_QUAL_COLORS
MARKER_CYCLE = ["o", "s", "^", "D", "P", "X", "v", "<", ">", "h"]

def _metric_value_format(metric: str) -> str:
    metric = str(metric).lower()
    if metric in {"pred_segments", "ref_segments", "n_cases"}:
        return "{:.0f}"
    if metric.startswith("ov_"):
        return "{:.3f}"
    return "{:.1f}"

def _phase_from_checkpoint(label: str) -> str:
    s = str(label).lower()
    if "phase1" in s or re.search(r"ep00[1-4]\b", s):
        return "phase1"
    if "phase2" in s or re.search(r"ep00[5-9]\b|ep010\b", s):
        return "phase2"
    if "phase3" in s or re.search(r"ep01[1-4]\b", s):
        return "phase3"
    return "other"

def _checkpoint_numeric_order(label: str) -> int:
    m = re.search(r"ep(\d+)", str(label).lower())
    return int(m.group(1)) if m else 999

def _annotation_indices(n_points: int, n_series: int) -> List[int]:
    if n_points <= 0:
        return []
    if n_points <= 3 and n_series <= 4:
        return list(range(n_points))
    if n_points <= 6 and n_series <= 3:
        return sorted(set([0, n_points - 1]))
    return [n_points - 1]

def _annotate_line_points(ax, xs: Sequence[float], ys: Sequence[float], metric: str, color: str, line_idx: int, n_series: int) -> None:
    fmt = _metric_value_format(metric)
    offset_cycle = [8, -10, 12, -14, 16, -18]
    y_offset = offset_cycle[line_idx % len(offset_cycle)]
    idxs = _annotation_indices(len(xs), n_series)
    for ii in idxs:
        x = xs[ii]
        y = ys[ii]
        if y is None or (isinstance(y, float) and np.isnan(y)):
            continue
        ax.annotate(
            fmt.format(float(y)),
            (x, y),
            textcoords="offset points",
            xytext=(0 if ii < len(xs) - 1 else 4, y_offset),
            ha="center" if ii < len(xs) - 1 else "left",
            fontsize=8.2,
            fontweight="bold",
            color="#111111",
            bbox=dict(boxstyle="round,pad=0.18", facecolor="white", edgecolor=color, linewidth=0.7, alpha=0.95),
            clip_on=False,
            zorder=20,
        )

def _apply_line_polish(line, style):
    line.set_markeredgecolor("white")
    line.set_markeredgewidth(style.get("markeredgewidth", 1.1))
    line.set_path_effects([
        pe.Stroke(linewidth=style["linewidth"] + 1.4, foreground="white", alpha=0.95),
        pe.Normal(),
    ])

def _plot_metric_lines(
    df: pd.DataFrame,
    group_col: str,
    metrics: Sequence[Tuple[str, str]],
    title: str,
    out_path: Path,
    legend_title: str,
    footnote: Optional[str] = None,
    max_cols: int = 2,
) -> None:
    if df.empty:
        return
    ckpt_order = sort_checkpoint_labels(df["checkpoint_label"].dropna().tolist())
    x_index = {label: i for i, label in enumerate(ckpt_order)}
    groups = [g for g in df[group_col].dropna().unique().tolist()]
    groups = sorted(groups, key=natural_key)

    n_metrics = len(metrics)
    ncols = min(max_cols, n_metrics)
    nrows = int(np.ceil(n_metrics / ncols))
    fig, axes = plt.subplots(nrows, ncols, figsize=(max(14, 4.8 * ncols), max(4.8, 3.9 * nrows)), squeeze=False)
    axes = axes.flatten()

    for ax_idx, (metric, ylabel) in enumerate(metrics):
        ax = axes[ax_idx]
        for gi, group in enumerate(groups):
            sdf = df[df[group_col] == group].copy()
            sdf = sdf.sort_values(by="checkpoint_label", key=lambda s: s.map(x_index))
            x = [x_index[v] for v in sdf["checkpoint_label"]]
            y = sdf[metric].astype(float).tolist()
            style = {
                "color": COLOR_CYCLE[gi % len(COLOR_CYCLE)],
                "linestyle": "-",
                "linewidth": 2.4,
                "marker": MARKER_CYCLE[gi % len(MARKER_CYCLE)],
                "markersize": 5.8,
                "alpha": 0.96,
                "zorder": 4,
            }
            line, = ax.plot(x, y, label=str(group), **style)
            _apply_line_polish(line, style)
            _annotate_line_points(ax, x, y, metric, style["color"], gi, len(groups))
        ax.set_title(ylabel, fontweight="bold")
        ax.set_ylabel(ylabel)
        ax.set_xticks(list(x_index.values()))
        ax.set_xticklabels(ckpt_order, rotation=40, ha="right")
        ax.margins(x=0.04)

    for ax in axes[n_metrics:]:
        ax.axis("off")

    handles, labels = axes[0].get_legend_handles_labels()
    if handles and len(handles) > 1:
        fig.legend(
            handles, labels, title=legend_title, loc="center left", bbox_to_anchor=(1.01, 0.5),
            frameon=True, fancybox=True, edgecolor="#D0D0D0"
        )

    fig.suptitle(title, fontsize=20, fontweight="bold", y=1.02)
    if footnote:
        fig.text(0.01, -0.02, footnote, ha="left", va="top", fontsize=10, color="#555555")
    fig.tight_layout()
    safe_mkdir(out_path.parent)
    fig.savefig(out_path, dpi=240)
    plt.close(fig)

def _checkpoint_line_style(label: str, color_idx: int) -> Dict[str, Any]:
    lower = str(label).lower()
    if lower == "baseline":
        return {
            "color": "#222222",
            "linestyle": "-",
            "linewidth": 3.0,
            "marker": "D",
            "markersize": 6.6,
            "markeredgewidth": 1.25,
            "alpha": 1.0,
            "zorder": 9,
        }
    if lower == "best_model":
        return {
            "color": "#D55E00",
            "linestyle": "-",
            "linewidth": 3.1,
            "marker": "o",
            "markersize": 6.8,
            "markeredgewidth": 1.25,
            "alpha": 1.0,
            "zorder": 10,
        }

    phase = _phase_from_checkpoint(lower)
    phase_pool = PHASE_COLOR_POOLS.get(phase, PAPER_QUAL_COLORS)
    if phase == "phase1":
        idx_in_phase = max(0, _checkpoint_numeric_order(lower) - 1)
    elif phase == "phase2":
        idx_in_phase = max(0, _checkpoint_numeric_order(lower) - 5)
    elif phase == "phase3":
        idx_in_phase = max(0, _checkpoint_numeric_order(lower) - 11)
    else:
        idx_in_phase = color_idx
    color = phase_pool[idx_in_phase % len(phase_pool)]
    marker_pool = PHASE_MARKERS.get(phase, ["o", "s", "^", "D"])
    marker = marker_pool[idx_in_phase % len(marker_pool)]
    return {
        "color": color,
        "linestyle": PHASE_LINESTYLES.get(phase, "-"),
        "linewidth": 2.35,
        "marker": marker,
        "markersize": 5.8,
        "markeredgewidth": 1.05,
        "alpha": 0.98,
        "zorder": 5,
    }

def _plot_checkpoint_lines_by_category(
    df: pd.DataFrame,
    category_col: str,
    metrics: Sequence[Tuple[str, str]],
    title: str,
    out_path: Path,
    x_label: str,
    legend_title: str = "Checkpoint",
    footnote: Optional[str] = None,
    max_cols: int = 2,
    category_order: Optional[Sequence[str]] = None,
) -> None:
    if df.empty:
        return

    ckpt_order = sort_checkpoint_labels(df["checkpoint_label"].dropna().astype(str).tolist())
    categories = list(category_order) if category_order is not None else sorted(df[category_col].dropna().astype(str).unique().tolist(), key=natural_key)
    if not categories:
        return

    x_index = {cat: i for i, cat in enumerate(categories)}
    n_metrics = len(metrics)
    ncols = min(max_cols, n_metrics)
    nrows = int(np.ceil(n_metrics / ncols))
    fig, axes = plt.subplots(nrows, ncols, figsize=(max(15, 5.2 * ncols), max(5.2, 4.1 * nrows)), squeeze=False)
    axes = axes.flatten()

    for ax_idx, (metric, ylabel) in enumerate(metrics):
        ax = axes[ax_idx]
        for ci, ckpt in enumerate(ckpt_order):
            sdf = df[df["checkpoint_label"].astype(str) == str(ckpt)].copy()
            if sdf.empty:
                continue
            val_map = {str(row[category_col]): row[metric] for _, row in sdf.iterrows()}
            xs = list(range(len(categories)))
            ys = [float(val_map.get(cat)) if pd.notna(val_map.get(cat)) else np.nan for cat in categories]
            style = _checkpoint_line_style(str(ckpt), ci)
            line, = ax.plot(xs, ys, label=str(ckpt), **style)
            _apply_line_polish(line, style)
            _annotate_line_points(ax, xs, ys, metric, style["color"], ci, len(ckpt_order))

        ax.set_title(ylabel, fontweight="bold")
        ax.set_ylabel(ylabel)
        ax.set_xlabel(x_label)
        ax.set_xticks(list(range(len(categories))))
        ax.set_xticklabels(categories, rotation=22, ha="right")
        ax.margins(x=0.05)

    for ax in axes[n_metrics:]:
        ax.axis("off")

    handles, labels = axes[0].get_legend_handles_labels()
    if handles:
        fig.legend(
            handles,
            labels,
            title=legend_title,
            loc="upper center",
            bbox_to_anchor=(0.5, -0.03),
            ncol=min(5, max(2, int(np.ceil(len(labels) / 5)))),
            frameon=True,
            fancybox=True,
            edgecolor="#D0D0D0",
        )

    fig.suptitle(title, fontsize=20, fontweight="bold", y=1.02)
    if footnote:
        fig.text(0.01, -0.08, footnote, ha="left", va="top", fontsize=10, color="#555555")
    fig.tight_layout()
    safe_mkdir(out_path.parent)
    fig.savefig(out_path, dpi=240)
    plt.close(fig)

def plot_test_labeled_class_curves(class_summary: pd.DataFrame, plots_dir: Path) -> None:
    if class_summary.empty:
        return
    footnote = "Each line is one checkpoint. X-axis is class from metadata_labelled. Segments = predicted final RTTM segment count from the unchanged evaluator."
    class_order = sorted(class_summary["group_name"].dropna().astype(str).unique().tolist(), key=natural_key)
    _plot_checkpoint_lines_by_category(
        class_summary,
        category_col="group_name",
        category_order=class_order,
        metrics=[("der", "DER %"), ("miss", "Miss %"), ("fa", "FA %"), ("conf", "Conf %")],
        title="test_labeled — DER Components by Class (Multiple Checkpoints per Plot)",
        out_path=plots_dir / "01_test_labeled_der_components_by_class_checkpoint_lines.png",
        x_label="Class",
        legend_title="Checkpoint",
        footnote=footnote,
    )
    _plot_checkpoint_lines_by_category(
        class_summary,
        category_col="group_name",
        category_order=class_order,
        metrics=[("ov_precision", "Overlap Precision"), ("ov_recall", "Overlap Recall"), ("ov_f1", "Overlap F1"), ("pred_segments", "Predicted Segments")],
        title="test_labeled — Overlap Metrics and Segment Count by Class (Multiple Checkpoints per Plot)",
        out_path=plots_dir / "02_test_labeled_overlap_and_segments_by_class_checkpoint_lines.png",
        x_label="Class",
        legend_title="Checkpoint",
        footnote=footnote,
    )

def plot_vivo_curves(vivo_summary: pd.DataFrame, vivo_spk_summary: pd.DataFrame, plots_dir: Path) -> None:
    if vivo_summary.empty:
        return
    footnote = "Overall vivo progression across checkpoints."
    _plot_metric_lines(
        vivo_summary.assign(series_name="vivo"),
        group_col="series_name",
        metrics=[("der", "DER %"), ("miss", "Miss %"), ("fa", "FA %"), ("conf", "Conf %")],
        title="test_data (vivo) — Overall DER Components Across Checkpoints",
        out_path=plots_dir / "03_vivo_der_components_overall.png",
        legend_title="Series",
        footnote=footnote,
        max_cols=2,
    )
    _plot_metric_lines(
        vivo_summary.assign(series_name="vivo"),
        group_col="series_name",
        metrics=[("ov_precision", "Overlap Precision"), ("ov_recall", "Overlap Recall"), ("ov_f1", "Overlap F1"), ("pred_segments", "Predicted Segments")],
        title="test_data (vivo) — Overall Overlap Metrics and Segment Count Across Checkpoints",
        out_path=plots_dir / "04_vivo_overlap_and_segments_overall.png",
        legend_title="Series",
        footnote=footnote,
        max_cols=2,
    )
    if not vivo_spk_summary.empty:
        footnote2 = "Each line is one checkpoint. X-axis is speaker count from metadata_test_data."
        df = vivo_spk_summary.copy()
        df["spk_group"] = df["n_speakers"].astype(int).astype(str) + " spk"
        spk_order = sorted(df["spk_group"].dropna().astype(str).unique().tolist(), key=natural_key)
        _plot_checkpoint_lines_by_category(
            df,
            category_col="spk_group",
            category_order=spk_order,
            metrics=[("der", "DER %"), ("miss", "Miss %"), ("fa", "FA %"), ("conf", "Conf %")],
            title="test_data (vivo) — DER Components by Number of Speakers (Multiple Checkpoints per Plot)",
            out_path=plots_dir / "05_vivo_der_components_by_nspeaker_checkpoint_lines.png",
            x_label="Speaker count",
            legend_title="Checkpoint",
            footnote=footnote2,
        )
        _plot_checkpoint_lines_by_category(
            df,
            category_col="spk_group",
            category_order=spk_order,
            metrics=[("pred_segments", "Predicted Segments"), ("ref_segments", "Reference Segments")],
            title="test_data (vivo) — Segment Count by Number of Speakers (Multiple Checkpoints per Plot)",
            out_path=plots_dir / "06_vivo_segments_by_nspeaker_checkpoint_lines.png",
            x_label="Speaker count",
            legend_title="Checkpoint",
            footnote=footnote2,
            max_cols=2,
        )

def plot_vivo_overlap_spk_curves(vivo_overlap_spk_summary: pd.DataFrame, plots_dir: Path) -> None:
    if vivo_overlap_spk_summary.empty:
        return
    footnote = "Each line is one checkpoint. X-axis is overlap bucket + number of speakers from metadata_test_data."
    df = vivo_overlap_spk_summary.copy()
    df["series_name"] = (
        df["overlap_bucket"].astype(str)
        + " | "
        + df["n_speakers"].astype(int).astype(str)
        + " spk"
    )
    overlap_order = []
    for ov in sorted(df["overlap_bucket"].dropna().astype(str).unique().tolist(), key=natural_key):
        sub = df[df["overlap_bucket"].astype(str) == ov]
        for spk in sorted(sub["n_speakers"].dropna().astype(int).unique().tolist()):
            overlap_order.append(f"{ov} | {spk} spk")
    _plot_checkpoint_lines_by_category(
        df,
        category_col="series_name",
        category_order=overlap_order,
        metrics=[("der", "DER %"), ("miss", "Miss %"), ("fa", "FA %"), ("conf", "Conf %")],
        title="test_data (vivo) — DER by Overlap Bucket and Number of Speakers (Multiple Checkpoints per Plot)",
        out_path=plots_dir / "07_vivo_der_components_by_overlap_and_nspeaker_checkpoint_lines.png",
        x_label="Overlap bucket | Speaker count",
        legend_title="Checkpoint",
        footnote=footnote,
    )
    _plot_checkpoint_lines_by_category(
        df,
        category_col="series_name",
        category_order=overlap_order,
        metrics=[("pred_segments", "Predicted Segments"), ("ref_segments", "Reference Segments")],
        title="test_data (vivo) — Segment Count by Overlap Bucket and Number of Speakers (Multiple Checkpoints per Plot)",
        out_path=plots_dir / "08_vivo_segments_by_overlap_and_nspeaker_checkpoint_lines.png",
        x_label="Overlap bucket | Speaker count",
        legend_title="Checkpoint",
        footnote=footnote,
        max_cols=2,
    )

def write_summary_markdown(
    out_path: Path,
    raw_df: pd.DataFrame,
    overall_summary: pd.DataFrame,
    class_summary: pd.DataFrame,
    vivo_spk_summary: pd.DataFrame,
    vivo_overlap_spk_summary: pd.DataFrame,
) -> None:
    lines: List[str] = []
    lines.append("# Checkpoint Evaluation Summary")
    lines.append("")
    lines.append("This report is produced by calling the original diarization evaluator unchanged for every sample/checkpoint, then aggregating the emitted JSON metrics.")
    lines.append("")

    if not overall_summary.empty:
        lines.append("## Overall summary by split and checkpoint")
        lines.append("")
        lines.append(overall_summary.to_markdown(index=False))
        lines.append("")

    if not class_summary.empty:
        lines.append("## test_labeled class summary")
        lines.append("")
        lines.append(class_summary.to_markdown(index=False))
        lines.append("")

    if not vivo_spk_summary.empty:
        lines.append("## test_data (vivo) by speaker count")
        lines.append("")
        lines.append(vivo_spk_summary.to_markdown(index=False))
        lines.append("")

    if not vivo_overlap_spk_summary.empty:
        lines.append("## test_data (vivo) by overlap bucket and speaker count")
        lines.append("")
        lines.append(vivo_overlap_spk_summary.to_markdown(index=False))
        lines.append("")

    lines.append("## Run counts")
    lines.append("")
    lines.append(f"- Total runs: {len(raw_df)}")
    lines.append(f"- Unique checkpoints: {raw_df['checkpoint_label'].nunique() if not raw_df.empty else 0}")
    lines.append(f"- Unique cases: {raw_df['case_id'].nunique() if not raw_df.empty else 0}")
    lines.append("")

    out_path.write_text("\n".join(lines), encoding="utf-8")

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate many checkpoints using the original diarization script unchanged and visualize metrics.")
    parser.add_argument("--root_dir", type=str, default=".", help="Project root containing cp/, data/, and the original evaluator script.")
    parser.add_argument("--python_exe", type=str, default=sys.executable, help="Python executable used to run the original evaluator.")
    parser.add_argument("--eval_script", type=str, default=EVAL_SCRIPT_DEFAULT, help="Original evaluator script path or filename under root_dir.")
    parser.add_argument("--cp_dir", type=str, default="cp", help="Checkpoint directory under root_dir or absolute path.")
    parser.add_argument("--data_root", type=str, default="data", help="Data root under root_dir or absolute path.")
    parser.add_argument("--output_dir", type=str, default="checkpoint_eval_report", help="Directory to store aggregated results, plots, and copied RTTMs.")
    parser.add_argument("--splits", nargs="+", default=["test_data", "test_labeled"], help="Which splits to evaluate.")
    parser.add_argument("--checkpoints", nargs="+", default=["all"], help="Checkpoint list. Use 'all' or names such as baseline ep005_phase2_top6_overlap.pth")
    parser.add_argument("--only_cases", nargs="*", default=None, help="Optional case IDs to restrict evaluation, e.g. data11 data1")
    parser.add_argument("--max_cases", type=int, default=None, help="Optional limit after filtering, for smoke tests.")
    parser.add_argument("--hf_model", type=str, default=HF_MODEL_DEFAULT, help="HF model id forwarded to the unchanged evaluator.")
    parser.add_argument("--force_rerun", action="store_true", help="Recompute runs even if per-run result.json already exists.")
    parser.add_argument("--quiet", action="store_true", help="Reduce console output.")
    parser.add_argument("--worker_manifest", type=str, default=None, help=argparse.SUPPRESS)
    return parser.parse_args()

def resolve_paths(args: argparse.Namespace) -> Dict[str, Path]:
    root_dir = Path(args.root_dir).resolve()
    python_exe = Path(args.python_exe).resolve()
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

    return {
        "root_dir": root_dir,
        "python_exe": python_exe,
        "eval_script": eval_script,
        "cp_dir": cp_dir,
        "data_root": data_root,
        "output_dir": output_dir,
    }

def coordinator_main(args: argparse.Namespace) -> None:
    paths = resolve_paths(args)
    root_dir = paths["root_dir"]
    python_exe = paths["python_exe"]
    eval_script = paths["eval_script"]
    cp_dir = paths["cp_dir"]
    data_root = paths["data_root"]
    output_dir = paths["output_dir"]

    if not python_exe.exists():
        raise FileNotFoundError(f"python_exe not found: {python_exe}")
    if not eval_script.exists():
        raise FileNotFoundError(f"eval_script not found: {eval_script}")
    if not cp_dir.exists():
        raise FileNotFoundError(f"cp_dir not found: {cp_dir}")
    if not data_root.exists():
        raise FileNotFoundError(f"data_root not found: {data_root}")

    runs_root = safe_mkdir(output_dir / "runs")
    plots_dir = safe_mkdir(output_dir / "plots")
    tables_dir = safe_mkdir(output_dir / "tables")
    rttm_root = safe_mkdir(output_dir / "rttm_cp_pred")
    manifests_dir = safe_mkdir(output_dir / "manifests")

    setup_plot_style()

    checkpoints = discover_checkpoints(cp_dir, args.checkpoints)
    samples = discover_samples(data_root, args.splits, verbose=not args.quiet)

    if args.only_cases:
        wanted = set(args.only_cases)
        samples = [s for s in samples if s.case_id in wanted]
    if args.max_cases is not None:
        samples = samples[: args.max_cases]

    if not checkpoints:
        raise RuntimeError("No checkpoints discovered.")
    if not samples:
        raise RuntimeError("No valid samples discovered.")

    if not args.quiet:
        print(f"[INFO] root_dir   : {root_dir}")
        print(f"[INFO] eval_script: {eval_script}")
        print(f"[INFO] python_exe : {python_exe}")
        print(f"[INFO] cp_dir     : {cp_dir}")
        print(f"[INFO] data_root  : {data_root}")
        print(f"[INFO] output_dir : {output_dir}")
        print(f"[INFO] checkpoints: {[c.label for c in checkpoints]}")
        print(f"[INFO] samples     : {len(samples)}")
        print(f"[INFO] splits      : {sorted(set(s.split for s in samples))}")

    failed_records: List[Dict[str, Any]] = []
    total = len(checkpoints) * len(samples)
    expected_done = 0
    this_script = Path(__file__).resolve()

    for ckpt in checkpoints:
        pending = pending_samples_for_checkpoint(ckpt, samples, runs_root, args.force_rerun)
        expected_done += len(pending)
        if not args.quiet:
            print(f"[CHECKPOINT] {ckpt.label} | pending={len(pending)} / {len(samples)}")
        if not pending:
            continue
        run_checkpoint_batch(
            python_exe=python_exe,
            this_script=this_script,
            eval_script=eval_script,
            checkpoint=ckpt,
            samples=pending,
            runs_root=runs_root,
            rttm_root=rttm_root,
            manifests_dir=manifests_dir,
            hf_model=args.hf_model,
            quiet=args.quiet,
            failed_records=failed_records,
        )

    raw_df, missing_records = gather_cached_rows(checkpoints, samples, runs_root, rttm_root, quiet=args.quiet)
    failed_records.extend(missing_records)
    if raw_df.empty:
        raise RuntimeError("No results were produced.")

    raw_df["checkpoint_label"] = pd.Categorical(
        raw_df["checkpoint_label"],
        categories=sort_checkpoint_labels(raw_df["checkpoint_label"].tolist()),
        ordered=True,
    )
    raw_df = raw_df.sort_values(["checkpoint_label", "split", "group_name", "case_id"]).reset_index(drop=True)

    overall_summary = aggregate_mean(raw_df, ["split", "checkpoint_label"])
    class_summary = aggregate_mean(raw_df[raw_df["split"] == "test_labeled"], ["split", "group_name", "checkpoint_label"])
    vivo_summary = aggregate_mean(raw_df[raw_df["split"] == "test_data"], ["split", "checkpoint_label"])
    vivo_spk_summary = aggregate_mean(raw_df[raw_df["split"] == "test_data"], ["split", "n_speakers", "checkpoint_label"])
    vivo_overlap_spk_summary = aggregate_mean(
        raw_df[raw_df["split"] == "test_data"],
        ["split", "overlap_bucket", "n_speakers", "checkpoint_label"],
    )

    raw_csv = tables_dir / "raw_case_results.csv"
    overall_csv = tables_dir / "overall_summary.csv"
    class_csv = tables_dir / "test_labeled_class_summary.csv"
    vivo_spk_csv = tables_dir / "test_data_vivo_by_nspeaker_summary.csv"
    vivo_overlap_spk_csv = tables_dir / "test_data_vivo_by_overlap_and_nspeaker_summary.csv"
    failed_csv = tables_dir / "failed_runs.csv"

    raw_df.to_csv(raw_csv, index=False, encoding="utf-8-sig")
    overall_summary.to_csv(overall_csv, index=False, encoding="utf-8-sig")
    class_summary.to_csv(class_csv, index=False, encoding="utf-8-sig")
    vivo_spk_summary.to_csv(vivo_spk_csv, index=False, encoding="utf-8-sig")
    vivo_overlap_spk_summary.to_csv(vivo_overlap_spk_csv, index=False, encoding="utf-8-sig")
    write_failed_runs_csv(failed_csv, failed_records)

    plot_test_labeled_class_curves(class_summary, plots_dir)
    plot_vivo_curves(vivo_summary, vivo_spk_summary, plots_dir)
    plot_vivo_overlap_spk_curves(vivo_overlap_spk_summary, plots_dir)

    write_summary_markdown(
        output_dir / "SUMMARY.md",
        raw_df=raw_df,
        overall_summary=overall_summary,
        class_summary=class_summary,
        vivo_spk_summary=vivo_spk_summary,
        vivo_overlap_spk_summary=vivo_overlap_spk_summary,
    )

    manifest = {
        "root_dir": str(root_dir),
        "python_exe": str(python_exe),
        "eval_script": str(eval_script),
        "cp_dir": str(cp_dir),
        "data_root": str(data_root),
        "output_dir": str(output_dir),
        "checkpoints": [
            {"label": c.label, "ckpt_path": None if c.ckpt_path is None else str(c.ckpt_path)}
            for c in checkpoints
        ],
        "n_samples": int(len(samples)),
        "expected_runs": int(len(checkpoints) * len(samples)),
        "cached_runs": int(len(raw_df)),
        "failed_runs": int(len(failed_records)),
        "splits": sorted(set(raw_df["split"].tolist())),
        "tables": {
            "raw_case_results": str(raw_csv),
            "overall_summary": str(overall_csv),
            "test_labeled_class_summary": str(class_csv),
            "test_data_vivo_by_nspeaker_summary": str(vivo_spk_csv),
            "test_data_vivo_by_overlap_and_nspeaker_summary": str(vivo_overlap_spk_csv),
            "failed_runs": str(failed_csv),
        },
        "plots_dir": str(plots_dir),
        "rttm_cp_pred": str(rttm_root),
    }
    (output_dir / "manifest.json").write_text(json.dumps(manifest, indent=2, ensure_ascii=False), encoding="utf-8")

    print("\n[DONE] Aggregation complete.")
    print(f"[DONE] Raw CSV          : {raw_csv}")
    print(f"[DONE] Overall CSV      : {overall_csv}")
    print(f"[DONE] Class CSV        : {class_csv}")
    print(f"[DONE] Vivo spk CSV     : {vivo_spk_csv}")
    print(f"[DONE] Vivo ov+spk CSV  : {vivo_overlap_spk_csv}")
    print(f"[DONE] Failed CSV       : {failed_csv}")
    print(f"[DONE] Plots directory  : {plots_dir}")
    print(f"[DONE] RTTM directory   : {rttm_root}")

if __name__ == "__main__":
    try:
        args = parse_args()
        if args.worker_manifest:
            sys.exit(worker_main(Path(args.worker_manifest).resolve()))
        coordinator_main(args)
    except Exception:
        traceback.print_exc()
        sys.exit(1)
import random
import logging
from pathlib import Path
from dataclasses import dataclass
from typing import List, Dict, Tuple, Optional
from collections import defaultdict, Counter, OrderedDict

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler
import torchaudio

log = logging.getLogger("ft")

try:
    if hasattr(torchaudio, "set_audio_backend"):
        for _backend in ("sox_io", "soundfile"):
            try:
                torchaudio.set_audio_backend(_backend)
                log.info(f"Using torchaudio backend: {_backend}")
                break
            except Exception:
                continue
except Exception:
    pass

from pyanote.ft_config import (
    FinetuneConfig, PhaseConfig,
    SOURCE_SELF, SOURCE_TIER_A, SOURCE_TIER_B, SOURCE_TIER_C, SOURCE_TIER_D,
    TIER_OVERLAP_MAP, TEST_MODE,
    SELF_LABELED_DIR, SELF_LABELED_META,
    DATA_MIX_DIR,  DATA_MIX_META,
    VAL_DATA_DIR, TEST_DATA_DIR, VAL_LABELED_DIR, TEST_LABELED_DIR,
)

SR              = 16000
FRAME_SHIFT_SEC = 0.02

def _load_audio(path: str):
    try:
        return torchaudio.load(path)
    except Exception as e:
        message = str(e)
        if "torchcodec" in message or "TorchCodec" in message:
            try:
                if hasattr(torchaudio, "set_audio_backend"):
                    for _backend in ("sox_io", "soundfile"):
                        try:
                            torchaudio.set_audio_backend(_backend)
                            return torchaudio.load(path)
                        except Exception:
                            continue
            except Exception:
                pass

            try:
                import soundfile as sf
                data, sr = sf.read(path, dtype="float32")
                data = np.asarray(data)
                if data.ndim == 1:
                    data = data[np.newaxis, :]
                else:
                    data = data.T
                return torch.from_numpy(data), sr
            except Exception:
                pass

            try:
                from scipy.io import wavfile
                sr, data = wavfile.read(path)
                if data.ndim == 1:
                    data = data[np.newaxis, :]
                else:
                    data = data.T
                if data.dtype != np.float32:
                    data = data.astype(np.float32)
                    if np.issubdtype(data.dtype, np.integer):
                        data /= np.iinfo(data.dtype).max
                return torch.from_numpy(data), sr
            except Exception:
                pass

        raise

@dataclass
class Sample:
    wav_path:       str
    rttm_path:      str
    source:         str
    n_speakers:     int
    load_speakers:  int
    duration:       float
    overlap_count:  int  = 0
    tape_index:     int  = 0
    crop_start:     float = 0.0
    crop_end:       float = -1.0
    enrollment_dir: str  = ""

    @property
    def spk_folder(self) -> str:

        return f"spk{self.n_speakers}" if self.n_speakers <= 4 else "spk5plus"

    @property
    def key(self) -> str:
        return f"{self.source}/{Path(self.wav_path).parent.parent.name}"

def parse_metadata(path: str) -> Dict[str, Dict]:

    out: Dict[str, Dict] = {}
    if not Path(path).exists():
        log.warning(f"  [meta] Not found: {path}")
        return out

    with open(path, encoding="utf-8") as f:
        for lineno, line in enumerate(f, 1):
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            parts = [p.strip() for p in line.split("|")]
            if len(parts) < 3:
                log.warning(f"  [meta] Line {lineno} malformed: {line!r}")
                continue
            try:
                name = parts[0]
                try:
                    overlap = int(parts[1])
                except ValueError:
                    overlap = 0

                try:
                    n_spk = int(parts[2])
                except (ValueError, IndexError):
                    log.warning(f"  [meta] Line {lineno} n_spk parse error — {line!r}")
                    continue

                try:
                    index = int(parts[3]) if len(parts) >= 4 else 0
                except (ValueError, IndexError):
                    index = 0

                out[name] = {"overlap": overlap, "n_spk": n_spk, "index": index}
            except ValueError as e:
                log.warning(f"  [meta] Line {lineno} parse error: {e} — {line!r}")

    log.info(f"  [meta] {len(out)} entries from {path}")
    return out

def _overlap_to_source(overlap: int) -> str:

    for src, (lo, hi) in TIER_OVERLAP_MAP.items():
        if lo <= overlap <= hi:
            return src
    return SOURCE_TIER_D

def _count_spk_from_rttm(path: str) -> int:

    spks = set()
    try:
        with open(path, encoding="utf-8") as f:
            for line in f:
                p = line.strip().split()
                if len(p) >= 8 and p[0] == "SPEAKER":
                    spks.add(p[7])
    except Exception:
        pass
    return max(len(spks), 2)

def _find_wav(data_dir: Path) -> Optional[Path]:

    audio_dir = data_dir / "audio"
    if audio_dir.exists():
        wavs = sorted(audio_dir.glob("*.wav"))
        if wavs:
            return wavs[0]

    for name in ("mixture.wav", "audio.wav"):
        p = data_dir / name
        if p.exists():
            return p

    wavs = sorted(data_dir.glob("*.wav"))
    return wavs[0] if wavs else None

def _find_rttm(data_dir: Path) -> Optional[Path]:

    labeled_dir = data_dir / "labeled"
    if labeled_dir.exists():
        rttms = sorted(labeled_dir.glob("*.rttm"))
        if rttms:
            return rttms[0]

    rttms = sorted(data_dir.glob("*.rttm"))
    return rttms[0] if rttms else None

def scan_directory(
    root_dir: str,
    source_name: str,
    meta_path: str,
    max_speakers: int = 4,
    max_n_files: Optional[int] = None,
    test_mode: bool = False,
) -> List[Sample]:

    root     = Path(root_dir)
    meta_map = parse_metadata(meta_path)
    samples: List[Sample] = []

    for data_dir in sorted(d for d in root.iterdir() if d.is_dir()):
        if max_n_files is not None and len(samples) >= max_n_files:
            break

        wav_path  = _find_wav(data_dir)
        rttm_path = _find_rttm(data_dir)
        if wav_path is None or rttm_path is None:
            continue

        try:
            import soundfile as sf
            sinfo = sf.info(str(wav_path))
            dur = sinfo.frames / sinfo.samplerate
        except ImportError:
            try:
                wav, sr = _load_audio(str(wav_path))
                dur = wav.shape[1] / sr
            except Exception as e2:
                log.error(f"  [torchaudio] Failed to load {wav_path}: {e2}")
                continue
        except Exception as e:
            log.error(f"  [soundfile] Failed to load {wav_path}: {e}")
            continue

        entry   = meta_map.get(data_dir.name, {})
        n_spk   = entry.get("n_spk", None)
        overlap = entry.get("overlap", 0)
        index   = entry.get("index", 0)

        if n_spk is None:
            n_spk = _count_spk_from_rttm(str(rttm_path))

        load_spk = min(n_spk, max_speakers)

        if source_name == SOURCE_SELF:
            src = SOURCE_SELF
        elif test_mode:
            src = SOURCE_TIER_A
        else:
            src = _overlap_to_source(overlap)

        enroll_dir = data_dir / "enrollment"

        samples.append(Sample(
            wav_path       = str(wav_path),
            rttm_path      = str(rttm_path),
            source         = src,
            n_speakers     = n_spk,
            load_speakers  = load_spk,
            duration       = dur,
            overlap_count  = overlap,
            tape_index     = index,
            crop_start     = 0.0,
            crop_end       = dur,
            enrollment_dir = str(enroll_dir) if enroll_dir.exists() else "",
        ))

    spk_dist = Counter(s.n_speakers for s in samples)
    tier_dist = Counter(s.source for s in samples)
    log.info(f"  [scan:{source_name}] {len(samples)} files from {root_dir}")
    log.info(f"    n_speakers dist: {dict(sorted(spk_dist.items()))}")
    log.info(f"    source tier dist: {dict(sorted(tier_dist.items()))}")
    n_capped = sum(1 for s in samples if s.n_speakers > max_speakers)
    if n_capped:
        log.info(
            f"    {n_capped} sessions have n_speakers > {max_speakers} "
            f"→ load_speakers capped to {max_speakers}"
        )
    return samples

def scan_self_labeled(
    root_dir: str = SELF_LABELED_DIR,
    meta_path: str = SELF_LABELED_META,
    max_speakers: int = 4,
    max_n_files: Optional[int] = None,
) -> List[Sample]:
    return scan_directory(
        root_dir, SOURCE_SELF, meta_path,
        max_speakers=max_speakers,
        max_n_files=max_n_files,
        test_mode=False,
    )

def scan_data_mix(
    root_dir: str = DATA_MIX_DIR,
    meta_path: str = DATA_MIX_META,
    max_speakers: int = 4,
    max_n_files: Optional[int] = None,
    test_mode: bool = TEST_MODE,
) -> List[Sample]:
    return scan_directory(
        root_dir, "tier",
        meta_path,
        max_speakers=max_speakers,
        max_n_files=max_n_files,
        test_mode=test_mode,
    )

def scan_all_sources(
    cfg: FinetuneConfig,
    max_n_files_per_source: Optional[int] = None,
) -> List[Sample]:

    if TEST_MODE and max_n_files_per_source is None:
        max_n_files_per_source = 2

    samples = []
    samples += scan_self_labeled(
        SELF_LABELED_DIR, SELF_LABELED_META,
        max_speakers=cfg.max_speakers,
        max_n_files=max_n_files_per_source,
    )
    samples += scan_data_mix(
        DATA_MIX_DIR, DATA_MIX_META,
        max_speakers=cfg.max_speakers,
        max_n_files=max_n_files_per_source,
        test_mode=TEST_MODE,
    )

    log.info(f"\n[Data] Total: {len(samples)} sessions")
    by_src: Dict[str, int] = defaultdict(int)
    for s in samples:
        by_src[s.source] += 1
    for src, n in sorted(by_src.items()):
        log.info(f"  {src:<28}: {n} sessions")
    return samples

def train_val_split(
    samples: List[Sample],
    val_ratio: float,
    seed: int,
    min_train_per_group: int = 1,
) -> Tuple[List[Sample], List[Sample]]:

    rng = random.Random(seed)

    by_group: Dict[str, List[Sample]] = defaultdict(list)
    seen: Dict[tuple, bool] = {}
    for s in samples:
        fkey = (s.source, s.n_speakers, s.wav_path)
        if fkey not in seen:
            seen[fkey] = True
            by_group[f"{s.source}/{s.spk_folder}"].append(s)

    train_files, val_files = [], []

    for group, files in sorted(by_group.items()):
        rng.shuffle(files)
        n_val = max(0, min(int(len(files) * val_ratio), len(files) - min_train_per_group))
        if n_val == 0:
            log.warning(f"  [split] {group}: {len(files)} file(s) → all train (too small for val)")
            train_files.extend(files)
        else:
            val_files.extend(files[:n_val])
            train_files.extend(files[n_val:])
            log.info(f"  [split] {group}: {len(files)} → train={len(files)-n_val}, val={n_val}")

    if not val_files and train_files:
        log.warning("  [split] No val files → using train files as val (TEST_MODE safety)")
        val_files = list(train_files)

    return train_files, val_files

def scan_fixed_split_dir(
    split_dir: str,
    meta_path: str,
    max_speakers: int = 4,
    source_name: str = SOURCE_SELF,
) -> List[Sample]:

    if not Path(split_dir).exists():
        log.warning(f"  [split_dir] Not found: {split_dir}")
        return []
    return scan_directory(
        split_dir, source_name, meta_path,
        max_speakers=max_speakers,
        max_n_files=None,
        test_mode=False,
    )

def stratified_sample_by_overlap(samples: List[Sample], target_count: int) -> List[Sample]:

    if not samples or target_count <= 0:
        return []
    if len(samples) <= target_count:
        return samples

    groups = defaultdict(list)
    for s in samples:
        groups[s.overlap_count].append(s)
    sorted_overlaps = sorted(groups.keys())

    selected = []
    pick_idx = {ov: 0 for ov in sorted_overlaps}
    while len(selected) < target_count:
        any_picked = False
        for ov in sorted_overlaps:
            if len(selected) >= target_count:
                break
            group_samples = groups[ov]
            if pick_idx[ov] < len(group_samples):
                selected.append(group_samples[pick_idx[ov]])
                pick_idx[ov] += 1
                any_picked = True
        if not any_picked:
            break

    log.info(f"  [sample] Stratified (overlap): {len(samples)} -> {len(selected)} files "
             f"across {len(sorted_overlaps)} overlap levels.")
    return selected

def stratified_sample_by_overlap_and_speakers(samples: List[Sample], target_count: int) -> List[Sample]:

    if not samples or target_count <= 0:
        return []
    if len(samples) <= target_count:
        return samples

    groups: Dict[tuple, List[Sample]] = defaultdict(list)
    for s in samples:
        tier = s.source
        n_spk_key = min(s.n_speakers, 4)
        groups[(tier, n_spk_key)].append(s)

    sorted_keys = sorted(groups.keys())
    selected: List[Sample] = []
    pick_idx = {k: 0 for k in sorted_keys}

    while len(selected) < target_count:
        any_picked = False
        for k in sorted_keys:
            if len(selected) >= target_count:
                break
            grp = groups[k]
            idx = pick_idx[k]
            if idx < len(grp):
                selected.append(grp[idx])
                pick_idx[k] = idx + 1
                any_picked = True
        if not any_picked:
            break

    tier_dist = Counter(f"{k[0]}/spk{k[1]}" for k in sorted_keys if pick_idx[k] > 0)
    log.info(
        f"  [sample] Stratified (overlap×spk): {len(samples)} -> {len(selected)} files | "
        f"groups={len(sorted_keys)} | dist={dict(tier_dist)}"
    )
    return selected

def parse_rttm(path: str) -> List[Dict]:
    segs = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            p = line.strip().split()
            if len(p) >= 8 and p[0] == "SPEAKER":
                segs.append({
                    "start":   float(p[3]),
                    "end":     float(p[3]) + float(p[4]),
                    "speaker": p[7],
                })
    return segs

def build_activity(
    segs: List[Dict],
    crop_start: float,
    crop_end: float,
    effective_speakers: int,
) -> np.ndarray:

    dur      = crop_end - crop_start
    n_frames = max(1, int(round(dur / FRAME_SHIFT_SEC)))
    activity = np.zeros((n_frames, effective_speakers), dtype=np.float32)

    cropped: List[Dict] = []
    for s in segs:
        s_c = max(s["start"] - crop_start, 0.0)
        e_c = min(s["end"]   - crop_start, dur)
        if e_c > s_c + 1e-4:
            cropped.append({"start": s_c, "end": e_c, "speaker": s["speaker"]})

    spk_map: Dict[str, int] = {}
    for s in sorted(cropped, key=lambda x: x["start"]):
        if s["speaker"] not in spk_map and len(spk_map) < effective_speakers:
            spk_map[s["speaker"]] = len(spk_map)

    for s in cropped:
        idx = spk_map.get(s["speaker"])
        if idx is None:
            continue
        sf = max(0, min(int(round(s["start"] / FRAME_SHIFT_SEC)), n_frames))
        ef = max(0, min(int(round(s["end"]   / FRAME_SHIFT_SEC)), n_frames))
        activity[sf:ef, idx] = 1.0

    return activity

def sliding_window_samples(
    samples: List[Sample],
    chunk_sec: float,
    stride_sec: float,
) -> List[Sample]:

    out: List[Sample] = []
    for s in samples:
        start = 0.0
        while start + chunk_sec * 0.5 <= s.duration:
            end = min(start + chunk_sec, s.duration)
            out.append(Sample(
                wav_path       = s.wav_path,
                rttm_path      = s.rttm_path,
                source         = s.source,
                n_speakers     = s.n_speakers,
                load_speakers  = s.load_speakers,
                duration       = s.duration,
                overlap_count  = s.overlap_count,
                tape_index     = s.tape_index,
                crop_start     = round(start, 3),
                crop_end       = round(end, 3),
                enrollment_dir = s.enrollment_dir,
            ))
            start += stride_sec
    return out

class AudioAugmenter:
    def __init__(self, sr: int = 16000):
        self.sr = sr

    def __call__(self, wav: torch.Tensor) -> torch.Tensor:
        if random.random() < 0.6:
            wav = wav * random.uniform(0.7, 1.3)
        if random.random() < 0.4:
            snr     = random.uniform(25, 40)
            sig_rms = wav.pow(2).mean().sqrt().clamp(min=1e-8)
            noise   = sig_rms / (10 ** (snr / 20))
            wav     = wav + torch.randn_like(wav) * noise
        if random.random() < 0.5:
            factor = random.uniform(0.92, 1.08)
            orig   = wav.shape[-1]
            new_sr = int(self.sr * factor)
            wav    = torchaudio.functional.resample(wav, self.sr, new_sr)
            if wav.shape[-1] < orig:
                wav = F.pad(wav, (0, orig - wav.shape[-1]))
            else:
                wav = wav[..., :orig]
        peak = wav.abs().max().clamp(min=1e-8)
        if peak > 1.0:
            wav = wav / peak
        return wav

class DiarizationDataset(Dataset):
    def __init__(
        self,
        samples: List[Sample],
        cfg: FinetuneConfig,
        stride_sec: float,
        augmenter: Optional[AudioAugmenter] = None,
        is_train: bool = True,
    ):
        self.crops     = sliding_window_samples(samples, cfg.chunk_sec, stride_sec)
        self.cfg       = cfg
        self.augmenter = augmenter
        self.is_train  = is_train
        self._wav_cache: OrderedDict[str, torch.Tensor] = OrderedDict()
        self._rttm_cache: OrderedDict[str, List[Dict]] = OrderedDict()
        log.info(
            f"    Dataset: {len(samples)} files → {len(self.crops)} crops "
            f"(stride={stride_sec}s, chunk={cfg.chunk_sec}s)"
        )

    def __len__(self) -> int:
        return len(self.crops)

    def _cache_get(self, cache: OrderedDict, key: str):
        if key in cache:
            value = cache.pop(key)
            cache[key] = value
            return value
        return None

    def _cache_put(self, cache: OrderedDict, key: str, value, max_size: int) -> None:
        if max_size <= 0:
            return
        if key in cache:
            cache.pop(key)
        cache[key] = value
        while len(cache) > max_size:
            cache.popitem(last=False)

    def _get_cached_wav(self, wav_path: str) -> torch.Tensor:
        cached = self._cache_get(self._wav_cache, wav_path)
        if cached is not None:
            return cached

        wav, sr = _load_audio(wav_path)
        if wav.shape[0] > 1:
            wav = wav.mean(0, keepdim=True)
        if sr != SR:
            wav = torchaudio.functional.resample(wav, sr, SR)
        wav = wav.contiguous()

        self._cache_put(
            self._wav_cache,
            wav_path,
            wav,
            int(getattr(self.cfg, "waveform_cache_size", 0)),
        )
        return wav

    def _get_cached_rttm(self, rttm_path: str) -> List[Dict]:
        cached = self._cache_get(self._rttm_cache, rttm_path)
        if cached is not None:
            return cached
        segs = parse_rttm(rttm_path)
        self._cache_put(
            self._rttm_cache,
            rttm_path,
            segs,
            int(getattr(self.cfg, "rttm_cache_size", 0)),
        )
        return segs

    def __getitem__(self, idx: int) -> Dict:
        crop = self.crops[idx]

        wav_full = self._get_cached_wav(crop.wav_path)

        s_smp = int(crop.crop_start * SR)
        e_smp = int(crop.crop_end   * SR)
        wav   = wav_full[:, s_smp:e_smp].contiguous()

        target = int(self.cfg.chunk_sec * SR)
        if wav.shape[1] < target:
            wav = F.pad(wav, (0, target - wav.shape[1]))

        if self.is_train and self.augmenter is not None:
            wav = self.augmenter(wav)

        segs     = self._get_cached_rttm(crop.rttm_path)
        activity = build_activity(
            segs, crop.crop_start, crop.crop_end,
            effective_speakers=crop.load_speakers,
        )

        if activity.shape[1] < self.cfg.max_speakers:
            pad = np.zeros(
                (activity.shape[0], self.cfg.max_speakers - activity.shape[1]),
                dtype=np.float32,
            )
            activity = np.concatenate([activity, pad], axis=1)
        labels = torch.from_numpy(activity)

        return {
            "waveform":       wav,
            "labels":         labels,
            "n_frames":       torch.tensor(labels.shape[0], dtype=torch.long),
            "n_speakers":     torch.tensor(crop.n_speakers,   dtype=torch.long),
            "load_speakers":  torch.tensor(crop.load_speakers, dtype=torch.long),
            "source":         crop.source,
            "spk_folder":     crop.spk_folder,
            "wav_path":       crop.wav_path,
            "enrollment_dir": crop.enrollment_dir,
            "crop_start":     crop.crop_start,
        }

def collate_fn(batch: List[Dict]) -> Dict:
    max_wav = max(b["waveform"].shape[1] for b in batch)
    max_lbl = max(b["labels"].shape[0]   for b in batch)
    B       = len(batch)
    S       = batch[0]["labels"].shape[1]

    waveforms = torch.zeros(B, 1, max_wav, dtype=batch[0]["waveform"].dtype)
    labels    = torch.zeros(B, max_lbl, S, dtype=batch[0]["labels"].dtype)
    n_frames  = torch.zeros(B, dtype=torch.long)

    for i, b in enumerate(batch):
        waveforms[i, :, :b["waveform"].shape[1]] = b["waveform"]
        labels[i,   :b["labels"].shape[0], :]    = b["labels"]
        n_frames[i] = b["labels"].shape[0]

    return {
        "waveform":        waveforms,
        "labels":          labels,
        "n_frames":        n_frames,
        "n_speakers":      torch.stack([b["n_speakers"]    for b in batch]),
        "load_speakers":   torch.stack([b["load_speakers"] for b in batch]),
        "sources":         [b["source"]         for b in batch],
        "spk_folders":     [b["spk_folder"]      for b in batch],
        "wav_paths":       [b["wav_path"]        for b in batch],
        "enrollment_dirs": [b["enrollment_dir"]  for b in batch],
        "crop_starts":     [b["crop_start"]      for b in batch],
    }

def build_train_loader(
    train_files: List[Sample],
    phase: PhaseConfig,
    epoch_idx: int,
    cfg: FinetuneConfig,
    augmenter: Optional[AudioAugmenter],
    stride_sec: float,
) -> DataLoader:
    mix    = phase.data_mix.get(epoch_idx)
    active = [s for s in train_files if mix.get(s.source, 0.0) > 0.0]

    if not active:
        log.warning(
            f"[build_train_loader] No files match mix={mix}. "
            f"Sources in train: {set(s.source for s in train_files)}. "
            f"Falling back to ALL train files."
        )
        active = train_files

    dataset = DiarizationDataset(active, cfg, stride_sec, augmenter, is_train=True)

    if len(dataset.crops) == 0:
        raise RuntimeError(
            f"0 crops after sliding window (stride={stride_sec}s, chunk={cfg.chunk_sec}s). "
            f"Files: {len(active)}. Reduce chunk_sec or check audio durations."
        )

    drop_last = len(dataset.crops) >= cfg.batch_size

    src_counts: Dict[str, int] = defaultdict(int)
    for c in dataset.crops:
        src_counts[c.source] += 1

    weights = [
        mix.get(c.source, 1.0) / max(src_counts[c.source], 1)
        for c in dataset.crops
    ]
    sampler = WeightedRandomSampler(
        torch.tensor(weights, dtype=torch.float),
        num_samples=len(dataset.crops),
        replacement=True,
    )

    loader_kwargs = dict(
        batch_size=cfg.batch_size,
        sampler=sampler,
        num_workers=cfg.num_workers,
        pin_memory=bool(getattr(cfg, "dataloader_pin_memory", True) and torch.cuda.is_available()),
        drop_last=drop_last,
        collate_fn=collate_fn,
    )
    if cfg.num_workers > 0:
        loader_kwargs["persistent_workers"] = bool(getattr(cfg, "dataloader_persistent_workers", False))
        prefetch = int(getattr(cfg, "dataloader_prefetch_factor", 2))
        if prefetch > 0:
            loader_kwargs["prefetch_factor"] = prefetch
    return DataLoader(dataset, **loader_kwargs)

def build_val_loader(
    val_files: List[Sample],
    cfg: FinetuneConfig,
) -> DataLoader:
    dataset = DiarizationDataset(
        val_files, cfg,
        stride_sec = cfg.val_stride_sec,
        augmenter  = None,
        is_train   = False,
    )
    loader_kwargs = dict(
        batch_size=1,
        shuffle=True,
        num_workers=cfg.num_workers,
        pin_memory=bool(getattr(cfg, "dataloader_pin_memory", True) and torch.cuda.is_available()),
        collate_fn=collate_fn,
    )
    if cfg.num_workers > 0:
        loader_kwargs["persistent_workers"] = bool(getattr(cfg, "dataloader_persistent_workers", False))
        prefetch = int(getattr(cfg, "dataloader_prefetch_factor", 2))
        if prefetch > 0:
            loader_kwargs["prefetch_factor"] = prefetch
    return DataLoader(dataset, **loader_kwargs)
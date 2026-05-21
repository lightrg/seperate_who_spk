from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List

import numpy as np
import soundfile as sf
import torch
import torchaudio
from torch.utils.data import Dataset
from transformers import WhisperProcessor

from config import DEFAULT_SAMPLE_RATE

def clean_text(text: str) -> str:
    return " ".join(str(text).strip().split())

def load_jsonl(path: str) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    with Path(path).open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows

def load_audio_segment(audio_path: str, start: float, end: float, target_sr: int = DEFAULT_SAMPLE_RATE) -> np.ndarray:

    if end <= start:
        raise ValueError(f"Invalid segment in {audio_path}: start={start:.3f} >= end={end:.3f}")
    info = sf.info(audio_path)
    start_frame = int(start * info.samplerate)
    end_frame = int(end * info.samplerate)
    audio, sr = sf.read(audio_path, start=start_frame, stop=end_frame)
    if audio.ndim == 2:
        audio = audio.mean(axis=1)
    if sr != target_sr:
        wav = torch.from_numpy(audio).float().unsqueeze(0)
        wav = torchaudio.functional.resample(wav, sr, target_sr)
        audio = wav.squeeze(0).numpy()
    return audio.astype(np.float32)

class WhisperManifestDataset(Dataset):
    def __init__(self, manifest_path: str, processor: WhisperProcessor):
        self.rows = load_jsonl(manifest_path)
        self.processor = processor

    def __len__(self) -> int:
        return len(self.rows)

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        row = self.rows[idx]
        audio = load_audio_segment(row["audio_path"], float(row["start"]), float(row["end"]), DEFAULT_SAMPLE_RATE)
        inputs = self.processor(audio=audio, sampling_rate=DEFAULT_SAMPLE_RATE, return_tensors="pt")
        label_ids = self.processor.tokenizer(clean_text(row["text"]), truncation=True, max_length=448, return_tensors="pt").input_ids[0]
        return {
            "input_features": inputs.input_features[0],
            "labels": label_ids,
            "text": row["text"],
            "sample_id": row["sample_id"],
        }

@dataclass
class WhisperDataCollator:
    processor: WhisperProcessor

    def __call__(self, features: List[Dict[str, Any]]) -> Dict[str, torch.Tensor]:
        input_features = [{"input_features": x["input_features"]} for x in features]
        batch = self.processor.feature_extractor.pad(input_features, return_tensors="pt")

        if "attention_mask" not in batch:
            batch["attention_mask"] = torch.ones(
                batch["input_features"].shape[:2], dtype=torch.long
            )

        label_features = [{"input_ids": x["labels"]} for x in features]
        labels_batch = self.processor.tokenizer.pad(label_features, return_tensors="pt")
        labels = labels_batch["input_ids"].masked_fill(labels_batch.attention_mask.ne(1), -100)
        batch["labels"] = labels
        return batch
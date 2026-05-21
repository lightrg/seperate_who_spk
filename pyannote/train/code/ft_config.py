from pathlib import Path
from dataclasses import dataclass, field
from typing import List, Dict, Optional

_HERE = Path(__file__).resolve().parent
_REPO_ROOT = _HERE.parents[1]
_DATA = _REPO_ROOT / "data"

SELF_LABELED_DIR  = str(_DATA / "data_labelled")
SELF_LABELED_META = str(_DATA / "data_labelled" / "metadata.txt")
DATA_MIX_DIR      = str(_DATA / "data" / "data_mix")
DATA_MIX_META     = str(_DATA / "data" / "data_mix" / "metadata.txt")

VAL_DATA_DIR      = str(_DATA / "val_data")
TEST_DATA_DIR     = str(_DATA / "test_data")
VAL_LABELED_DIR   = str(_DATA / "val_labeled")
TEST_LABELED_DIR  = str(_DATA / "test_labeled")

TEST_MODE = False

SOURCE_SELF   = "self_labeled"
SOURCE_TIER_A = "tier_A_0pct"
SOURCE_TIER_B = "tier_B_5_10pct"
SOURCE_TIER_C = "tier_C_15pct"
SOURCE_TIER_D = "tier_D_20_30pct"

ALL_SOURCES = [SOURCE_SELF, SOURCE_TIER_A, SOURCE_TIER_B, SOURCE_TIER_C, SOURCE_TIER_D]

TIER_OVERLAP_MAP: Dict[str, tuple] = {
    SOURCE_TIER_A: (0, 0),
    SOURCE_TIER_B: (5, 10),
    SOURCE_TIER_C: (15, 15),
    SOURCE_TIER_D: (20, 30),
}

@dataclass
class PhaseDataMix:
    epoch_mixes: List[Dict[str, float]]

    @staticmethod
    def fixed(mix: Dict[str, float], n_epochs: int) -> "PhaseDataMix":
        return PhaseDataMix([dict(mix) for _ in range(n_epochs)])

    @staticmethod
    def curriculum(mixes: List[Dict[str, float]]) -> "PhaseDataMix":
        return PhaseDataMix(mixes)

    def get(self, epoch_idx: int) -> Dict[str, float]:
        return self.epoch_mixes[min(epoch_idx, len(self.epoch_mixes) - 1)]

@dataclass
class PhaseConfig:
    name: str
    epochs: int
    lr: float
    warmup_steps: int
    freeze_strategy: str
    unfreeze_top_n: int
    data_mix: PhaseDataMix
    use_augmentation: bool
    grad_clip: float = 1.0
    weight_decay: float = 0.01
    overlap_loss_weight: float = 1.5
    triple_loss_weight: float = 0.0
    silence_loss_weight: float = 1.0
    boundary_ignore_sec: float = 0.5

def build_phase_configs_test() -> List[PhaseConfig]:
    return [
        PhaseConfig(
            name="phase1_top4_smoke",
            epochs=2, lr=5e-5, warmup_steps=20,
            freeze_strategy="top_N", unfreeze_top_n=4,
            data_mix=PhaseDataMix.fixed({SOURCE_SELF: 0.90, SOURCE_TIER_B: 0.10}, n_epochs=2),
            use_augmentation=False, grad_clip=0.5, weight_decay=0.01,
            overlap_loss_weight=1.4, triple_loss_weight=0.0, silence_loss_weight=1.0,
            boundary_ignore_sec=0.30,
        ),
        PhaseConfig(
            name="phase2_top6_smoke",
            epochs=1, lr=2e-5, warmup_steps=10,
            freeze_strategy="top_N", unfreeze_top_n=6,
            data_mix=PhaseDataMix.fixed(
                {SOURCE_SELF: 0.75, SOURCE_TIER_B: 0.20, SOURCE_TIER_C: 0.05}, n_epochs=1
            ),
            use_augmentation=False, grad_clip=0.5, weight_decay=0.01,
            overlap_loss_weight=1.7, triple_loss_weight=0.0, silence_loss_weight=1.0,
            boundary_ignore_sec=0.25,
        ),
        PhaseConfig(
            name="phase3_top6_smoke",
            epochs=1, lr=1e-5, warmup_steps=5,
            freeze_strategy="top_N", unfreeze_top_n=6,
            data_mix=PhaseDataMix.fixed(
                {SOURCE_SELF: 0.65, SOURCE_TIER_B: 0.25, SOURCE_TIER_C: 0.10}, n_epochs=1
            ),
            use_augmentation=False, grad_clip=0.5, weight_decay=0.01,
            overlap_loss_weight=1.7, triple_loss_weight=0.0, silence_loss_weight=1.0,
            boundary_ignore_sec=0.20,
        ),
    ]

def build_phase_configs_full() -> List[PhaseConfig]:
    return [
        PhaseConfig(
            name="phase1_top4_anchor",
            epochs=4, lr=5e-5, warmup_steps=100,
            freeze_strategy="top_N", unfreeze_top_n=4,
            data_mix=PhaseDataMix.fixed(
                {SOURCE_SELF: 0.80, SOURCE_TIER_B: 0.15, SOURCE_TIER_C: 0.05}, n_epochs=4
            ),
            use_augmentation=False, grad_clip=0.5, weight_decay=0.01,
            overlap_loss_weight=1.4, triple_loss_weight=0.0, silence_loss_weight=1.0,
            boundary_ignore_sec=0.30,
        ),
        PhaseConfig(
            name="phase2_top6_overlap",
            epochs=6, lr=2e-5, warmup_steps=150,
            freeze_strategy="top_N", unfreeze_top_n=6,
            data_mix=PhaseDataMix.fixed(
                {SOURCE_SELF: 0.60, SOURCE_TIER_B: 0.30, SOURCE_TIER_C: 0.10}, n_epochs=6
            ),
            use_augmentation=True, grad_clip=0.5, weight_decay=0.01,
            overlap_loss_weight=1.7, triple_loss_weight=0.0, silence_loss_weight=1.0,
            boundary_ignore_sec=0.25,
        ),
        PhaseConfig(
            name="phase3_top6_consolidate",
            epochs=4, lr=1e-5, warmup_steps=60,
            freeze_strategy="top_N", unfreeze_top_n=6,
            data_mix=PhaseDataMix.fixed(
                {SOURCE_SELF: 0.55, SOURCE_TIER_B: 0.25, SOURCE_TIER_C: 0.20}, n_epochs=4
            ),
            use_augmentation=True, grad_clip=0.5, weight_decay=0.01,
            overlap_loss_weight=1.7, triple_loss_weight=0.0, silence_loss_weight=1.0,
            boundary_ignore_sec=0.20,
        ),
    ]

def build_phase_configs() -> List[PhaseConfig]:
    return build_phase_configs_test() if TEST_MODE else build_phase_configs_full()

@dataclass
class FinetuneConfig:
    pretrained_model: str = "pyannote/speaker-diarization-community-1"

    output_dir: str = str(
        _REPO_ROOT
        / "checkpoints"
        / "training"
        / ("smoke_test_pyannote_fixed" if TEST_MODE else "diarizen_vi_pyannote_community1_fixed")
    )

    sample_rate: int = 16000
    chunk_sec: float = 10.0
    val_stride_sec: float = 8.0

    max_speakers: int = 4

    model_max_speakers_per_chunk: int = 3
    model_max_speakers_per_frame: int = 2

    eval_max_total_speakers: int = 4

    batch_size: int = 8 if TEST_MODE else 128
    grad_accum: int = 1 if TEST_MODE else 1
    num_workers: int = 0 if TEST_MODE else 4
    seed: int = 42
    fp16: bool = True
    amp_dtype: str = "float16" if TEST_MODE else "bfloat16"
    require_cuda: bool = True

    validation_decode_mode: str = "argmax_powerset"

    strict_task_constraints: bool = True
    constraint_violation_policy: str = "warn"
    skip_invalid_frames_in_loss: bool = True

    val_every_n_epochs: int = 1
    save_every_n_epochs: int = 1
    patience: int = 5
    max_val_files: int = 4 if TEST_MODE else 20

    train_stride_sec: float = 8.0
    train_stride_sec_p3: float = 8.0
    max_val_crops: int = 80 if TEST_MODE else 500

    heavy_validate_enabled: bool = True
    heavy_validate_every: int = 3
    heavy_max_files_per_split: int = 0
    heavy_chunk_hop_sec: float = 8.0

    val_ratio: float = 0.4 if TEST_MODE else 0.10
    use_fixed_val_dir: bool = True
    use_fixed_test_dir: bool = True

    enable_tf32: bool = True
    cudnn_benchmark: bool = True
    matmul_precision: str = "high"
    compile_model: bool = False
    compile_mode: str = "max-autotune"

    dataloader_persistent_workers: bool = False if TEST_MODE else True
    dataloader_prefetch_factor: int = 2 if TEST_MODE else 1
    dataloader_pin_memory: bool = True

    waveform_cache_size: int = 2 if TEST_MODE else 8
    rttm_cache_size: int = 64 if TEST_MODE else 512
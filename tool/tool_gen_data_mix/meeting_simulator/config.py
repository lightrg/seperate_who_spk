from dataclasses import dataclass, field
from typing import List, Tuple, Dict
from pathlib import Path

@dataclass
class AudioConfig:

    sample_rate: int = 16000
    bit_depth: int = 16
    channels: int = 1

@dataclass
class SpeakerConfig:

    vivos_speakers: List[str] = field(
        default_factory=lambda: [
            "VIVOSSPK04",
            "VIVOSSPK07",
            "VIVOSSPK08",
            "VIVOSSPK44",
            "VIVOSSPK45",
            "VIVOSSPK46",
        ]
    )

    lsvsc_speakers: List[str] = field(
        default_factory=lambda: ["0F0NM", "0F3NM", "2F0NM", "0M0NM", "0M3NM", "2M0NM"]
    )

    cv_speakers: List[str] = field(
        default_factory=lambda: ["23926", "25092", "25226", "25257", "25258", "25275"]
    )

    @property
    def male_speakers(self) -> List[str]:

        return ["VIVOSSPK04", "VIVOSSPK07", "VIVOSSPK08"]

    @property
    def female_speakers(self) -> List[str]:

        return ["VIVOSSPK44", "VIVOSSPK45", "VIVOSSPK46"]

    @property
    def all_speakers(self) -> List[str]:

        return self.vivos_speakers + self.lsvsc_speakers + self.cv_speakers

    @property
    def datasets(self) -> Dict[str, List[str]]:

        return {
            "VIVOS": self.vivos_speakers,
            "LSVSC": self.lsvsc_speakers,
            "Common Voice": self.cv_speakers,
        }

@dataclass
class MeetingConfig:

    min_duration_minutes: float = 30.0
    max_duration_minutes: float = 35.0

    min_speakers: int = 2
    max_speakers: int = 4

    silence_ratio: Tuple[float, float] = (0.20, 0.30)
    overlap_ratio: Tuple[float, float] = (0.10, 0.20)

@dataclass
class OverlapConfig:

    turn_taking_ratio: float = 0.40
    turn_taking_min_sec: float = 0.5
    turn_taking_max_sec: float = 2.5

    interruption_ratio: float = 0.60
    interruption_min_sec: float = 2.0
    interruption_max_sec: float = 6.0

@dataclass
class SegmentConfig:

    merge_ratio: float = 0.70
    merge_min_segments: int = 2
    merge_max_segments: int = 3

    max_merged_duration_sec: float = 15.0

@dataclass
class SilenceConfig:

    short_pause_min_sec: float = 0.3
    short_pause_max_sec: float = 1.0
    short_pause_ratio: float = 0.60

    long_pause_min_sec: float = 1.5
    long_pause_max_sec: float = 5.0
    long_pause_ratio: float = 0.40

@dataclass
class NoiseConfig:

    noise_types: List[str] = field(
        default_factory=lambda: ["white", "pink", "brown", "office"]
    )

    snr_min_db: float = 20.0
    snr_max_db: float = 35.0

    noise_probability: float = 0.50

@dataclass
class ReverbConfig:

    room_width_range: Tuple[float, float] = (4.0, 10.0)
    room_length_range: Tuple[float, float] = (5.0, 12.0)
    room_height_range: Tuple[float, float] = (2.5, 4.0)

    rt60_range: Tuple[float, float] = (0.15, 0.4)

    reverb_probability: float = 0.40

@dataclass
class TimelineConfig:

    single_speaker_ratio: float = 0.60
    silence_ratio: float = 0.35
    overlap_ratio: float = 0.05

    silence_duration_range: Tuple[float, float] = (0.3, 2.0)
    overlap_duration_range: Tuple[float, float] = (
        1.0,
        4.0,
    )

def _get_base_path() -> Path:

    import platform

    if platform.system() == "Windows":
        return Path("D:/cachyos/du_an1/data")
    else:

        linux_paths = [
            Path("/mnt/sda1/du_an1/data"),
            Path("/run/media/phuong/HDD/cachyos/du_an1/data"),
            Path("/mnt/Data/cachyos/du_an1/data"),
            Path("/mnt/d/cachyos/du_an1/data"),
            Path.home() / "cachyos/du_an1/data",
        ]
        for p in linux_paths:
            if p.exists():
                return p

        return Path("/mnt/sda1/du_an1/data")

@dataclass
class DatasetConfig:

    total_hours: float = 20.0

    num_sets: int = 3

    vivos_path: Path = field(default_factory=lambda: _get_base_path() / "vivos")
    lsvsc_path: Path = field(default_factory=lambda: _get_base_path() / "LSVSC")
    cv_path: Path = field(default_factory=lambda: _get_base_path() / "cv_corpus")
    output_path: Path = field(default_factory=lambda: _get_base_path() / "output")

    random_seed: int = 42

@dataclass
class Config:

    audio: AudioConfig = field(default_factory=AudioConfig)
    speaker: SpeakerConfig = field(default_factory=SpeakerConfig)
    meeting: MeetingConfig = field(default_factory=MeetingConfig)
    overlap: OverlapConfig = field(default_factory=OverlapConfig)
    segment: SegmentConfig = field(default_factory=SegmentConfig)
    silence: SilenceConfig = field(default_factory=SilenceConfig)
    noise: NoiseConfig = field(default_factory=NoiseConfig)
    reverb: ReverbConfig = field(default_factory=ReverbConfig)
    timeline: TimelineConfig = field(default_factory=TimelineConfig)
    dataset: DatasetConfig = field(default_factory=DatasetConfig)

    def __post_init__(self):

        min_silence = self.meeting.silence_ratio[0]
        max_overlap = self.meeting.overlap_ratio[1]
        assert (
            min_silence + max_overlap <= 1.0
        ), "silence + overlap không được vượt quá 100%"

        total_speakers = len(self.speaker.all_speakers)
        assert (
            total_speakers >= self.meeting.max_speakers
        ), f"Cần ít nhất {self.meeting.max_speakers} speakers"

DEFAULT_CONFIG = Config()

def get_config(**overrides) -> Config:

    config = Config()

    for key, value in overrides.items():
        if hasattr(config, key):
            setattr(config, key, value)

    return config

if __name__ == "__main__":

    cfg = DEFAULT_CONFIG
    print(f"Sample rate: {cfg.audio.sample_rate}")
    print(f"All speakers: {cfg.speaker.all_speakers}")
    print(
        f"Meeting duration: {cfg.meeting.min_duration_minutes}-{cfg.meeting.max_duration_minutes} min"
    )
    print(f"Silence ratio: {cfg.meeting.silence_ratio}")
    print(f"Overlap ratio: {cfg.meeting.overlap_ratio}")
    print(f"Total dataset: {cfg.dataset.total_hours} hours")
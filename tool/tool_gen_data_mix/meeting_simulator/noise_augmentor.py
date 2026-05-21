import random
import numpy as np
from pathlib import Path
from typing import List, Dict, Tuple, Optional
from dataclasses import dataclass
import logging

from config import Config, DEFAULT_CONFIG
from utils import load_audio, logger

class SNRCalculator:

    @staticmethod
    def calculate_rms(audio: np.ndarray) -> float:

        return np.sqrt(np.mean(audio ** 2))

    @staticmethod
    def calculate_snr(signal: np.ndarray, noise: np.ndarray) -> float:
        signal_rms = SNRCalculator.calculate_rms(signal)
        noise_rms = SNRCalculator.calculate_rms(noise)

        if noise_rms < 1e-10:
            return float('inf')

        return 20 * np.log10(signal_rms / noise_rms)

    @staticmethod
    def adjust_noise_for_snr(
        signal: np.ndarray,
        noise: np.ndarray,
        target_snr_db: float
    ) -> np.ndarray:

        signal_rms = SNRCalculator.calculate_rms(signal)
        noise_rms = SNRCalculator.calculate_rms(noise)

        if noise_rms < 1e-10:
            return noise

        target_noise_rms = signal_rms / (10 ** (target_snr_db / 20))

        scale = target_noise_rms / noise_rms

        return noise * scale

    @staticmethod
    def mix_with_snr(
        signal: np.ndarray,
        noise: np.ndarray,
        snr_db: float,
        noise_loader=None
    ) -> Tuple[np.ndarray, float]:

        if len(signal) == 0:
            return signal, 0.0
        if len(noise) == 0:
            return signal, float('inf')

        if len(noise) < len(signal):

            if noise_loader is not None:

                remaining_samples = len(signal) - len(noise)
                remaining_duration = remaining_samples / noise_loader.sample_rate

                noise_segments = [noise]

                while len(np.concatenate(noise_segments)) < len(signal):

                    new_segment = noise_loader.get_random_noise(remaining_duration)
                    if new_segment is None:

                        repeats = int(np.ceil(len(signal) / len(noise)))
                        noise = np.tile(noise, repeats)
                        break
                    noise_segments.append(new_segment.audio_data)
                else:

                    noise = np.concatenate(noise_segments)
            else:

                repeats = int(np.ceil(len(signal) / len(noise)))
                noise = np.tile(noise, repeats)

        noise = noise[:len(signal)]

        adjusted_noise = SNRCalculator.adjust_noise_for_snr(
            signal, noise, snr_db
        )

        mixed = signal + adjusted_noise

        max_val = np.max(np.abs(mixed))
        if max_val > 1.0:
            mixed = mixed / max_val * 0.95

        actual_snr = SNRCalculator.calculate_snr(signal, adjusted_noise)

        return mixed, actual_snr

@dataclass
class NoiseSegment:

    audio_data: np.ndarray
    sample_rate: int
    source_path: Optional[Path] = None

    @property
    def duration(self) -> float:
        return len(self.audio_data) / self.sample_rate

class RealNoiseLoader:

    def __init__(
        self,
        noise_dir: Path,
        sample_rate: int = 16000
    ):
        self.noise_dir = Path(noise_dir)
        self.sample_rate = sample_rate

        self._noise_files: List[Path] = []
        self._scanned = False

    def scan(self) -> None:

        if self._scanned:
            return

        if not self.noise_dir.exists():
            logger.warning(f"Noise directory not found: {self.noise_dir}")
            return

        self._noise_files = list(self.noise_dir.rglob("*.wav"))
        logger.info(f"Found {len(self._noise_files)} noise files in {self.noise_dir}")

        self._scanned = True

    @property
    def num_files(self) -> int:

        self.scan()
        return len(self._noise_files)

    def get_random_noise(self, duration: float = None) -> Optional[NoiseSegment]:

        self.scan()

        if not self._noise_files:
            return None

        noise_file = random.choice(self._noise_files)
        audio, _ = load_audio(noise_file, self.sample_rate)

        if duration and len(audio) / self.sample_rate > duration:

            max_start = len(audio) - int(duration * self.sample_rate)
            start = random.randint(0, max(0, max_start))
            audio = audio[start:start + int(duration * self.sample_rate)]

        return NoiseSegment(
            audio_data=audio,
            sample_rate=self.sample_rate,
            source_path=noise_file
        )

    def get_noise_by_index(self, index: int, duration: float = None) -> Optional[NoiseSegment]:

        self.scan()

        if not self._noise_files or index >= len(self._noise_files):
            return None

        noise_file = self._noise_files[index]
        audio, _ = load_audio(noise_file, self.sample_rate)

        if duration and len(audio) / self.sample_rate > duration:
            max_start = len(audio) - int(duration * self.sample_rate)
            start = random.randint(0, max(0, max_start))
            audio = audio[start:start + int(duration * self.sample_rate)]

        return NoiseSegment(
            audio_data=audio,
            sample_rate=self.sample_rate,
            source_path=noise_file
        )

class NoiseAugmentor:

    DEFAULT_NOISE_DIR = Path("data/noise/speech-noise-dataset/noise_only")

    def __init__(self, config: Config = DEFAULT_CONFIG, noise_dir: Path = None):
        self.config = config
        self.sample_rate = config.audio.sample_rate

        if noise_dir is None:

            noise_dir = getattr(config.dataset, 'noise_dir', None)
            if noise_dir:
                noise_dir = Path(noise_dir)
            else:

                project_root = Path(__file__).parent.parent
                noise_dir = project_root / self.DEFAULT_NOISE_DIR

        self.noise_loader = RealNoiseLoader(noise_dir, self.sample_rate)

    def augment(
        self,
        audio: np.ndarray,
        snr_db: float = None,
        probability: float = 1.0
    ) -> Tuple[np.ndarray, Dict]:

        metadata = {
            "noise_applied": False,
            "noise_file": None,
            "target_snr": None,
            "actual_snr": None
        }

        if random.random() > probability:
            return audio, metadata

        if snr_db is None:
            snr_range = getattr(self.config.noise, 'snr_range', (15.0, 30.0))
            snr_db = random.uniform(*snr_range)

        duration = len(audio) / self.sample_rate
        noise_segment = self.noise_loader.get_random_noise(duration)

        if noise_segment is None:
            logger.warning("No noise files available, returning original audio")
            return audio, metadata

        augmented, actual_snr = SNRCalculator.mix_with_snr(
            audio, noise_segment.audio_data, snr_db, self.noise_loader
        )

        metadata = {
            "noise_applied": True,
            "noise_file": str(noise_segment.source_path.name) if noise_segment.source_path else None,
            "target_snr": snr_db,
            "actual_snr": actual_snr
        }

        return augmented, metadata

    def augment_batch(
        self,
        audios: List[np.ndarray],
        snr_range: Tuple[float, float] = None
    ) -> List[Tuple[np.ndarray, Dict]]:

        results = []

        for audio in audios:
            snr = random.uniform(*snr_range) if snr_range else None
            augmented, metadata = self.augment(audio, snr_db=snr)
            results.append((augmented, metadata))

        return results

    def apply_variable_noise(
        self,
        audio: np.ndarray,
        snr_schedule: List[Tuple[float, float, float]]
    ) -> np.ndarray:

        result = audio.copy()

        for start_time, end_time, snr_db in snr_schedule:
            start_sample = int(start_time * self.sample_rate)
            end_sample = int(end_time * self.sample_rate)

            start_sample = max(0, start_sample)
            end_sample = min(len(audio), end_sample)

            segment = audio[start_sample:end_sample]
            duration = len(segment) / self.sample_rate

            noise_segment = self.noise_loader.get_random_noise(duration)
            if noise_segment is not None:
                augmented, _ = SNRCalculator.mix_with_snr(
                    segment, noise_segment.audio_data, snr_db, self.noise_loader
                )
                result[start_sample:end_sample] = augmented

        return result

def add_noise_to_meeting(
    audio: np.ndarray,
    config: Config = DEFAULT_CONFIG
) -> Tuple[np.ndarray, Dict]:

    augmentor = NoiseAugmentor(config)

    return augmentor.augment(
        audio,
        probability=config.noise.noise_probability
    )

if __name__ == "__main__":
    print("Testing noise_augmentor.py...")
    print("=" * 60)

    config = Config()

    print("\n1. Testing SNR Calculator...")
    signal = np.random.randn(16000) * 0.5
    noise = np.random.randn(16000) * 0.1

    snr = SNRCalculator.calculate_snr(signal, noise)
    print(f"   Original SNR: {snr:.1f} dB")

    target_snr = 10.0
    adjusted_noise = SNRCalculator.adjust_noise_for_snr(signal, noise, target_snr)
    actual_snr = SNRCalculator.calculate_snr(signal, adjusted_noise)
    print(f"   Target SNR: {target_snr:.1f} dB")
    print(f"   Actual SNR: {actual_snr:.1f} dB")

    mixed, mix_snr = SNRCalculator.mix_with_snr(signal, noise, 15.0)
    print(f"   Mixed SNR (target 15dB): {mix_snr:.1f} dB")

    print("\n2. Testing RealNoiseLoader...")
    project_root = Path(__file__).parent.parent
    noise_dir = project_root / "data/noise/speech-noise-dataset/noise_only"

    loader = RealNoiseLoader(noise_dir, 16000)
    print(f"   Found {loader.num_files} noise files")

    if loader.num_files > 0:
        segment = loader.get_random_noise(3.0)
        if segment:
            print(f"   Random noise: {segment.source_path.name}, "
                  f"duration={segment.duration:.2f}s")

    print("\n3. Testing NoiseAugmentor...")
    augmentor = NoiseAugmentor(config)

    test_audio = np.random.randn(48000) * 0.5

    for target_snr in [5, 10, 15, 20]:
        augmented, metadata = augmentor.augment(
            test_audio,
            snr_db=target_snr,
            probability=1.0
        )
        if metadata['noise_applied']:
            print(f"   SNR={target_snr}dB: actual={metadata['actual_snr']:.1f}dB, "
                  f"noise_file={metadata['noise_file']}")
        else:
            print(f"   SNR={target_snr}dB: No noise applied (no files)")

    print("\n4. Testing Random Augmentation...")
    for i in range(5):
        augmented, metadata = augmentor.augment(test_audio, probability=0.8)
        if metadata['noise_applied']:
            print(f"   Trial {i+1}: noise={metadata['noise_file']}, "
                  f"SNR={metadata['actual_snr']:.1f}dB")
        else:
            print(f"   Trial {i+1}: No noise applied")

    print("\n5. Testing Variable Noise...")
    long_audio = np.random.randn(160000) * 0.5

    snr_schedule = [
        (0.0, 3.0, 20.0),
        (3.0, 7.0, 10.0),
        (7.0, 10.0, 5.0),
    ]

    variable_noisy = augmentor.apply_variable_noise(long_audio, snr_schedule)
    print(f"   Applied variable noise to {len(long_audio)/16000:.1f}s audio")
    print(f"   Schedule: {len(snr_schedule)} segments")

    print("\n" + "=" * 60)
    print("✅ noise_augmentor.py tests passed!")
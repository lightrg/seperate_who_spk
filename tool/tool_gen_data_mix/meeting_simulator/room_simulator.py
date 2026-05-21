import random
import numpy as np
from typing import Tuple, Optional, Dict, List
from dataclasses import dataclass
import logging

try:
    import pyroomacoustics as pra

    PRA_AVAILABLE = False
    logging.info("pyroomacoustics disabled - using fast fallback reverb")
except ImportError:
    PRA_AVAILABLE = False
    logging.warning("pyroomacoustics not installed. Using fallback reverb.")

from scipy import signal
from config import Config, DEFAULT_CONFIG
from utils import logger

@dataclass
class RoomConfig:

    width: float
    length: float
    height: float
    rt60: float

    @property
    def dimensions(self) -> List[float]:
        return [self.width, self.length, self.height]

    @property
    def volume(self) -> float:
        return self.width * self.length * self.height

    def __str__(self) -> str:
        return (
            f"Room({self.width:.1f}m x {self.length:.1f}m x {self.height:.1f}m, "
            f"RT60={self.rt60:.2f}s, Vol={self.volume:.1f}m³)"
        )

@dataclass
class MicrophoneConfig:

    position: np.ndarray

    @classmethod
    def center_of_room(cls, room: RoomConfig, height: float = 1.0) -> 'MicrophoneConfig':

        return cls(
            position=np.array([
                room.width / 2,
                room.length / 2,
                height
            ])
        )

@dataclass
class SpeakerPosition:

    speaker_id: str
    position: np.ndarray
    distance_to_mic: float

    def __str__(self) -> str:
        return f"{self.speaker_id} at ({self.position[0]:.1f}, {self.position[1]:.1f}, {self.position[2]:.1f}), dist={self.distance_to_mic:.1f}m"

class RoomSimulator:

    def __init__(self, config: Config = DEFAULT_CONFIG):
        self.config = config
        self.sample_rate = config.audio.sample_rate

        self.room_config: Optional[RoomConfig] = None
        self.mic_config: Optional[MicrophoneConfig] = None
        self.speaker_positions: Dict[str, SpeakerPosition] = {}

        self._rir_cache: Dict[str, np.ndarray] = {}

    def generate_random_room(self) -> RoomConfig:
        cfg = self.config.reverb

        width = random.uniform(*cfg.room_width_range)
        length = random.uniform(*cfg.room_length_range)
        height = random.uniform(*cfg.room_height_range)
        rt60 = random.uniform(*cfg.rt60_range)

        self.room_config = RoomConfig(
            width=width,
            length=length,
            height=height,
            rt60=rt60
        )

        self.mic_config = MicrophoneConfig.center_of_room(self.room_config)

        self._rir_cache.clear()
        self.speaker_positions.clear()

        logger.info(f"Generated room: {self.room_config}")

        return self.room_config

    def setup_speaker_positions(
        self,
        speaker_ids: List[str],
        arrangement: str = "circular"
    ) -> Dict[str, SpeakerPosition]:

        if self.room_config is None:
            self.generate_random_room()

        room = self.room_config
        mic_pos = self.mic_config.position

        num_speakers = len(speaker_ids)

        if arrangement == "circular":
            positions = self._circular_arrangement(num_speakers, room, mic_pos)
        else:
            positions = self._random_arrangement(num_speakers, room, mic_pos)

        self.speaker_positions = {}
        for i, speaker_id in enumerate(speaker_ids):
            pos = positions[i]
            distance = np.linalg.norm(pos - mic_pos)

            self.speaker_positions[speaker_id] = SpeakerPosition(
                speaker_id=speaker_id,
                position=pos,
                distance_to_mic=distance
            )

            logger.debug(f"  {self.speaker_positions[speaker_id]}")

        return self.speaker_positions

    def _circular_arrangement(
        self,
        num_speakers: int,
        room: RoomConfig,
        mic_pos: np.ndarray
    ) -> List[np.ndarray]:

        positions = []

        base_radius = random.uniform(1.5, 2.5)

        for i in range(num_speakers):

            angle = (2 * np.pi * i / num_speakers) + random.uniform(-0.2, 0.2)

            radius = base_radius + random.uniform(-0.3, 0.5)

            x = mic_pos[0] + radius * np.cos(angle)
            y = mic_pos[1] + radius * np.sin(angle)
            z = random.uniform(1.1, 1.4)

            x = np.clip(x, 0.5, room.width - 0.5)
            y = np.clip(y, 0.5, room.length - 0.5)
            z = np.clip(z, 0.5, room.height - 0.5)

            positions.append(np.array([x, y, z]))

        return positions

    def _random_arrangement(
        self,
        num_speakers: int,
        room: RoomConfig,
        mic_pos: np.ndarray
    ) -> List[np.ndarray]:

        positions = []

        for _ in range(num_speakers):

            x = random.uniform(0.5, room.width - 0.5)
            y = random.uniform(0.5, room.length - 0.5)
            z = random.uniform(1.1, 1.4)

            positions.append(np.array([x, y, z]))

        return positions

    def compute_rir(
        self,
        speaker_id: str
    ) -> np.ndarray:

        if speaker_id in self._rir_cache:
            return self._rir_cache[speaker_id]

        if speaker_id not in self.speaker_positions:
            raise ValueError(f"Speaker {speaker_id} not set up")

        speaker_pos = self.speaker_positions[speaker_id]

        if PRA_AVAILABLE:
            rir = self._compute_rir_pra(speaker_pos)
        else:
            rir = self._compute_rir_fallback(speaker_pos)

        self._rir_cache[speaker_id] = rir

        return rir

    def _compute_rir_pra(
        self,
        speaker_pos: SpeakerPosition
    ) -> np.ndarray:

        room = self.room_config

        e_absorption, max_order = pra.inverse_sabine(room.rt60, room.dimensions)

        max_order = min(max_order, 3)

        pra_room = pra.ShoeBox(
            room.dimensions,
            fs=self.sample_rate,
            materials=pra.Material(e_absorption),
            max_order=max_order
        )

        pra_room.add_microphone(self.mic_config.position)

        pra_room.add_source(speaker_pos.position)

        pra_room.compute_rir()

        rir = pra_room.rir[0][0]

        rir = rir / np.max(np.abs(rir))

        return rir

    def _compute_rir_fallback(
        self,
        speaker_pos: SpeakerPosition
    ) -> np.ndarray:

        room = self.room_config
        rt60 = room.rt60

        rir_length = int(rt60 * self.sample_rate * 1.5)

        speed_of_sound = 343.0
        direct_delay = speaker_pos.distance_to_mic / speed_of_sound
        direct_delay_samples = int(direct_delay * self.sample_rate)

        rir = np.zeros(rir_length)

        if direct_delay_samples < rir_length:
            rir[direct_delay_samples] = 1.0

        num_reflections = 20
        for i in range(num_reflections):
            delay = direct_delay_samples + int(random.uniform(0.002, 0.05) * self.sample_rate)
            if delay < rir_length:
                amplitude = random.uniform(0.1, 0.4) * (0.8 ** i)
                rir[delay] += amplitude * random.choice([-1, 1])

        decay_rate = 6.91 / rt60
        late_start = int(0.08 * self.sample_rate)

        t = np.arange(late_start, rir_length) / self.sample_rate
        decay = np.exp(-decay_rate * (t - late_start / self.sample_rate))
        noise = np.random.randn(rir_length - late_start) * 0.1

        rir[late_start:] += noise * decay

        rir = rir / np.max(np.abs(rir))

        return rir

    def apply_reverb(
        self,
        audio: np.ndarray,
        speaker_id: str,
        dry_wet_ratio: float = 0.3
    ) -> np.ndarray:

        if len(audio) == 0:
            return audio

        rir = self.compute_rir(speaker_id)

        wet = signal.fftconvolve(audio, rir, mode='full')

        wet = wet[:len(audio)]

        if np.max(np.abs(wet)) > 0:
            wet = wet / np.max(np.abs(wet)) * np.max(np.abs(audio))

        output = (1 - dry_wet_ratio) * audio + dry_wet_ratio * wet

        max_val = np.max(np.abs(output))
        if max_val > 1.0:
            output = output / max_val * 0.95

        return output

    def apply_distance_attenuation(
        self,
        audio: np.ndarray,
        speaker_id: str,
        reference_distance: float = 1.0
    ) -> np.ndarray:

        if speaker_id not in self.speaker_positions:
            return audio

        distance = self.speaker_positions[speaker_id].distance_to_mic

        attenuation = reference_distance / max(distance, reference_distance)

        return audio * attenuation

    def process_meeting_audio(
        self,
        speaker_audios: Dict[str, np.ndarray],
        apply_reverb: bool = True,
        apply_distance: bool = True
    ) -> Dict[str, np.ndarray]:

        processed = {}

        for speaker_id, audio in speaker_audios.items():
            output = audio.copy()

            if apply_distance and speaker_id in self.speaker_positions:
                output = self.apply_distance_attenuation(output, speaker_id)

            if apply_reverb and speaker_id in self.speaker_positions:
                output = self.apply_reverb(output, speaker_id)

            processed[speaker_id] = output

        return processed

    def get_room_info(self) -> Dict:

        if self.room_config is None:
            return {"configured": False}

        return {
            "configured": True,
            "dimensions": {
                "width": self.room_config.width,
                "length": self.room_config.length,
                "height": self.room_config.height,
            },
            "volume": self.room_config.volume,
            "rt60": self.room_config.rt60,
            "mic_position": self.mic_config.position.tolist(),
            "speakers": {
                spk_id: {
                    "position": pos.position.tolist(),
                    "distance_to_mic": pos.distance_to_mic
                }
                for spk_id, pos in self.speaker_positions.items()
            },
            "pra_available": PRA_AVAILABLE
        }

def create_room_for_meeting(
    speaker_ids: List[str],
    config: Config = DEFAULT_CONFIG
) -> RoomSimulator:

    simulator = RoomSimulator(config)

    if random.random() < config.reverb.reverb_probability:
        simulator.generate_random_room()
        simulator.setup_speaker_positions(speaker_ids)
        logger.info(f"Room reverb enabled: {simulator.room_config}")
    else:
        logger.info("Room reverb disabled for this meeting")

    return simulator

if __name__ == "__main__":
    print("Testing room_simulator.py...")
    print("=" * 60)
    print(f"pyroomacoustics available: {PRA_AVAILABLE}")
    print()

    config = Config()
    simulator = RoomSimulator(config)

    room = simulator.generate_random_room()
    print(f"Generated room: {room}")
    print(f"  Volume: {room.volume:.1f} m³")

    speakers = ["SPK01", "SPK02", "SPK03", "SPK04"]
    positions = simulator.setup_speaker_positions(speakers)

    print(f"\nSpeaker positions:")
    for spk_id, pos in positions.items():
        print(f"  {pos}")

    print(f"\nComputing RIRs...")
    for spk_id in speakers:
        rir = simulator.compute_rir(spk_id)
        print(f"  {spk_id}: RIR length = {len(rir)} samples ({len(rir)/16000:.3f}s)")

    print(f"\nTesting reverb application...")
    test_audio = np.random.randn(16000) * 0.5

    reverbed = simulator.apply_reverb(test_audio, "SPK01", dry_wet_ratio=0.5)
    print(f"  Input length: {len(test_audio)}")
    print(f"  Output length: {len(reverbed)}")
    print(f"  Input RMS: {np.sqrt(np.mean(test_audio**2)):.4f}")
    print(f"  Output RMS: {np.sqrt(np.mean(reverbed**2)):.4f}")

    print(f"\nTesting distance attenuation...")
    for spk_id in speakers[:2]:
        attenuated = simulator.apply_distance_attenuation(test_audio, spk_id)
        distance = positions[spk_id].distance_to_mic
        attenuation_db = 20 * np.log10(np.sqrt(np.mean(attenuated**2)) / np.sqrt(np.mean(test_audio**2)))
        print(f"  {spk_id}: distance={distance:.2f}m, attenuation={attenuation_db:.1f}dB")

    print(f"\nRoom info:")
    info = simulator.get_room_info()
    print(f"  Dimensions: {info['dimensions']}")
    print(f"  RT60: {info['rt60']:.2f}s")
    print(f"  Num speakers: {len(info['speakers'])}")

    print("\n" + "=" * 60)
    print("✅ room_simulator.py tests passed!")
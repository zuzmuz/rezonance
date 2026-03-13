import logging
import torch
from torch.types import Number, Tensor
from torch.utils.data import Dataset
from src.utils import (
    freq_from_pitch,
    get_pitch_of_rank,
    get_rank_of_pitch,
    pitch_from_freq,
    current_device,
)
from src.waveform import Timbre, WaveformSynth
from src.noise_generators import NoiseSynth

logger = logging.getLogger(__name__)


class NoisySineWaveformDataset(Dataset):
    """
    Synthetic dataset of sinewaveforms with varying pitch and phase, adding various noises.
    Parameters:
        nb_pitches (int): number of different pitches to generate, divides `min_pitch` to `max_pitch`
        np_phases (int): number of different phases to generate, divides -1 to 1
        noises (list[NoiseSynth]): list of noise synthesizers to add to the sinewaveforms
        sample_rate (float): the sample rate of the generated waveform
        buffer_size (int): the buffer size of the generated waveform
        A4 (float): the reference frequency of the A4 note
        min_pitch (float): the minimum pitch number (MIDI standard)
        max_pitch (float | None): the maximum pitch number (MIDI standard).
    """

    def __init__(
        self,
        nb_pitches: int,
        nb_phases: int,
        /,
        noises: list[NoiseSynth],
        *,
        sample_rate: Number = 16_000.0,
        buffer_size: int = 1024,
        A4: Number = 440.0,
        min_pitch: Number = 20,
        max_pitch: Number | None = None,
        seed: int | None = None,
    ):
        if seed:
            torch.manual_seed(seed)

        self.synth = WaveformSynth(
            sample_rate=sample_rate,
            buffer_size=buffer_size,
            A4=A4,
        )

        # Creating all noises
        # shape `(nb_noises, buffer_size)`
        self.noises = torch.stack(
            [synth(buffer_size) for synth in noises]
        )

        # max pitch is necessary to prevent aliasing
        # adding sines with frequencies close and higher than Shannon frequency
        # will add lower frequencies which are undesired
        max_possible_pitch = pitch_from_freq(
            0.25 * sample_rate, A4=A4
        )
        if not max_pitch:
            max_pitch = max_possible_pitch
        elif max_pitch > max_possible_pitch:
            logger.warning(
                "Provided max_pitch %.2f "
                "exceeds the maximum possible pitch %.2f"
                "for the given sample rate %.2f.",
                max_pitch,
                max_possible_pitch,
                sample_rate,
            )
            max_pitch = max_possible_pitch

        self.pitches = torch.linspace(
            min_pitch, max_pitch, nb_pitches, dtype=torch.float32
        )

        self.phases = torch.linspace(
            -1, 1 - 1 / nb_phases, nb_phases, dtype=torch.float32
        )

        # Combining pitches, phases, and noises into a 2D tensor
        # shape `(nb_pitches * nb_phases * nb_noises, 2)`
        # drop noise dimension
        self.data = torch.cartesian_prod(
            self.pitches,
            self.phases,
            torch.zeros(self.noises.size(0)),
        )[:, 0:2]

        self.data[:, 0] = freq_from_pitch(self.data[:, 0], A4=A4)

        self.data = self.synth.gen_mono(self.data)
        # shape `(nb_pitches * nb_phases * nb_noises, buffer_size)`

        repeated_noises = self.noises.repeat(
            self.pitches.size(0) * self.phases.size(0), 1
        )

        self.data += repeated_noises

        self.data /= self.data.std(dim=1, keepdim=True)

        self.data.to(current_device)

    def __len__(self):
        return self.data.size(0)

    def __getitem__(self, idx):
        pitch = self.pitches[
            idx // (self.phases.size(0) * self.noises.size(0))
        ]
        waveform = self.data[idx]

        return waveform, pitch


class LazyNoisySineWaveformDataset(Dataset):
    def __init__(
        self,
        nb_pitches: int,
        nb_phases: int,
        /,
        noises: list[NoiseSynth],
        *,
        sample_rate: Number = 16_000.0,
        buffer_size: int = 1024,
        A4: Number = 440.0,
        min_pitch: Number = 20,
        max_pitch: Number | None = None,
        seed: int | None = None,
    ):

        if seed:
            torch.manual_seed(seed)

        self.synth = WaveformSynth(
            sample_rate=sample_rate, buffer_size=buffer_size, A4=A4
        )

        max_possible_pitch = pitch_from_freq(
            0.25 * sample_rate, A4=A4
        )

        if not max_pitch:
            max_pitch = max_possible_pitch
        elif max_pitch > max_possible_pitch:
            logger.warning(
                "Provided max_pitch %.2f "
                "exceeds the maximum possible pitch %.2f"
                "for the given sample rate %.2f.",
                max_pitch,
                max_possible_pitch,
                sample_rate,
            )

        self.pitches = torch.linspace(
            min_pitch, max_pitch, nb_pitches, dtype=torch.float32
        )

        self.freqencies = freq_from_pitch(self.pitches, A4=A4)

        self.phases = torch.linspace(
            -1, 2 - 1 / nb_phases, nb_phases, dtype=torch.float32
        )

        self.noises = torch.stack(
            [synth(buffer_size) for synth in noises]
        )

        print(
            f"{self.pitches.shape=} {self.phases.shape=} {self.noises.shape=}"
        )

    def __len__(self):
        return (
            self.pitches.size(0)
            * self.phases.size(0)
            * self.noises.size(0)
        )

    def __getitem__(self, idx):
        pitch = self.pitches[
            idx // (self.phases.size(0) * self.noises.size(0))
        ]
        freq = self.freqencies[
            idx // (self.phases.size(0) * self.noises.size(0))
        ]
        phase = self.phases[
            (idx // self.noises.size(0)) % self.phases.size(0)
        ]
        noise = self.noises[idx % self.noises.size(0)]

        waveform = self.synth.gen_mono(
            torch.tensor([freq, phase]).unsqueeze(0)
        )

        # print(f"sizes {waveform[0].shape}  pi {pitch.shape}")

        return waveform[0] + noise, pitch


class RandomTimbralDataSet(Dataset):
    def __init__(
        self,
        nb_pitches: int,
        nb_harm_dist: int,
        nb_noises: int,
        *,
        sample_rate: Number = 16_000,
        buffer_size: int = 1024,
        A4: Number = 440.0,
        min_pitch: Number = 20,
        max_pitch: Number | None = None,
        seed: int | None = None,
    ):

        if seed:
            torch.manual_seed(seed)

        self.sample_rate = sample_rate
        self.A4 = A4

        self.synth = WaveformSynth(
            sample_rate=sample_rate, buffer_size=buffer_size, A4=A4
        )

        max_possible_pitch = pitch_from_freq(
            0.25 * sample_rate, A4=A4
        )

        if not max_pitch:
            max_pitch = max_possible_pitch
        elif max_pitch < max_possible_pitch:
            logger.warning(
                "Provided max_pitch %.2f "
                "exceeds the maximum possible pitch %.2f"
                "for the given sample rate %.2f.",
                max_pitch,
                max_possible_pitch,
                sample_rate,
            )
        
        pitches = torch.linspace(
            min_pitch, max_pitch, nb_pitches, dtype=torch.float32
        )

        ranks = get_rank_of_pitch(
            pitches,
            sample_rate=sample_rate,
            A4=A4,
        )

        # rank_pitches[0] = min_pitch
        # rank_pitches[-1] = max_pitch

        



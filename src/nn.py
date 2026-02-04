import time
import logging
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torch.types import Number, Tensor


from src.utils import (
    pitch_from_freq,
)
from src.waveform import WaveformSynth, NoiseSynth

logger = logging.getLogger(__name__)


class NoisySineWaveformDataset(Dataset):
    def __init__(
        self,
        nb_pitches: int,
        nb_phases: int,
        /,
        noises: list[NoiseSynth],
        *,
        sample_rate: Number,
        buffer_size: int,
        A4: Number,
        seed: int | None = None,
        min_pitch: Number = 20,
        max_pitch: Number | None = None,
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
            -1, 1, nb_phases, dtype=torch.float32
        )

        # Combining pitches, phases, and noises into a 2D tensor
        # shape `(2, nb_pitches * nb_phases * nb_noises)`
        # drop noise dimension
        self.data = torch.cartesian_prod(
            self.pitches,
            self.phases,
            torch.zeros(self.noises.size(0)),
        ).T[0:2]

        self.data = self.synth.gen_mono(self.data)

        repeated_noises = self.noises.repeat(
            self.pitches.size(0) * self.phases.size(0), 1
        )

        self.data += repeated_noises

        self.data /= self.data.std(dim=1, keepdim=True)

    def __len__(self):
        return self.data.size(0)

    def __getitem__(self, idx):
        pitch = self.pitches[
            idx // (self.phases.size(0) * self.noises.size(0))
        ]
        waveform = self.data[idx]
        return waveform, pitch


class SineWaveformDataset(Dataset):
    """
    Simple synthetic dataset of sinewaveforms with varying pitch and phase.
    Parameters:
        nb_pitches (int): number of different pitches to generate, divides `min_pitch` to `max_pitch`
        np_phases (int): number of different phases to generate, divides -1 to 1
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
        *,
        sample_rate: Number,
        buffer_size: int,
        A4: Number,
        min_pitch: Number = 20,
        max_pitch: Number | None = None,
    ):
        self.synth = WaveformSynth(
            sample_rate=sample_rate,
            buffer_size=buffer_size,
            A4=A4,
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
            max_pitch = max_possible_pitch

        # max pitch is necessary to prevent aliasing
        # adding sines with frequencies close and higher than Shannon frequency
        # will add lower frequencies which are undesired

        self.pitches = torch.linspace(
            min_pitch,
            max_pitch,
            nb_pitches,
            dtype=torch.float32,
        )
        self.phases = torch.linspace(
            -1, 1, nb_phases, dtype=torch.float32
        )

        # Combining pitches and phases into a 2D tensor
        # shape `(2, nb_pitches * nb_phases)`
        self.data = torch.cartesian_prod(self.pitches, self.phases).T

        # Generating waveforms from pitches and phases
        # shape `(nb_pitches * nb_phases, buffer_size)`
        self.data = self.synth.gen_mono(self.data)

        # Standardizing waveforms
        self.data /= self.data.std(dim=1, keepdim=True)

    def __len__(self):
        return self.data.size(0)

    def __getitem__(self, idx) -> tuple[Tensor, Tensor]:
        pitch = self.pitches[idx // self.phases.size(0)]
        waveform = self.data[idx]
        return waveform, pitch


class LinearModel1(nn.Module):
    def __init__(self, buffer_size: int):
        super(LinearModel1, self).__init__()

        self.model = nn.Sequential(
            nn.Linear(buffer_size, 512),
            nn.Tanh(),
            nn.Linear(512, 256),
            nn.Tanh(),
            nn.Linear(256, 128),
            nn.Tanh(),
            nn.Linear(128, 1),
        )

    def forward(self, X):
        return self.model(X)


class Trainer:
    def __init__(self, model):
        self.model = model
        self.criterion = nn.MSELoss()
        self.optimizer = optim.Adam(self.model.parameters(), lr=0.001)

    def train(self, nb_epoch: int, dataset: Dataset):
        data_loader = DataLoader(
            dataset,
            batch_size=32,
            shuffle=True,
            generator=torch.Generator(
                device=torch.get_default_device()
            ),
        )

        perf_counter = time.perf_counter()
        for epoch in range(nb_epoch):
            self.model.train()
            epoch_loss = 0
            for batch_X, batch_y in data_loader:
                self.optimizer.zero_grad()
                hat_y = self.model.forward(batch_X).squeeze()
                loss = self.criterion(hat_y, batch_y)
                loss.backward()
                self.optimizer.step()

                epoch_loss += loss.item()

            if (epoch + 1) % 2 == 0:
                num_batches = len(data_loader)
                print(
                    f"Epoch {epoch + 1}: "
                    f"Mean Squared Error = {epoch_loss / num_batches:.5f}, "
                    f"Time = {time.perf_counter() - perf_counter:.2f} seconds"
                )
                perf_counter = time.perf_counter()
        return []

    def save_model(self, path: str):
        torch.save(self.model.state_dict(), path)

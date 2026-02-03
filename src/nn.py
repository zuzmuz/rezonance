import time
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torch.types import Number, Tensor
import numpy as np

from src.utils import (
    pitch_from_freq,
)
from src.waveform import WaveformSynth, NoiseSynth


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

        print(f"Noises shape: {self.noises.shape}")

        # max pitch is necessary to prevent aliasing
        # adding sines with frequencies close and higher than Shannon frequency
        # will add lower frequencies which are undesired
        max_pitch = pitch_from_freq(0.25 * sample_rate, A4=A4)

        self.pitches = torch.linspace(
            20, max_pitch, nb_pitches, dtype=torch.float32
        )
        print(f"Pitches shape: {self.pitches.shape}")

        self.phases = torch.linspace(
            -1, 1, nb_phases, dtype=torch.float32
        )
        print(f"Phases shape: {self.phases.shape}")

        # Combining pitches, phases, and noises into a 2D tensor
        # shape `(2, nb_pitches * nb_phases * nb_noises)`
        # drop noise dimension
        self.data = torch.cartesian_prod(
            self.pitches,
            self.phases,
            torch.zeros(self.noises.size(0)),
        ).T[0:2]

        print(
            f"Data shape before waveform generation: {self.data.shape}"
        )

        self.data = self.synth.gen_mono(self.data)
        print(
            f"Data shape after waveform generation: {self.data.shape}"
        )

        repeated_noises = self.noises.repeat(
            self.pitches.size(0) * self.phases.size(0), 1
        )
        print(f"Noises shape after repeat: {repeated_noises.shape}")

        self.data += repeated_noises

        self.data /= self.data.std(dim=1, keepdim=True)

    def __len__(self):
        return self.data.size(0)

    def __getitem__(self, idx):
        pitch = self.pitches[
            idx // (self.phases.size(0) + self.noises.size(0))
        ]
        waveform = self.data[idx]
        return waveform, pitch


class SineWaveformDataset(Dataset):
    def __init__(
        self,
        nb_pitches: int,
        nb_phases: int,
        /,
        *,
        sample_rate: Number,
        buffer_size: int,
        A4: Number,
    ):
        self.synth = WaveformSynth(
            sample_rate=sample_rate,
            buffer_size=buffer_size,
            A4=A4,
        )

        # max pitch is necessary to prevent aliasing
        # adding sines with frequencies close and higher than Shannon frequency
        # will add lower frequencies which are undesired
        max_pitch = pitch_from_freq(0.25 * sample_rate, A4=A4)

        self.pitches = torch.linspace(
            20,
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

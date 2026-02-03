import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import numpy as np
import numpy.typing as npt

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
        noises: list,
        *,
        sample_rate: np.floating,
        buffer_size: np.int16,
        A4: np.float32,
        seed: int | None = None,
    ):
        if seed:
            np.random.seed(seed)

        self.nb_pitches = nb_pitches
        self.nb_phases = nb_phases

        self.noise_synth = NoiseSynth(
            sample_rate=sample_rate, buffer_size=buffer_size
        )
        # Saving the noise function and not the noise signal
        # Saving the noise signal might
        # self.noises = [
        #     noise(noise_synth.gaussian)
        #     for noise_func, noise_power in noises
        # ]
        self.noises = noises

        self.synth = WaveformSynth(
            sample_rate=sample_rate,
            buffer_size=buffer_size,
            A4=A4,
        )

        # max pitch is necessary to prevent aliasing
        # adding sines with frequencies close and higher than Shannon frequency
        # will add lower frequencies which are undesired
        max_pitch = pitch_from_freq(0.25 * sample_rate, A4=A4)  # type: ignore

        # TODO: consider performance benefits of torch tensors here
        self.pitches = np.linspace(
            0, max_pitch, num=nb_pitches, dtype=np.float32
        )
        self.phases = np.linspace(
            -1, 1, num=nb_phases, dtype=np.float32
        )

    def __len__(self):
        return self.nb_pitches * self.nb_phases * len(self.noises)

    def __getitem__(self, idx):
        pitch = self.pitches[
            idx // (self.nb_phases * len(self.noises))
        ]
        phase = self.phases[
            (idx // len(self.noises)) % self.nb_phases
        ]

        waveform = self.synth.gen_single(pitch, phase)  # type: ignore

        # if self.noises:  # in case of non empty noise list
        # noise_func, noise_power = self.noises[
        #     idx % len(self.noises)
        # ]
        # noise = (
        #     noise_func(self.noise_synth.gaussian) * noise_power
        # )
        # waveform += noise

        waveform /= waveform.std()
        waveform = torch.tensor(waveform, dtype=torch.float32)

        return waveform, pitch


class SineWaveformDataset(Dataset):
    def __init__(
        self,
        nb_pitches: int,
        nb_phases: int,
        /,
        noises: list,
        *,
        sample_rate: np.floating,
        buffer_size: np.int16,
        A4: np.float32,
        seed: int | None = None,
    ):
        if seed:
            np.random.seed(seed)

        self.nb_pitches = nb_pitches
        self.nb_phases = nb_phases

        self.noise_synth = NoiseSynth(
            sample_rate=sample_rate, buffer_size=buffer_size
        )

        self.synth = WaveformSynth(
            sample_rate=sample_rate,
            buffer_size=buffer_size,
            A4=A4,
        )

        # max pitch is necessary to prevent aliasing
        # adding sines with frequencies close and higher than Shannon frequency
        # will add lower frequencies which are undesired
        max_pitch = pitch_from_freq(0.25 * sample_rate, A4=A4)  # type: ignore

        # TODO: consider performance benefits of torch tensors here
        self.pitches = np.linspace(
            0, max_pitch, num=nb_pitches, dtype=np.float32
        )
        self.phases = np.linspace(
            -1, 1, num=nb_phases, dtype=np.float32
        )

    def __len__(self):
        return self.nb_pitches * self.nb_phases

    def __getitem__(self, idx):
        pitch = self.pitches[idx // self.nb_phases]
        phase = self.phases[idx % self.nb_phases]

        waveform = self.synth.gen_single(pitch, phase)  # type: ignore

        waveform /= waveform.std()
        waveform = torch.tensor(waveform, dtype=torch.float32)

        return waveform, pitch


class MonophonicModel(nn.Module):
    def __init__(self, buffer_size: np.int16):
        super(MonophonicModel, self).__init__()

        self.model = nn.Sequential(
            nn.Linear(buffer_size, 512),
            nn.Tanh(),
            # nn.Linear(512, 512),
            # nn.Tanh(),
            nn.Linear(512, 256),
            nn.Tanh(),
            # nn.Linear(256, 256),
            # nn.Tanh(),
            nn.Linear(256, 128),
            nn.Tanh(),
            # nn.Linear(128, 128),
            # nn.Tanh(),
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
                )
        return []

    def save_model(self, path: str):
        torch.save(self.model.state_dict(), path)

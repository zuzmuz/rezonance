import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import numpy as np
import numpy.typing as npt
import matplotlib.pyplot as plt

from src.utils import pitch_from_freq, freq_from_pitch, gen_wav_from_spectrum
from src.spectrum import SpectrumSynth


class SineWaveformDataset(Dataset):
    def __init__(
        self,
        nb_pitches: int,
        nb_phases: int,
        /,
        sample_rate: np.floating,
        buffer_size: np.int16,
        A4: np.float32,
    ):
        self.nb_pitches = nb_pitches
        self.nb_phases = nb_phases

        self.synth = SpectrumSynth(
            sample_rate=sample_rate, buffer_size=buffer_size, A4=A4
        )

        max_pitch = pitch_from_freq(0.25 * sample_rate, A4=A4) # type: ignore

        # TODO: consider performance benefits of torch tensors here
        self.pitches = np.linspace(
            0, max_pitch, num=nb_pitches, dtype=np.float32
        )
        self.phases = np.linspace(
            -1, 1, num=nb_phases, dtype=np.float32
        )

    def __len__(self):
        return self.nb_pitches * self.nb_phases**2

    def __getitem__(self, idx):
        pitch_idx = idx // (self.nb_phases**2)
        pitch = self.pitches[pitch_idx]
        phase_start_idx = (idx // self.nb_phases) % self.nb_phases
        phase_end_idx = idx % self.nb_phases
        phase_start = self.phases[phase_start_idx]
        phase_end = (
            self.synth.buffer_size
            * self.phases[phase_end_idx]
            / self.nb_phases
        )

        spectrum = self.synth.gen_sin(pitch, phase_start, phase_end) # type
        waveform = gen_wav_from_spectrum(spectrum)
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
        data_loader = DataLoader(dataset, batch_size=32, shuffle=True)
        history = []
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
                # epoch_accuracy += accuracy_score(
                #     batch_y.detach().numpy(),
                #     hat_y.detach().numpy().argmax(axis=1)
                # )
            history.append(epoch_loss / len(data_loader))
            if (epoch + 1) % 20 == 0:
                print(
                    f"Epoch {epoch + 1}: Mean Squared Error = {history[-1]:.5f}"
                )
        return history

    def save_model(self, path: str):
        torch.save(self.model.state_dict(), path)

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import matplotlib.pyplot as plt
import numpy as np
import matplotlib.pyplot as plt

from src.generator import Synthesizer


class WaveformDataset(Dataset):
    def __init__(
        self,
        length: int,
        /,
        sample_rate: np.floating,
        buffer_size: np.int16,
        seed: int | None = None,
    ):
        self.length = length
        if seed:
            np.random.seed(seed)

        self.synth = Synthesizer(
            sample_rate=sample_rate, buffer_size=buffer_size
        )

        self.pitches = np.linspace(1, 127, num=length, dtype=np.float32)

    def __len__(self):
        return self.length

    def __getitem__(self, idx):
        pitch = self.pitches[idx]
        spectrum = self.synth.generate_spectrum_from_pitch(pitch)
        waveform = self.synth.generate_waveform_from_spectrum(spectrum)
        waveform = torch.tensor(waveform, dtype=torch.float32)
        return waveform, pitch


class MonophonicModel(nn.Module):
    def __init__(self, buffer_size: int):
        super(MonophonicModel, self).__init__()

        self.model = nn.Sequential(
            nn.Linear(buffer_size, 512),
            nn.Tanh(),
            nn.Linear(512, 512),
            nn.Tanh(),
            nn.Linear(512, 256),
            nn.Tanh(),
            nn.Linear(256, 128),
            nn.Tanh(),
            nn.Linear(128, 1)
        )

    def forward(self, X):
        return self.model(X)


class Trainer:
    def __init__(self, model):
        self.model = model
        self.criterion = nn.MSELoss()
        self.optimizer = optim.Adam(self.model.parameters(), lr=0.001)

    def train(
        self,
        nb_epoch: int,
        dataset: Dataset
    ):
        data_loader = DataLoader(dataset, batch_size=32, shuffle=True)
        history = []
        for epoch in range(nb_epoch):
            self.model.train()
            epoch_loss = 0
            for batch_X, batch_y in data_loader:
                self.optimizer.zero_grad()
                hat_y = self.model.forward(batch_X)
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
                print(f"Epoch {epoch+1}: Mean Squared Error = {history[-1]:.5f}")
        return history

import time
from pathlib import Path
import pandas as pd

import torch
from torch import Tensor
from torch.types import Number
from torch.utils.data import Dataset

import torchaudio

from rezonance.logger import logger


class NSynthDataset(Dataset):
    def __init__(
        self,
        folder: Path,
        sample_rate: Number,
        buffer_size: int,
        element_per_file: int,
    ):
        self.sample_rate = sample_rate
        self.buffer_size = buffer_size
        self.element_per_file = element_per_file

        self.folder = folder
        json_file = folder / "examples.json"

        self.data = pd.read_json(json_file).T.reset_index()

        self.data: pd.DataFrame = self.data[  # type: ignore
            self.data["sample_rate"] == self.sample_rate
        ]

    def __len__(self):
        return self.data.shape[0] * self.element_per_file

    def __getitem__(self, idx):

        row = self.data.iloc[idx // self.element_per_file]

        pitch = row["pitch"]

        file_name = self.folder / "audio" / f"{row['note_str']}.wav"

        signal, _ = torchaudio.load(file_name)

        if signal.size(0) == 2:
            signal = signal.mean(0)
        else:
            signal = signal[0]
        signal /= signal.std()
        return signal[
            (2 + idx % self.element_per_file) * self.buffer_size : (
                2 + idx % self.element_per_file + 1
            )
            * self.buffer_size
        ], pitch

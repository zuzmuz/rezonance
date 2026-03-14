import time
from pathlib import Path
import pandas as pd

import torch
from torch import Tensor
from torch.types import Number 
from torch.utils.data import Dataset

import torchaudio
from torchcodec.decoders import AudioDecoder

from src.logger import logger

class NSynthDataset(Dataset):
    def __init__(
        self,
        folder: Path,
        sample_rate: Number,
        buffer_size: int,
    ):
        self.sample_rate = sample_rate
        self.buffer_size = buffer_size

        self.folder = folder
        json_file = folder / "examples.json"

        self.data = pd.read_json(json_file).T.reset_index()

        self.data: pd.DataFrame = self.data[  # type: ignore
            self.data["sample_rate"] == self.sample_rate
        ]

        file_name = self._get_file_name(0)

        c = time.perf_counter()
        decoder = AudioDecoder(file_name)
        logger.debug(f"{decoder.metadata.duration_seconds=}")
        logger.debug(f"it took {time.perf_counter() - c}")

        c = time.perf_counter()
        signal, _ = torchaudio.load(file_name)
        logger.debug(f"{signal.shape}")
        logger.debug(f"it took {time.perf_counter() - c}")

    def _get_file_name(self, idx) -> Path:
        return (
            self.folder
            / "audio"
            / f"{self.data['note_str'].iloc[idx]}.wav"
        )

    def __len__(self):
        return self.data.shape[0]

    def __getitem__(self, idx):
        logger.debug(self.data["note_str"].iloc[idx])
        pitch = self.data["pitch"].iloc[idx]
        file_name = (
            self.folder
            / "audio"
            / f"{self.data['note_str'].iloc[idx]}.wav"
        )

        signal, _ = torchaudio.load(file_name)

        return signal, pitch
